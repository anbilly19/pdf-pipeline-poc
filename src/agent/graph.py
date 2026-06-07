"""LangGraph agent definition with persistent session memory.

Roadmap #5: integrates MemorySaver checkpointing with isolated
conversation-history and retrieval-context keys.
Roadmap #6: build_tools now receives Self-RAG config kwargs.

State schema (see memory.AgentState)
-------------------------------------
    messages           -- conversation history (append-only via add_messages)
    retrieval_context  -- raw bbox/text snippets from tool calls (replaced)

Memory isolation guarantee
---------------------------
The LLM node only reads from ``state["messages"]``.  Retrieval context
is stored in ``state["retrieval_context"]`` and is only injected into the
chat window as a trimmed SystemMessage -- never as part of the raw message
history.  This prevents bbox coordinate bleed between turns.

Backward compatibility
-----------------------
If ``checkpointer=None`` (the default), the graph behaves exactly as
before: stateless, no memory, one agent object per query.  Pass a
MemorySaver (or SqliteSaver) to enable multi-turn memory.

GPU / CPU note
--------------
Ollama defaults to CPU-only (num_gpu=0) + cpu_avx2 backend (set in
src/silence.py which runs first).  phi4-mini-reasoning also gets a reduced
context window (2048 tokens) to keep RAM usage manageable on CPU.
Set OLLAMA_NUM_GPU=-1 in .env to re-enable GPU auto-detect.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.memory import AgentState, trim_retrieval_context
from src.agent.tools import build_tools
from src.models import Chunk

if TYPE_CHECKING:
    from src.agent.domain_config import DomainSpec
    from src.retrieval.retriever import BBoxRetriever

# ---------------------------------------------------------------------------
# GPU control
# ---------------------------------------------------------------------------
# 0  = CPU-only (default, safe for all models)
# -1 = Ollama auto-detect (use when GPU has enough VRAM)
_OLLAMA_NUM_GPU: int = int(os.environ.get("OLLAMA_NUM_GPU", "0"))

# Per-model context window overrides.
# phi4-mini-reasoning is a reasoning model that generates verbose chain-of-
# thought; a large num_ctx eats RAM fast on CPU, so we cap it at 2048.
_MODEL_NUM_CTX: dict[str, int] = {
    "phi4-mini-reasoning:3.8b": 2048,
}
_DEFAULT_NUM_CTX: int = 4096


def build_agent(
    retriever: BBoxRetriever,
    provider: str = "ollama",
    model: str = "gemma4:e2b",
    domain_spec: DomainSpec | None = None,
    graph: object = None,
    all_chunks: list[Chunk] | None = None,
    checkpointer: Any = None,
    self_rag_enabled: bool = True,
    self_rag_bm25_gate: float = 0.5,
) -> Any:
    """Construct and compile the LangGraph ReAct agent.

    Args:
        retriever: Hybrid FAISS+BM25 retriever.
        provider: LLM provider (``'ollama'`` or ``'openai'``).
        model: Model name string.
        domain_spec: Optional domain specialist configuration.
        graph: Optional NetworkX DiGraph for graph-based chunk expansion.
        all_chunks: Full ordered corpus for the graph expander.
        checkpointer: LangGraph checkpoint saver for persistent memory.
            Pass ``None`` (default) for stateless / single-query mode.
        self_rag_enabled: Enable the Self-RAG relevance filter. Set False
            to disable all extra Ollama calls (useful in tests / low-RAM).
        self_rag_bm25_gate: BM25 gate threshold; chunks above this score
            skip the Self-RAG LLM call.

    Returns:
        Compiled LangGraph runnable.
    """
    tools = build_tools(
        retriever,
        graph=graph,
        all_chunks=all_chunks,
        self_rag_model=model,
        self_rag_enabled=self_rag_enabled,
        self_rag_bm25_gate=self_rag_bm25_gate,
    )
    tool_node = ToolNode(tools)

    llm = _build_llm(provider, model)
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = _system_prompt(domain_spec)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        retrieval_ctx = state.get("retrieval_context", [])
        trimmed = trim_retrieval_context(retrieval_ctx)

        messages: list = [SystemMessage(content=system_prompt)]
        if trimmed:
            ctx_text = "\n\n".join(trimmed)
            messages.append(
                SystemMessage(
                    content=(
                        "Folgende Dokumentenausz\u00fcge wurden f\u00fcr diese Anfrage abgerufen:\n\n"
                        + ctx_text
                    )
                )
            )
        messages += state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END  # type: ignore[return-value]

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


def _build_llm(provider: str, model: str) -> Any:
    """Instantiate the LLM.

    For Ollama:
    - num_gpu=0 forces CPU-only (OLLAMA_NUM_GPU env var overrides).
    - OLLAMA_LLM_LIBRARY=cpu_avx2 is set in silence.py before any imports.
    - num_ctx is set per-model: phi4-mini-reasoning gets 2048 to cap RAM
      usage; all other models get 4096 (Ollama default).

    Override via .env:
        OLLAMA_NUM_GPU=-1   # auto GPU
        OLLAMA_NUM_GPU=0    # CPU-only (default)
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=0)

    from langchain_ollama import ChatOllama  # noqa: PLC0415
    num_ctx = _MODEL_NUM_CTX.get(model, _DEFAULT_NUM_CTX)
    return ChatOllama(
        model=model,
        temperature=0,
        num_gpu=_OLLAMA_NUM_GPU,
        num_ctx=num_ctx,
    )


def _system_prompt(domain_spec: DomainSpec | None) -> str:
    base = (
        "Du bist ein pr\u00e4ziser Dokumentenanalyst. "
        "Beantworte Fragen ausschlie\u00dflich auf Basis des bereitgestellten Dokuments. "
        "Wenn die Information nicht im Dokument steht, sage das klar. "
        "Antworte auf Deutsch."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
