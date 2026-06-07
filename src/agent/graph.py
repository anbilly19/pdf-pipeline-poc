"""LangGraph agent definition with persistent session memory.

GPU / CPU policy
-----------------
All models use GPU by default (num_gpu=-1 = Ollama auto-detect).
phi4-mini-reasoning is explicitly forced to CPU-only (num_gpu=0) because
it cannot be reliably offloaded to GPU and produces garbage output in
hybrid mode.  OLLAMA_LLM_LIBRARY=cpu_avx2 is also set at runtime for
the phi model only, scoped to that process.
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
# Per-model CPU-only overrides
# ---------------------------------------------------------------------------
# Models listed here are always run on CPU regardless of OLLAMA_NUM_GPU.
# All other models use GPU auto-detect (-1).
_CPU_ONLY_MODELS: frozenset[str] = frozenset({
    "phi4-mini-reasoning:3.8b",
})

# Reduced context window for CPU-bound models to keep RAM usage manageable.
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

    GPU policy:
      - Default: num_gpu=-1 (Ollama auto-detects available VRAM).
      - phi4-mini-reasoning: num_gpu=0 + OLLAMA_LLM_LIBRARY=cpu_avx2,
        forced regardless of any env var, because the model cannot be
        reliably GPU-offloaded.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=0)

    from langchain_ollama import ChatOllama  # noqa: PLC0415

    cpu_only = model in _CPU_ONLY_MODELS
    if cpu_only:
        # Set library before ChatOllama is constructed so the Ollama server
        # picks up the cpu_avx2 backend for this process.
        os.environ["OLLAMA_LLM_LIBRARY"] = "cpu_avx2"
        num_gpu = 0
    else:
        # Let Ollama decide how many layers to offload based on available VRAM.
        os.environ.pop("OLLAMA_LLM_LIBRARY", None)
        num_gpu = int(os.environ.get("OLLAMA_NUM_GPU", "-1"))

    num_ctx = _MODEL_NUM_CTX.get(model, _DEFAULT_NUM_CTX)
    return ChatOllama(
        model=model,
        temperature=0,
        num_gpu=num_gpu,
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
