"""LangGraph agent definition with persistent session memory.

GPU / CPU policy
-----------------
All models use GPU by default (num_gpu=-1 = Ollama auto-detect).
No per-model CPU overrides are needed now that phi4-mini-reasoning
has been removed (it does not support the tools API).

Qwen3 thinking mode
-------------------
Qwen3 models default to extended <think>...</think> reasoning chains
which massively increase latency without improving RAG answer quality.
We disable thinking via extra_body={"think": False} when the model
name contains "qwen3" or "qwen2.5". This maps to Ollama's /api/chat
"think" parameter introduced in Ollama 0.6.5.
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

# GPU layers: -1 = Ollama auto-detect (use all VRAM available).
# Override via OLLAMA_NUM_GPU in .env (e.g. 0 for CPU-only).
_OLLAMA_NUM_GPU: int = int(os.environ.get("OLLAMA_NUM_GPU", "-1"))

# Default context window kept low to stay within 5 GB free RAM on this machine.
_DEFAULT_NUM_CTX: int = 512

# Models that support (and default to) extended thinking chains.
# We disable thinking by default for faster RAG responses.
_THINKING_MODEL_SUBSTRINGS = ("qwen3", "qwen2.5", "deepseek-r", "phi4-reasoning")


def _is_thinking_model(model: str) -> bool:
    lower = model.lower()
    return any(s in lower for s in _THINKING_MODEL_SUBSTRINGS)


def build_agent(
    retriever: BBoxRetriever,
    provider: str = "ollama",
    model: str = "qwen3:4b",
    domain_spec: DomainSpec | None = None,
    graph: object = None,
    all_chunks: list[Chunk] | None = None,
    checkpointer: Any = None,
    self_rag_enabled: bool = True,
    self_rag_bm25_gate: float = 0.5,
    num_ctx: int = _DEFAULT_NUM_CTX,
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

    llm = _build_llm(provider, model, num_ctx=num_ctx)
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
                        "Folgende Dokumentenauszüge wurden für diese Anfrage abgerufen:\n\n"
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


def _build_llm(provider: str, model: str, num_ctx: int = _DEFAULT_NUM_CTX) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=0)

    from langchain_ollama import ChatOllama  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "num_gpu": _OLLAMA_NUM_GPU,
        "num_ctx": num_ctx,
    }

    # Disable extended thinking chains on Qwen3 / DeepSeek-R / similar models.
    # This cuts first-token latency from 30-120s down to ~2-5s with no RAG quality loss.
    # Ollama passes extra_body fields directly to the /api/chat payload.
    if _is_thinking_model(model):
        kwargs["extra_body"] = {"think": False}

    return ChatOllama(**kwargs)


def _system_prompt(domain_spec: DomainSpec | None) -> str:
    base = (
        "Du bist ein präziser Dokumentenanalyst. "
        "Beantworte Fragen ausschließlich auf Basis des bereitgestellten Dokuments. "
        "Wenn die Information nicht im Dokument steht, sage das klar. "
        "Antworte auf Deutsch."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
