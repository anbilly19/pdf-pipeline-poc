"""LangGraph agent definition with persistent session memory.

Qwen3 sampling parameters
--------------------------
Official Qwen3 non-thinking recommendations (Unsloth / Qwen docs):
  temperature = 0.7   (NOT 0 — greedy causes loops and repetition)
  top_p       = 0.8
  top_k       = 20
  min_p       = 0.0

Thinking mode is disabled by injecting /nothink into the system
prompt — the only reliable method for Ollama regardless of
langchain-ollama version.

Loop guard
----------
MAX_TOOL_ITERATIONS caps tool calls per turn to prevent infinite loops.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.memory import AgentState, trim_retrieval_context
from src.agent.tools import build_tools
from src.models import Chunk

if TYPE_CHECKING:
    from src.agent.domain_config import DomainSpec
    from src.retrieval.retriever import BBoxRetriever

logger = logging.getLogger(__name__)

_OLLAMA_NUM_GPU: int = int(os.environ.get("OLLAMA_NUM_GPU", "-1"))
_DEFAULT_NUM_CTX: int = 2048
MAX_TOOL_ITERATIONS: int = 4

_THINKING_MODEL_SUBSTRINGS = ("qwen3", "qwen2.5", "deepseek-r", "phi4-reasoning")


def _is_thinking_model(model: str) -> bool:
    return any(s in model.lower() for s in _THINKING_MODEL_SUBSTRINGS)


def _count_tool_calls(messages: list) -> int:
    return sum(
        1 for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )


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
    system_prompt = _system_prompt(domain_spec, model)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        retrieval_ctx = state.get("retrieval_context", [])
        trimmed = trim_retrieval_context(retrieval_ctx)

        messages: list = [SystemMessage(content=system_prompt)]
        if trimmed:
            ctx_text = "\n\n".join(trimmed)
            messages.append(
                SystemMessage(content="Dokumentausz\u00fcge:\n\n" + ctx_text)
            )
        messages += state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END  # type: ignore[return-value]
        if _count_tool_calls(state["messages"]) >= MAX_TOOL_ITERATIONS:
            return END  # type: ignore[return-value]
        return "tools"

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

    if _is_thinking_model(model):
        # Qwen3 official non-thinking recommended params.
        # temperature=0 (greedy) must NOT be used — causes loops and repetition.
        return ChatOllama(
            model=model,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            num_gpu=_OLLAMA_NUM_GPU,
            num_ctx=num_ctx,
        )

    return ChatOllama(
        model=model,
        temperature=0,
        num_gpu=_OLLAMA_NUM_GPU,
        num_ctx=num_ctx,
    )


def _system_prompt(domain_spec: DomainSpec | None, model: str = "") -> str:
    # /nothink disables Qwen3 thinking mode at the prompt level —
    # works reliably in Ollama regardless of langchain-ollama version.
    no_think_tag = " /nothink" if _is_thinking_model(model) else ""

    base = (
        f"Du bist ein Dokumentenanalyst.{no_think_tag}\n\n"
        "Deine einzige Aufgabe: Fragen zum bereitgestellten Vertragsdokument beantworten.\n\n"
        "REGELN — strikt einhalten:\n"
        "1. Benutze IMMER zuerst 'search_term', um relevante Abschnitte abzurufen.\n"
        "2. Beantworte die Frage DIREKT auf Deutsch, nur auf Basis der abgerufenen Abschnitte.\n"
        "3. Leere Formularfelder oder fehlende Infos: sag genau das — erfinde NICHTS.\n"
        "4. Gib KEINE Liste m\u00f6glicher Aktionen oder Hilfsangebote aus.\n"
        "5. Frage NICHT nach, was du tun sollst. Antworte sofort.\n"
        "6. Maximal 2x 'search_term' pro Frage aufrufen, dann antworten."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
