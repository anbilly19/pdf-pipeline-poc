"""LangGraph agent definition with persistent session memory.

Qwen3 thinking mode
-------------------
Disabled via ChatOllama(thinking=False) — supported in langchain-ollama >= 0.2.3.
Falls back gracefully if the param is rejected (older installs).

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
    system_prompt = _system_prompt(domain_spec)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        retrieval_ctx = state.get("retrieval_context", [])
        trimmed = trim_retrieval_context(retrieval_ctx)

        messages: list = [SystemMessage(content=system_prompt)]
        if trimmed:
            ctx_text = "\n\n".join(trimmed)
            messages.append(
                SystemMessage(
                    content="Dokumentauszüge:\n\n" + ctx_text
                )
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

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "num_gpu": _OLLAMA_NUM_GPU,
        "num_ctx": num_ctx,
    }

    if _is_thinking_model(model):
        # langchain-ollama >= 0.2.3 supports thinking=False natively.
        # This maps to Ollama's /api/chat "think": false parameter.
        try:
            kwargs["thinking"] = False
            test = ChatOllama(**kwargs)
            logger.info("Thinking mode disabled for model '%s'", model)
        except TypeError:
            # Older langchain-ollama — remove unsupported param and warn.
            kwargs.pop("thinking", None)
            logger.warning(
                "langchain-ollama does not support thinking=False — "
                "upgrade with: uv add langchain-ollama --upgrade"
            )

    return ChatOllama(**kwargs)


def _system_prompt(domain_spec: DomainSpec | None) -> str:
    base = (
        "Du bist ein Dokumentenanalyst. Deine einzige Aufgabe ist es, "
        "Fragen zum bereitgestellten Vertragsdokument zu beantworten.\n\n"
        "REGELN — halte dich strikt daran:\n"
        "1. Benutze IMMER zuerst das Tool 'search_term', um relevante Abschnitte abzurufen.\n"
        "2. Beantworte die Frage DIREKT auf Deutsch, basierend NUR auf den abgerufenen Abschnitten.\n"
        "3. Wenn ein Formularfeld leer ist oder eine Information nicht im Dokument steht, "
        "sage genau das — erfinde NICHTS.\n"
        "4. Gib KEINE Liste von m\u00f6glichen Aktionen oder Hilfsangeboten aus.\n"
        "5. Frage NICHT nach, was du tun soll. Beantworte die Frage sofort.\n"
        "6. Rufe 'search_term' maximal 2x pro Frage auf, dann antworte."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
