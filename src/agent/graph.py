"""LangGraph ReAct agent graph.

Supports OpenAI and Ollama at runtime.
The system prompt and model are injected at build time from a DomainSpec,
enabling the router to select the right specialist per query.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from src.retrieval.retriever import BBoxRetriever
from src.agent.tools import build_tools
from src.agent.domain_config import DomainSpec

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """Du bist ein Dokumentenassistent.

REGELN:
1. Rufe IMMER zuerst search_term auf mit top_k=10 und passenden Schl\u00fcsselw\u00f6rtern.
2. Lies ALLE zur\u00fcckgegebenen Textabschnitte vollst\u00e4ndig durch.
3. Antworte NUR mit Inhalten aus den Tool-Ergebnissen. Erfinde nichts.
4. Wenn kein Abschnitt die Frage beantwortet: 'Dazu enth\u00e4lt das Dokument keine explizite Information.'
5. Beende jede Antwort mit: [Quelle: Seite <N>, Bboxes: <bboxes>]
6. Antworte in der Sprache der Frage.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _build_llm(provider: str, model: str, temperature: float) -> object:
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "ollama":
        from langchain_ollama import ChatOllama  # noqa: PLC0415
        return ChatOllama(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'openai' or 'ollama'.")


def build_agent(
    retriever: BBoxRetriever,
    provider: str = "ollama",
    model: str = "qwen2.5:3b",
    temperature: float = 0.1,
    domain_spec: DomainSpec | None = None,
) -> object:
    """Build and compile a LangGraph ReAct agent.

    Args:
        retriever: FAISS-backed retriever supplying the search_term tool.
        provider: LLM provider ('ollama' or 'openai').
        model: Model name. Overridden by domain_spec.model if provided.
        temperature: Sampling temperature.
        domain_spec: If provided, uses its system_prompt and model.
                     Falls back to _DEFAULT_SYSTEM_PROMPT and model arg.
    """
    effective_model = domain_spec.model if domain_spec else model
    effective_prompt = domain_spec.system_prompt if domain_spec else _DEFAULT_SYSTEM_PROMPT
    domain_name = domain_spec.name if domain_spec else "general"

    tools = build_tools(retriever)
    llm = _build_llm(provider, effective_model, temperature)
    llm_with_tools = llm.bind_tools(tools)  # type: ignore[union-attr]
    tool_node = ToolNode(tools)
    system_message = SystemMessage(content=effective_prompt)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        turn_messages = _current_turn_messages(state["messages"])
        response = llm_with_tools.invoke([system_message] + turn_messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info(
        "Agent compiled (provider=%s, model=%s, domain=%s, tracing=%s)",
        provider, effective_model, domain_name,
        os.getenv("LANGCHAIN_TRACING_V2", "false"),
    )
    return compiled
