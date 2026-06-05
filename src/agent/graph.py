"""LangGraph ReAct agent graph — supports OpenAI and Ollama at runtime."""
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

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Du bist ein Dokumentenassistent f\u00fcr deutsche Vertr\u00e4ge.

REGELN (alle zwingend):

1. Rufe IMMER zuerst search_term mit deutschen Schl\u00fcsselw\u00f6rtern auf, bevor du antwortest.
   - Suche nach dem genauen Begriff aus der Frage (z.B. "Verz\u00f6gerung", "K\u00fcndigung", "Laufzeit").
   - Falls die erste Suche keine passenden Ergebnisse liefert, suche nochmal mit anderen Begriffen.

2. Beantworte die Frage NUR mit dem Text aus den Tool-Ergebnissen.
   - Zitiere Abschnitte w\u00f6rtlich oder fasse sie zusammen.
   - Erw\u00e4hne KEINE Gesetze (BGB, HGB etc.), die nicht im gefundenen Text stehen.
   - Verwende KEINE Paragraphennummern, die nicht im gefundenen Text stehen.
   - Erfinde NICHTS.

3. Wenn der gefundene Text die Frage nicht beantwortet, antworte NUR:
   "Dazu enth\u00e4lt der Vertrag keine explizite Regelung."

4. Beende jede Antwort mit Quellenangaben aus den Tool-Ergebnissen:
   [Quelle: Seite <N>, Bboxes: <bboxes>]
   Kopiere Seitenzahl und Bboxes exakt aus den Tool-Ergebnissen.

5. Antworte in der Sprache der Frage.
"""

_SYSTEM_MESSAGE = SystemMessage(content=_SYSTEM_PROMPT)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Extract only the current turn's messages (last HumanMessage onward)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _build_llm(provider: str, model: str, temperature: float) -> object:
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"],
        )
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
) -> object:
    tools = build_tools(retriever)
    llm = _build_llm(provider, model, temperature)
    llm_with_tools = llm.bind_tools(tools)  # type: ignore[union-attr]
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        turn_messages = _current_turn_messages(state["messages"])
        response = llm_with_tools.invoke([_SYSTEM_MESSAGE] + turn_messages)
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
        "Agent compiled (provider=%s, model=%s, tracing=%s, stateless=True)",
        provider, model, os.getenv("LANGCHAIN_TRACING_V2", "false"),
    )
    return compiled
