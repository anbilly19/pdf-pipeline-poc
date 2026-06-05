"""LangGraph ReAct agent graph — supports OpenAI and Ollama at runtime.

LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY are set in the environment.

Each invocation is fully isolated — no cross-turn memory. The agent only
sees the system prompt + messages from the current question's turn
(the last HumanMessage and any tool call/result messages that follow it).
This prevents context contamination where prior questions bleed into the
current query's search term generation.
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

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an intelligent document assistant specialised in German legal and contract documents.
Answer questions based ONLY on the document content using the available tools.

Mandatory rules — never skip any of these:

1. ALWAYS call search_term first with relevant German keywords before answering.
   Search strategy:
   a) Start with the EXACT legal term or section topic (e.g. "K\u00fcndigung", "Vertragsstrafe", "Gew\u00e4hrleistung").
   b) If the question refers to a named topic like "K\u00fcndigung", also search for its synonyms:
      "Beendigung", "Vertragsende", "Laufzeit" — whichever apply.
   c) If the first search returns off-topic results (e.g. you asked about K\u00fcndigung but
      got results about Verg\u00fctung or Verl\u00e4ngerung), do a SECOND search_term call
      with different keywords before answering.
   d) NEVER use results from a search about topic X to answer a question about topic Y.
      If retrieved chunks are about a different clause, discard them and search again.

2. Read ALL chunks returned by the tool carefully before composing your answer.
   Do NOT stop at the first chunk — the most relevant passage may be further down.
   Section numbers in the document (e.g. \u00a715, \u00a713.1) are strong signals of relevance.

3. STRICT RELEVANCE — only include a chunk in your answer if it DIRECTLY answers
   the question asked. Do NOT pad your answer with loosely related clauses.
   If a chunk is about a different topic than the question, skip it entirely.

4. If no chunk directly answers the question after two searches, say ONLY:
   "Dazu enth\u00e4lt der Vertrag keine explizite Regelung."
   Do NOT invent, infer, or paraphrase content not directly present in a chunk.

5. ALWAYS end your answer with a source citation for EVERY chunk you actually used:
   [Quelle: Seite <N>, Bboxes: <bboxes>]
   Copy page numbers and bboxes VERBATIM from the tool results — do not modify them.
   One citation per chunk. Do NOT cite chunks you did not use in your answer.

6. Do NOT mention how many results were returned or count tool outputs in your answer.
   Just answer the question and cite sources.

7. For tables use extract_table_to_csv.
8. To point the user to a specific location use highlight_section.
9. Respond in the same language the user writes in.
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
    model: str = "gemma4:e2b",
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
