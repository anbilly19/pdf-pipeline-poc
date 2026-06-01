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
   If the first search returns no useful results, call search_term again with
   alternative synonyms (e.g. "Beendigung" if "Vertragsende" found nothing,
   "Subunternehmer" if "Unterauftragnehmer" found nothing,
   "Vertragsstrafe" if "Konsequenzen Reaktionszeit" found nothing).

2. Read ALL chunks returned by the tool carefully before composing your answer.
   Do NOT stop at the first chunk — the most relevant passage may be further down.

3. STRICT RELEVANCE — only include a chunk in your answer if it DIRECTLY answers
   the question asked. Do NOT pad your answer with loosely related clauses about
   billing, scheduling, or other topics just because they appeared in the tool results.
   If only one chunk is truly relevant, cite only that one.

4. If no chunk directly answers the question after two searches, say ONLY:
   "Dazu enthält der Vertrag keine explizite Regelung."
   Do NOT invent, infer, or paraphrase content not directly present in a chunk.

5. ALWAYS end your answer with a source citation for EVERY chunk you actually used:
   [Quelle: Seite <N>, Bboxes: <bboxes>]
   Copy page numbers and bboxes verbatim from the tool results. One citation per chunk.

6. For tables use extract_table_to_csv.
7. To point the user to a specific location use highlight_section.
8. Respond in the same language the user writes in.
"""

_SYSTEM_MESSAGE = SystemMessage(content=_SYSTEM_PROMPT)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Extract only the current turn's messages (last HumanMessage onward).

    Scans backwards to find the last HumanMessage, then returns everything
    from that point forward. This isolates the current question from all
    prior conversation history, preventing context contamination.

    Args:
        messages: Full accumulated message list from AgentState.

    Returns:
        Slice starting at the last HumanMessage, or all messages if none found.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _build_llm(provider: str, model: str, temperature: float) -> object:
    """Instantiate the correct LLM based on provider.

    Args:
        provider: 'openai' or 'ollama'.
        model: Model identifier.
        temperature: Sampling temperature.

    Returns:
        LangChain chat model instance.
    """
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
    """Build and compile the LangGraph ReAct agent.

    Each call to the agent is fully isolated — the LLM only sees the system
    prompt and the current question's turn, never prior questions.

    Args:
        retriever: Initialised BBoxRetriever.
        provider: 'openai' or 'ollama'.
        model: Model identifier matching the provider.
        temperature: Sampling temperature.

    Returns:
        Compiled LangGraph graph (no checkpointer — stateless per invocation).
    """
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

    compiled = graph.compile()  # no checkpointer — fully stateless
    logger.info(
        "Agent compiled (provider=%s, model=%s, tracing=%s, stateless=True)",
        provider, model, os.getenv("LANGCHAIN_TRACING_V2", "false"),
    )
    return compiled
