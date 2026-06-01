"""LangGraph ReAct agent graph.

Builds a compiled StateGraph with:
  - tool_node: executes bound tools
  - agent_node: ChatOllama with tools bound
  - conditional edge: loop until no more tool calls
  - MemorySaver checkpoint: preserves conversation across turns
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from src.retrieval.retriever import BBoxRetriever
from src.agent.tools import build_tools

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an intelligent document assistant for SG Magazin.
Answer questions based on the document content using the available tools.

Rules:
- Always use search_term to find relevant passages before answering.
- Always cite the page number and bounding box coordinates of your source.
- For tables use extract_table_to_csv.
- To point the user to a specific location use highlight_section.
- Respond in the same language the user writes in.
"""


class AgentState(TypedDict):
    """State schema for the ReAct agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(
    retriever: BBoxRetriever,
    model: str = "gemma4:e2b",
    temperature: float = 0.1,
) -> object:
    """Build and compile the LangGraph ReAct agent.

    Args:
        retriever: Initialised BBoxRetriever connected to the vector store.
        model: Ollama model identifier.
        temperature: LLM sampling temperature (low = more deterministic).

    Returns:
        Compiled LangGraph graph with MemorySaver checkpointing.
    """
    tools = build_tools(retriever)

    llm = ChatOllama(model=model, temperature=temperature)
    llm_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        """Invoke the LLM with the current message history."""
        messages = state["messages"]
        from langchain_core.messages import SystemMessage  # noqa: PLC0415
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Route to tools if the LLM made tool calls, else end."""
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

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Agent graph compiled (model=%s)", model)
    return compiled
