"""LangGraph ReAct agent graph — supports OpenAI and Ollama at runtime.

LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY are set in the environment.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
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
   "Subunternehmer" if "Unterauftragnehmer" found nothing).
2. ALWAYS end your answer with a source citation in EXACTLY this format:
   [Quelle: Seite <N>, Bboxes: <bboxes>]
   Copy the page number and bboxes verbatim from the tool result.
3. If no relevant passage is found after two searches, say so explicitly.
   Do NOT invent or paraphrase content not present in the tool results.
4. For tables use extract_table_to_csv.
5. To point the user to a specific location use highlight_section.
6. Respond in the same language the user writes in.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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

    Args:
        retriever: Initialised BBoxRetriever.
        provider: 'openai' or 'ollama'.
        model: Model identifier matching the provider.
        temperature: Sampling temperature.

    Returns:
        Compiled LangGraph graph with MemorySaver checkpointing.
    """
    tools = build_tools(retriever)
    llm = _build_llm(provider, model, temperature)
    llm_with_tools = llm.bind_tools(tools)  # type: ignore[union-attr]
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        from langchain_core.messages import SystemMessage  # noqa: PLC0415
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
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

    compiled = graph.compile(checkpointer=MemorySaver())
    logger.info("Agent compiled (provider=%s, model=%s, tracing=%s)", provider, model, os.getenv("LANGCHAIN_TRACING_V2", "false"))
    return compiled
