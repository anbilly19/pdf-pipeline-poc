"""LangGraph agent definition."""
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.tools import build_tools
from src.models import Chunk

if TYPE_CHECKING:
    from src.agent.domain_config import DomainSpec
    from src.retrieval.retriever import BBoxRetriever


def build_agent(
    retriever: BBoxRetriever,
    provider: str = "ollama",
    model: str = "gemma4:e2b",
    domain_spec: DomainSpec | None = None,
    graph: object = None,
    all_chunks: list[Chunk] | None = None,
) -> object:
    """Construct and compile the LangGraph ReAct agent.

    Args:
        retriever: Hybrid FAISS+BM25 retriever.
        provider: LLM provider ('ollama' or 'openai').
        model: Model name.
        domain_spec: Optional domain specialist configuration.
        graph: Optional NetworkX DiGraph for graph-based expansion.
        all_chunks: Full ordered corpus for the expander.

    Returns:
        Compiled LangGraph runnable.
    """
    tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
    tool_node = ToolNode(tools)

    llm = _build_llm(provider, model)
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = _system_prompt(domain_spec)

    def call_model(state: MessagesState) -> dict:  # type: ignore[type-arg]
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END  # type: ignore[return-value]

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def _build_llm(provider: str, model: str) -> object:
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=0)
    from langchain_ollama import ChatOllama  # noqa: PLC0415
    return ChatOllama(model=model, temperature=0)


def _system_prompt(domain_spec: DomainSpec | None) -> str:  # type: ignore[return-value]
    base = (
        "Du bist ein präziser Dokumentenanalyst. "
        "Beantworte Fragen ausschließlich auf Basis des bereitgestellten Dokuments. "
        "Wenn die Information nicht im Dokument steht, sage das klar. "
        "Antworte auf Deutsch."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
