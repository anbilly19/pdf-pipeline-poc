"""LangGraph agent definition with persistent session memory.

Roadmap #5: integrates MemorySaver checkpointing with isolated
conversation-history and retrieval-context keys.

State schema (see memory.AgentState)
-------------------------------------
    messages           -- conversation history (append-only via add_messages)
    retrieval_context  -- raw bbox/text snippets from tool calls (replaced)

Memory isolation guarantee
---------------------------
The LLM node only reads from ``state["messages"]``.  Retrieval context
is stored in ``state["retrieval_context"]`` and is only injected into the
chat window as a trimmed SystemMessage — never as part of the raw message
history.  This prevents bbox coordinate bleed between turns.

Backward compatibility
-----------------------
If ``checkpointer=None`` (the default), the graph behaves exactly as
before: stateless, no memory, one agent object per query.  Pass a
MemorySaver (or SqliteSaver) to enable multi-turn memory.
"""
from __future__ import annotations

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


def build_agent(
    retriever: BBoxRetriever,
    provider: str = "ollama",
    model: str = "gemma4:e2b",
    domain_spec: DomainSpec | None = None,
    graph: object = None,
    all_chunks: list[Chunk] | None = None,
    checkpointer: Any = None,
) -> Any:
    """Construct and compile the LangGraph ReAct agent.

    Args:
        retriever: Hybrid FAISS+BM25 retriever.
        provider: LLM provider (``'ollama'`` or ``'openai'``).
        model: Model name string.
        domain_spec: Optional domain specialist configuration.
        graph: Optional NetworkX DiGraph for graph-based chunk expansion.
        all_chunks: Full ordered corpus for the graph expander.
        checkpointer: LangGraph checkpoint saver for persistent memory.
            Pass ``None`` (default) for stateless / single-query mode.
            Pass a ``MemorySaver`` or ``SqliteSaver`` for multi-turn sessions.

    Returns:
        Compiled LangGraph runnable.  If ``checkpointer`` is not None,
        the runnable expects a ``config`` kwarg with a ``thread_id`` key
        (use :func:`src.agent.memory.make_thread_config`).
    """
    tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
    tool_node = ToolNode(tools)

    llm = _build_llm(provider, model)
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = _system_prompt(domain_spec)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        # Build context prefix from retrieval_context key (budget-trimmed)
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


def _build_llm(provider: str, model: str) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=0)
    from langchain_ollama import ChatOllama  # noqa: PLC0415
    return ChatOllama(model=model, temperature=0)


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
