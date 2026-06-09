"""LangGraph agent definition with persistent session memory.

GPU / CPU policy
-----------------
All models use GPU by default (num_gpu=-1 = Ollama auto-detect).

Qwen3 thinking mode
-------------------
Qwen3 models default to extended <think>...</think> reasoning chains
which massively increase latency without improving RAG answer quality.
We disable thinking via extra_body={"think": False}.

Loop guard
----------
The agent is limited to MAX_TOOL_ITERATIONS tool calls per turn.
After the limit is hit the graph routes directly to END so the LLM
formulates an answer from whatever context it has accumulated,
preventing infinite retrieve→LLM→retrieve loops.
"""
from __future__ import annotations

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

_OLLAMA_NUM_GPU: int = int(os.environ.get("OLLAMA_NUM_GPU", "-1"))
_DEFAULT_NUM_CTX: int = 2048

# Maximum tool-call iterations per user turn before forcing a final answer.
MAX_TOOL_ITERATIONS: int = 4

_THINKING_MODEL_SUBSTRINGS = ("qwen3", "qwen2.5", "deepseek-r", "phi4-reasoning")


def _is_thinking_model(model: str) -> bool:
    lower = model.lower()
    return any(s in lower for s in _THINKING_MODEL_SUBSTRINGS)


def _count_tool_calls(messages: list) -> int:
    """Count how many AIMessages with tool_calls exist in the current message list."""
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
                    content=(
                        "Folgende Dokumentenausz\u00fcge wurden f\u00fcr diese Anfrage abgerufen:\n\n"
                        + ctx_text
                    )
                )
            )
        messages += state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END  # type: ignore[return-value]
        # Loop guard: if we've already hit the iteration cap, force END.
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
        kwargs["extra_body"] = {"think": False}

    return ChatOllama(**kwargs)


def _system_prompt(domain_spec: DomainSpec | None) -> str:
    base = (
        "Du bist ein pr\u00e4ziser Dokumentenanalyst. "
        "Beantworte Fragen ausschlie\u00dflich auf Basis des bereitgestellten Dokuments. "
        "Wenn die Information nicht im Dokument steht, sage das klar. "
        "Antworte auf Deutsch. "
        "WICHTIG: Nachdem du Abschnitte abgerufen hast, formuliere SOFORT eine Antwort. "
        "Rufe das Such-Tool nicht mehr als 2 Mal pro Frage auf."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
