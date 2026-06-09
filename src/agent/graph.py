"""LangGraph agent definition with persistent session memory.

Answer generation strategy
---------------------------
qwen3:4b ignores system prompt instructions when it has tool-call freedom
and defaults to its RLHF assistant persona (summaries, help offers, etc.).

Fix: after retrieval is done, we do NOT ask the agent to freely generate
an answer. Instead we call the LLM once with a hardcoded fill-in-the-blank
template that leaves no room for deviation:

    Dokumentauszüge:
    [chunks]

    Frage: <question>
    Antwort (max. 2 Sätze, nur aus obigem Text):

The model is forced to complete the sentence after "Antwort:" — it cannot
start a new paragraph, offer help, or write a summary.

The agent graph still handles multi-hop tool calls (search_term can be
called up to MAX_TOOL_ITERATIONS times). Only the FINAL answer step is
replaced by the template.

Qwen3 sampling parameters
--------------------------
  temperature = 0.7, top_p = 0.8, top_k = 20  (Unsloth/Qwen official)

Thinking disabled via /nothink appended to last HumanMessage.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
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


def _append_nothink(messages: list) -> list:
    """Append /nothink to the last HumanMessage so Ollama skips the think block."""
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if isinstance(result[i], HumanMessage):
            content = result[i].content
            if "/nothink" not in content:
                result[i] = HumanMessage(content=content + " /nothink")
            break
    return result


def _get_last_human_question(messages: list) -> str | None:
    """Return the text of the most recent HumanMessage (without /nothink tag)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content.replace(" /nothink", "").strip()
    return None


def _build_answer_prompt(question: str, ctx_text: str) -> str:
    """Build the fill-in-the-blank prompt used for final answer generation.

    The prompt ends with 'Antwort:' on its own line so the model is forced
    to complete it inline — no room to start a new topic or summary.
    """
    return (
        f"Dokumentausz\u00fcge:\n{ctx_text}\n\n"
        f"Frage: {question}\n"
        f"Antwort (maximal 2 S\u00e4tze, ausschlie\u00dflich aus den obigen Ausz\u00fcgen, "
        f"Zahlen und Begriffe w\u00f6rtlich zitieren):"
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
    # Plain LLM (no tools) used for the final answer template call.
    llm_plain = _build_llm(provider, model, num_ctx=num_ctx)
    no_think = _is_thinking_model(model)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        retrieval_ctx = state.get("retrieval_context", [])
        trimmed = trim_retrieval_context(retrieval_ctx)
        already_called_tools = _count_tool_calls(state["messages"]) > 0

        # --- Final answer phase: chunks available, no more tool calls needed ---
        if trimmed and already_called_tools:
            question = _get_last_human_question(state["messages"]) or ""
            ctx_text = "\n\n".join(trimmed)
            prompt = _build_answer_prompt(question, ctx_text)
            # Single HumanMessage ending with 'Antwort:' — model completes inline
            msg = HumanMessage(content=prompt + (" /nothink" if no_think else ""))
            response = llm_plain.invoke([msg])
            return {"messages": [response]}

        # --- Tool-call phase: let agent decide which tools to call ---
        system_prompt = (
            "Du bist ein Vertragsanalyst. "
            "Rufe search_term auf, um relevante Abschnitte zu finden. "
            "Formuliere KEINE Antwort selbst — das passiert nach dem Tool-Aufruf automatisch."
        )
        messages: list = [SystemMessage(content=system_prompt)]
        history = list(state["messages"])
        if no_think:
            history = _append_nothink(history)
        messages += history
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

    if _is_thinking_model(model):
        return ChatOllama(
            model=model,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            num_gpu=_OLLAMA_NUM_GPU,
            num_ctx=num_ctx,
        )

    return ChatOllama(
        model=model,
        temperature=0,
        num_gpu=_OLLAMA_NUM_GPU,
        num_ctx=num_ctx,
    )
