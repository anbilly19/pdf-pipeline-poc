"""LangGraph agent definition with persistent session memory.

Qwen3 sampling parameters
--------------------------
  temperature = 0.7, top_p = 0.8, top_k = 20  (Unsloth/Qwen official)

Thinking disabled via /nothink appended to last HumanMessage.

Loop guard: MAX_TOOL_ITERATIONS = 4
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
    no_think = _is_thinking_model(model)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        retrieval_ctx = state.get("retrieval_context", [])
        trimmed = trim_retrieval_context(retrieval_ctx)

        messages: list = [SystemMessage(content=system_prompt)]
        if trimmed:
            ctx_text = "\n\n".join(trimmed)
            messages.append(
                SystemMessage(content="Dokumentausz\u00fcge:\n\n" + ctx_text)
            )
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


def _system_prompt(domain_spec: DomainSpec | None) -> str:
    # Few-shot examples teach the model the exact expected behaviour.
    # This is more reliable than rule lists for small models like qwen3:4b.
    base = (
        "Du bist ein spezialisierter Vertragsanalyst. "
        "Beantworte AUSSCHLIESSLICH Fragen zum vorliegenden Vertragsdokument.\n\n"
        # Hard negative examples — show the model exactly what NOT to do
        "VERBOTEN — diese Antwortmuster sind absolut unzul\u00e4ssig:\n"
        "- \"Welche Hilfe ben\u00f6tigen Sie?\" → VERBOTEN\n"
        "- \"Was m\u00f6chten Sie mit diesem Dokument tun?\" → VERBOTEN\n"
        "- \"Ich kann Ihnen helfen mit: ...\" → VERBOTEN\n"
        "- Zusammenfassungen von Abschnitten ohne Bezug zur Frage → VERBOTEN\n"
        "- Antworten ohne vorherigen search_term-Aufruf → VERBOTEN\n\n"
        # Positive few-shot
        "BEISPIEL KORREKTE ANTWORT:\n"
        "Frage: \"Was ist die vereinbarte Verg\u00fctung?\"\n"
        "Antwort: \"Die Verg\u00fctung erfolgt nach Aufwand pro Stunde. "
        "Ein konkreter Betrag ist im Formular nicht eingetragen (Felder leer).\"\n\n"
        "BEISPIEL KORREKTE ANTWORT:\n"
        "Frage: \"Wann endet der Vertrag?\"\n"
        "Antwort: \"Der Vertrag endet am 31.12.2025 gem\u00e4\u00df \u00a715. "
        "Eine Verl\u00e4ngerungsoption bis 31.12.2026 muss bis 30.09.2025 aus\u00fcbt werden.\"\n\n"
        "ABLAUF PRO FRAGE:\n"
        "1. search_term aufrufen (max. 2x)\n"
        "2. Frage direkt und pr\u00e4zise auf Deutsch beantworten\n"
        "3. Wenn Information fehlt: \"Diese Information ist im Dokument nicht enthalten.\"\n"
        "4. FERTIG. Keine weiteren Angebote, keine R\u00fcckfragen."
    )
    if domain_spec and domain_spec.system_prompt:
        return f"{base}\n\n{domain_spec.system_prompt}"
    return base
