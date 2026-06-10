"""LangGraph agent — two-phase call_model.

Phase 1 (no ToolMessage yet): agent calls search_term.
Phase 2 (ToolMessage in history): extract chunks, build answer prompt,
        call llm_plain once. Language of the question is detected and
        the prompt is written in the same language.

Key fixes (Jun 2026)
--------------------
- char_limit raised to 12000: all 23 chunks now fit, no relevant
  content discarded.
- trim strategy changed: keeps highest-content (longest) chunks instead
  of most-recent, preventing early-document sections from being dropped.
- Language detection: English questions now get an English prompt so
  'When must...' queries are answered correctly.
- Bilingual Phase-1 system prompt so tool call fires for any language.
- _ANSWER_NUM_PREDICT 1024: enough budget for verbose models.
"""
from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
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
_ANSWER_NUM_PREDICT: int = 1024

_THINKING_MODEL_SUBSTRINGS = ("qwen3", "qwen2.5", "deepseek-r", "phi4-reasoning")
_INSTRUCT_SUFFIXES = ("-instruct", ":instruct")

_METADATA_RE = re.compile(
    r"\[source:.*?\]|bboxes=\[.*?\]|image_path=.*?(?=\n|$)",
    re.DOTALL,
)
_SECTION_HEADER_RE = re.compile(
    r"^GEFUNDENE ABSCHNITTE.*$|^--- Abschnitt \d+ ---$",
    re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# German indicator words — if any appear in the question, use DE prompt
_DE_INDICATORS = re.compile(
    r"\b(was|wie|wann|wo|wer|welch|warum|bitte|ist|sind|hat|haben|werden|kann|darf|muss|soll|gilt|steht|ab|bis|nach|vor|unter|gem\u00e4\u00df|vertrag|auftragnehmer|auftraggeber|leistung|vergütung|k\u00fcndigung|frist|zahlung)\b",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    """Return 'de' if the text looks German, else 'en'."""
    return "de" if _DE_INDICATORS.search(text) else "en"


def _is_thinking_model(model: str) -> bool:
    lower = model.lower()
    if any(lower.endswith(s) for s in _INSTRUCT_SUFFIXES):
        return False
    return any(s in lower for s in _THINKING_MODEL_SUBSTRINGS)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _count_tool_calls(messages: list) -> int:
    return sum(
        1 for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )


def _split_tool_output(raw: str) -> list[str]:
    parts = re.split(r"--- Abschnitt \d+ ---", raw)
    return [p.strip() for p in parts if p.strip() and not p.strip().startswith("GEFUNDENE")]


def _get_tool_outputs(messages: list) -> list[str]:
    chunks: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.content:
            content = m.content if isinstance(m.content, str) else str(m.content)
            chunks.extend(_split_tool_output(content))
    return chunks


def _strip_chunk_metadata(text: str) -> str:
    cleaned = _METADATA_RE.sub("", text)
    cleaned = _SECTION_HEADER_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _append_nothink(messages: list) -> list:
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if isinstance(result[i], HumanMessage):
            content = result[i].content
            if "/nothink" not in content:
                result[i] = HumanMessage(content=content + " /nothink")
            break
    return result


def _get_last_human_question(messages: list) -> str | None:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content.replace(" /nothink", "").strip()
    return None


def _build_answer_prompt(question: str, ctx_text: str) -> str:
    lang = _detect_language(question)
    if lang == "de":
        return (
            "Du bist ein präziser Vertragsanalyst. "
            "Unten stehen Auszüge aus einem Vertrag.\n"
            "Beantworte die Frage ausschließlich mit den Informationen aus den Auszügen, "
            "die DIREKT die Frage beantworten.\n"
            "Ignoriere Abschnitte, die nicht zur Frage passen.\n"
            "Zahlen, Zeiten und Begriffe wörtlich aus dem Text zitieren.\n"
            "Wenn kein Abschnitt die Frage direkt beantwortet, schreibe: "
            "'Die Antwort ist in den vorliegenden Auszügen nicht enthalten.'\n\n"
            f"Vertragsauszüge:\n{ctx_text}\n\n"
            f"Frage: {question}\n"
            "Antwort:"
        )
    # English fallback
    return (
        "You are a precise contract analyst. "
        "Below are excerpts from a contract (which may be in German).\n"
        "Answer the question using ONLY the information in the excerpts that directly addresses it.\n"
        "Ignore sections that are not relevant to the question.\n"
        "Quote numbers, dates, and terms verbatim from the text.\n"
        "If no excerpt directly answers the question, write: "
        "'The answer is not contained in the provided excerpts.'\n\n"
        f"Contract excerpts:\n{ctx_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_agent(
    retriever: BBoxRetriever,
    provider: str = "ollama",
    model: str = "FieldMouse-AI/qwen3.5:4b-instruct",
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
    llm_with_tools = _build_llm(provider, model, num_ctx=num_ctx).bind_tools(tools)
    llm_plain = _build_llm(provider, model, num_ctx=num_ctx, num_predict=_ANSWER_NUM_PREDICT)
    logger.info("Agent model: %s (thinking=%s)", model, _is_thinking_model(model))
    no_think = _is_thinking_model(model)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        messages = state["messages"]
        tool_outputs = _get_tool_outputs(messages)

        if tool_outputs:
            question = _get_last_human_question(messages) or ""
            trimmed = trim_retrieval_context(tool_outputs)
            if not trimmed:
                trimmed = tool_outputs[:5]
            clean_chunks = [_strip_chunk_metadata(t) for t in trimmed]
            ctx_text = "\n\n---\n\n".join(c for c in clean_chunks if c)
            prompt = _build_answer_prompt(question, ctx_text)
            msg = HumanMessage(content=prompt)
            logger.info(
                "Final-answer phase: %d chunks, ctx ~%d chars, lang=%s, question=%.60s",
                len(clean_chunks), len(ctx_text), _detect_language(question), question,
            )
            response = llm_plain.invoke([msg])
            raw_content = response.content if isinstance(response.content, str) else ""
            answer = _strip_think(raw_content)
            if answer != raw_content:
                response = response.model_copy(update={"content": answer})
            return {"messages": [response]}

        # Phase 1 — bilingual so tool call fires for any question language
        system_prompt = (
            "You are a contract analyst / Du bist ein Vertragsanalyst.\n"
            "Call the search_term tool to find relevant contract sections. "
            "Do NOT write an answer yourself — that happens automatically after the tool call.\n"
            "Rufe search_term auf, um relevante Abschnitte zu finden. "
            "Formuliere KEINE Antwort selbst."
        )
        call_messages: list = [SystemMessage(content=system_prompt)]
        history = list(messages)
        if no_think:
            history = _append_nothink(history)
        call_messages += history
        return {"messages": [llm_with_tools.invoke(call_messages)]}

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


def _build_llm(
    provider: str,
    model: str,
    num_ctx: int = _DEFAULT_NUM_CTX,
    num_predict: int | None = None,
) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415
        return ChatOpenAI(model=model, temperature=0)

    from langchain_ollama import ChatOllama  # noqa: PLC0415

    base_kwargs: dict[str, Any] = {
        "model": model,
        "num_gpu": _OLLAMA_NUM_GPU,
        "num_ctx": num_ctx,
    }
    if num_predict is not None:
        base_kwargs["num_predict"] = num_predict

    if _is_thinking_model(model):
        return ChatOllama(**base_kwargs, temperature=0.7, top_p=0.8, top_k=20)

    return ChatOllama(**base_kwargs, temperature=0)
