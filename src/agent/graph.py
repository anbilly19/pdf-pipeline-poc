"""LangGraph agent — two-phase call_model.

Dual-model architecture (Jun 2026)
------------------------------------
Phase 1  (tool calling):   llm_tools  = qwen3.5:2b          — native tool-call support
Phase 2  (answer gen):     llm_plain  = 4b-instruct variant  — better instruction following

This split avoids the problem where FieldMouse-AI/qwen3.5:4b-instruct
(fine-tuned for instructions, not function-calling) would sometimes skip
the tool call and answer directly from parametric knowledge.

Other fixes present
--------------------
- char_limit 12000: all chunks fit
- Longest-first trim strategy
- Language-aware answer prompt (DE/EN)
- Bilingual Phase-1 system prompt
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

# Default tool-calling model: small, fast, native tool support
_DEFAULT_TOOL_MODEL = "qwen3.5:2b"

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

_DE_INDICATORS = re.compile(
    r"\b(was|wie|wann|wo|wer|welch|warum|bitte|ist|sind|hat|haben|werden|kann|darf|muss|soll|gilt|steht|ab|bis|nach|vor|unter|gem\u00e4\u00df|vertrag|auftragnehmer|auftraggeber|leistung|verg\u00fcung|k\u00fcndigung|frist|zahlung)\b",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
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
            "Du bist ein pr\u00e4ziser Vertragsanalyst. "
            "Unten stehen Ausz\u00fcge aus einem Vertrag.\n"
            "Beantworte die Frage ausschlie\u00dflich mit den Informationen aus den Ausz\u00fcgen, "
            "die DIREKT die Frage beantworten.\n"
            "Ignoriere Abschnitte, die nicht zur Frage passen.\n"
            "Zahlen, Zeiten und Begriffe w\u00f6rtlich aus dem Text zitieren.\n"
            "Wenn kein Abschnitt die Frage direkt beantwortet, schreibe: "
            "'Die Antwort ist in den vorliegenden Ausz\u00fcgen nicht enthalten.'\n\n"
            f"Vertragsausz\u00fcge:\n{ctx_text}\n\n"
            f"Frage: {question}\n"
            "Antwort:"
        )
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
    tool_model: str | None = None,
    domain_spec: DomainSpec | None = None,
    graph: object = None,
    all_chunks: list[Chunk] | None = None,
    checkpointer: Any = None,
    self_rag_enabled: bool = True,
    self_rag_bm25_gate: float = 0.5,
    num_ctx: int = _DEFAULT_NUM_CTX,
) -> Any:
    """Build the two-phase LangGraph agent.

    Args:
        model:       Answer-generation model (Phase 2). Shown in UI as selected model.
        tool_model:  Tool-calling model (Phase 1). Defaults to qwen3.5:2b.
                     Pass None to use the same model for both phases.
    """
    answer_model = model
    tc_model = tool_model if tool_model is not None else _DEFAULT_TOOL_MODEL
    # For OpenAI, both phases use the same model (tool calling is native)
    if provider == "openai":
        tc_model = answer_model

    logger.info(
        "Agent models — tool_caller=%s  answer=%s  (thinking_tc=%s  thinking_ans=%s)",
        tc_model, answer_model,
        _is_thinking_model(tc_model), _is_thinking_model(answer_model),
    )

    tools = build_tools(
        retriever,
        graph=graph,
        all_chunks=all_chunks,
        self_rag_model=answer_model,
        self_rag_enabled=self_rag_enabled,
        self_rag_bm25_gate=self_rag_bm25_gate,
    )
    tool_node = ToolNode(tools)

    # Phase 1: small model wired to tools
    llm_tools = _build_llm(provider, tc_model, num_ctx=num_ctx).bind_tools(tools)
    # Phase 2: answer model, no tools, generous output budget
    llm_plain = _build_llm(provider, answer_model, num_ctx=num_ctx, num_predict=_ANSWER_NUM_PREDICT)

    # /nothink suffix needed only when the tool-calling model is a thinking model
    no_think_tc = _is_thinking_model(tc_model)

    def call_model(state: AgentState) -> dict:  # type: ignore[type-arg]
        messages = state["messages"]
        tool_outputs = _get_tool_outputs(messages)

        # ── Phase 2: we have retrieval results, generate the answer ──
        if tool_outputs:
            question = _get_last_human_question(messages) or ""
            trimmed = trim_retrieval_context(tool_outputs)
            if not trimmed:
                trimmed = tool_outputs[:5]
            clean_chunks = [_strip_chunk_metadata(t) for t in trimmed]
            ctx_text = "\n\n---\n\n".join(c for c in clean_chunks if c)
            prompt = _build_answer_prompt(question, ctx_text)
            logger.info(
                "Phase 2 (answer): model=%s  chunks=%d  ctx=%d chars  lang=%s  q=%.60s",
                answer_model, len(clean_chunks), len(ctx_text),
                _detect_language(question), question,
            )
            response = llm_plain.invoke([HumanMessage(content=prompt)])
            raw_content = response.content if isinstance(response.content, str) else ""
            answer = _strip_think(raw_content)
            if answer != raw_content:
                response = response.model_copy(update={"content": answer})
            return {"messages": [response]}

        # ── Phase 1: no tool results yet, call search_term ──
        system_prompt = (
            "You are a contract analyst / Du bist ein Vertragsanalyst.\n"
            "Call the search_term tool to find relevant contract sections. "
            "Do NOT write an answer yourself — that happens automatically after the tool call.\n"
            "Rufe search_term auf, um relevante Abschnitte zu finden. "
            "Formuliere KEINE Antwort selbst."
        )
        call_messages: list = [SystemMessage(content=system_prompt)]
        history = list(messages)
        if no_think_tc:
            history = _append_nothink(history)
        call_messages += history
        logger.info("Phase 1 (tool call): model=%s  q=%.60s", tc_model, messages[-1].content if messages else "")
        return {"messages": [llm_tools.invoke(call_messages)]}

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
