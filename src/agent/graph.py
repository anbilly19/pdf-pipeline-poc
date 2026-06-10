"""LangGraph agent — two-phase call_model.

Dual-model architecture
-----------------------
Phase 1  (tool calling):  llm_tools = qwen3.5:2b
Phase 2  (answer gen):    llm_plain = selected answer model

Self-RAG removed — same-machine LLM calls during retrieval caused
latency and occasional deadlocks. Pure FAISS+BM25+reranker pipeline.

Prompt design
-------------
- No 'Antwort:'/'Answer:' completion cue: gemma4 renders this inside
  the user turn causing empty responses.
- Language-aware: DE → DE prompt, EN → EN prompt.
- answer model has full num_predict=2048 budget.
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
_DEFAULT_NUM_CTX: int = 4096
_ANSWER_NUM_PREDICT: int = 2048
_DEFAULT_TOOL_MODEL = "qwen3.5:2b"
MAX_TOOL_ITERATIONS: int = 4

_THINKING_MODEL_SUBSTRINGS = ("qwen3", "qwen2.5", "deepseek-r", "phi4-reasoning")
_INSTRUCT_SUFFIXES = ("-instruct", ":instruct")

_METADATA_RE = re.compile(
    r"\[source:.*?\]|bboxes=\[.*?\]|image_path=.*?(?=\n|$)",
    re.DOTALL,
)
_SECTION_HEADER_RE = re.compile(
    r"^FOUND SECTIONS.*$|^GEFUNDENE ABSCHNITTE.*$|^--- Abschnitt \d+ ---$",
    re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_DE_INDICATORS = re.compile(
    r"\b(was|wie|wann|wo|wer|welch|warum|bitte|ist|sind|hat|haben|werden|kann|darf|muss|soll|gilt|steht|ab|bis|nach|vor|unter|gem\u00e4\u00df|vertrag|auftragnehmer|auftraggeber|leistung|verg\u00fctung|k\u00fcndigung|frist|zahlung)\b",
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
    return [
        p.strip() for p in parts
        if p.strip() and not re.match(r"FOUND SECTIONS|GEFUNDENE ABSCHNITTE", p.strip())
    ]


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
            "Unten stehen Ausz\u00fcge aus einem Vertrag.\n\n"
            "Regeln:\n"
            "- Beantworte die Frage ausschlie\u00dflich mit Informationen aus den Ausz\u00fcgen.\n"
            "- Zitiere Zahlen, Zeiten und Begriffe w\u00f6rtlich.\n"
            "- Wenn kein Abschnitt die Frage direkt beantwortet: "
            "'Die Antwort ist in den vorliegenden Ausz\u00fcgen nicht enthalten.'\n"
            "- Fasse dich kurz und pr\u00e4zise.\n\n"
            f"Vertragsausz\u00fcge:\n{ctx_text}\n\n"
            f"Frage: {question}"
        )
    return (
        "You are a precise contract analyst. "
        "Below are excerpts from a contract (text may be in German).\n\n"
        "Rules:\n"
        "- Answer using ONLY information from the excerpts.\n"
        "- Quote numbers, dates, and terms verbatim.\n"
        "- If no excerpt directly answers the question: "
        "'The answer is not contained in the provided excerpts.'\n"
        "- Be concise and precise.\n\n"
        f"Contract excerpts:\n{ctx_text}\n\n"
        f"Question: {question}"
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
    self_rag_enabled: bool = False,   # kept for API compat, ignored
    self_rag_bm25_gate: float = 0.5,  # kept for API compat, ignored
    num_ctx: int = _DEFAULT_NUM_CTX,
) -> Any:
    answer_model = model
    tc_model = tool_model if tool_model is not None else _DEFAULT_TOOL_MODEL
    if provider == "openai":
        tc_model = answer_model

    logger.info("Agent — tool_caller=%s  answer=%s", tc_model, answer_model)

    tools = build_tools(retriever, graph=graph, all_chunks=all_chunks)
    tool_node = ToolNode(tools)
    llm_tools = _build_llm(provider, tc_model, num_ctx=num_ctx).bind_tools(tools)
    llm_plain = _build_llm(provider, answer_model, num_ctx=num_ctx, num_predict=_ANSWER_NUM_PREDICT)
    no_think_tc = _is_thinking_model(tc_model)

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
            logger.info(
                "Phase 2 (answer): model=%s  chunks=%d  ctx=%d chars  lang=%s",
                answer_model, len(clean_chunks), len(ctx_text), _detect_language(question),
            )
            response = llm_plain.invoke([HumanMessage(content=prompt)])
            raw_content = response.content if isinstance(response.content, str) else ""
            answer = _strip_think(raw_content)
            if answer != raw_content:
                response = response.model_copy(update={"content": answer})
            return {"messages": [response]}

        system_prompt = (
            "You are a contract analyst. "
            "Call the search_term tool with a precise query to find relevant contract sections. "
            "Do NOT answer the question yourself — always call the tool first.\n"
            "Du bist ein Vertragsanalyst. "
            "Rufe search_term mit einer pr\u00e4zisen Suchanfrage auf. "
            "Beantworte die Frage NICHT selbst."
        )
        call_messages: list = [SystemMessage(content=system_prompt)]
        history = list(messages)
        if no_think_tc:
            history = _append_nothink(history)
        call_messages += history
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
        return ChatOllama(**base_kwargs, temperature=0.6, top_p=0.9, top_k=20)

    # gemma4 and instruct models: slightly raised temperature for more
    # natural, complete responses instead of cut-off answers
    return ChatOllama(**base_kwargs, temperature=0.1)
