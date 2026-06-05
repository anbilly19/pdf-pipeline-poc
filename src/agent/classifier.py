"""Document type classifier.

Runs once at index time on the first N chunks of a document.
Uses keyword matching first (fast, no LLM), then falls back to
a lightweight LLM call if confidence is too low.

Returns a DocTypeConfig that is persisted to outputs/faiss_index/domain_config.json.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.agent.domain_config import DocTypeConfig, list_available_doc_types, load_doc_type, save_active_config

if TYPE_CHECKING:
    from src.models import Chunk

logger = logging.getLogger(__name__)

_SAMPLE_CHUNKS = 15       # number of chunks to sample for classification
_KW_THRESHOLD = 3         # min keyword hits to accept a doc type without LLM
_LLM_FALLBACK_MODEL = "qwen2.5:3b"


def _keyword_score(text: str, hints: list[str]) -> int:
    """Count how many detection hints appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for hint in hints if hint.lower() in text_lower)


def _classify_by_keywords(sample_text: str) -> tuple[str, int]:
    """Score all doc types by keyword hits. Returns (best_doc_type, score)."""
    best_type = "fallback"
    best_score = 0
    for doc_type in list_available_doc_types():
        if doc_type == "fallback":
            continue
        config = load_doc_type(doc_type)
        score = _keyword_score(sample_text, config.detection_hints)
        logger.debug("Doc type '%s' keyword score: %d", doc_type, score)
        if score > best_score:
            best_score = score
            best_type = doc_type
    return best_type, best_score


def _classify_by_llm(sample_text: str, provider: str, model: str) -> str:
    """Ask an LLM to classify the document type.

    Returns one of the available doc type strings.
    """
    available = [t for t in list_available_doc_types() if t != "fallback"]
    options_str = ", ".join(available)

    prompt = (
        f"Classify this document into exactly one of these types: {options_str}.\n"
        f"Respond with ONLY the type name, nothing else.\n\n"
        f"Document excerpt:\n{sample_text[:2000]}"
    )

    try:
        if provider == "openai":
            import os  # noqa: PLC0415
            from langchain_openai import ChatOpenAI  # noqa: PLC0415
            llm = ChatOpenAI(model=model, temperature=0.0, api_key=os.environ["OPENAI_API_KEY"])
        else:
            from langchain_ollama import ChatOllama  # noqa: PLC0415
            llm = ChatOllama(model=model, temperature=0.0)

        response = llm.invoke(prompt)
        raw = response.content.strip().lower()
        # Extract first matching doc type from response
        for doc_type in available:
            if doc_type in raw:
                logger.info("LLM classified document as: %s", doc_type)
                return doc_type
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM classification failed: %s", e)

    return "fallback"


def classify_document(
    chunks: list[Chunk],
    provider: str = "ollama",
    model: str = _LLM_FALLBACK_MODEL,
    save: bool = True,
) -> DocTypeConfig:
    """Classify a document from its chunks and return the matching DocTypeConfig.

    Strategy:
    1. Sample first N chunks into a text blob.
    2. Score all doc types by keyword hit count.
    3. If best score >= threshold: accept without LLM.
    4. Otherwise: call LLM classifier for a final decision.
    5. Persist the result to outputs/faiss_index/domain_config.json.

    Args:
        chunks: All chunks from the indexed document.
        provider: LLM provider for fallback ('ollama' or 'openai').
        model: Model name for LLM fallback classification.
        save: Whether to persist the config to disk.

    Returns:
        The matching DocTypeConfig.
    """
    sample = chunks[:_SAMPLE_CHUNKS]
    sample_text = "\n".join(c.text for c in sample)

    doc_type, score = _classify_by_keywords(sample_text)
    logger.info("Keyword classification: doc_type='%s' score=%d (threshold=%d)", doc_type, score, _KW_THRESHOLD)

    if score < _KW_THRESHOLD:
        logger.info("Score below threshold, falling back to LLM classifier")
        doc_type = _classify_by_llm(sample_text, provider=provider, model=model)

    config = load_doc_type(doc_type)
    logger.info("Document classified as: %s (%s)", config.doc_type, config.display_name)

    if save:
        save_active_config(config)

    return config
