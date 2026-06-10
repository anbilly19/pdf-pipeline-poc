"""LangGraph checkpoint-backed persistent memory with isolated keys."""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

_RetrievalCtx = list[str]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieval_context: list[str]


def make_checkpointer(backend: str = "memory") -> Any:
    if backend == "memory":
        logger.info("Using in-process MemorySaver (state lost on process exit)")
        return MemorySaver()

    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: PLC0415
            saver = SqliteSaver.from_conn_string("./checkpoints.db")
            logger.info("Using SqliteSaver (./checkpoints.db)")
            return saver
        except ImportError as exc:
            logger.warning(
                "langgraph-checkpoint-sqlite not installed (%s); falling back to MemorySaver.", exc
            )
            return MemorySaver()

    raise ValueError(f"Unknown checkpointer backend: {backend!r}. Valid values: 'memory', 'sqlite'.")


def make_thread_config(thread_id: str | None = None) -> dict[str, Any]:
    tid = thread_id or str(uuid.uuid4())
    return {"configurable": {"thread_id": tid}}


# Raised from 4500 to 12000 so all 23 retrieval chunks fit comfortably.
# At ~400 chars/chunk * 23 = ~9200 chars; 12000 gives headroom.
_DEFAULT_CONTEXT_CHAR_LIMIT = 12_000


def trim_retrieval_context(
    snippets: list[str],
    char_limit: int = _DEFAULT_CONTEXT_CHAR_LIMIT,
) -> list[str]:
    """Keep the highest-content chunks that fit within char_limit.

    Strategy: sort by length descending (longer = more content), keep
    greedily until budget exhausted, then restore original order.
    This avoids discarding early-document chunks (e.g. Ziffer 5.1)
    just because they were indexed first.
    """
    if not snippets:
        return []

    # Pair each snippet with its original index so we can restore order
    indexed = list(enumerate(snippets))
    # Sort by length descending — longest chunks carry the most information
    indexed_by_len = sorted(indexed, key=lambda x: len(x[1]), reverse=True)

    kept_indices: list[int] = []
    total = 0
    for orig_idx, snippet in indexed_by_len:
        needed = len(snippet)
        if total + needed > char_limit:
            continue  # skip this one, try smaller chunks
        kept_indices.append(orig_idx)
        total += needed

    # Restore original order
    kept_indices_set = set(kept_indices)
    result = [snippets[i] for i in range(len(snippets)) if i in kept_indices_set]

    dropped = len(snippets) - len(result)
    if dropped:
        logger.debug("trim_retrieval_context: dropped %d chunks (budget %d chars)", dropped, char_limit)
    return result
