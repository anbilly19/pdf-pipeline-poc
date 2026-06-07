"""LangGraph checkpoint-backed persistent memory with isolated keys.

This module implements Roadmap #5: replace the stateless per-query agent
with a session-scoped agent that persists conversation history across turns
using LangGraph's MemorySaver checkpointer.

Key design decisions
--------------------
**Isolated keys**: conversation history and retrieval context are stored
under separate keys in the graph state:

    messages          -- LangChain message objects (HumanMessage, AIMessage …)
    retrieval_context -- raw text blobs from tool calls (bbox data etc.)

This prevents bbox data from earlier retrieval steps bleeding into the
conversation window that the LLM sees. The LLM only ever reads `messages`;
retrieval_context is private bookkeeping used by tools.

**Thread isolation**: each user session gets its own `thread_id` so
memory from one conversation never contaminates another.

**In-process store**: MemorySaver keeps state in RAM. It disappears when
the process exits. For persistent across-session memory, swap in
SqliteSaver (see `make_checkpointer`). No filesystem writes needed by
default.

**Retrieval context window discipline**: CLAUDE.md mandates that no more
than 4,000–5,000 characters of context are passed to the LLM per turn.
The `trim_retrieval_context` helper enforces this budget by keeping the
most recent N retrieval snippets that fit inside the limit.

Public API
----------
    make_checkpointer(backend="memory") -> BaseCheckpointSaver
    make_thread_config(thread_id)       -> dict   (LangGraph RunnableConfig)
    trim_retrieval_context(ctx, limit)  -> list[str]
    AgentState                          -- TypedDict for graph state
"""
from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------

_RetrievalCtx = list[str]  # ordered list of raw text snippets


# -----------------------------------------------------------------------
# Graph state — two isolated keys
# -----------------------------------------------------------------------

class AgentState(TypedDict):
    """LangGraph state schema with isolated conversation and retrieval keys.

    Attributes:
        messages: Conversation history (LangChain messages). Managed by
            the `add_messages` reducer — each update appends, never
            replaces the list.
        retrieval_context: Raw retrieval snippets produced by tool calls.
            Managed by a simple replace reducer so the agent can update
            or clear the context between turns without touching messages.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    retrieval_context: list[str]  # replaced wholesale on each update


# -----------------------------------------------------------------------
# Checkpointer factory
# -----------------------------------------------------------------------

def make_checkpointer(backend: str = "memory") -> Any:
    """Create a LangGraph checkpoint saver.

    Args:
        backend: ``"memory"`` (default, in-process RAM) or ``"sqlite"``
            (persistent file at ``./checkpoints.db``).

    Returns:
        A LangGraph BaseCheckpointSaver instance.

    Raises:
        ValueError: If an unknown backend is requested.
    """
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
                "langgraph-checkpoint-sqlite not installed (%s); "
                "falling back to MemorySaver. "
                "Run: uv add langgraph-checkpoint-sqlite",
                exc,
            )
            return MemorySaver()

    raise ValueError(
        f"Unknown checkpointer backend: {backend!r}. "
        "Valid values: 'memory', 'sqlite'."
    )


# -----------------------------------------------------------------------
# Thread config
# -----------------------------------------------------------------------

def make_thread_config(thread_id: str | None = None) -> dict[str, Any]:
    """Create a LangGraph RunnableConfig for a specific conversation thread.

    Each unique thread_id gets an isolated memory namespace in the
    checkpointer.  If no thread_id is given, a fresh UUID is generated.

    Args:
        thread_id: Stable identifier for the conversation session.  Use
            the same ID across turns to preserve memory.  Pass ``None``
            to start a fresh, anonymous session.

    Returns:
        A dict suitable for passing as the ``config`` kwarg to
        ``agent.invoke()`` or ``agent.stream()``.
    """
    tid = thread_id or str(uuid.uuid4())
    return {"configurable": {"thread_id": tid}}


# -----------------------------------------------------------------------
# Context budget enforcement
# -----------------------------------------------------------------------

_DEFAULT_CONTEXT_CHAR_LIMIT = 4_500  # CLAUDE.md: 4000-5000 chars max


def trim_retrieval_context(
    snippets: list[str],
    char_limit: int = _DEFAULT_CONTEXT_CHAR_LIMIT,
) -> list[str]:
    """Trim retrieval snippets to fit inside the LLM context budget.

    Keeps snippets in order, dropping from the *oldest* end until the
    total character count is within `char_limit`. The most recent
    snippets are always preferred.

    Args:
        snippets: Ordered list of retrieval text snippets (oldest first).
        char_limit: Maximum total character budget (default 4 500).

    Returns:
        Trimmed list that fits inside the budget (may be empty).
    """
    if not snippets:
        return []

    kept: list[str] = []
    total = 0
    for snippet in reversed(snippets):  # most recent first
        needed = len(snippet) + (1 if kept else 0)  # +1 for separator
        if total + needed > char_limit:
            break
        kept.append(snippet)
        total += needed

    kept.reverse()  # restore chronological order
    dropped = len(snippets) - len(kept)
    if dropped:
        logger.debug("trim_retrieval_context: dropped %d old snippets", dropped)
    return kept
