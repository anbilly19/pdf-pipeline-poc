"""Domain router.

Reads the active DocTypeConfig (written at index time) and keyword-matches
the user query against each domain's keywords to return the best DomainSpec.

No LLM involved — pure keyword scoring, fully testable offline.
"""
from __future__ import annotations

import logging

from src.agent.domain_config import DocTypeConfig, DomainSpec, load_active_config

logger = logging.getLogger(__name__)


def _score_query(query: str, spec: DomainSpec) -> int:
    """Count how many domain keywords appear in the query (case-insensitive)."""
    q = query.lower()
    return sum(1 for kw in spec.all_keywords if kw in q)


def route_query(
    query: str,
    config: DocTypeConfig | None = None,
) -> DomainSpec:
    """Return the best-matching DomainSpec for the given query.

    Args:
        query: The user's question.
        config: DocTypeConfig to route within. Loads active config if None.

    Returns:
        The highest-scoring DomainSpec, or the 'general' fallback.
    """
    if config is None:
        config = load_active_config()

    best_spec = config.get_domain("general")
    best_score = 0

    for name, spec in config.domains.items():
        if name == "general":
            continue
        score = _score_query(query, spec)
        logger.debug("Domain '%s' score: %d", name, score)
        if score > best_score:
            best_score = score
            best_spec = spec

    logger.info(
        "Routed query to domain='%s' (score=%d, doc_type='%s')",
        best_spec.name, best_score, config.doc_type,
    )
    return best_spec
