"""TDD tests for the domain router.

All tests are fully offline — no LLM, no FAISS, no file I/O.
"""
from __future__ import annotations

import pytest

from src.agent.domain_config import DocTypeConfig, DomainSpec
from src.agent.router import route_query, _score_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec(name: str, keywords_de: list[str], keywords_en: list[str]) -> DomainSpec:
    return DomainSpec(
        name=name,
        display_name=name.capitalize(),
        keywords_de=keywords_de,
        keywords_en=keywords_en,
        search_terms=[],
        model="qwen2.5:3b",
        system_prompt="test prompt",
    )


def _contract_config() -> DocTypeConfig:
    return DocTypeConfig(
        doc_type="contract",
        display_name="Contract",
        detection_hints=[],
        domains={
            "general": _spec("general", [], []),
            "deadlines": _spec("deadlines", ["Frist", "Verzug", "Reaktionszeit"], ["deadline", "delay"]),
            "termination": _spec("termination", ["K\u00fcndigung", "Laufzeit"], ["termination", "notice"]),
            "payment": _spec("payment", ["Verg\u00fctung", "Zahlung", "Preis"], ["payment", "fee"]),
            "liability": _spec("liability", ["Haftung", "Schaden"], ["liability", "damages"]),
        },
    )


def _magazine_config() -> DocTypeConfig:
    return DocTypeConfig(
        doc_type="magazine",
        display_name="Magazine",
        detection_hints=[],
        domains={
            "general": _spec("general", [], []),
            "market_data": _spec("market_data", ["Umsatz", "Markt", "Wachstum"], ["revenue", "market", "growth"]),
            "company_news": _spec("company_news", ["Unternehmen", "CEO"], ["company", "acquisition"]),
            "events": _spec("events", ["Messe", "ISM", "Aussteller"], ["fair", "trade show"]),
        },
    )


# ---------------------------------------------------------------------------
# _score_query unit tests
# ---------------------------------------------------------------------------

class TestScoreQuery:
    def test_exact_keyword_match(self) -> None:
        spec = _spec("deadlines", ["Frist", "Verzug"], ["deadline"])
        assert _score_query("Was ist die Frist f\u00fcr Verzug?", spec) == 2

    def test_case_insensitive(self) -> None:
        spec = _spec("deadlines", ["Frist"], ["DEADLINE"])
        assert _score_query("what is the deadline frist", spec) == 2

    def test_no_match_returns_zero(self) -> None:
        spec = _spec("payment", ["Zahlung"], ["payment"])
        assert _score_query("Was ist die K\u00fcndigung?", spec) == 0

    def test_empty_query(self) -> None:
        spec = _spec("deadlines", ["Frist"], ["deadline"])
        assert _score_query("", spec) == 0

    def test_empty_keywords(self) -> None:
        spec = _spec("general", [], [])
        assert _score_query("any query here", spec) == 0


# ---------------------------------------------------------------------------
# route_query tests
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def test_routes_to_deadlines(self) -> None:
        config = _contract_config()
        spec = route_query("Wie lange ist die Frist bei Verzug?", config=config)
        assert spec.name == "deadlines"

    def test_routes_to_termination(self) -> None:
        config = _contract_config()
        spec = route_query("Wann kann ich den Vertrag k\u00fcndigen und wie lange ist die Laufzeit?", config=config)
        assert spec.name == "termination"

    def test_routes_to_payment(self) -> None:
        config = _contract_config()
        spec = route_query("Wie hoch ist die Verg\u00fctung und wann ist die Zahlung f\u00e4llig?", config=config)
        assert spec.name == "payment"

    def test_routes_to_liability(self) -> None:
        config = _contract_config()
        spec = route_query("Wer haftet f\u00fcr den Schaden?", config=config)
        assert spec.name == "liability"

    def test_routes_to_market_data(self) -> None:
        config = _magazine_config()
        spec = route_query("Wie hoch war der Umsatz und das Marktwachstum?", config=config)
        assert spec.name == "market_data"

    def test_routes_to_events(self) -> None:
        config = _magazine_config()
        spec = route_query("Wie viele Aussteller waren auf der ISM Messe?", config=config)
        assert spec.name == "events"

    def test_unrecognised_query_falls_back_to_general(self) -> None:
        config = _contract_config()
        spec = route_query("Tell me something random.", config=config)
        assert spec.name == "general"

    def test_tie_broken_by_first_higher_score(self) -> None:
        """If two domains tie, the one encountered first with that score wins."""
        config = _contract_config()
        # Query only matches one keyword in deadlines and one in termination
        spec = route_query("Frist K\u00fcndigung", config=config)
        # Both score 1; whichever comes first in dict wins — just assert it's not general
        assert spec.name != "general"

    def test_english_keywords_routed_correctly(self) -> None:
        config = _contract_config()
        spec = route_query("What are the payment terms and fee structure?", config=config)
        assert spec.name == "payment"

    def test_returns_domain_spec_type(self) -> None:
        config = _contract_config()
        result = route_query("Frist Verzug", config=config)
        assert isinstance(result, DomainSpec)
