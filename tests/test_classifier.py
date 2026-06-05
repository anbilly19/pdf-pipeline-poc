"""TDD tests for doc classifier and domain config system.

All tests are fully offline — no Ollama, no OpenAI, no FAISS.
LLM fallback path is tested via mocks only.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models import Chunk
from src.agent.domain_config import (
    DocTypeConfig,
    DomainSpec,
    load_doc_type,
    load_active_config,
    save_active_config,
    list_available_doc_types,
)
from src.agent.classifier import (
    classify_document,
    _classify_by_keywords,
    _keyword_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str, page: int = 1) -> Chunk:
    return Chunk(
        text=text,
        page_number=page,
        bboxes=[[0.0, 0.0, 100.0, 20.0]],
        chunk_type="text",
        confidence=0.9,
        image_path="",
    )


# ---------------------------------------------------------------------------
# DomainSpec unit tests
# ---------------------------------------------------------------------------

class TestDomainSpec:
    def test_all_keywords_lowercased(self) -> None:
        spec = DomainSpec(
            name="test",
            display_name="Test",
            keywords_de=["Frist", "VERZUG"],
            keywords_en=["Deadline"],
            search_terms=[],
            model="qwen2.5:3b",
            system_prompt="",
        )
        assert "frist" in spec.all_keywords
        assert "verzug" in spec.all_keywords
        assert "deadline" in spec.all_keywords
        # originals not mutated
        assert "Frist" in spec.keywords_de

    def test_all_keywords_combined(self) -> None:
        spec = DomainSpec(
            name="x",
            display_name="X",
            keywords_de=["Haftung"],
            keywords_en=["liability"],
            search_terms=[],
            model="qwen2.5:3b",
            system_prompt="",
        )
        assert len(spec.all_keywords) == 2


# ---------------------------------------------------------------------------
# DocTypeConfig unit tests
# ---------------------------------------------------------------------------

class TestDocTypeConfig:
    def _make_config(self) -> DocTypeConfig:
        general = DomainSpec(
            name="general", display_name="General",
            keywords_de=[], keywords_en=[], search_terms=[],
            model="qwen2.5:3b", system_prompt="",
        )
        deadlines = DomainSpec(
            name="deadlines", display_name="Fristen",
            keywords_de=["Frist"], keywords_en=["deadline"], search_terms=[],
            model="gemma4:e2b", system_prompt="",
        )
        return DocTypeConfig(
            doc_type="contract",
            display_name="Contract",
            detection_hints=["Vertrag"],
            domains={"general": general, "deadlines": deadlines},
        )

    def test_get_domain_known(self) -> None:
        config = self._make_config()
        assert config.get_domain("deadlines").name == "deadlines"

    def test_get_domain_unknown_falls_back_to_general(self) -> None:
        config = self._make_config()
        assert config.get_domain("nonexistent").name == "general"


# ---------------------------------------------------------------------------
# load_doc_type tests (reads real JSON files)
# ---------------------------------------------------------------------------

class TestLoadDocType:
    @pytest.mark.parametrize("doc_type", ["contract", "magazine", "technical", "research", "fallback"])
    def test_all_bundled_configs_load(self, doc_type: str) -> None:
        config = load_doc_type(doc_type)
        assert config.doc_type == doc_type
        assert "general" in config.domains
        assert config.domains["general"].system_prompt

    def test_unknown_type_falls_back_to_fallback(self) -> None:
        config = load_doc_type("totally_unknown_type_xyz")
        assert config.doc_type == "fallback"

    def test_contract_has_required_domains(self) -> None:
        config = load_doc_type("contract")
        for domain in ["deadlines", "termination", "payment", "liability", "general"]:
            assert domain in config.domains, f"Missing domain: {domain}"

    def test_magazine_has_required_domains(self) -> None:
        config = load_doc_type("magazine")
        for domain in ["market_data", "company_news", "products", "events", "general"]:
            assert domain in config.domains, f"Missing domain: {domain}"

    def test_domain_spec_fields_populated(self) -> None:
        config = load_doc_type("contract")
        spec = config.domains["deadlines"]
        assert spec.keywords_de
        assert spec.keywords_en
        assert spec.search_terms
        assert spec.model
        assert spec.system_prompt

    def test_list_available_includes_all_bundled(self) -> None:
        available = list_available_doc_types()
        for expected in ["contract", "magazine", "technical", "research", "fallback"]:
            assert expected in available


# ---------------------------------------------------------------------------
# save/load active config tests (uses tmp_path)
# ---------------------------------------------------------------------------

class TestActiveConfig:
    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Point active config path to tmp_path
        import src.agent.domain_config as dc_module
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", tmp_path / "domain_config.json")

        config = load_doc_type("contract")
        save_active_config(config)
        loaded = load_active_config()
        assert loaded.doc_type == "contract"
        assert "deadlines" in loaded.domains

    def test_load_active_config_missing_falls_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agent.domain_config as dc_module
        missing_path = tmp_path / "nonexistent" / "domain_config.json"
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", missing_path)
        config = load_active_config()
        assert config.doc_type == "fallback"

    def test_save_creates_parent_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agent.domain_config as dc_module
        deep_path = tmp_path / "a" / "b" / "c" / "domain_config.json"
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", deep_path)
        config = load_doc_type("magazine")
        save_active_config(config)  # must not raise
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# _keyword_score unit tests
# ---------------------------------------------------------------------------

class TestKeywordScore:
    def test_exact_match(self) -> None:
        assert _keyword_score("Dieser Vertrag regelt die Vereinbarung", ["Vertrag", "Vereinbarung"]) == 2

    def test_case_insensitive(self) -> None:
        assert _keyword_score("VERTRAG und vereinbarung", ["Vertrag", "Vereinbarung"]) == 2

    def test_no_match(self) -> None:
        assert _keyword_score("something completely different", ["Vertrag"]) == 0

    def test_partial_word_match(self) -> None:
        # 'Vertrag' should match inside 'Vertragspartner'
        assert _keyword_score("Vertragspartner", ["Vertrag"]) == 1


# ---------------------------------------------------------------------------
# _classify_by_keywords tests
# ---------------------------------------------------------------------------

class TestClassifyByKeywords:
    def test_contract_text_detected(self) -> None:
        text = "Dieser Vertrag ist eine Vereinbarung zwischen den Vertragsparteien. Klausel 1 regelt die Obligation."
        doc_type, score = _classify_by_keywords(text)
        assert doc_type == "contract"
        assert score >= 3

    def test_magazine_text_detected(self) -> None:
        text = "Redaktion: Albert Angerer. Impressum. Jahrgang 64. Heft 3. Verlag SWEETS GLOBAL NETWORK."
        doc_type, score = _classify_by_keywords(text)
        assert doc_type == "magazine"
        assert score >= 3

    def test_low_confidence_returns_fallback(self) -> None:
        text = "Lorem ipsum dolor sit amet."
        doc_type, score = _classify_by_keywords(text)
        # score will be low; type may be anything but score should be < threshold
        assert score < 3


# ---------------------------------------------------------------------------
# classify_document integration tests (no LLM, no FAISS)
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    def test_contract_chunks_classified_correctly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agent.domain_config as dc_module
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", tmp_path / "domain_config.json")

        chunks = [
            _make_chunk("Dieser Vertrag regelt die Vereinbarung zwischen den Parteien."),
            _make_chunk("Klausel 3: Die Vertragspartei verpflichtet sich zur Einhaltung der Obligation."),
            _make_chunk("agreement clause party liability"),
        ]
        config = classify_document(chunks, save=True)
        assert config.doc_type == "contract"

    def test_magazine_chunks_classified_correctly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agent.domain_config as dc_module
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", tmp_path / "domain_config.json")

        chunks = [
            _make_chunk("Redaktion: Albert Angerer. Impressum. Jahrgang 64."),
            _make_chunk("Heft 3. Verlag SWEETS GLOBAL NETWORK. Ausgabe M\u00e4rz 2026."),
            _make_chunk("editorial magazine publisher issue edition"),
        ]
        config = classify_document(chunks, save=True)
        assert config.doc_type == "magazine"

    def test_save_false_does_not_write_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agent.domain_config as dc_module
        target = tmp_path / "domain_config.json"
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", target)

        chunks = [_make_chunk("Vertrag Vereinbarung Klausel")]
        classify_document(chunks, save=False)
        assert not target.exists()

    def test_empty_chunks_returns_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.agent.domain_config as dc_module
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", tmp_path / "domain_config.json")
        config = classify_document([], save=False)
        assert config.doc_type == "fallback"

    def test_llm_fallback_called_on_low_score(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When keyword score < threshold, LLM classifier should be invoked."""
        import src.agent.domain_config as dc_module
        import src.agent.classifier as clf_module
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", tmp_path / "domain_config.json")

        mock_llm_classify = MagicMock(return_value="technical")
        monkeypatch.setattr(clf_module, "_classify_by_llm", mock_llm_classify)

        chunks = [_make_chunk("Lorem ipsum dolor sit amet consectetur.")]
        config = classify_document(chunks, save=False)
        mock_llm_classify.assert_called_once()
        assert config.doc_type == "technical"

    def test_llm_fallback_exception_returns_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If LLM call throws, result must be 'fallback', not a crash."""
        import src.agent.domain_config as dc_module
        import src.agent.classifier as clf_module
        monkeypatch.setattr(dc_module, "_ACTIVE_CONFIG_PATH", tmp_path / "domain_config.json")

        def _raise(*a, **kw):  # noqa: ANN001
            raise RuntimeError("Ollama not available")
        monkeypatch.setattr(clf_module, "_classify_by_llm", _raise)

        # Patch keyword score to always return 0 to force LLM path
        monkeypatch.setattr(clf_module, "_classify_by_keywords", lambda _text: ("fallback", 0))

        chunks = [_make_chunk("irrelevant text")]
        # Should not raise
        config = classify_document(chunks, save=False)
        assert config.doc_type == "fallback"
