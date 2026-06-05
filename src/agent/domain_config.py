"""Domain configuration loader.

Loads doc-type taxonomy JSONs from src/agent/domains/ and persists
the active config for the current indexed document alongside the
FAISS index in outputs/faiss_index/domain_config.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DOMAINS_DIR = Path(__file__).resolve().parent / "domains"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ACTIVE_CONFIG_PATH = _REPO_ROOT / "outputs" / "faiss_index" / "domain_config.json"


@dataclass
class DomainSpec:
    """A single domain within a doc type (e.g. 'deadlines' within 'contract')."""
    name: str
    display_name: str
    keywords_de: list[str]
    keywords_en: list[str]
    search_terms: list[str]
    model: str
    system_prompt: str

    @property
    def all_keywords(self) -> list[str]:
        return [k.lower() for k in self.keywords_de + self.keywords_en]


@dataclass
class DocTypeConfig:
    """Full configuration for a detected document type."""
    doc_type: str
    display_name: str
    detection_hints: list[str]
    domains: dict[str, DomainSpec] = field(default_factory=dict)

    def get_domain(self, name: str) -> DomainSpec:
        return self.domains.get(name, self.domains["general"])


def load_doc_type(doc_type: str) -> DocTypeConfig:
    """Load a domain config from src/agent/domains/<doc_type>.json.

    Falls back to 'fallback' if the file does not exist.
    """
    path = _DOMAINS_DIR / f"{doc_type}.json"
    if not path.exists():
        logger.warning("No domain config for '%s', using fallback", doc_type)
        path = _DOMAINS_DIR / "fallback.json"

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    domains = {
        name: DomainSpec(
            name=name,
            display_name=spec["display_name"],
            keywords_de=spec["keywords_de"],
            keywords_en=spec["keywords_en"],
            search_terms=spec["search_terms"],
            model=spec["model"],
            system_prompt=spec["system_prompt"],
        )
        for name, spec in data["domains"].items()
    }
    return DocTypeConfig(
        doc_type=data["doc_type"],
        display_name=data["display_name"],
        detection_hints=data["detection_hints"],
        domains=domains,
    )


def list_available_doc_types() -> list[str]:
    """Return all doc types with a JSON file in the domains directory."""
    return [p.stem for p in _DOMAINS_DIR.glob("*.json")]


def save_active_config(config: DocTypeConfig) -> None:
    """Persist the active doc type config alongside the FAISS index."""
    _ACTIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_type": config.doc_type,
        "display_name": config.display_name,
    }
    _ACTIVE_CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved active domain config: %s", config.doc_type)


def load_active_config() -> DocTypeConfig:
    """Load the active doc type config written at index time.

    Falls back to 'fallback' if no config has been saved yet.
    """
    if _ACTIVE_CONFIG_PATH.exists():
        data = json.loads(_ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
        return load_doc_type(data["doc_type"])
    logger.warning("No active domain config found, using fallback")
    return load_doc_type("fallback")
