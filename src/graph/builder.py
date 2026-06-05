"""Knowledge graph builder.

Builds a directed NetworkX graph from a list of Chunk objects.
No LLM, no spaCy — pure regex over chunk text.

Node types
----------
chunk     id=chunk_{i}    page, chunk_type, text_preview
section   id=sec_{label}  label, title, page, level

Edge types
----------
chunk_of        chunk   -> section   chunk belongs to this section
subsection_of   section -> section   §10.1 -> §10
references      chunk   -> section   body text "gemäß §14.3"
sequential      chunk   -> chunk     chunk i -> chunk i+1 (same page)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import networkx as nx
    from networkx.readwrite import json_graph
except ImportError as e:  # pragma: no cover
    raise ImportError("networkx is required: uv add networkx") from e

if TYPE_CHECKING:
    from src.models import Chunk

logger = logging.getLogger(__name__)

# Matches: §10, §10.1, §10.1.2  (with optional trailing space/punctuation)
_SECTION_REF_RE = re.compile(r"§\s*(\d+(?:\.\d+)*)", re.UNICODE)

# Heading detector: line starts with §X or "X." pattern and is short (<= 120 chars)
_HEADING_LINE_RE = re.compile(
    r"^(?:§\s*\d+(?:\.\d+)*|\d+(?:\.\d+)+)\.?\s+\S",
    re.MULTILINE | re.UNICODE,
)


def _section_level(label: str) -> int:
    """Depth of a section label. '10' -> 1, '10.1' -> 2, '10.1.2' -> 3."""
    return label.count(".") + 1


def _parent_label(label: str) -> str | None:
    """Return parent label or None for top-level. '10.1.2' -> '10.1'."""
    parts = label.rsplit(".", 1)
    return parts[0] if len(parts) == 2 else None  # noqa: PLR2004


def _extract_section_label(text: str) -> str | None:
    """Extract section label from the first heading line of a chunk, if any."""
    match = _HEADING_LINE_RE.search(text)
    if not match:
        return None
    # Pull the leading §X.Y or X.Y out of the matched line
    line = match.group(0).strip()
    ref = _SECTION_REF_RE.match(line)
    if ref:
        return ref.group(1)
    # Plain numeric heading: "10.1 Title" -> "10.1"
    num_match = re.match(r"(\d+(?:\.\d+)+)", line)
    return num_match.group(1) if num_match else None


def _extract_section_title(text: str, label: str) -> str:
    """Extract a short title from the heading line."""
    first_line = text.strip().split("\n")[0][:120]
    return first_line


def _extract_cross_refs(text: str) -> list[str]:
    """Return all §-reference labels found in body text."""
    return _SECTION_REF_RE.findall(text)


def build_graph(chunks: list[Chunk]) -> nx.DiGraph:  # type: ignore[name-defined]
    """Build a directed knowledge graph from a list of Chunk objects.

    Args:
        chunks: Ordered list of Chunk objects from the pipeline.

    Returns:
        nx.DiGraph with chunk/section nodes and typed edges.
    """
    g: nx.DiGraph = nx.DiGraph()

    section_nodes: dict[str, str] = {}   # label -> node_id
    chunk_section: dict[int, str] = {}   # chunk_index -> section node_id
    current_section_id: str | None = None

    # ------------------------------------------------------------------
    # Pass 1: add chunk nodes + section nodes + subsection_of hierarchy
    # ------------------------------------------------------------------
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        g.add_node(
            chunk_id,
            node_type="chunk",
            chunk_index=i,
            page=chunk.page_number,
            chunk_type=chunk.chunk_type,
            text_preview=chunk.text[:80],
        )

        label = _extract_section_label(chunk.text)
        if label:
            sec_id = f"sec_{label}"
            if sec_id not in g:
                g.add_node(
                    sec_id,
                    node_type="section",
                    label=label,
                    title=_extract_section_title(chunk.text, label),
                    page=chunk.page_number,
                    level=_section_level(label),
                )
                section_nodes[label] = sec_id
                # subsection_of edge
                parent = _parent_label(label)
                if parent and parent in section_nodes:
                    g.add_edge(sec_id, section_nodes[parent], edge_type="subsection_of")
            current_section_id = sec_id

        if current_section_id:
            chunk_section[i] = current_section_id
            g.add_edge(chunk_id, current_section_id, edge_type="chunk_of")

    # ------------------------------------------------------------------
    # Pass 2: cross-ref edges + sequential edges
    # ------------------------------------------------------------------
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"

        # references edges: body text §-refs
        for ref_label in _extract_cross_refs(chunk.text):
            target_sec_id = section_nodes.get(ref_label)
            if target_sec_id and target_sec_id != chunk_section.get(i):
                if not g.has_edge(chunk_id, target_sec_id):
                    g.add_edge(chunk_id, target_sec_id, edge_type="references")

        # sequential edge to next chunk on the same page
        if i + 1 < len(chunks) and chunks[i + 1].page_number == chunk.page_number:
            next_id = f"chunk_{i + 1}"
            g.add_edge(chunk_id, next_id, edge_type="sequential")

    logger.info(
        "Graph built: %d nodes (%d chunks, %d sections), %d edges",
        g.number_of_nodes(),
        sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "chunk"),
        sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "section"),
        g.number_of_edges(),
    )
    return g


def save_graph(graph: nx.DiGraph, path: Path) -> None:  # type: ignore[name-defined]
    """Persist graph to JSON (NetworkX node-link format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(graph)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Graph saved to %s (%d nodes, %d edges)", path, graph.number_of_nodes(), graph.number_of_edges())


def load_graph(path: Path) -> nx.DiGraph:  # type: ignore[name-defined]
    """Load graph from JSON. Returns empty DiGraph if file missing."""
    if not path.exists():
        logger.warning("Graph file not found at %s, returning empty graph", path)
        return nx.DiGraph()
    data = json.loads(path.read_text(encoding="utf-8"))
    graph = json_graph.node_link_graph(data, directed=True)
    logger.info("Graph loaded from %s (%d nodes, %d edges)", path, graph.number_of_nodes(), graph.number_of_edges())
    return graph
