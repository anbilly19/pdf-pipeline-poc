"""Graph-based chunk expander.

Takes the FAISS top-k chunks and expands them using the knowledge graph
to surface related chunks the embedding search may have missed.

Expansion strategy (depth-1 only to avoid noise):
  For each seed chunk:
    1. references edges  -> section node -> all chunks in that section
    2. chunk_of edge     -> section node -> sibling chunks (same section)
    3. sequential edges  -> immediate neighbours (same page)

All expanded chunks are deduplicated and appended after the original seeds,
preserving seed order (seeds always come first).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    import networkx as nx
except ImportError as e:  # pragma: no cover
    raise ImportError("networkx is required: uv add networkx") from e

if TYPE_CHECKING:
    from src.models import Chunk

logger = logging.getLogger(__name__)

_MAX_EXPANDED = 25  # hard cap to avoid flooding the LLM context


def _chunks_in_section(section_id: str, graph: nx.DiGraph, chunks: list[Chunk]) -> list[int]:  # type: ignore[name-defined]
    """Return chunk indices whose chunk_of edge points to section_id."""
    indices = []
    for pred in graph.predecessors(section_id):
        data = graph.nodes[pred]
        if data.get("node_type") == "chunk":
            edge_data = graph.edges[pred, section_id]
            if edge_data.get("edge_type") == "chunk_of":
                idx = data.get("chunk_index")
                if idx is not None and idx < len(chunks):
                    indices.append(idx)
    return indices


def expand_chunks(
    seed_chunks: list[Chunk],
    graph: nx.DiGraph,  # type: ignore[name-defined]
    all_chunks: list[Chunk],
    max_expanded: int = _MAX_EXPANDED,
) -> list[Chunk]:
    """Expand seed chunks using the knowledge graph.

    Args:
        seed_chunks: Chunks returned by FAISS retrieval (ordered by relevance).
        graph: Built by build_graph(all_chunks).
        all_chunks: The complete ordered list of chunks from the indexed document.
        max_expanded: Hard cap on total returned chunks.

    Returns:
        Deduplicated list with seeds first, then graph-expanded additions.
        Never exceeds max_expanded total.
    """
    if not seed_chunks or graph.number_of_nodes() == 0:
        return seed_chunks

    # Build reverse lookup: text identity -> chunk index in all_chunks
    text_to_index: dict[str, int] = {c.text: i for i, c in enumerate(all_chunks)}

    seed_indices: list[int] = []
    for c in seed_chunks:
        idx = text_to_index.get(c.text)
        if idx is not None:
            seed_indices.append(idx)

    seen: set[int] = set(seed_indices)
    expanded_indices: list[int] = list(seed_indices)  # seeds first

    for seed_idx in seed_indices:
        if len(expanded_indices) >= max_expanded:
            break

        chunk_id = f"chunk_{seed_idx}"
        if chunk_id not in graph:
            continue

        # 1. Follow chunk_of -> section -> sibling chunks
        for _, sec_id, edata in graph.out_edges(chunk_id, data=True):
            if edata.get("edge_type") != "chunk_of":
                continue
            for sibling_idx in _chunks_in_section(sec_id, graph, all_chunks):
                if sibling_idx not in seen:
                    seen.add(sibling_idx)
                    expanded_indices.append(sibling_idx)

        # 2. Follow references -> target section -> chunks in that section
        for _, sec_id, edata in graph.out_edges(chunk_id, data=True):
            if edata.get("edge_type") != "references":
                continue
            for ref_idx in _chunks_in_section(sec_id, graph, all_chunks):
                if ref_idx not in seen:
                    seen.add(ref_idx)
                    expanded_indices.append(ref_idx)

        # 3. Follow sequential -> immediate page neighbours
        for _, next_id, edata in graph.out_edges(chunk_id, data=True):
            if edata.get("edge_type") != "sequential":
                continue
            ndata = graph.nodes[next_id]
            next_idx = ndata.get("chunk_index")
            if next_idx is not None and next_idx not in seen:
                seen.add(next_idx)
                expanded_indices.append(next_idx)

    result = [all_chunks[i] for i in expanded_indices[:max_expanded]]
    added = len(result) - len(seed_chunks)
    if added > 0:
        logger.info("Graph expansion: %d seeds -> %d total (+%d)", len(seed_chunks), len(result), added)
    return result
