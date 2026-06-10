"""Agent tools for the PDF Q&A pipeline."""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from typing import Annotated

from langchain_core.tools import tool

from src.models import Chunk
from src.retrieval.retriever import BBoxRetriever

logger = logging.getLogger(__name__)

# Reduced from 10 — fewer, higher-quality chunks beat more noisy ones
_TOP_K = 8


@dataclass
class ToolResult:
    content: str
    bboxes: list[list[float]]
    page_number: int
    image_path: str

    def __str__(self) -> str:
        return (
            f"{self.content}\n"
            f"[source: page {self.page_number}, "
            f"bboxes={self.bboxes}, "
            f"image_path={self.image_path!r}]"
        )


NO_RESULTS = "Keine relevanten Abschnitte gefunden."
NO_TABLE = "No table found."
NO_SECTION = "Section not found."
NO_REGION = "No region found on page."


def build_tools(
    retriever: BBoxRetriever,
    graph: object = None,
    all_chunks: list[Chunk] | None = None,
    self_rag_model: str = "gemma4:e2b",
    self_rag_enabled: bool = True,
    self_rag_bm25_gate: float = 0.5,
) -> list[object]:
    from src.agent.self_rag import ScoredChunk, make_self_rag_filter  # noqa: PLC0415

    _rag_filter = make_self_rag_filter(
        model=self_rag_model,
        bm25_gate=self_rag_bm25_gate,
        enabled=self_rag_enabled,
    )
    _graph_enabled = graph is not None and all_chunks is not None

    def _expand(chunks: list[Chunk]) -> list[Chunk]:
        if not _graph_enabled or not chunks:
            return chunks
        try:
            from src.graph.expander import expand_chunks  # noqa: PLC0415
            return expand_chunks(chunks, graph, all_chunks)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph expansion failed (non-fatal): %s", exc)
            return chunks

    def _self_rag_filter(query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Run Self-RAG; on failure or all-filtered, return top-3 by original rank."""
        try:
            scored = [ScoredChunk(chunk=c, bm25_score=0.0) for c in chunks]
            results = _rag_filter.filter(query, scored)
            kept = [r.chunk for r in results]
            if not kept:
                # Return top-3 best-ranked originals rather than the full list
                logger.info("Self-RAG filtered all — falling back to top-3 originals")
                return chunks[:3]
            return kept
        except Exception as exc:  # noqa: BLE001
            logger.warning("Self-RAG filter error (non-fatal): %s", exc)
            return chunks[:3]

    @tool
    def search_term(
        query: Annotated[str, "Search query (German or English)"],
    ) -> str:
        """Search the document for relevant text sections.

        Returns up to 8 of the most relevant sections.
        Read ALL returned sections — the answer may be in section 3, 5 or 8.
        """
        chunks = retriever.retrieve(query, top_k=_TOP_K)
        if not chunks:
            return NO_RESULTS
        chunks = _expand(chunks)
        chunks = _self_rag_filter(query, chunks)

        parts = [f"FOUND SECTIONS ({len(chunks)} total — read all):\n"]
        for i, c in enumerate(chunks, 1):
            result = ToolResult(
                content=c.text,
                bboxes=c.bboxes,
                page_number=c.page_number,
                image_path=c.image_path,
            )
            parts.append(f"--- Abschnitt {i} ---\n{result}")
        return "\n\n".join(parts)

    @tool
    def extract_table_to_csv(
        query: Annotated[str, "Description of the table to find and extract"],
    ) -> str:
        """Find a table in the document and return it as CSV."""
        chunks = retriever.retrieve(query, top_k=5, filter_chunk_type="table")
        if not chunks:
            chunks = retriever.retrieve(query, top_k=3)
        if not chunks:
            return NO_TABLE

        best: Chunk = chunks[0]
        result = ToolResult(
            content=_markdown_table_to_csv(best.text),
            bboxes=best.bboxes,
            page_number=best.page_number,
            image_path=best.image_path,
        )
        return str(result)

    @tool
    def summarize_section(
        title: Annotated[str, "Title or topic of the section to summarise"],
    ) -> str:
        """Retrieve and summarise a named section from the document."""
        chunks = retriever.retrieve(title, top_k=4)
        if not chunks:
            return NO_SECTION
        chunks = _expand(chunks)

        combined = "\n\n".join(c.text for c in chunks)
        all_bboxes = [bbox for c in chunks for bbox in c.bboxes]
        result = ToolResult(
            content=combined,
            bboxes=all_bboxes,
            page_number=chunks[0].page_number,
            image_path=chunks[0].image_path,
        )
        return str(result)

    @tool
    def highlight_section(
        page_number: Annotated[int, "Page number (1-based)"],
        query: Annotated[str, "Query to find the specific region to highlight"],
    ) -> str:
        """Return the bounding boxes for a region on a specific page."""
        chunks = retriever.retrieve(query, top_k=10)
        page_chunks = [c for c in chunks if c.page_number == page_number] or chunks[:3]
        if not page_chunks:
            return NO_REGION

        all_bboxes = [bbox for c in page_chunks for bbox in c.bboxes]
        result = ToolResult(
            content=f"Highlighted region on page {page_number}",
            bboxes=all_bboxes,
            page_number=page_number,
            image_path=page_chunks[0].image_path,
        )
        return str(result)

    @tool
    def rerank_and_filter(
        query: Annotated[str, "Search query to retrieve and filter sections"],
        top_k: Annotated[int, "Max sections after filter (1-8)"] = 4,
    ) -> str:
        """Retrieve, rerank, then Self-RAG filter for highest precision.

        Use when search_term returns too many borderline results.
        """
        top_k = max(1, min(top_k, 8))
        chunks = retriever.retrieve(query, top_k=_TOP_K)
        if not chunks:
            return NO_RESULTS
        chunks = _expand(chunks)
        chunks = _self_rag_filter(query, chunks)
        chunks = chunks[:top_k]

        parts = [f"FILTERED SECTIONS (top {len(chunks)}, Self-RAG verified):\n"]
        for i, c in enumerate(chunks, 1):
            result = ToolResult(
                content=c.text,
                bboxes=c.bboxes,
                page_number=c.page_number,
                image_path=c.image_path,
            )
            parts.append(f"--- Abschnitt {i} ---\n{result}")
        return "\n\n".join(parts)

    @tool
    def verify_relevance(
        query: Annotated[str, "The question or query"],
        chunk_text: Annotated[str, "The text chunk to check"],
    ) -> str:
        """Check whether a single text chunk is relevant to the query."""
        from src.models import Chunk as _Chunk  # noqa: PLC0415
        dummy = _Chunk(
            text=chunk_text,
            page_number=0,
            bboxes=[],
            chunk_type="text",
            confidence=1.0,
            image_path="",
        )
        result = _rag_filter.check_one(query=query, chunk=dummy, bm25_score=0.0)
        return (
            f"relevant={result.is_relevant} | "
            f"score={result.score:.2f} | "
            f"reason={result.reason!r}"
        )

    return [
        search_term,
        extract_table_to_csv,
        summarize_section,
        highlight_section,
        rerank_and_filter,
        verify_relevance,
    ]


def _markdown_table_to_csv(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    table_lines = [
        line for line in lines
        if line.startswith("|") and set(line.replace("|", "").replace("-", "").replace(" ", "")) != set()
    ]
    if not table_lines:
        return text

    output = io.StringIO()
    writer = csv.writer(output)
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        writer.writerow(cells)
    return output.getvalue()
