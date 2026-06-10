"""Agent tools for the PDF Q&A pipeline."""
from __future__ import annotations

import csv
import io
import logging
import os
from dataclasses import dataclass
from typing import Annotated

from langchain_core.tools import tool

from src.models import Chunk
from src.retrieval.retriever import BBoxRetriever

logger = logging.getLogger(__name__)
_DBG = os.environ.get("DEBUG_PIPELINE", "0") == "1"

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
    self_rag_enabled: bool = False,
    self_rag_bm25_gate: float = 0.5,
) -> list[object]:
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

        # --- CHECKPOINT 6: what the tool actually sends ---
        if _DBG:
            logger.info(
                "[DBG-CP6-TOOL] search_term retrieved %d chunks for query=%r",
                len(chunks), query,
            )
            for i, c in enumerate(chunks[:3], 1):
                logger.info(
                    "[DBG-CP6-TOOL] chunk %d: page=%d  bboxes=%s  image_path=%r  text[:60]=%r",
                    i, c.page_number, c.bboxes, c.image_path, c.text[:60],
                )

        parts = [f"FOUND SECTIONS ({len(chunks)} total \u2014 read all):\n"]
        for i, c in enumerate(chunks, 1):
            result = ToolResult(
                content=c.text,
                bboxes=c.bboxes,
                page_number=c.page_number,
                image_path=c.image_path,
            )
            parts.append(f"--- Abschnitt {i} ---\n{result}")

        raw_output = "\n\n".join(parts)
        if _DBG:
            logger.info(
                "[DBG-CP6-TOOL] raw ToolMessage snippet (first 400 chars):\n%s",
                raw_output[:400],
            )
        return raw_output

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
        best = chunks[0]
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

    return [search_term, extract_table_to_csv, summarize_section, highlight_section]


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
