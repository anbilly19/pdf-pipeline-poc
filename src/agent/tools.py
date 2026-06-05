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

_TOP_K = 10  # hardcoded, model cannot override


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


def build_tools(retriever: BBoxRetriever) -> list[object]:

    @tool
    def search_term(
        query: Annotated[str, "Suchanfrage auf Deutsch"],
    ) -> str:
        """Durchsucht das Dokument nach relevanten Textabschnitten.

        Gibt die 10 relevantesten Abschnitte zurück.
        WICHTIG: Lies ALLE Abschnitte (1-10) vollständig durch.
        Der gesuchte Inhalt kann in Abschnitt 3, 5 oder 8 stehen — nicht nur in Abschnitt 1.
        Benutze den Text aus diesen Abschnitten für deine Antwort.
        """
        chunks = retriever.retrieve(query, top_k=_TOP_K)
        if not chunks:
            return NO_RESULTS

        parts = ["GEFUNDENE ABSCHNITTE (lies alle durch):\n"]
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
