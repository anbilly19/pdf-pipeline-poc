"""Agent tools for the PDF Q&A pipeline.

All tools follow the contract from CLAUDE.md:
    return (result, source_bboxes, page_number, image_path)

Each tool is a LangChain @tool that returns a ToolResult dataclass
so the agent can render highlights in the frontend.
"""
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


@dataclass
class ToolResult:
    """Standardised tool output carrying result + source location.

    Args:
        content: The tool's primary output (text, CSV string, etc.).
        bboxes: Bounding boxes of the source region on the page.
        page_number: 1-based source page.
        image_path: Path to rendered page PNG for overlay.
    """

    content: str
    bboxes: list[list[float]]
    page_number: int
    image_path: str

    def __str__(self) -> str:
        return (
            f"{self.content}\n"
            f"[source: page {self.page_number}, "
            f"bboxes={self.bboxes[:2]}{'...' if len(self.bboxes) > 2 else ''}]"
        )


NO_RESULTS = "No relevant passages found."
NO_TABLE = "No table found."
NO_SECTION = "Section not found."
NO_REGION = "No region found on page."


def build_tools(retriever: BBoxRetriever) -> list[object]:
    """Build all agent tools bound to a retriever instance.

    Args:
        retriever: Initialised BBoxRetriever connected to the vector store.

    Returns:
        List of LangChain tool objects ready to bind to an LLM.
    """

    @tool
    def search_term(
        query: Annotated[str, "Natural language search query in German or English"],
        top_k: Annotated[int, "Number of results to return (default 5)"] = 5,
    ) -> str:
        """Search the document for text relevant to a query.

        Returns the most relevant passages with their page locations
        and bounding boxes for highlight rendering.
        """
        chunks = retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return NO_RESULTS

        results = [str(ToolResult(
            content=c.text,
            bboxes=c.bboxes,
            page_number=c.page_number,
            image_path=c.image_path,
        )) for c in chunks]
        return "\n\n---\n\n".join(results)

    @tool
    def extract_table_to_csv(
        query: Annotated[str, "Description of the table to find and extract"],
    ) -> str:
        """Find a table in the document and return it as CSV.

        Searches for table chunks matching the query, then formats
        the best match as CSV. Returns the CSV string and source location.
        """
        chunks = retriever.retrieve(query, top_k=5, filter_chunk_type="table")
        if not chunks:
            chunks = retriever.retrieve(query, top_k=3)

        if not chunks:
            return NO_TABLE

        best: Chunk = chunks[0]
        csv_str = _markdown_table_to_csv(best.text)

        result = ToolResult(
            content=csv_str,
            bboxes=best.bboxes,
            page_number=best.page_number,
            image_path=best.image_path,
        )
        return str(result)

    @tool
    def summarize_section(
        title: Annotated[str, "Title or topic of the section to summarise"],
    ) -> str:
        """Retrieve and summarise a named section from the document.

        Returns the raw text of the section with its page location.
        The LLM calling this tool should summarise the returned text.
        """
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
        """Return the bounding boxes for a region on a specific page.

        Use this when the user asks to highlight or point to a specific
        part of the document. Returns bbox coordinates and the page image path.
        """
        chunks = retriever.retrieve(query, top_k=10)
        page_chunks = [c for c in chunks if c.page_number == page_number]

        if not page_chunks:
            page_chunks = chunks[:3]

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
    """Convert a markdown table string to CSV format.

    Falls back to returning the original text if it is not a valid
    markdown table.

    Args:
        text: Markdown table string (pipe-delimited rows).

    Returns:
        CSV string, or original text if conversion fails.
    """
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
