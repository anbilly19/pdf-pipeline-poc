# CLAUDE.md – Agentic PDF Pipeline with Bounding‑Box Citation

## 1. Project Overview

You are helping build a **modular, agentic document Q&A system** for complex German‑language PDFs (SG Magazin).  
The system must answer natural‑language questions and **return precise bounding‑box coordinates** of the evidence on the original page.

**Primary goal:** Every retrieved answer must be accompanied by exact page regions that can be highlighted visually.  
**Secondary goal:** Evolve into a tool‑using agent (ReAct loop) that can explore documents, extract tables, and perform multi‑step reasoning.

---

## 2. Core Principles (Never Violate)

- **Bounding‑box first** – Every piece of text, table, or figure must retain its original page coordinates from the moment of PDF extraction.
- **Parser‑agnostic** – Swap extraction engines without rewriting downstream logic (MinerU, PyMuPDF, etc.).
- **Fully local / on‑prem capable** – Prefer local models (LLM, embeddings) unless privacy is explicitly waived.
- **Agentic design** – The system must support tool use, multi‑turn interactions, and context preservation.

---

## 3. Technology Stack (Current Choices)

| Component          | Recommended                                      | Fallback / Alternative                     |
| ------------------ | ------------------------------------------------ | ------------------------------------------ |
| PDF extraction     | MinerU (VLM+OCR, GPU)                            | PyMuPDF + pymupdf4llm (CPU)                |
| Chunking           | Layout‑aware splitter (headings + table preservation) | -                                          |
| Embeddings         | `intfloat/multilingual-e5-small`                | `all-MiniLM-L6-v2`                         |
| Vector store       | Chroma (with metadata filters) or FAISS          | -                                          |
| LLM (local)        | Llama‑3.2‑1B / LeoLM via Ollama or llama.cpp     | GPT‑4o mini (cloud, only if allowed)       |
| Frontend           | Streamlit or Gradio (canvas overlays)            | FastAPI + React                            |
| Containerisation   | Docker (optional)                                | -                                          |

---

## 4. Core Data Models (Immutable Contracts)

### Page (after normalisation)
```python
@dataclass
class Page:
    page_number: int
    image_path: str          # rendered PNG of the page
    elements: List[Element]

@dataclass
class Element:
    type: Literal["text", "table", "image"]
    text: str                # plain or markdown
    bbox: List[float]        # [x0, y0, x1, y1] in points or pixels
    confidence: float        # parser confidence (0..1)
```

### Chunk (retrieval unit)
```python
@dataclass
class Chunk:
    text: str
    page_number: int
    bboxes: List[List[float]]   # aggregated from all elements in this chunk
    chunk_type: Literal["text", "table", "figure"]
    confidence: float
    image_path: str
```

### QA Response (final output)
```python
@dataclass
class QAResponse:
    answer: str
    sources: List[Source]

@dataclass
class Source:
    text: str
    page: int
    bboxes: List[List[float]]
    image: str   # path to page image
```

---

## 5. Coding Standards & Quality Gates

- **Type hints** required for all function signatures.
- **Docstrings** (Google style) for every public function and class.
- **Logging** using Python's `logging` module – no `print()` in production paths.
- **Error handling** – extraction failures, missing fonts, corrupted PDFs must not crash the pipeline.
- **Unit tests** for:
  - Normalisation layer (parser → Page)
  - Chunking (bounding box aggregation and alignment)
  - Retrieval (metadata integrity)
  - Bounding‑box IoU against ground‑truth (≥0.7 acceptable for most cases).

---

## 6. Key Constraints When You Generate Code

1. **Never discard bounding boxes** – Every transformation (chunking, embedding storage, LLM response) must carry them forward.
2. **Prefer local inference** – If a code path requires an API key (e.g., OpenAI), make it optional and clearly documented.
3. **Parser fallback logic** – When MinerU fails (low confidence), automatically rerun with PyMuPDF and merge results, keeping the highest‑confidence bounding boxes.
4. **Agent tools must return metadata** – Any tool that extracts information (table, summary) must also return the source bounding boxes.

---

## 7. Agent Tool Definitions (To Be Implemented)

Once the basic QA pipeline works, you will implement these tools in a ReAct loop:

```python
tools = [
    Tool(name="highlight_section", 
         func=lambda page, bbox: ...),   # already in frontend
    Tool(name="extract_table_to_csv",
         func=lambda page, bbox: ...),
    Tool(name="search_term",
         func=lambda term: ...),          # full‑text search
    Tool(name="summarize_section",
         func=lambda title: ...)
]
```

All tools **must** return `(result, source_bboxes, page_number, image_path)`.

---

## 8. Testing & Evaluation Metrics

- **Extraction quality** – BLEU, edit distance (text), IoU (bounding boxes) against manually annotated ground truth.
- **Retrieval accuracy** – hit rate of top‑k chunks containing the correct answer.
- **End‑to‑end** – user studies on SG Magazin PDFs to verify that highlighted regions are correct and useful.

---

## 9. Important Reminders for You (Claude)

- When asked to modify or extend the pipeline, **always preserve the `bbox` → `Page` → `Chunk` → `Response` chain**.
- Prefer **small, testable functions** over monolithic scripts.
- If you are uncertain about coordinate systems (points vs pixels, PDF native vs rendered image), ask for clarification.
- Never assume a parser will produce perfect reading order – the normalisation layer should not rely on element ordering.
- The project is **German‑first** – ensure text processing (tokenisation, stopwords, LLM prompts) works well with German umlauts and compound nouns.
