# CLAUDE.md – Agentic PDF Pipeline with Bounding‑Box Citation

## 1. Project Overview

You are helping build a **modular, agentic document Q&A system** for complex German‑language PDFs (SG Magazin).  
The system must answer natural‑language questions and **return precise bounding‑box coordinates** of the evidence on the original page.

**Primary goal:** Every retrieved answer must be accompanied by exact page regions that can be highlighted visually.  
**Secondary goal:** Evolve into a tool‑using agent (ReAct loop) that can explore documents, extract tables, and perform multi‑step reasoning.  
**Hard constraint:** The entire stack must run **100% offline, no GPU required, no HuggingFace API calls**.

---

## 2. Core Principles (Never Violate)

- **Bounding‑box first** – Every piece of text, table, or figure must retain its original page coordinates from the moment of PDF extraction.
- **Parser‑agnostic** – Swap extraction engines without rewriting downstream logic.
- **Fully local / on‑prem** – All models (LLM, embeddings, reranker) run via Ollama or are bundled with the library. No HuggingFace API, no cloud inference unless explicitly opted in.
- **No GPU required** – All components must run on CPU. GPU is a nice-to-have, never a requirement.
- **Agentic design** – The system must support tool use, multi‑turn interactions, and context preservation.

---

## 3. Technology Stack (Current Choices)

| Component          | Current / Target                                      | Fallback / Notes                           |
| ------------------ | ----------------------------------------------------- | ------------------------------------------ |
| PDF extraction     | **Kreuzberg** (Rust-based, precise bboxes, CPU)       | PyMuPDF + pymupdf4llm (CPU)                |
| Table extraction   | **Camelot** or **pdfplumber** (cell-level bboxes)     | pymupdf4llm markdown tables                |
| Chunking           | Layout-aware splitter with **sliding window overlap** (10–15%) | –                               |
| Embeddings         | `intfloat/multilingual-e5-small` via Ollama           | Ollama default embedding model             |
| Vector store       | **FAISS** (fully offline, no server)                  | –                                          |
| Retrieval          | Hybrid BM25 + FAISS → **Ollama reranker** (cross-encoder pass) | Raw FAISS order                 |
| Knowledge filter   | **Self-RAG loop** via local Ollama call (relevance check per chunk before generation) | Score threshold on BM25     |
| LLM (local)        | Gemma / LeoLM via **Ollama** (CPU)                    | GPT-4o mini (cloud, only if allowed)       |
| Memory             | **LangGraph checkpoint** with isolated keys (retrieval ctx ≠ conversation history) | Stateless per query          |
| Frontend           | Streamlit (canvas overlays)                           | FastAPI + React                            |
| Containerisation   | Docker (optional)                                     | –                                          |

> ⚠️ **MinerU is retired** as a target. It requires GPU and heavy dependencies that violate the no-GPU, no-HuggingFace constraint. Do not reintroduce it.

---

## 4. Roadmap — Prioritised Upgrade Order

Follow this order. Do not skip ahead.

1. **Cross-encoder reranking via Ollama** — Add a second-pass reranker after BM25+FAISS retrieval using an Ollama-hosted reranker model (e.g. `bge-reranker`). This is the single highest-ROI improvement for small model accuracy.
2. **Embedding upgrade to `intfloat/multilingual-e5-small`** — Pull the model via Ollama. Directly improves German umlaut and compound noun retrieval quality.
3. **Sliding window chunk overlap (10–15%)** — Fix answer truncation at chunk boundaries. Zero infrastructure cost.
4. **Kreuzberg extraction layer** — Replace the `pdfmux`/`pymupdf-layout` extraction path with Kreuzberg for precise, Rust-speed bboxes. Validate bbox contract compatibility with `Page → Chunk → Response` chain before switching.
5. **LangGraph persistent memory with isolated keys** — Re-introduce `langgraph-checkpoint`. Store conversation history in a separate key from retrieval context. Enables natural follow-up questions without bbox bleed.
6. **Self-RAG knowledge filter** — After retrieval, run a local Ollama call per chunk to verify relevance before passing to the final generator. Only enable this if latency is acceptable after steps 1–5.

---

## 5. Core Data Models (Immutable Contracts)

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

> **Citation metadata always comes from the child chunk**, even when parent-chunk context is expanded for generation.

---

## 6. Coding Standards & Quality Gates

- **Type hints** required for all function signatures.
- **Docstrings** (Google style) for every public function and class.
- **Logging** using Python's `logging` module – no `print()` in production paths.
- **Error handling** – extraction failures, missing fonts, corrupted PDFs must not crash the pipeline.
- **Context window discipline** – never pass more than **4,000–5,000 characters** of context to the LLM in a single generation call. Small models lose citation accuracy beyond this.
- **Unit tests** for:
  - Normalisation layer (parser → Page)
  - Chunking (bounding box aggregation and alignment)
  - Retrieval (metadata integrity)
  - Bounding‑box IoU against ground‑truth (≥0.7 acceptable for most cases).

---

## 7. Key Constraints When You Generate Code

1. **Never discard bounding boxes** – Every transformation (chunking, embedding storage, LLM response) must carry them forward.
2. **Prefer local inference** – All model calls go through Ollama. If a code path requires an API key (e.g., OpenAI), make it optional and clearly documented.
3. **No HuggingFace API** – Do not add `sentence-transformers` or any library that requires a HuggingFace API token at runtime. Offline model weights pulled via Ollama are acceptable.
4. **No GPU assumption** – All default code paths must run on CPU. GPU optimisations may exist as optional branches but must never be required.
5. **Parser fallback logic** – When Kreuzberg fails (low confidence or parse error), automatically rerun with PyMuPDF and merge results, keeping the highest‑confidence bounding boxes.
6. **Agent tools must return metadata** – Any tool that extracts information (table, summary) must also return the source bounding boxes.
7. **Self-RAG filter is latency-gated** – Only invoke the relevance filter Ollama call if the chunk's BM25 score is below a configurable threshold. Do not call it unconditionally.

---

## 8. Agent Tool Definitions (Current + Planned)

### Currently implemented
```python
tools = [
    Tool(name="search_term",          # semantic + BM25 hybrid search
         func=lambda query, top_k: ...),
    Tool(name="extract_table_to_csv",  # find table chunk → CSV
         func=lambda query: ...),
    Tool(name="summarize_section",     # retrieve + combine section chunks
         func=lambda title: ...),
    Tool(name="highlight_section",     # return bboxes for a page region
         func=lambda page_number, query: ...),
]
```

### Planned (add after roadmap step 1–3)
```python
tools += [
    Tool(name="rerank_and_filter",     # cross-encoder rerank via Ollama
         func=lambda query, chunks: ...),
    Tool(name="verify_relevance",      # Self-RAG chunk relevance check
         func=lambda query, chunk: ...),  # returns (is_relevant: bool, score: float)
]
```

All tools **must** return `(result, source_bboxes, page_number, image_path)`.

---

## 9. Testing & Evaluation Metrics

- **Extraction quality** – BLEU, edit distance (text), IoU (bounding boxes) against manually annotated ground truth.
- **Retrieval accuracy** – hit rate of top‑k chunks containing the correct answer; establish a baseline before any roadmap change.
- **End‑to‑end** – user studies on SG Magazin PDFs to verify that highlighted regions are correct and useful.
- **Latency budget** – Self-RAG filter (step 6) is only acceptable if total response time stays under 10s on CPU.

---

## 10. Important Reminders for You (Claude)

- When asked to modify or extend the pipeline, **always preserve the `bbox` → `Page` → `Chunk` → `Response` chain**.
- Prefer **small, testable functions** over monolithic scripts.
- If you are uncertain about coordinate systems (points vs pixels, PDF native vs rendered image), ask for clarification.
- Never assume a parser will produce perfect reading order – the normalisation layer should not rely on element ordering.
- The project is **German‑first** – ensure text processing (tokenisation, stopwords, LLM prompts) works well with German umlauts and compound nouns.
- **Do not suggest MinerU** under any circumstances. It is permanently retired from this project.
- **Do not suggest HuggingFace API-dependent libraries** (e.g. `sentence-transformers` with a remote token). Offline weight files pulled via Ollama are the only acceptable model delivery mechanism.
