# PDF Pipeline PoC

Agentic document Q&A system for German-language contract PDFs with **bounding-box citation**. Every answer is accompanied by the exact page coordinates of its source, highlighted visually in the UI.

**Hard constraints:** 100% offline · CPU-first (4 GB VRAM budget) · no HuggingFace API.

---

## Current Status (June 2026)

### ✅ Working
- **PDF indexing** — PyMuPDF extraction, chunking, FAISS embedding, knowledge graph build all complete successfully
- **Hybrid retrieval** — BM25 + FAISS fusion returning 23 chunks per query, correctly split and passed to the agent
- **Tool calling (Phase 1)** — all tested models reliably call `search_term` with a relevant German query
- **Bbox citation** — source page numbers extracted and rendered as clickable pills; page overlay viewer works
- **Self-RAG filter** — per-chunk relevance gate functional, stats shown in status bar
- **Domain routing** — query type detection dispatches correct contract-domain prompt
- **LangSmith tracing** — full run traces visible when `LANGCHAIN_TRACING_V2=true`
- **Streamlit UI** — sidebar model/provider selector, context window slider, reranker toggle all functional
- **OpenAI fallback** — `gpt-4o-mini` / `gpt-4o` work when `OPENAI_API_KEY` is set

### ⚠️ Known Issues

| Issue | Root cause | Workaround |
|---|---|---|
| `gemma4:e2b` answers cut off (`done_reason: length`) | Phase-2 `num_predict` was 300, Gemma outputs more verbose responses | Fixed in latest commit — raised to 512 |
| `qwen3.5:4b` (thinking variant) sometimes returns empty content | `<think>` block consumes the entire token budget | Use `qwen3.5:4b` with ctx=2048, `_strip_think()` removes the block |
| Irrelevant chunks selected in Phase-2 answer (e.g. Nutzungsrechte instead of Servicezeiten) | Model picks longest/most-similar chunk instead of most directly relevant | Prompt now explicitly instructs model to ignore off-topic sections |
| Page number questions ("Auf welcher Seite...") may answer incorrectly | `[source: page N]` metadata is stripped before the answer prompt | Known limitation — page metadata not yet surfaced in answer context |
| `done_reason: length` with empty `content: ""` | Token budget exhausted before model writes any output | Fixed by raising `_ANSWER_NUM_PREDICT` to 512 |

### ❌ Not Yet Implemented
- Table extraction to CSV (`extract_table_to_csv` tool exists but untested end-to-end)
- Multi-document indexing (single PDF only)
- Kreuzberg/MinerU extraction backend
- Persistent conversation history across browser sessions

---

## Models

| Model | Size | Role | Notes |
|---|---|---|---|
| `gemma4:e2b` | 7.2 GB | Primary (UI default) | Good German instruction-following; set ctx=1024 to avoid OOM |
| `gemma4:e4b` | 9.6 GB | Alternate | Larger; set ctx=1024 only |
| `qwen3.5:4b` | 3.4 GB | Lightweight default | Strong tool-calling + German; has thinking mode — `_strip_think()` handles it |
| `qwen3.5:2b` | 2.7 GB | Fallback | CPU-only option |
| `nomic-embed-text` | — | Embeddings | Local via Ollama; no remote calls |

Switch models at runtime from the Streamlit sidebar. OpenAI (`gpt-4o-mini`) is available as an optional fallback — requires `OPENAI_API_KEY` in `.env`.

> **Gemma4 memory note:** `gemma4:e2b` requires ~7 GB RAM. Set the context window slider to **1024** in the sidebar to avoid out-of-memory errors on machines with ≤8 GB free RAM.

---

## Features

### Document Processing
- **PyMuPDF-based extraction** (`pymupdf4llm`) with layout-aware parsing — preserves reading order, tables, and per-element bounding boxes
- **Sliding window chunking (10–15% overlap)** — prevents answer truncation at chunk boundaries
- **Bbox-first pipeline** — every chunk carries `[x0, y0, x1, y1]` coordinates from extraction through to the final response

### Retrieval
- **Hybrid BM25 + FAISS** — keyword recall fused with semantic similarity for German contract vocabulary
- **Ollama `nomic-embed-text` embeddings** — local, no remote calls
- **Query expansion** for German legal terms to improve BM25 hit rate on compound nouns
- **Page-rank decay boost** — chunks from earlier pages get a mild boost
- **Self-RAG filter** — per-chunk relevance verification via a local Ollama call, latency-gated by a configurable BM25 threshold

### Agent
- **Two-phase LangGraph ReAct agent:**
  - Phase 1 — tool call (`search_term`)
  - Phase 2 — focused answer prompt with trimmed, deduplicated chunks; model instructed to use only directly relevant passages
- **Domain routing** — German contract-specific prompts per query type
- **`_strip_think()`** — removes `<think>...</think>` blocks from thinking models

### UI
- **Streamlit** frontend with right-hand page viewer
- **Bbox overlay** — highlighted source regions rendered over the original PDF page image
- **Model/provider selector**, context window slider, reranker toggle in sidebar
- **LangSmith tracing** indicator in sidebar

---

## Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Dev dependencies
uv sync --group dev

# Copy env file and add keys if needed
cp .env.example .env
```

## Running

```bash
uv run streamlit run src/app.py
```

## Testing

```bash
uv run pytest
```

---

## Technical Decisions

**PyMuPDF over Kreuzberg/MinerU** — MinerU requires GPU and heavy HuggingFace dependencies. PyMuPDF delivers reliable bboxes today with zero extra dependencies.

**FAISS over a vector DB server** — No server process, no network overhead, serialises to a single file. Sufficient at PoC scale.

**BM25 hybrid over pure semantic** — German legal text has high keyword density (`Auftraggeber`, `Vertragsstrafe`, clause numbers). BM25 recovers exact term matches cheaply.

**Ollama over sentence-transformers** — Keeps the entire inference stack in one runtime. No HuggingFace token, no CUDA dependency.

**LangGraph over plain LangChain** — Explicit state graph with typed keys makes it straightforward to isolate conversation history from retrieval context.

---

## Project Structure

```
pdf-pipeline-poc/
├── src/
│   ├── extraction/      # PyMuPDF-based PDF parsing with bbox preservation
│   ├── chunking/        # Layout-aware chunking with sliding window overlap
│   ├── retrieval/       # Hybrid BM25+FAISS, Self-RAG filter, query expansion
│   ├── agent/           # LangGraph ReAct agent + bbox-preserving tools
│   └── app.py           # Streamlit UI with bbox overlay and model selector
├── tests/
├── data/                # PDFs (gitignored)
├── CLAUDE.md            # Coding standards
└── pyproject.toml
```
