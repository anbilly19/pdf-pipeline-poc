# PDF Pipeline PoC

Agentic document Q&A system for German-language contract PDFs with **bounding-box citation**. Every answer is accompanied by the exact page coordinates of its source, highlighted visually in the UI.

**Hard constraints:** 100% offline · CPU-first (4 GB VRAM budget) · no HuggingFace API.

---

## Features

### Document Processing
- **PyMuPDF-based extraction** (`pymupdf4llm`) with layout-aware parsing — preserves reading order, tables, and per-element bounding boxes
- **Sliding window chunking (10–15% overlap)** — prevents answer truncation at chunk boundaries
- **Bbox-first pipeline** — every chunk carries `[x0, y0, x1, y1]` coordinates from extraction through to the final response; coordinates are never discarded

### Retrieval
- **Hybrid BM25 + FAISS** — keyword recall (BM25) fused with semantic similarity (FAISS) for German contract vocabulary
- **Ollama `nomic-embed-text` embeddings** — local, no remote calls, good multilingual coverage
- **Query expansion** for German legal terms (e.g. `Verschwiegenheit`, `Auftragnehmer`) to improve BM25 hit rate on compound nouns
- **Page-rank decay boost** — chunks from earlier pages get a mild boost to prefer introductory/definitional text
- **Self-RAG filter** — per-chunk relevance verification via a local Ollama call, latency-gated by a configurable BM25 score threshold

### Agent
- **LangGraph ReAct agent** — tool-using loop with multi-turn context preservation
- **LangGraph persistent memory** — conversation history and retrieval context stored under isolated keys to prevent bleed between turns
- **Domain routing** — German contract-specific prompts dispatched based on query type (clause lookup, date extraction, penalty calculation, etc.)
- **Bbox-preserving tools:**
  - `search_term` — hybrid semantic + BM25 search
  - `extract_table_to_csv` — table chunk → structured CSV
  - `summarize_section` — multi-chunk section summary
  - `highlight_section` — bbox return for a page region
  - `rerank_and_filter` — post-retrieval cross-encoder rerank
  - `verify_relevance` — Self-RAG chunk relevance check

### UI
- **Streamlit** frontend with a right-hand page viewer
- **Bbox overlay** — highlighted source regions rendered over the original PDF page image
- **Self-RAG status panel** — shows per-chunk relevance scores
- **BM25 gate slider** — tune the hybrid retrieval balance at runtime
- **Model selector** — switch between local Ollama models from the sidebar

---

## Models

| Model | Role | Notes |
|---|---|---|
| `qwen3:4b` | Primary LLM | Strong German tool-calling; fits 4 GB VRAM |
| `gemma4:e4b` | Alternate LLM | Good instruction following; similar VRAM profile |
| `qwen2.5:3b` | Lightweight fallback | CPU-only option for low-resource runs |
| `nomic-embed-text` | Embeddings | Local via Ollama; no remote calls |

Switch models at runtime from the Streamlit sidebar. OpenAI (`gpt-4o-mini`) is available as an optional fallback — requires `OPENAI_API_KEY` in `.env`.

---

## Technical Decisions

**PyMuPDF over Kreuzberg/MinerU** — MinerU requires GPU and heavy HuggingFace dependencies, violating the offline/CPU constraint. Kreuzberg (Rust-based) is planned but not yet stable for our bbox contract. PyMuPDF delivers reliable bboxes today with zero extra dependencies.

**FAISS over a vector DB server** — No server process, no network overhead, serialises to a single file. At PoC scale (hundreds of pages) this is faster than standing up Qdrant or Chroma.

**BM25 hybrid over pure semantic** — German legal text has high keyword density (`Auftraggeber`, `Vertragsstrafe`, clause numbers). Pure semantic retrieval on a 4B model misses exact term matches. BM25 recovers these cheaply.

**Ollama over sentence-transformers** — Keeps the entire inference stack in one runtime. No HuggingFace token, no CUDA dependency, works CPU-only out of the box.

**LangGraph over plain LangChain** — Provides explicit state graph with typed keys, making it straightforward to isolate conversation history from retrieval context and to add new tool nodes without rewriting the loop.

**Qwen3:4b as default over 8B** — 8B models exceed the 4 GB VRAM budget and are too slow on CPU-only runs. Qwen3:4b benchmarks close to Qwen2.5:7B on instruction-following and German tool-calling at half the size.

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
