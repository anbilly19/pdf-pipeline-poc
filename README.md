# PDF Pipeline PoC

Modular, agentic document Q&A system for complex German-language PDFs (SG Magazin) with **bounding-box citation**.

Every answer is accompanied by exact page coordinates that can be highlighted visually.

**Hard constraints:** 100% offline · no GPU required · no HuggingFace API.

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv sync

# Dev dependencies
uv sync --group dev
```

## Running

```bash
uv run streamlit run src/app.py
```

## Testing

```bash
uv run pytest
```

## Project Structure

```
pdf-pipeline-poc/
├── src/
│   ├── extraction/      # PDF parsers (Kreuzberg → PyMuPDF fallback)
│   ├── chunking/        # Layout-aware chunking with sliding window overlap
│   ├── retrieval/       # Hybrid BM25+FAISS + Ollama cross-encoder reranker
│   ├── agent/           # LangGraph ReAct agent + bbox-preserving tools
│   └── app.py           # Streamlit frontend with bbox overlay
├── tests/
├── data/                # PDFs (gitignored)
├── CLAUDE.md            # Architecture, principles, and roadmap
└── pyproject.toml
```

## Architecture

See [CLAUDE.md](./CLAUDE.md) for full design principles, data models, roadmap, and coding standards.

## Upgrade Roadmap

Ordered by impact. All steps are CPU-only and fully offline.

| # | Upgrade | Why |
|---|---------|-----|
| 1 | **Ollama cross-encoder reranking** | Second-pass rerank over BM25+FAISS candidates — highest-ROI accuracy gain for small models |
| 2 | **`multilingual-e5-small` embeddings via Ollama** | German umlaut + compound noun retrieval quality |
| 3 | **Sliding window chunk overlap (10–15%)** | Fix answer truncation at chunk boundaries |
| 4 | **Kreuzberg extraction layer** | Rust-speed, precise cell-level bboxes; replaces pdfmux/pymupdf-layout |
| 5 | **LangGraph persistent memory (isolated keys)** | Multi-turn follow-up questions without retrieval context bleed |
| 6 | **Self-RAG knowledge filter** | Per-chunk relevance verification via local Ollama call (latency-gated) |

## LLM Providers

Switch at runtime from the Streamlit sidebar:
- **Ollama** (default, fully local) — `gemma4:e2b` or any locally pulled model
- **OpenAI** (optional, requires `OPENAI_API_KEY` in `.env`) — `gpt-4o-mini`

Copy `.env.example` to `.env` and fill in keys as needed.
