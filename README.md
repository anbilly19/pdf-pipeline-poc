# PDF Pipeline PoC

Modular, agentic document Q&A system for complex German-language PDFs (SG Magazin) with **bounding-box citation**.

Every answer is accompanied by exact page coordinates that can be highlighted visually.

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv sync

# With GPU extras (MinerU)
uv sync --extra gpu

# Dev dependencies
uv sync --extra dev
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
│   ├── extraction/      # PDF parsers (MinerU, PyMuPDF)
│   ├── chunking/        # Layout-aware chunking
│   ├── retrieval/       # Vector store + metadata filters
│   ├── agent/           # ReAct agent + tools
│   └── app.py           # Streamlit frontend
├── tests/
├── data/                # PDFs (gitignored)
├── CLAUDE.md
└── pyproject.toml
```

## Architecture

See [CLAUDE.md](./CLAUDE.md) for full design principles, data models, and coding standards.
