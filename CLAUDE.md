# CLAUDE.md – Coding Standards

This file defines **coding standards only**. It does not describe architecture, libraries, or project roadmap.

---

## 1. General Code Style

- **Python 3.10+** syntax only.
- **Type hints** required on all function signatures — parameters and return types.
- **Docstrings** (Google style) on every public function and class.
- **No `print()` in production paths** — use Python's `logging` module exclusively.
- **No magic numbers** — constants go in a dedicated `config.py` or at the module top.
- Line length: **100 characters max**.
- Imports: stdlib → third-party → local, separated by blank lines.

---

## 2. Error Handling

- Extraction failures, missing fonts, and corrupted PDFs must **never crash the pipeline** — catch and log, then continue or fall back.
- Use specific exception types, not bare `except Exception`.
- Every fallback path must emit a `logging.warning()` with the reason.

---

## 3. Bounding Box Contract

- **Never discard bounding boxes.** Every transformation (chunking, embedding, LLM response) must carry `bboxes: List[List[float]]` forward.
- `bbox` format is always `[x0, y0, x1, y1]` in PDF points unless explicitly noted.
- Citation metadata always comes from the **child chunk**, even when parent-chunk context is expanded for generation.
- Any tool that returns extracted information must also return `(result, source_bboxes, page_number, image_path)`.

---

## 4. LLM / Inference Rules

- All model calls go through **Ollama**. No API keys at runtime unless explicitly optional and documented.
- **No HuggingFace API** — do not add `sentence-transformers` or any library requiring a remote HuggingFace token.
- **No GPU assumption** — all default code paths must run on CPU. GPU branches are optional and clearly gated.
- **Context window discipline** — never pass more than **4,000–5,000 characters** of context in a single LLM call. Small models lose citation accuracy beyond this.
- **Self-RAG filter is latency-gated** — only invoke the relevance-check Ollama call if the chunk's BM25 score is below a configurable threshold. Do not call it unconditionally.

---

## 5. Function and Module Design

- Prefer **small, single-responsibility functions** over monolithic scripts.
- Module names are `snake_case`; class names are `PascalCase`.
- Do not put business logic in `app.py` — it is for UI wiring only.
- Agent tools must be **stateless** — they receive inputs and return outputs with no hidden side effects.

---

## 6. Testing Standards

- Tests live in `tests/` mirroring the `src/` structure.
- Every public function in `extraction/`, `chunking/`, and `retrieval/` must have a unit test.
- Required test coverage areas:
  - Normalisation layer (parser → `Page`)
  - Chunking (bbox aggregation and alignment)
  - Retrieval (metadata integrity through the pipeline)
  - Bbox IoU ≥ 0.7 against ground truth for extraction tests
- Use `pytest` — no `unittest` boilerplate.
- Mocks for Ollama calls are required in all unit tests (do not call live models in CI).

---

## 7. Commit & PR Standards

- Commits are **atomic** — one logical change per commit.
- Commit message format: `<type>: <short description>` (e.g. `fix: handle empty bbox list in chunker`).
- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Do not commit commented-out code or debug `print()` statements.
