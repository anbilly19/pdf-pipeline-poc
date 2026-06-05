"""Streamlit frontend for the PDF Q&A pipeline."""
from __future__ import annotations

# Must be first import — silences transformers __path__ spam before any model loads
import src.silence  # noqa: F401

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.agent.graph import build_agent
from src.indexer import DocumentIndexer
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.store import FAISSStore
from src.ui.overlay import render_page_with_bboxes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_OLLAMA_MODELS = [
    "qwen2.5:3b",
    "qwen2.5:7b",
    "gemma4:e2b",
    "llama3.2:3b",
    "llama3.1:8b",
    "mistral:7b",
]
_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
_DEFAULT_TOP_K = 15
_MAX_SOURCE_PILLS = 3


def _init_session() -> None:
    defaults = {
        "messages": [],
        "agent": None,
        "indexed_doc": None,
        "overlay_source": None,
        "latest_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource
def _get_shared_components() -> tuple[FAISSStore, ChunkEmbedder]:
    store = FAISSStore(persist_dir=OUTPUT_DIR / "faiss_index")
    embedder = ChunkEmbedder()
    return store, embedder


def _clear_index(store: FAISSStore) -> None:
    index_dir = OUTPUT_DIR / "faiss_index"
    for f in index_dir.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    store._index = None
    store._metadata = []
    store._texts = []


def main() -> None:
    st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="wide")
    _init_session()
    store, embedder = _get_shared_components()

    with st.sidebar:
        st.title("📄 PDF Q&A")
        st.caption("Agentic document assistant with bbox citation")
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            st.caption("🔍 LangSmith tracing enabled")
        st.divider()

        provider = st.radio("LLM provider", ["ollama", "openai"], horizontal=True)
        if provider == "openai":
            model = st.selectbox("Model", _OPENAI_MODELS)
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("OPENAI_API_KEY not set in .env")
        else:
            model = st.selectbox("Model", _OLLAMA_MODELS)

        st.divider()
        uploaded = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded and st.button("Index document", type="primary"):
            pdf_path = DATA_DIR / uploaded.name
            pdf_path.write_bytes(uploaded.read())
            with st.spinner("Extracting and indexing..."):
                _clear_index(store)
                indexer = DocumentIndexer(embedder=embedder, store=store)
                n = indexer.index(pdf_path)
            st.success(f"Indexed {n} chunks from {uploaded.name}")
            st.session_state.indexed_doc = pdf_path.stem
            st.session_state.messages = []
            st.session_state.overlay_source = None
            st.session_state.latest_sources = []
            retriever = BBoxRetriever(store=store, embedder=embedder, top_k=_DEFAULT_TOP_K)
            st.session_state.agent = build_agent(
                retriever=retriever, provider=provider, model=model
            )

        if st.session_state.indexed_doc:
            st.info(f"Active doc: **{st.session_state.indexed_doc}**")
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.overlay_source = None
            st.session_state.latest_sources = []
            st.rerun()

    chat_col, view_col = st.columns([3, 2])

    with chat_col:
        st.subheader("Chat")
        if not st.session_state.agent:
            st.info("Upload and index a PDF to start.")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask a question about the document..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        from langchain_core.messages import HumanMessage  # noqa: PLC0415
                        result = st.session_state.agent.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                        )
                        last = result["messages"][-1]
                        answer = last.content
                    st.markdown(answer)
                sources = _extract_sources_from_messages(result["messages"])
                st.session_state.latest_sources = sources
                if sources:
                    st.session_state.overlay_source = sources[0]
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                st.rerun()

    with view_col:
        st.subheader("Page View")

        latest_sources = st.session_state.get("latest_sources", [])
        if latest_sources:
            st.caption("Sources (latest query):")
            cols = st.columns(len(latest_sources))
            for i, src in enumerate(latest_sources):
                with cols[i]:
                    if st.button(f"Page {src['page']}", key=f"pill_{i}"):
                        st.session_state.overlay_source = src
                        st.rerun()

        src = st.session_state.get("overlay_source")
        if src:
            img = render_page_with_bboxes(str(src["image_path"]), src["bboxes"])
            if img:
                st.image(img, caption=f"Page {src['page']}", use_container_width=True)
            else:
                st.warning(
                    f"Page {src['page']} image not found at `{src['image_path']}`. "
                    "Re-index the document to regenerate page images."
                )
        else:
            st.info("Source pages will appear here after a query.")


def _extract_sources_from_messages(messages: list[object]) -> list[dict[str, object]]:
    from langchain_core.messages import ToolMessage  # noqa: PLC0415
    import re, ast  # noqa: PLC0415, E401

    sources: list[dict[str, object]] = []
    seen_pages: set[int] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for match in re.finditer(
            r"\[source: page (\d+), bboxes=(\[\[.*?\]\]), image_path=('.*?'|\".*?\")",
            str(msg.content),
            re.DOTALL,
        ):
            if len(sources) >= _MAX_SOURCE_PILLS:
                break
            try:
                page = int(match.group(1))
                if page in seen_pages:
                    continue
                bboxes = ast.literal_eval(match.group(2))
                image_path = ast.literal_eval(match.group(3))
                if not image_path or not Path(image_path).exists():
                    fallback = OUTPUT_DIR / "pages" / f"page_{page:04d}.png"
                    if fallback.exists():
                        image_path = str(fallback)
                sources.append({"page": page, "bboxes": bboxes, "image_path": image_path})
                seen_pages.add(page)
            except Exception:  # noqa: BLE001
                pass
        if len(sources) >= _MAX_SOURCE_PILLS:
            break
    return sources


if __name__ == "__main__":
    main()
