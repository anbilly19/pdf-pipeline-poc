"""Streamlit frontend for the PDF Q&A pipeline.

Features:
- PDF upload and indexing
- Multi-turn chat with the LangGraph agent
- Page image rendering with bbox highlight overlays
- Source citations with clickable page references
"""
from __future__ import annotations

import logging
import uuid
import warnings
from pathlib import Path

# suppress noisy transformers __path__ deprecation warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

from src.agent.graph import build_agent
from src.indexer import DocumentIndexer
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.store import ChromaStore
from src.ui.overlay import render_page_with_bboxes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role, content, sources}
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "indexed_doc" not in st.session_state:
        st.session_state.indexed_doc = None


@st.cache_resource
def _get_shared_components() -> tuple[ChromaStore, ChunkEmbedder]:
    """Initialise heavy components once and cache across reruns."""
    store = ChromaStore(persist_dir=OUTPUT_DIR / "chroma_db")
    embedder = ChunkEmbedder()
    return store, embedder


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PDF Q&A",
        page_icon="📄",
        layout="wide",
    )
    _init_session()

    store, embedder = _get_shared_components()

    # -----------------------------------------------------------------------
    # Sidebar: upload + index
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("📄 PDF Q&A")
        st.caption("Agentic document assistant with bbox citation")
        st.divider()

        uploaded = st.file_uploader("Upload a PDF", type="pdf")
        model = st.selectbox(
            "Ollama model",
            ["gemma4:e2b", "llama3.2:1b", "mistral:7b"],
            index=0,
        )

        if uploaded and st.button("Index document", type="primary"):
            pdf_path = DATA_DIR / uploaded.name
            pdf_path.write_bytes(uploaded.read())

            with st.spinner("Extracting and indexing..."):
                indexer = DocumentIndexer(
                    embedder=embedder,
                    store=store,
                )
                n = indexer.index(pdf_path)

            st.success(f"Indexed {n} chunks from {uploaded.name}")
            st.session_state.indexed_doc = pdf_path.stem
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())

            retriever = BBoxRetriever(store=store, embedder=embedder, top_k=5)
            st.session_state.agent = build_agent(retriever=retriever, model=model)

        if st.session_state.indexed_doc:
            st.info(f"Active doc: **{st.session_state.indexed_doc}**")

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    # -----------------------------------------------------------------------
    # Main area: chat + overlay
    # -----------------------------------------------------------------------
    chat_col, view_col = st.columns([3, 2])

    with chat_col:
        st.subheader("Chat")

        if not st.session_state.agent:
            st.info("Upload and index a PDF to start.")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        _render_source_pills(msg["sources"], view_col)

            if prompt := st.chat_input("Ask a question about the document..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        from langchain_core.messages import HumanMessage  # noqa: PLC0415
                        result = st.session_state.agent.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config={"configurable": {"thread_id": st.session_state.thread_id}},
                        )
                        last = result["messages"][-1]
                        answer = last.content

                    st.markdown(answer)
                    sources = _extract_sources_from_messages(result["messages"])
                    if sources:
                        _render_source_pills(sources, view_col)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })


def _extract_sources_from_messages(messages: list[object]) -> list[dict[str, object]]:
    """Parse ToolMessage results to extract page/bbox source info."""
    from langchain_core.messages import ToolMessage  # noqa: PLC0415
    import re, ast  # noqa: PLC0415, E401
    sources: list[dict[str, object]] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        match = re.search(r"\[source: page (\d+), bboxes=(\[.*?\])", str(msg.content))
        if match:
            try:
                page = int(match.group(1))
                bboxes = ast.literal_eval(match.group(2))
                sources.append({"page": page, "bboxes": bboxes, "image_path": _find_image_path(page)})
            except Exception:  # noqa: BLE001
                pass
    return sources


def _find_image_path(page_number: int) -> str:
    path = OUTPUT_DIR / "pages" / f"page_{page_number:04d}.png"
    return str(path) if path.exists() else ""


def _render_source_pills(sources: list[dict[str, object]], view_col: object) -> None:
    """Render source page buttons and update the overlay column on click."""
    if not sources:
        return
    st.caption("Sources:")
    cols = st.columns(len(sources))
    for i, src in enumerate(sources):
        with cols[i]:
            if st.button(f"Page {src['page']}", key=f"src_{uuid.uuid4()}"):
                st.session_state["overlay_source"] = src
                st.rerun()
    if sources and "overlay_source" not in st.session_state:
        st.session_state["overlay_source"] = sources[0]


if __name__ == "__main__":
    main()
