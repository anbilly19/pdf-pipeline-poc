"""Streamlit frontend for the PDF Q&A pipeline."""
from __future__ import annotations

import src.silence  # noqa: F401  — must be first

import logging
import os
import re
import ast
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.domain_config import DocTypeConfig, load_active_config
from src.agent.graph import build_agent
from src.agent.router import route_query
from src.graph.builder import load_graph
from src.indexer import DocumentIndexer
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.reranker import OllamaReranker
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.store import FAISSStore
from src.ui.overlay import render_page_with_bboxes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_MODEL_MAP: dict[str, str] = {
    "qwen3.5:4b":  "FieldMouse-AI/qwen3.5:4b-instruct",
    "qwen3.5:2b":  "qwen3.5:2b",
    "gemma4:e4b":  "gemma4:e4b-it-qat",
    "gemma4:e2b":  "gemma4:e2b",
}
_OLLAMA_MODELS = list(_MODEL_MAP.keys())
_LARGE_MODELS = {"gemma4:e4b", "gemma4:e2b"}
_OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
_DEFAULT_TOP_K = 15
_MAX_SOURCE_PILLS = 5
_CTX_OPTIONS = [128, 256, 512, 1024, 2048, 4096]
_DEFAULT_CTX = 2048

# Regex to extract source metadata from ToolMessage content.
# Handles: bboxes=[], bboxes=[[...]], bboxes=[x,y,x,y] (any list format)
_SOURCE_RE = re.compile(
    r"\[source: page (\d+), bboxes=(\[.*?\]), image_path=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")",
    re.DOTALL,
)


def _resolve_model(display_name: str) -> str:
    return _MODEL_MAP.get(display_name, display_name)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
  --bg:            #171614;
  --surface:       #1c1b19;
  --surface-2:     #22211f;
  --border:        #393836;
  --text:          #cdccca;
  --text-muted:    #797876;
  --primary:       #4f98a3;
  --primary-glow:  rgba(79,152,163,.15);
  --accent:        #e8af34;
  --radius:        10px;
  --font:          'Inter', sans-serif;
}

html, body, [class*="css"] { font-family: var(--font) !important; }
.stApp { background: var(--bg) !important; color: var(--text) !important; }

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  margin-bottom: .5rem !important;
}

[data-testid="stChatInput"] textarea {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px var(--primary-glow) !important;
}

.stButton > button {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
  font-weight: 500 !important;
  transition: all .15s ease !important;
}
.stButton > button:hover {
  border-color: var(--primary) !important;
  color: var(--primary) !important;
  background: var(--primary-glow) !important;
}
.stButton > button[kind="primary"] {
  background: var(--primary) !important;
  border-color: var(--primary) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover { background: #3a7f8a !important; }

[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: .75rem 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: .75rem !important; }

[data-baseweb="select"] > div {
  background: var(--surface-2) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius) !important;
}

[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

.source-strip { display: flex; gap: .4rem; flex-wrap: wrap; margin-top: .4rem; }
.source-pill {
  background: var(--primary-glow);
  border: 1px solid var(--primary);
  color: var(--primary);
  border-radius: 999px;
  padding: .15rem .6rem;
  font-size: .72rem;
  font-weight: 600;
  cursor: pointer;
  transition: background .15s;
}
.source-pill:hover { background: var(--primary); color: #fff; }

.domain-badge {
  display: inline-block;
  background: rgba(232,175,52,.12);
  border: 1px solid var(--accent);
  color: var(--accent);
  border-radius: 999px;
  padding: .1rem .55rem;
  font-size: .68rem;
  font-weight: 600;
  margin-left: .4rem;
}

.status-bar {
  display: flex; gap: 1.5rem; align-items: center;
  padding: .45rem .75rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: .72rem;
  color: var(--text-muted);
  margin-bottom: .75rem;
}
.status-bar span { display: flex; align-items: center; gap: .3rem; }
.dot-green  { width:7px; height:7px; border-radius:50%; background:#6daa45; display:inline-block; }
.dot-yellow { width:7px; height:7px; border-radius:50%; background:#e8af34; display:inline-block; }
.dot-red    { width:7px; height:7px; border-radius:50%; background:#dd6974; display:inline-block; }

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""


def _init_session() -> None:
    defaults: dict = {
        "messages": [],
        "indexed_doc": None,
        "overlay_source": None,
        "latest_sources": [],
        "active_provider": "ollama",
        "active_model": _resolve_model(_OLLAMA_MODELS[0]),
        "doc_config": None,
        "active_domain": None,
        "index_stats": {},
        "self_rag_stats": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource
def _get_shared_components() -> tuple[FAISSStore, ChunkEmbedder]:
    store = FAISSStore(persist_dir=OUTPUT_DIR / "faiss_index")
    embedder = ChunkEmbedder()
    return store, embedder


def _get_retriever(
    store: FAISSStore,
    embedder: ChunkEmbedder,
    enable_reranker: bool = True,
    reranker_model: str = "bge-reranker-v2-m3",
) -> BBoxRetriever:
    reranker = None
    if enable_reranker:
        reranker = OllamaReranker(model=reranker_model)
        if not reranker.is_available():
            reranker = None
    return BBoxRetriever(store=store, embedder=embedder, top_k=_DEFAULT_TOP_K, reranker=reranker)


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


def _extract_sources(messages: list) -> list[dict]:
    """Parse [source: page N, bboxes=..., image_path=...] from ToolMessages.

    Uses broadened regex that handles empty bboxes=[], flat lists, and
    nested lists. Skips duplicate pages.
    """
    sources: list[dict] = []
    seen: set[int] = set()
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        for m in _SOURCE_RE.finditer(str(msg.content)):
            if len(sources) >= _MAX_SOURCE_PILLS:
                break
            try:
                page = int(m.group(1))
                if page in seen:
                    continue
                bboxes_raw = m.group(2)
                bboxes = ast.literal_eval(bboxes_raw)
                # Normalise: flat [x,y,x,y] → [[x,y,x,y]], empty [] stays []
                if bboxes and not isinstance(bboxes[0], list):
                    bboxes = [bboxes]
                image_path = ast.literal_eval(m.group(3))
                if not image_path or not Path(image_path).exists():
                    fallback = OUTPUT_DIR / "pages" / f"page_{page:04d}.png"
                    if fallback.exists():
                        image_path = str(fallback)
                if not image_path:
                    continue  # no image — skip pill
                sources.append({"page": page, "bboxes": bboxes, "image_path": image_path})
                seen.add(page)
            except Exception:  # noqa: BLE001
                pass
    return sources


def _status_bar_html(
    indexed_doc: str | None,
    stats: dict,
    self_rag_enabled: bool,
    self_rag_stats: dict,
    reranker_on: bool,
) -> str:
    if not indexed_doc:
        return (
            '<div class="status-bar">'
            '<span><span class="dot-red"></span>No document indexed</span>'
            '</div>'
        )
    chunks  = stats.get("chunks", "\u2014")
    nodes   = stats.get("nodes",  "\u2014")
    edges   = stats.get("edges",  "\u2014")
    kept    = self_rag_stats.get("kept",    "\u2014")
    dropped = self_rag_stats.get("dropped", "\u2014")
    rag_dot   = "dot-green"  if self_rag_enabled else "dot-yellow"
    rag_label = f"Self-RAG \u25b8 {kept} kept / {dropped} dropped" if self_rag_enabled else "Self-RAG off"
    reranker_dot = "dot-green" if reranker_on else "dot-yellow"
    return (
        '<div class="status-bar">'
        f'<span><span class="dot-green"></span><b>{indexed_doc}</b></span>'
        f'<span>\U0001f4e6 {chunks} chunks</span>'
        f'<span>\U0001f578 {nodes} nodes \u00b7 {edges} edges</span>'
        f'<span><span class="{rag_dot}"></span>{rag_label}</span>'
        f'<span><span class="{reranker_dot}"></span>Reranker {"on" if reranker_on else "off"}</span>'
        '</div>'
    )


def main() -> None:
    st.set_page_config(
        page_title="PDF Q&A",
        page_icon="\U0001f4c4",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_session()
    store, embedder = _get_shared_components()

    with st.sidebar:
        st.markdown("## \U0001f4c4 PDF Q&A")
        st.caption("Agentic document assistant \u00b7 bbox citation \u00b7 knowledge graph")
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            st.caption("\U0001f50d LangSmith tracing on")
        st.divider()

        provider = st.radio("Provider", ["ollama", "openai"], horizontal=True)
        model_display = st.selectbox(
            "Model",
            _OLLAMA_MODELS if provider == "ollama" else _OPENAI_MODELS,
        )
        model = _resolve_model(model_display) if provider == "ollama" else model_display

        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            st.warning("OPENAI_API_KEY not set in .env")
        if provider == "ollama" and model_display in _LARGE_MODELS:
            st.warning("\u26a0\ufe0f Large model (6\u201310 GB). Set ctx=1024 to avoid OOM.")

        st.divider()
        enable_reranker = st.toggle("Cross-encoder reranker", value=True)
        enable_self_rag = st.toggle("Self-RAG filter", value=True)
        if enable_self_rag:
            self_rag_gate = st.slider(
                "BM25 gate (skip Self-RAG above)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            )
        else:
            self_rag_gate = 0.5

        ctx_limit = st.select_slider(
            "Context window (tokens)",
            options=_CTX_OPTIONS,
            value=_DEFAULT_CTX,
            help="Lower = less RAM. 2048 for qwen3.5, 1024 for Gemma4.",
        )

        st.divider()
        uploaded = st.file_uploader("Upload PDF", type="pdf")
        if uploaded and st.button("\u26a1 Index document", type="primary", use_container_width=True):
            pdf_path = DATA_DIR / uploaded.name
            pdf_path.write_bytes(uploaded.read())
            with st.spinner("Indexing \u2014 extraction \u2192 chunks \u2192 embeddings \u2192 graph \u2026"):
                _clear_index(store)
                indexer = DocumentIndexer(
                    embedder=embedder,
                    store=store,
                    llm_provider=provider,
                    llm_model=model,
                )
                n = indexer.index(pdf_path)
                doc_config: DocTypeConfig = indexer.last_doc_type or load_active_config()

            graph_path = store._persist_dir / "graph.json"
            kg = load_graph(graph_path)
            st.session_state.index_stats = {
                "chunks": n,
                "nodes": kg.number_of_nodes(),
                "edges": kg.number_of_edges(),
            }
            st.session_state.indexed_doc     = pdf_path.stem
            st.session_state.doc_config      = doc_config
            st.session_state.active_provider = provider
            st.session_state.active_model    = model
            st.session_state.messages        = []
            st.session_state.overlay_source  = None
            st.session_state.latest_sources  = []
            st.session_state.self_rag_stats  = {}
            st.session_state.active_domain   = None
            st.success(f"\u2705 {n} chunks indexed \u00b7 {kg.number_of_nodes()} graph nodes")
            st.info(f"Doc type: **{doc_config.display_name}**")

        if st.session_state.indexed_doc:
            st.divider()
            doc_cfg: DocTypeConfig | None = st.session_state.doc_config
            st.markdown(f"**Active:** `{st.session_state.indexed_doc}`")
            if doc_cfg:
                domains = list(doc_cfg.domains.values())
                st.caption("Domains: " + " \u00b7 ".join(d.display_name for d in domains))
            if st.session_state.active_domain:
                st.caption(f"Last domain: **{st.session_state.active_domain}**")

        st.divider()
        if st.button("\U0001f5d1 Clear conversation", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.overlay_source = None
            st.session_state.latest_sources = []
            st.session_state.active_domain  = None
            st.session_state.self_rag_stats = {}
            st.rerun()

    st.markdown(
        _status_bar_html(
            st.session_state.indexed_doc,
            st.session_state.index_stats,
            enable_self_rag,
            st.session_state.self_rag_stats,
            enable_reranker,
        ),
        unsafe_allow_html=True,
    )

    chat_col, view_col = st.columns([3, 2], gap="large")

    with chat_col:
        if not st.session_state.indexed_doc:
            st.info("\U0001f448 Upload and index a PDF to start chatting.")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        meta_parts = []
                        if msg.get("domain"):
                            meta_parts.append(
                                f'<span class="domain-badge">{msg["domain"]}</span>'
                            )
                        if msg.get("sources"):
                            pills = "".join(
                                f'<span class="source-pill">p.{s["page"]}</span>'
                                for s in msg["sources"]
                            )
                            meta_parts.append(f'<span class="source-strip">{pills}</span>')
                        if meta_parts:
                            st.markdown(" ".join(meta_parts), unsafe_allow_html=True)

            if prompt := st.chat_input("Ask a question about the document \u2026"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                doc_config = st.session_state.doc_config or load_active_config()
                domain_spec = route_query(prompt, config=doc_config)
                st.session_state.active_domain = domain_spec.display_name

                with st.chat_message("assistant"):
                    with st.spinner(f"Thinking \u00b7 {domain_spec.display_name} \u2026"):
                        retriever = _get_retriever(store, embedder, enable_reranker=enable_reranker)
                        graph_path = store._persist_dir / "graph.json"
                        kg = load_graph(graph_path)
                        all_chunks = store.get_all_chunks()
                        agent = build_agent(
                            retriever=retriever,
                            provider=st.session_state.active_provider,
                            model=st.session_state.active_model,
                            domain_spec=domain_spec,
                            graph=kg,
                            all_chunks=all_chunks,
                            self_rag_enabled=enable_self_rag,
                            self_rag_bm25_gate=self_rag_gate,
                            num_ctx=ctx_limit,
                        )
                        result = agent.invoke(
                            {"messages": [HumanMessage(content=prompt)]}
                        )
                        answer = result["messages"][-1].content

                    st.markdown(answer)
                    sources = _extract_sources(result["messages"])
                    if sources:
                        pills_html = '<div class="source-strip">' + "".join(
                            f'<span class="source-pill">p.{s["page"]}</span>' for s in sources
                        ) + "</div>"
                        st.markdown(pills_html, unsafe_allow_html=True)
                    st.markdown(
                        f'<span class="domain-badge">{domain_spec.display_name}</span>',
                        unsafe_allow_html=True,
                    )

                st.session_state.self_rag_stats  = _parse_self_rag_stats(result["messages"])
                st.session_state.latest_sources  = sources
                if sources:
                    st.session_state.overlay_source = sources[0]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "domain": domain_spec.display_name,
                })
                st.rerun()

    with view_col:
        st.markdown("#### \U0001f50d Source Viewer")
        latest = st.session_state.latest_sources
        if latest:
            st.caption("Click a page to view highlighted source:")
            btn_cols = st.columns(len(latest))
            for i, src in enumerate(latest):
                with btn_cols[i]:
                    if st.button(f"Page {src['page']}", key=f"pill_{i}", use_container_width=True):
                        st.session_state.overlay_source = src
                        st.rerun()

        src = st.session_state.overlay_source
        if src:
            img = render_page_with_bboxes(str(src["image_path"]), src["bboxes"])
            if img:
                st.image(img, caption=f"Page {src['page']} \u2014 highlighted citation", use_container_width=True)
            else:
                st.warning(
                    f"Page image not found at `{src['image_path']}`. "
                    "Re-index to regenerate page images."
                )
            with st.expander("\U0001f4d0 Raw bounding boxes"):
                st.json(src["bboxes"])
        else:
            st.info("Source pages appear here after a query.")

        stats = st.session_state.index_stats
        if stats:
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Chunks",       stats.get("chunks", "\u2014"))
            c2.metric("Graph nodes",  stats.get("nodes",  "\u2014"))
            c3.metric("Graph edges",  stats.get("edges",  "\u2014"))


def _parse_self_rag_stats(messages: list) -> dict:
    kept = dropped = 0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            for m in re.finditer(r"Self-RAG filter: (\d+) kept, (\d+) dropped", str(msg.content)):
                kept    += int(m.group(1))
                dropped += int(m.group(2))
    return {"kept": kept, "dropped": dropped} if kept or dropped else {}


if __name__ == "__main__":
    main()
