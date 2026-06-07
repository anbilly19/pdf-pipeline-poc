"""CLI runner — multi-turn session with persistent LangGraph memory.

Roadmap #5: stable thread_id, MemorySaver/SqliteSaver, /new command.
Roadmap #6: --no-self-rag flag + self_rag_bm25_gate env var.

Usage
-----
    python -m src.agent.runner
    python -m src.agent.runner --memory sqlite
    python -m src.agent.runner --no-self-rag
    python -m src.agent.runner --no-reranker
    python -m src.agent.runner --provider openai --model gpt-4o-mini
"""
from __future__ import annotations

import src.silence  # noqa: F401

import argparse
import logging
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from src.agent.graph import build_agent
from src.agent.memory import make_checkpointer, make_thread_config
from src.agent.router import route_query
from src.agent.domain_config import load_active_config
from src.graph.builder import load_graph
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.reranker import OllamaReranker
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.store import FAISSStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_cli(
    provider: str = "ollama",
    model: str = "gemma4:e2b",
    reranker_model: str = "bge-reranker-v2-m3",
    enable_reranker: bool = True,
    memory_backend: str = "memory",
    self_rag_enabled: bool = True,
    self_rag_bm25_gate: float = 0.5,
) -> None:
    """Start an interactive multi-turn CLI session.

    Args:
        provider: LLM provider (``'ollama'`` or ``'openai'``).
        model: Model name.
        reranker_model: Ollama cross-encoder model name.
        enable_reranker: Whether to load the reranker.
        memory_backend: ``'memory'`` or ``'sqlite'``.
        self_rag_enabled: Enable Self-RAG filter (disable to reduce latency).
        self_rag_bm25_gate: BM25 gate threshold above which Self-RAG is skipped.
    """
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")

    embedder = ChunkEmbedder()
    store = FAISSStore()

    reranker: OllamaReranker | None = None
    if enable_reranker:
        reranker = OllamaReranker(model=reranker_model)
        if reranker.is_available():
            logger.info("Cross-encoder reranker enabled (model=%s)", reranker_model)
        else:
            logger.warning(
                "Ollama reranker not available (model=%s) — falling back to BM25 order. "
                "Run: ollama pull %s", reranker_model, reranker_model,
            )

    retriever = BBoxRetriever(
        store=store,
        embedder=embedder,
        top_k=5,
        reranker=reranker,
    )
    doc_config = load_active_config()

    graph_path = store._persist_dir / "graph.json"
    knowledge_graph = load_graph(graph_path)
    all_chunks = store.get_all_chunks()
    if knowledge_graph.number_of_nodes() > 0:
        logger.info(
            "Knowledge graph loaded: %d nodes, %d edges",
            knowledge_graph.number_of_nodes(),
            knowledge_graph.number_of_edges(),
        )
    else:
        logger.info("No knowledge graph — graph expansion disabled")

    checkpointer = make_checkpointer(backend=memory_backend)
    thread_id = str(uuid.uuid4())
    thread_config = make_thread_config(thread_id)
    logger.info("Session thread_id=%s (memory=%s)", thread_id, memory_backend)

    agent = build_agent(
        retriever=retriever,
        provider=provider,
        model=model,
        domain_spec=None,
        graph=knowledge_graph,
        all_chunks=all_chunks,
        checkpointer=checkpointer,
        self_rag_enabled=self_rag_enabled,
        self_rag_bm25_gate=self_rag_bm25_gate,
    )

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    self_rag_status = f"bm25_gate={self_rag_bm25_gate}" if self_rag_enabled else "off"
    print(
        f"\nPDF Agent ready "
        f"(provider={provider}, doc_type={doc_config.doc_type}, "
        f"reranker={'on' if reranker else 'off'}, "
        f"self_rag={self_rag_status}, "
        f"memory={memory_backend}, tracing={tracing})"
    )
    print("Type 'exit' to quit. Type '/new' to start a fresh thread.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if not user_input:
            continue
        if user_input.lower() == "/new":
            thread_id = str(uuid.uuid4())
            thread_config = make_thread_config(thread_id)
            print(f"  [new session — thread_id={thread_id}]\n")
            continue

        domain_spec = route_query(user_input, config=doc_config)
        print(f"  [domain: {domain_spec.display_name} | model: {domain_spec.model}]")

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=thread_config,
        )
        print(f"\nAssistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--reranker-model", default="bge-reranker-v2-m3")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--memory", dest="memory_backend", default="memory",
                        choices=["memory", "sqlite"])
    parser.add_argument("--no-self-rag", action="store_true",
                        help="Disable Self-RAG relevance filter")
    parser.add_argument("--self-rag-gate", type=float, default=0.5,
                        dest="self_rag_bm25_gate",
                        help="BM25 gate threshold (default 0.5)")
    args = parser.parse_args()
    run_cli(
        provider=args.provider,
        model=args.model,
        reranker_model=args.reranker_model,
        enable_reranker=not args.no_reranker,
        memory_backend=args.memory_backend,
        self_rag_enabled=not args.no_self_rag,
        self_rag_bm25_gate=args.self_rag_bm25_gate,
    )
