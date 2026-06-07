"""CLI runner — smoke-test before the UI."""
from __future__ import annotations

# Must be first import — silences transformers __path__ spam
import src.silence  # noqa: F401

import argparse
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from src.agent.graph import build_agent
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
) -> None:
    """Start an interactive CLI session with domain-aware routing."""
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")

    embedder = ChunkEmbedder()
    store = FAISSStore()

    # Cross-encoder reranker (non-fatal if Ollama unavailable)
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

    # Load knowledge graph (non-fatal if missing)
    graph_path = store._persist_dir / "graph.json"
    knowledge_graph = load_graph(graph_path)
    all_chunks = store.get_all_chunks()
    if knowledge_graph.number_of_nodes() > 0:
        logger.info("Knowledge graph loaded: %d nodes, %d edges", knowledge_graph.number_of_nodes(), knowledge_graph.number_of_edges())
    else:
        logger.info("No knowledge graph found at %s — graph expansion disabled", graph_path)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    print(f"\nPDF Agent ready (provider={provider}, doc_type={doc_config.doc_type}, reranker={'on' if reranker else 'off'}, tracing={tracing})")
    print("Type 'exit' to quit.\n")

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

        domain_spec = route_query(user_input, config=doc_config)
        print(f"  [domain: {domain_spec.display_name} | model: {domain_spec.model}]")

        agent = build_agent(
            retriever=retriever,
            provider=provider,
            model=model,
            domain_spec=domain_spec,
            graph=knowledge_graph,
            all_chunks=all_chunks,
        )
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print(f"\nAssistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--reranker-model", default="bge-reranker-v2-m3")
    parser.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranking")
    args = parser.parse_args()
    run_cli(
        provider=args.provider,
        model=args.model,
        reranker_model=args.reranker_model,
        enable_reranker=not args.no_reranker,
    )
