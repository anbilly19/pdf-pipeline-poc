"""CLI runner — smoke-test before the UI."""
from __future__ import annotations

import argparse
import logging
import os
import uuid
import warnings

warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from src.agent.graph import build_agent
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.store import FAISSStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_cli(provider: str = "ollama", model: str = "gemma4:e2b") -> None:
    """Start an interactive CLI session."""
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")

    embedder = ChunkEmbedder()
    store = FAISSStore()
    retriever = BBoxRetriever(store=store, embedder=embedder, top_k=5)
    agent = build_agent(retriever=retriever, provider=provider, model=model)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    print(f"\nPDF Agent ready (provider={provider}, model={model}, tracing={tracing})")
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
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print(f"\nAssistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="gemma4:e2b")
    args = parser.parse_args()
    run_cli(provider=args.provider, model=args.model)
