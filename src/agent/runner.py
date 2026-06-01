"""Simple CLI runner for the agent — smoke-test before the UI."""
from __future__ import annotations

import logging
import uuid
import warnings

# suppress noisy transformers __path__ deprecation warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from langchain_core.messages import HumanMessage

from src.agent.graph import build_agent
from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.retriever import BBoxRetriever
from src.retrieval.store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_cli(model: str = "gemma4:e2b") -> None:
    """Start an interactive CLI session with the agent."""
    embedder = ChunkEmbedder()
    store = ChromaStore()
    retriever = BBoxRetriever(store=store, embedder=embedder, top_k=5)
    agent = build_agent(retriever=retriever, model=model)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nPDF Agent ready (model={model}, thread={thread_id[:8]})")
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
        last = result["messages"][-1]
        print(f"\nAssistant: {last.content}\n")


if __name__ == "__main__":
    run_cli()
