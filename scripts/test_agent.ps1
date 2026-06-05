<#
.SYNOPSIS
    Tests the agent end-to-end from CLI, printing all tool calls and results.
#>

$python = @'
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
logging.basicConfig(level=logging.WARNING)

from src.retrieval.embedder import ChunkEmbedder
from src.retrieval.store import FAISSStore
from src.retrieval.retriever import BBoxRetriever
from src.indexer import DocumentIndexer
from src.agent.graph import build_agent
from langchain_core.messages import HumanMessage

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

store = FAISSStore(persist_dir=OUTPUT_DIR / "faiss_index")
embedder = ChunkEmbedder()

if store.count() == 0:
    print("Index is empty, indexing now...")
    pdfs = list(Path("data").glob("*.pdf"))
    if not pdfs:
        print("No PDFs in data/")
        sys.exit(1)
    indexer = DocumentIndexer(embedder=embedder, store=store)
    n = indexer.index(pdfs[0])
    print(f"Indexed {n} chunks")
else:
    print(f"Using existing index ({store.count()} chunks)")

# Direct retrieval check first
retriever = BBoxRetriever(store=store, embedder=embedder, top_k=10)
print("\n=== DIRECT RETRIEVAL CHECK: Verzug ===")
chunks = retriever.retrieve("Verzug Verzoegerungsschaden Auftraggeber", top_k=10)
for i, c in enumerate(chunks, 1):
    print(f"  [{i}] p{c.page_number}: {c.text[:100].replace(chr(10), ' ')}")

agent = build_agent(retriever=retriever, provider="ollama", model="qwen2.5:3b")

for query in [
    "Welche Fristen gelten bei Verzug und was kann der Auftraggeber verlangen?",
    "Was sind die Regelungen zur Laufzeit und Kuendigung?",
]:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    for msg in result["messages"]:
        cls = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                print(f"  -> tool_call: {tc[\"name\"]}({tc[\"args\"]})")
    print(f"ANSWER: {result[\"messages\"][-1].content}")
'@

Write-Host "`nRunning agent debug...`n" -ForegroundColor Cyan
$python | uv run python -
