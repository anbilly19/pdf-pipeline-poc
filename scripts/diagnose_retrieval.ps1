<#
.SYNOPSIS
    Retrieval diagnostic — queries the live FAISS index.
.USAGE
    .\scripts\diagnose_retrieval.ps1
    .\scripts\diagnose_retrieval.ps1 -Query "Vertragsstrafe"
    .\scripts\diagnose_retrieval.ps1 -All
#>
param(
    [string]$Query = "",
    [switch]$All
)

$scriptBlock = @'
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
import logging, os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

from src.retrieval.store import FAISSStore
from src.retrieval.embedder import ChunkEmbedder

store = FAISSStore(persist_dir=Path("outputs/faiss_index"))
embedder = ChunkEmbedder()

total = store.count()
print(f"\n{'='*60}")
print(f"Index contains {total} chunks")
print(f"{'='*60}\n")

if total == 0:
    print("ERROR: Index is empty. Upload and index a PDF in Streamlit first.")
    sys.exit(1)

queries = QUERIES_PLACEHOLDER

for q in queries:
    print(f"{'='*60}")
    print(f"QUERY: {q}")
    print(f"{'='*60}")
    embedding = embedder.embed_query(q)
    chunks = store.query(embedding, n_results=5)
    if not chunks:
        print("  [no results]")
    for i, c in enumerate(chunks, 1):
        bbox_ok = all(len(b)==4 and (b[2]-b[0])*(b[3]-b[1]) >= 50 for b in c.bboxes)
        img_exists = Path(c.image_path).exists() if c.image_path else False
        print(f"  [{i}] Page {c.page_number:>2} | bboxes={len(c.bboxes)} {'OK' if bbox_ok else 'WARN'} | img={'OK' if img_exists else 'MISSING'} | {c.text[:100].replace(chr(10),' ')}")
    print()
'@

$builtinQueries = @(
    "Kundigung Fristen",
    "Laufzeit Kundigung",
    "Vertragsstrafe Prozent",
    "Insolvenz Auftragnehmer",
    "Geheimhaltung Verschwiegenheit",
    "Tabelle Definitionen"
)

if ($Query -ne "") {
    $testQueries = @($Query)
} elseif ($All) {
    $testQueries = $builtinQueries
} else {
    $testQueries = @("Kundigung Fristen", "Laufzeit Kundigung")
}

$pyList = "[" + (($testQueries | ForEach-Object { "'$_'" }) -join ", ") + "]"
$pythonScript = $scriptBlock -replace "QUERIES_PLACEHOLDER", $pyList

Write-Host "`nRunning retrieval diagnostic (uv)...`n" -ForegroundColor Cyan
$pythonScript | uv run python -

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFailed. Index a PDF in Streamlit first." -ForegroundColor Red
} else {
    Write-Host "`nDone. Look for:" -ForegroundColor Green
    Write-Host "  - Kundigung content on page 12-13 in top 3 results" -ForegroundColor Yellow
    Write-Host "  - WARN bboxes or MISSING images = re-index needed" -ForegroundColor Yellow
}
