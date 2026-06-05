<#
.SYNOPSIS
    Diagnostic script for PDF pipeline retrieval debugging.
    Tests what chunks are returned for key queries and checks
    page numbers, bboxes, and image paths.

.USAGE
    # From repo root:
    .\scripts\diagnose_retrieval.ps1

    # Test a custom query:
    .\scripts\diagnose_retrieval.ps1 -Query "Vertragsstrafe"

    # Test all built-in queries:
    .\scripts\diagnose_retrieval.ps1 -All
#>
param(
    [string]$Query = "",
    [switch]$All
)

$scriptBlock = @'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

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
        bbox_ok = all(len(b)==4 and (b[2]-b[0])*(b[3]-b[1]) >= 100 for b in c.bboxes)
        img_exists = Path(c.image_path).exists() if c.image_path else False
        print(f"  [{i}] Page {c.page_number:>2} | bboxes={len(c.bboxes)} {'OK' if bbox_ok else 'WARN:invalid'} | img={'OK' if img_exists else 'MISSING'} | {c.text[:100].replace(chr(10),' ')}")
    print()
'@

# --- Build query list ---
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
    # Default: the two problem queries
    $testQueries = @("Kundigung Fristen", "Laufzeit Kundigung")
}

# Encode query list as Python list literal
$pyList = "[" + (($testQueries | ForEach-Object { "'$_'" }) -join ", ") + "]"
$pythonScript = $scriptBlock -replace "QUERIES_PLACEHOLDER", $pyList

# --- Run ---
Write-Host "`nRunning retrieval diagnostic...`n" -ForegroundColor Cyan
$pythonScript | uv run -

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDiagnostic failed. Make sure you have indexed a PDF first." -ForegroundColor Red
} else {
    Write-Host "`nDone. Look for:" -ForegroundColor Green
    Write-Host "  - Wrong page numbers (e.g. Kundigung showing up on page 2 instead of 12)" -ForegroundColor Yellow
    Write-Host "  - WARN:invalid bboxes (zero-area, need re-index)" -ForegroundColor Yellow
    Write-Host "  - MISSING image paths (need re-index)" -ForegroundColor Yellow
}
