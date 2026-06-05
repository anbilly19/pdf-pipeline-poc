<#
.SYNOPSIS
    Deep diagnostic: checks embedder output variance and dumps all chunk texts.
    Run this when all queries return the same results.
.USAGE
    .\scripts\diagnose_deep.ps1
#>

$python = @'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import numpy as np

# ── 1. Dump all chunks ──────────────────────────────────────────────
print("\n" + "="*60)
print("ALL INDEXED CHUNKS")
print("="*60)

meta_path = Path("outputs/faiss_index/metadata.json")
if not meta_path.exists():
    print("ERROR: No index found. Index a PDF in Streamlit first.")
    sys.exit(1)

data = json.loads(meta_path.read_text(encoding="utf-8"))
metadata = data["metadata"]
texts = data["texts"]

print(f"Total chunks: {len(texts)}")
for i, (t, m) in enumerate(zip(texts, metadata)):
    bbox_areas = [(b[2]-b[0])*(b[3]-b[1]) for b in m['bboxes'] if len(b)==4]
    area_str = f"areas={[round(a,0) for a in bbox_areas]}"
    print(f"\n[{i+1:02d}] Page {m['page_number']:>2} | type={m['chunk_type']:<5} | {area_str}")
    print(f"      {t[:200].replace(chr(10), ' ')}")

# ── 2. Check embedder variance ──────────────────────────────────────
print("\n" + "="*60)
print("EMBEDDER VARIANCE CHECK")
print("="*60)

from src.retrieval.embedder import ChunkEmbedder
embedder = ChunkEmbedder()

test_queries = [
    "Kundigung",
    "Vertragsstrafe",
    "Haftung",
    "Insolvenz",
    "Geheimhaltung",
]

vectors = []
for q in test_queries:
    v = embedder.embed_query(q)
    vectors.append(v)
    norm = float(np.linalg.norm(v))
    print(f"  '{q}': dim={len(v)}, norm={norm:.4f}, first3={[round(x,4) for x in v[:3]]}")

# Cosine similarity matrix
vecs = np.array(vectors, dtype=np.float32)
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
norms[norms == 0] = 1e-9
normed = vecs / norms
sim = normed @ normed.T

print("\nCosine similarity matrix (1.0 = identical):")
print(f"  {'':20s}" + "".join(f"{q[:8]:>10s}" for q in test_queries))
for i, q in enumerate(test_queries):
    row = "".join(f"{sim[i,j]:>10.4f}" for j in range(len(test_queries)))
    print(f"  {q:<20s}{row}")

max_off_diag = float(np.max(sim - np.eye(len(test_queries))))
if max_off_diag > 0.98:
    print(f"\n*** EMBEDDER BROKEN: all queries produce near-identical vectors (max sim={max_off_diag:.4f}) ***")
    print("    The model is likely outputting zero/constant vectors.")
    print("    Check ChunkEmbedder model path and initialization.")
elif max_off_diag > 0.90:
    print(f"\nWARN: Queries are very similar in embedding space (max sim={max_off_diag:.4f}).")
    print("    Consider a better German embedding model.")
else:
    print(f"\nEmbedder OK: queries are distinct (max off-diag sim={max_off_diag:.4f})")
    print("    Problem is in chunking/indexing, not the embedder.")
'@

Write-Host "`nRunning deep diagnostic...`n" -ForegroundColor Cyan
$python | uv run -

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFailed. Make sure the PDF is indexed in Streamlit first." -ForegroundColor Red
}
