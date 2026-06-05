<#
.SYNOPSIS
    Deep diagnostic: embedder variance, Ollama health, chunk content.
.USAGE
    .\scripts\diagnose_deep.ps1
#>

$python = @'
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np

warnings.filterwarnings("ignore", message=r"Accessing `__path__`")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# -- 1. Dump all chunks ---------------------------------------------------
print("\n" + "="*60)
print("ALL INDEXED CHUNKS")
print("="*60)
meta_path = Path("outputs/faiss_index/metadata.json")
if not meta_path.exists():
    print("ERROR: No index found. Index a PDF in Streamlit first.")
    sys.exit(1)
data = json.loads(meta_path.read_text(encoding="utf-8"))
metadata, texts = data["metadata"], data["texts"]
print(f"Total chunks: {len(texts)}")
for i, (t, m) in enumerate(zip(texts, metadata)):
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in m["bboxes"] if len(b)==4]
    print(f"\n[{i+1:02d}] Page {m['page_number']:>2} | {[round(a,0) for a in areas]}")
    print(f"      {t[:200].replace(chr(10), ' ')}")

# -- 2. Ollama health check -----------------------------------------------
print("\n" + "="*60)
print("OLLAMA HEALTH CHECK")
print("="*60)
try:
    import requests
    r = requests.get("http://localhost:11434/api/tags", timeout=3)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"Ollama running. Available models: {models}")
    ollama_ok = True
except Exception as e:
    print(f"Ollama not reachable: {e}")
    ollama_ok = False

if ollama_ok:
    try:
        import requests
        payload = {"model": "nomic-embed-text", "input": ["Kundigung", "Vertragsstrafe", "Haftung"]}
        r = requests.post("http://localhost:11434/api/embed", json=payload, timeout=10)
        embs = r.json().get("embeddings", [])
        if embs and len(embs) >= 2:
            v0 = np.array(embs[0])
            v1 = np.array(embs[1])
            n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
            sim = float(np.dot(v0/n0, v1/n1)) if n0 > 1e-6 and n1 > 1e-6 else 1.0
            print(f"nomic-embed-text: dim={len(embs[0])}, norm0={n0:.4f}")
            print(f"  cos_sim('Kundigung','Vertragsstrafe') = {sim:.4f}")
            if sim > 0.999:
                print("  WARN: Ollama still returning identical vectors for different inputs.")
                print("  Try: ollama pull nomic-embed-text")
            else:
                print("  Ollama embeddings look healthy.")
    except Exception as e:
        print(f"nomic-embed-text call failed: {e}")

# -- 3. sentence-transformers direct test ----------------------------------
print("\n" + "="*60)
print("SENTENCE-TRANSFORMERS DIRECT TEST")
print("="*60)
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    test_queries = ["Kundigung", "Vertragsstrafe", "Haftung", "Insolvenz", "Geheimhaltung"]
    vecs = model.encode(test_queries, normalize_embeddings=True, show_progress_bar=False)
    sim = vecs @ vecs.T
    max_off = float(np.max(sim - np.eye(len(test_queries))))
    print(f"dim={vecs.shape[1]}, max off-diag sim={max_off:.4f}")
    if max_off < 0.98:
        print("sentence-transformers OK")
    else:
        print("ERROR: ST also returning constant vectors")
except Exception as e:
    print(f"sentence-transformers failed: {e}")
    print("Fix: uv add sentence-transformers")

# -- 4. Live pipeline dry-run (no Streamlit) -------------------------------
print("\n" + "="*60)
print("PIPELINE DRY-RUN (first PDF in data/)")
print("="*60)
pdfs = list(Path("data").glob("*.pdf"))
if not pdfs:
    print("No PDFs in data/ — skip")
else:
    pdf = pdfs[0]
    print(f"Testing: {pdf.name}")
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from src.pipeline import PDFPipeline
    pipeline = PDFPipeline()
    chunks = pipeline.run(pdf)
    print(f"Chunks produced: {len(chunks)}")
    sec15 = [c for c in chunks if "k\u00fcndigung" in c.text.lower() or "laufzeit" in c.text.lower()]
    print(f"Chunks containing 'Laufzeit'/'K\u00fcndigung': {len(sec15)}")
    for c in sec15:
        print(f"  Page {c.page_number}: {c.text[:150].replace(chr(10), ' ')}")
'@

Write-Host "`nRunning deep diagnostic (uv)...`n" -ForegroundColor Cyan
$python | uv run python -

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nFailed. Make sure the PDF is in data/ and dependencies are installed." -ForegroundColor Red
}
