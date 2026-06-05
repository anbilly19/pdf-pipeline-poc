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

# suppress noise
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
        payload = {"model": "nomic-embed-text", "input": ["test"]}
        r = requests.post("http://localhost:11434/api/embed", json=payload, timeout=10)
        data_r = r.json()
        emb = data_r.get("embeddings", [[]])[0]
        norm = float(np.linalg.norm(emb)) if emb else 0
        print(f"nomic-embed-text test: dim={len(emb)}, norm={norm:.4f}")
        if norm < 1e-6:
            print("WARN: nomic-embed-text returning zero vector! Model may not be pulled.")
            print("Fix: ollama pull nomic-embed-text")
    except Exception as e:
        print(f"nomic-embed-text call failed: {e}")

# -- 3. Force sentence-transformers directly (bypass Ollama) ---------------
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
    print(f"Model loaded. dim={vecs.shape[1]}")
    sim = vecs @ vecs.T
    max_off = float(np.max(sim - np.eye(len(test_queries))))
    print(f"Max off-diagonal cosine similarity: {max_off:.4f}")
    for q, v in zip(test_queries, vecs):
        print(f"  '{q}': first3={[round(x,4) for x in v[:3].tolist()]}")
    if max_off < 0.98:
        print("\nsentence-transformers OK — fallback will work correctly.")
        print("ACTION: Re-index the PDF in Streamlit to rebuild embeddings with ST.")
    else:
        print("\nERROR: sentence-transformers also returning constant vectors.")
except Exception as e:
    print(f"sentence-transformers failed: {e}")
    print("Fix: pip install sentence-transformers")
'@

Write-Host "`nRunning deep diagnostic...`n" -ForegroundColor Cyan
$python | python -
