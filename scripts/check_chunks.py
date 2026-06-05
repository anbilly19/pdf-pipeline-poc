"""Quick sanity check on pipeline chunk output."""
import json
import pathlib
import sys

output_dir = pathlib.Path("output")
candidates = list(output_dir.glob("**/*.json")) if output_dir.exists() else []
if not candidates:
    sys.exit("No JSON files found in output/")

chunk_file = candidates[0]
chunks = json.loads(chunk_file.read_text(encoding="utf-8"))

print(f"File: {chunk_file}")
print(f"Total chunks: {len(chunks)}")

keywords = ["Laufzeit", "Kuendigung", "K\u00fcndigung", "Schlechtleistung"]
hits = [c for c in chunks if any(k in c.get("text", "") for k in keywords)]
print(f"Chunks with key contract terms: {len(hits)}")
for h in hits:
    page = h.get("page_number", "?")
    preview = h.get("text", "")[:120].replace("\n", " ")
    print(f"  p{page}: {preview}")
