"""Run the pipeline on all PDFs in data/ and write outputs/chunks.json.

Usage:
    uv run python scripts/check_chunks.py
"""
import dataclasses
import json
import logging
import pathlib
import sys

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(pathlib.Path(".").resolve()))

from src.pipeline import PDFPipeline, PipelineConfig  # noqa: E402

data_dir = pathlib.Path("data")
pdfs = list(data_dir.glob("*.pdf"))
if not pdfs:
    sys.exit("No PDFs found in data/")

outputs_dir = pathlib.Path("outputs")
outputs_dir.mkdir(exist_ok=True)

pipeline = PDFPipeline(config=PipelineConfig())

all_chunks = []
for pdf in pdfs:
    print(f"Processing {pdf.name}...")
    chunks = pipeline.run(pdf)
    print(f"  -> {len(chunks)} chunks")
    all_chunks.extend(chunks)

out_file = outputs_dir / "chunks.json"
out_file.write_text(
    json.dumps([dataclasses.asdict(c) for c in all_chunks], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"\nWrote {len(all_chunks)} chunks to {out_file}")

keywords = ["Laufzeit", "K\u00fcndigung", "Schlechtleistung"]
hits = [c for c in all_chunks if any(k in c.text for k in keywords)]
print(f"Chunks with key contract terms: {len(hits)}")
for h in hits:
    preview = h.text[:120].replace("\n", " ")
    print(f"  p{h.page_number}: {preview}")
