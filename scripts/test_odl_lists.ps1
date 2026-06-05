<#
.SYNOPSIS
    Directly tests ODL list extraction bypassing the pipeline.
#>

$python = @'
import sys, json, warnings, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

pdfs = list(Path("data").glob("*.pdf"))
pdf_path = pdfs[0]

from opendataloader_pdf import convert
with tempfile.TemporaryDirectory() as tmp:
    convert(str(pdf_path.resolve()), tmp)
    json_out = Path(tmp) / (pdf_path.stem + ".json")
    raw = json.loads(json_out.read_text(encoding="utf-8"))

kids = raw.get("kids", [])
page12 = [k for k in kids if k.get("page number") == 12]

print(f"Page 12 elements from raw ODL JSON: {len(page12)}")
for e in page12:
    print(f"  type={e.get('type')} content={repr(e.get('content','')[:60])} kids={len(e.get('kids',[]))}")
    for child in e.get("kids", [])[:3]:
        print(f"    child type={child.get('type')} content={repr(child.get('content','')[:80])}")

print()
print("--- Now test _flatten_list directly ---")
import importlib, importlib.util
spec = importlib.util.spec_from_file_location(
    "odl_ext", Path("src/extraction/opendataloader_extractor.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for e in page12:
    if e.get("type", "").lower() == "list":
        text = mod._flatten_list(e)
        print(f"  _flatten_list result ({len(text)} chars): {repr(text[:200])}")
'@

Write-Host "`nTesting ODL list extraction directly...`n" -ForegroundColor Cyan
$python | uv run python -
