<#
.SYNOPSIS
    Dumps all keys/values of ODL list elements on page 12.
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

print("=== ALL KEYS on page 12 list elements ===")
for e in page12:
    if e.get("type", "").lower() == "list":
        print(f"\nList element keys: {list(e.keys())}")
        print(json.dumps(e, ensure_ascii=False, indent=2)[:2000])
        break  # just first list is enough

print("\n=== Top-level JSON keys ===")
print(list(raw.keys()))
print("\n=== First top-level kid full dump ===")
if kids:
    print(json.dumps(kids[0], ensure_ascii=False, indent=2)[:1000])
'@

Write-Host "`nDumping ODL list element structure...`n" -ForegroundColor Cyan
$python | uv run python -
