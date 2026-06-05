<#
.SYNOPSIS
    Dumps the raw ODL JSON for a PDF so you can see exactly what ODL extracts
    per page, including page numbers, element types, and text snippets.
.USAGE
    .\scripts\inspect_odl_output.ps1
    .\scripts\inspect_odl_output.ps1 -Pages 11,12,13
#>
param(
    [int[]]$Pages = @()  # empty = all pages
)

$python = @'
import sys, json, warnings, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

pdfs = list(Path("data").glob("*.pdf"))
if not pdfs:
    print("No PDFs in data/")
    sys.exit(1)
pdf_path = pdfs[0]
print(f"Inspecting: {pdf_path.name}\n")

try:
    from opendataloader_pdf import convert
except ImportError:
    print("ODL not installed")
    sys.exit(1)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    convert(str(pdf_path.resolve()), str(tmp_path))
    json_out = tmp_path / (pdf_path.stem + ".json")
    if not json_out.exists():
        print("ODL produced no JSON")
        sys.exit(1)
    raw = json.loads(json_out.read_text(encoding="utf-8"))

kids = raw.get("kids", [])
page_filter = PAGES_PLACEHOLDER

# Group by page number
from collections import defaultdict
by_page = defaultdict(list)
for kid in kids:
    pn = kid.get("page number", "?")
    by_page[pn].append(kid)

all_pages = sorted(by_page.keys())
print(f"ODL reports {len(all_pages)} distinct page numbers: {all_pages}")
print(f"Total elements: {len(kids)}\n")

for pn in all_pages:
    if page_filter and pn not in page_filter:
        continue
    elems = by_page[pn]
    print(f"=== ODL Page {pn} ({len(elems)} elements) ===")
    for e in elems:
        bbox = e.get("bounding box", [])
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) if len(bbox)==4 else 0
        text = (e.get("content") or "").replace("\n", " ")[:120]
        print(f"  [{e.get('type','?'):10s}] bbox_area={area:>8.0f} | {text}")
    print()
'@

$pyList = if ($Pages.Count -gt 0) { "[" + ($Pages -join ", ") + "]" } else { "[]" }
$script = $python -replace "PAGES_PLACEHOLDER", $pyList

Write-Host "`nInspecting ODL raw output...`n" -ForegroundColor Cyan
$script | uv run python -
