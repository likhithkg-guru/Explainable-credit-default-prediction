from pathlib import Path
from document_verification import analyze_document

for path in sorted(Path("data/sample_docs").glob("*.png")):
    with open(path, "rb") as f:
        result = analyze_document(f)
    print(f"\n===== {path.name} =====")
    print("Mean OCR confidence:", result["mean_confidence"],
          "| low quality:", result["low_quality"])
    print("--- raw text ---")
    print(result["text"])
    print("--- extracted fields ---")
    for name, info in result["fields"].items():
        print(f"{name}: {info['value']:,.2f}   (from line: {info['line']!r})")
print("\n" + result["note"])