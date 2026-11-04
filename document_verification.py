import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OCR_NOTE = (
    "This is text extraction only. OCR reads the words on an image; it does "
    "not prove that a document is genuine or unaltered."
)

# Which words on a line identify which field
FIELD_PATTERNS = {
    "net_salary": r"net\s*(salary|pay|payable|amount)?|take[\s-]*home",
    "gross_salary": r"gross\s*(salary|earnings|pay|total)?",
    "basic_salary": r"\bbasic\b",
    "total_deductions": r"total\s*deductions?",
}
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d{1,2})?)")
LOW_CONFIDENCE = 60  # mean OCR confidence below this = ask for a clearer image


def _configure_tesseract():
    if shutil.which("tesseract"):
        return
    if Path(WINDOWS_DEFAULT).exists():
        pytesseract.pytesseract.tesseract_cmd = WINDOWS_DEFAULT


_configure_tesseract()


def preprocess(img):
    """Clean-up with OpenCV: grayscale, enlarge, smooth, black-and-white."""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    if w < 1200:
        scale = 1200 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def _ocr(img):
    config = "--psm 6"  # treat the page as one block of text lines
    data = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"])
             if t.strip() and float(c) >= 0]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    text = pytesseract.image_to_string(img, config=config)
    return text, mean_conf


def parse_amount(text):
    """'Rs. 35,000.00' -> 35000.0 (uses the last number on the line)."""
    numbers = NUMBER.findall(text)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def extract_fields(text):
    """Look for known labels line by line. Keeps the original line so the
    user can see exactly what was read."""
    fields = {}
    for line in text.splitlines():
        low = line.lower()
        for name, pattern in FIELD_PATTERNS.items():
            if name in fields:
                continue
            if re.search(pattern, low):
                value = parse_amount(line)
                if value is not None:
                    fields[name] = {"value": value, "line": line.strip()}
                    break  # one field per line
    return fields


def analyze_document(file_like):
    """file_like: an open file or Streamlit upload. Works in memory only;
    nothing is written to disk."""
    img = ImageOps.exif_transpose(Image.open(file_like))
    candidates = [_ocr(img.convert("L")), _ocr(preprocess(img))]
    text, conf = max(candidates, key=lambda c: c[1])
    return {
        "text": text,
        "mean_confidence": round(conf, 1),
        "low_quality": conf < LOW_CONFIDENCE,
        "fields": extract_fields(text),
        "note": OCR_NOTE,
    }
    
    # ---------- Layer 2: information matching ----------
MATCH_TOLERANCE_PCT = 1.0    # up to 1% apart counts as a match
REVIEW_TOLERANCE_PCT = 5.0   # 1-5% apart: could be rounding or an OCR misread


def check_consistency(fields):
    """Does gross - deductions = net? This catches OCR misreads, NOT forgeries:
    a forged slip can have numbers that add up. Returns True, False, or None
    when the three amounts are not all available."""
    keys = ("gross_salary", "total_deductions", "net_salary")
    if not all(k in fields for k in keys):
        return None
    expected_net = fields["gross_salary"]["value"] - fields["total_deductions"]["value"]
    actual_net = fields["net_salary"]["value"]
    return abs(expected_net - actual_net) <= max(1.0, 0.01 * abs(actual_net))


def compare_salary(entered, salary_type, doc_result):
    """Compare the amount the user typed with the amount read from the document.
    salary_type is 'gross' or 'net'. Returns a dict with a status of
    'match', 'differs' or 'needs_review'."""
    field = "gross_salary" if salary_type == "gross" else "net_salary"
    label = "gross" if salary_type == "gross" else "net (take-home)"
    fields = doc_result["fields"]

    reasons = []
    if doc_result["low_quality"]:
        reasons.append("The image quality is low, so numbers may have been misread.")
    if check_consistency(fields) is False:
        reasons.append("The gross, deductions and net amounts read from the "
                       "document do not add up, which can mean a misread number.")

    if field not in fields:
        return {"status": "needs_review", "entered": entered, "extracted": None,
                "difference": None, "difference_pct": None, "reasons": reasons,
                "message": f"Could not find a {label} salary in the document. "
                           "Please review manually."}

    extracted = fields[field]["value"]
    difference = entered - extracted
    if extracted == 0:
        pct = 0.0 if entered == 0 else float("inf")
    else:
        pct = abs(difference) / extracted * 100

    if pct <= MATCH_TOLERANCE_PCT:
        status = "match"
        message = f"The {label} salary you entered matches the document."
    elif pct <= REVIEW_TOLERANCE_PCT:
        status = "needs_review"
        message = (f"Your entry is close to the document value but not identical "
                   f"(Rs. {entered:,.0f} vs Rs. {extracted:,.0f}). Please review.")
    else:
        status = "differs"
        message = (f"Discrepancy detected. You entered Rs. {entered:,.0f} but the "
                   f"document shows Rs. {extracted:,.0f}. Please review your information.")

    # A match from an unreliable reading cannot be trusted either
    if status == "match" and reasons:
        status = "needs_review"
        message = ("The values match, but the document reading is not fully "
                   "reliable. Please review manually.")

    return {"status": status, "entered": entered, "extracted": extracted,
            "difference": difference, "difference_pct": pct,
            "reasons": reasons, "message": message}