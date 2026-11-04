from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = Path("data/sample_docs")
OUT.mkdir(parents=True, exist_ok=True)


def get_font(size):
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_salary_slip(gross=35000, deductions=7000):
    basic = gross * 0.5
    hra = gross * 0.2
    special = gross - basic - hra
    net = gross - deductions

    img = Image.new("RGB", (1000, 1100), "white")
    d = ImageDraw.Draw(img)
    title, body = get_font(40), get_font(30)

    d.text((50, 40), "SAMPLE CORP PVT LTD", font=title, fill="black")
    d.text((50, 100), "SALARY SLIP - SYNTHETIC SAMPLE", font=body, fill="black")
    d.text((50, 140), "(not a real document)", font=body, fill="black")

    rows = [
        ("Employee Name", "Test Employee"),
        ("Pay Period", "October 2026"),
        (None, None),
        ("Basic", f"Rs. {basic:,.2f}"),
        ("HRA", f"Rs. {hra:,.2f}"),
        ("Special Allowance", f"Rs. {special:,.2f}"),
        ("Gross Earnings", f"Rs. {gross:,.2f}"),
        ("Total Deductions", f"Rs. {deductions:,.2f}"),
        ("Net Pay", f"Rs. {net:,.2f}"),
    ]
    y = 240
    for label, value in rows:
        if label:
            d.text((50, y), label, font=body, fill="black")
            d.text((560, y), value, font=body, fill="black")
        y += 70
    return img


if __name__ == "__main__":
    clean = make_salary_slip()
    clean.save(OUT / "salary_slip_clean.png")

    # A rougher version, like a phone photo: slightly tilted and blurred
    rough = clean.rotate(2.5, expand=True, fillcolor="white")
    rough = rough.filter(ImageFilter.GaussianBlur(1.2))
    rough.save(OUT / "salary_slip_rough.png")
    print("Saved sample documents in", OUT)