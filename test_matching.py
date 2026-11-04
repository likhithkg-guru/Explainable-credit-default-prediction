from document_verification import compare_salary


def fields(gross=35000, deductions=7000, net=28000):
    return {
        "gross_salary": {"value": gross, "line": ""},
        "total_deductions": {"value": deductions, "line": ""},
        "net_salary": {"value": net, "line": ""},
    }


def doc(f, low_quality=False):
    return {"fields": f, "low_quality": low_quality}


cases = [
    ("Exact match (gross)",                    35000, "gross", doc(fields())),
    ("Your example: typed 35,000 as net",      35000, "net",   doc(fields())),
    ("Close but not identical",                28500, "net",   doc(fields())),
    ("Low image quality",                      35000, "gross", doc(fields(), True)),
    ("Field not found",                        35000, "gross", doc({})),
    ("Numbers don't add up (misread?)",        35000, "gross", doc(fields(net=82000))),
]

for name, entered, kind, d in cases:
    r = compare_salary(entered, kind, d)
    print(f"\n{name}\n  status: {r['status']}\n  {r['message']}")
    for reason in r["reasons"]:
        print("  note:", reason)