"""Financial Health Calculator.
Plain arithmetic only. It is NOT connected to the ML model, which was not
trained on income, expense or debt-payment data."""


def validate_inputs(income, expenses, debt):
    errors = []
    for name, value in (("Monthly income", income),
                        ("Monthly expenses", expenses),
                        ("Existing debt payments", debt)):
        if value < 0:
            errors.append(f"{name} cannot be negative.")
    if income <= 0:
        errors.append("Monthly income must be greater than 0 to calculate ratios.")
    return errors


def summarize(income, expenses, debt):
    remaining = income - expenses - debt
    return {
        "remaining": remaining,
        "dti_pct": debt / income * 100,          # debt-to-income ratio
        "expense_pct": expenses / income * 100,
        "remaining_pct": remaining / income * 100,
    }


def dti_band(dti_pct):
    """General rule-of-thumb ranges for explanation only. They are not
    lender criteria, and lenders differ."""
    if dti_pct < 20:
        return "Low", "#2E9E5B"
    if dti_pct < 36:
        return "Moderate", "#E0A030"
    if dti_pct < 50:
        return "High", "#E07A30"
    return "Very high", "#D64545"


def plain_summary(income, expenses, debt):
    s = summarize(income, expenses, debt)
    lines = [
        f"Out of every Rs. 100 of income, about Rs. {s['expense_pct']:.0f} goes to "
        f"expenses and Rs. {s['dti_pct']:.0f} to debt payments.",
    ]
    if s["remaining"] >= 0:
        lines.append(f"About Rs. {s['remaining_pct']:.0f} of every Rs. 100 is left over.")
    else:
        lines.append("Spending and debt payments together are higher than income "
                     "this month.")
    return lines