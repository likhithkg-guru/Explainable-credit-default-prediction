import pandas as pd
import plotly.express as px
import streamlit as st

from financial_calculator import (validate_inputs, summarize, dti_band,
                                  plain_summary)
from ui_helpers import card


def calculator_page():
    st.title("Financial Health Calculator")
    st.info("This calculator is separate from the risk model. The model was not "
            "trained on income, expenses or debt payments, so nothing you enter "
            "here changes any risk score.")

    c1, c2, c3 = st.columns(3)
    income = c1.number_input("Monthly income (Rs.)", min_value=0.0,
                             value=50000.0, step=1000.0)
    expenses = c2.number_input("Monthly expenses (Rs.)", min_value=0.0,
                               value=25000.0, step=1000.0,
                               help="Living costs, excluding debt payments.")
    debt = c3.number_input("Existing debt payments (Rs. per month)",
                           min_value=0.0, value=8000.0, step=500.0,
                           help="Loan EMIs and minimum card payments.")

    errors = validate_inputs(income, expenses, debt)
    if errors:
        for e in errors:
            st.error(e)
        return

    s = summarize(income, expenses, debt)
    band, band_color = dti_band(s["dti_pct"])
    r1, r2, r3 = st.columns(3)
    with r1:
        card("Remaining each month", f"Rs. {s['remaining']:,.0f}",
             "#2E9E5B" if s["remaining"] >= 0 else "#D64545")
    with r2:
        card("Debt-to-income ratio", f"{s['dti_pct']:.0f}% ({band})", band_color)
    with r3:
        card("Expenses as share of income", f"{s['expense_pct']:.0f}%")

    if s["remaining"] < 0:
        st.warning("Expenses and debt payments are higher than income.")
    for line in plain_summary(income, expenses, debt):
        st.write("- " + line)

    df = pd.DataFrame({"Category": ["Expenses", "Debt payments", "Remaining"],
                       "Amount": [expenses, debt, s["remaining"]]})
    st.plotly_chart(px.bar(
        df, x="Category", y="Amount", color="Category",
        color_discrete_map={"Expenses": "#2E86AB", "Debt payments": "#E0A030",
                            "Remaining": "#2E9E5B"}))
    st.caption("Debt-to-income here = monthly debt payments / monthly income. "
               "Lenders usually use gross (before-tax) income, so entering "
               "take-home pay makes the ratio look higher. The ranges are general "
               "rules of thumb, not lender criteria.")


def awareness_page():
    st.title("Financial Awareness")
    st.info("General education only. Following these ideas does not guarantee "
            "loan approval or lower your actual risk of default.")

    topics = {
        "Pay on time": """
- Paying at least the minimum by the due date avoids late fees and late-payment marks on your record.
- Set reminders or an auto-pay for the minimum amount as a safety net.
- In this prototype, the most recent repayment status was the strongest signal for all three models. That describes how the models behave on this dataset, not what will happen to you.""",
        "Understand your credit card bill": """
- **Total due** is the full balance. **Minimum due** is the smallest amount that keeps the account in good standing.
- Paying only the minimum usually means interest is charged on the rest.
- Check the statement date, due date, interest rate and any fees each month.""",
        "Manage existing debt": """
- List every debt with its balance, interest rate and monthly payment.
- Many people focus extra payments on the highest-interest debt first, while paying the minimum on the others.
- If you expect to miss a payment, contacting the lender early is usually better than staying silent.""",
        "Credit utilisation": """
- Utilisation is how much of your credit limit you are using, for example a Rs. 30,000 balance on a Rs. 100,000 limit is 30%.
- Lower utilisation is generally viewed more favourably, but each lender has its own approach.
- Spending less relative to your limit is a healthier habit than chasing a higher limit.""",
        "Monitor your repayment history": """
- Check your statements regularly for charges you do not recognise.
- Request your credit report from a credit bureau and look for errors or accounts you do not know.
- Keep records of payments and dispute mistakes in writing.""",
    }
    for title, body in topics.items():
        with st.expander(title):
            st.markdown(body)

    st.caption("This prototype is trained on a 2005 Taiwanese credit card dataset. "
               "It is an educational project, not financial advice or a credit "
               "approval system.")