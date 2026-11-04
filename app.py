import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from explainability import DISCLAIMER
from predict import (MODEL_LABELS, build_row_dict, row_to_frame, predict_scores,
                     risk_band)
from validation import validate_row
from document_verification import analyze_document, compare_salary, OCR_NOTE
from ui_helpers import card, get_explainer, get_global_importance
from ui_theme import (inject_css, brand_html, page_header, section_title, pill,
                      show_fig, gauge_figure, contribution_chart, importance_chart,
                      BAND_COLORS, COLORS, BRAND_NAME)
from ui_dashboard import (dashboard_page, model_performance_page, shap_page,
                          lime_page, customer_analytics_page, about_page)
from ui_extra import calculator_page, awareness_page

st.set_page_config(page_title=BRAND_NAME + " | Explainable Credit Risk",
                   page_icon="📊", layout="wide")

REQUIRED_FILES = [
    "models/feature_columns.json", "models/decision_tree.joblib",
    "models/random_forest.joblib", "models/xgboost.joblib",
    "data/X_train.csv", "data/X_test.csv", "data/y_test.csv",
]
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error("Some generated files are missing: " + ", ".join(missing) +
             ". Run `python train_model.py` first (see README).")
    st.stop()

inject_css()

EDUCATION = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE = {1: "Married", 2: "Single", 3: "Other"}
FIELD_NAMES = {
    "gross_salary": "Gross salary",
    "net_salary": "Net salary",
    "basic_salary": "Basic salary",
    "total_deductions": "Total deductions",
}


# ------------------------------------------------- Credit Risk Prediction
def borrower_page():
    page_header("Credit Risk Prediction",
                "Enter a customer's card profile and six months of repayment "
                "history to get a model risk score and an explanation.",
                eyebrow="Prediction")
    st.caption(DISCLAIMER)

    model_name = st.selectbox("Model", list(MODEL_LABELS),
                              format_func=MODEL_LABELS.get, index=1)

    with st.form("borrower_form"):
        section_title("Customer profile")
        c1, c2, c3, c4 = st.columns(4)
        limit_bal = c1.number_input("Credit limit (NT$)", min_value=0.0,
                                    value=100000.0, step=5000.0)
        age = c2.number_input("Age", min_value=0, max_value=120, value=30)
        education = c3.selectbox("Education", list(EDUCATION),
                                 format_func=EDUCATION.get, index=1)
        marriage = c4.selectbox("Marital status", list(MARRIAGE),
                                format_func=MARRIAGE.get, index=1)

        section_title("Credit history (last six months)")
        with st.expander("How to fill the repayment status"):
            st.markdown("""
- `-1` = paid in full on time
- `1` = payment delayed by 1 month, `2` = delayed by 2 months, ... up to `8`
- `0` and `-2` are not described in the official dataset notes. They are commonly interpreted as *paid the minimum on a revolving balance* (`0`) and *no card use* (`-2`). Treat this as an assumption.
- Month 1 is the most recent month.
""")
        default_months = pd.DataFrame({
            "Repayment status": [0] * 6,
            "Bill amount": [50000, 48000, 47000, 45000, 43000, 42000],
            "Amount paid": [2000] * 6,
        }, index=["Month 1 (most recent)"] + ["Month " + str(i) for i in range(2, 7)])
        edited = st.data_editor(default_months)

        confirmed = st.checkbox(
            "I confirm these details are entered as I intend. I understand the "
            "estimate is only as reliable as the numbers I provide.")
        submitted = st.form_submit_button("Estimate default risk")

    if not submitted:
        return

    months = [{"status": r["Repayment status"], "bill": r["Bill amount"],
               "paid": r["Amount paid"]} for _, r in edited.iterrows()]
    row = build_row_dict(limit_bal, education, marriage, age, months)
    errors, warnings = validate_row(row)

    if errors:
        for e in errors:
            st.error(e)
        return
    for w in warnings:
        st.warning(w)
    if not confirmed:
        st.error("Please tick the confirmation box before estimating.")
        return

    frame = row_to_frame(row)
    score = float(predict_scores(model_name, frame)[0])
    band, _ = risk_band(score)
    color = BAND_COLORS[band]

    log = st.session_state.setdefault("activity", [])
    log.append({"time": datetime.now().strftime("%H:%M:%S"),
                "model": MODEL_LABELS[model_name], "score": score, "band": band})
    del log[:-20]

    with st.spinner("Computing explanations..."):
        explainer = get_explainer(model_name)
        shap_df = explainer.shap_local(frame)
        lime_df = explainer.lime_local(frame)

    section_title("Result")
    left, right = st.columns([1, 1.3])
    with left:
        show_fig(gauge_figure(score, color))
    with right:
        r1, r2 = st.columns(2)
        with r1:
            card("Risk band (demo)", band, color)
        with r2:
            card("Model used", MODEL_LABELS[model_name])
        ups = shap_df[shap_df["shap_value"] > 0].head(3)
        downs = shap_df[shap_df["shap_value"] < 0].head(3)
        st.markdown('<div class="muted small">Weighed most toward a higher estimate</div>' +
                    "".join(pill("▲ " + lab, "up") for lab in ups["label"]),
                    unsafe_allow_html=True)
        st.markdown('<div class="muted small">Weighed most toward a lower estimate</div>' +
                    "".join(pill("▼ " + lab, "down") for lab in downs["label"]),
                    unsafe_allow_html=True)
        st.caption("This is a model score, not a calibrated probability, and not "
                   "a decision about any loan or card.")

    tab1, tab2, tab3 = st.tabs(["SHAP explanation", "LIME explanation",
                                "Overall feature importance"])
    with tab1:
        show_fig(contribution_chart(shap_df, "label", "shap_value",
                                    "SHAP: what moved this estimate",
                                    "SHAP value (" + explainer.units + ")"))
        with st.expander("Show values"):
            st.dataframe(shap_df[["label", "shap_value"]].head(10))
    with tab2:
        show_fig(contribution_chart(lime_df, "rule", "weight",
                                    "LIME: local approximation", "LIME weight"))
        with st.expander("Show values"):
            st.dataframe(lime_df)
    with tab3:
        show_fig(importance_chart(get_global_importance(model_name)))
    st.caption("Charts show what the model relied on. They do not show what "
               "causes default, and they are not advice.")


# --------------------------------------------------------- Document Check
def document_page():
    page_header("Document Check",
                "Read a salary slip with OCR and compare it with the amount you "
                "entered.", eyebrow="Verification")
    st.warning(
        OCR_NOTE + " Matching your entry against a document only checks that "
        "two things agree. It cannot tell whether the document is real.")
    st.caption("Salary is not an input of the risk model. This page is separate "
               "from the risk estimate.")

    c1, c2 = st.columns(2)
    entered = c1.number_input("Monthly salary you entered (Rs.)", min_value=0.0,
                              value=35000.0, step=500.0)
    salary_type = c2.radio(
        "This amount is my...", ["gross", "net"], horizontal=True,
        format_func=lambda x: "Gross salary (before deductions)"
        if x == "gross" else "Net / take-home salary")

    uploaded = st.file_uploader("Upload a salary slip image (PNG or JPG)",
                                type=["png", "jpg", "jpeg"])
    st.caption("Files are read in memory only and are not saved. Use synthetic "
               "sample documents while testing.")

    if uploaded is None:
        st.session_state.pop("doc_key", None)
        st.session_state.pop("doc_result", None)
        st.info("Upload a document to begin.")
        return

    key = uploaded.name + "-" + str(uploaded.size)
    if st.session_state.get("doc_key") != key:
        with st.spinner("Reading document..."):
            st.session_state["doc_result"] = analyze_document(
                io.BytesIO(uploaded.getvalue()))
            st.session_state["doc_key"] = key
    result = st.session_state["doc_result"]

    left, right = st.columns(2)
    left.image(uploaded.getvalue(), caption="Uploaded document")
    with right:
        section_title("1. Text extracted (OCR)")
        st.metric("OCR confidence", str(result["mean_confidence"]) + "%")
        if result["low_quality"]:
            st.warning("Low OCR confidence. Try a clearer, straighter image.")
        if result["fields"]:
            table = pd.DataFrame([
                {"Field": FIELD_NAMES.get(k, k), "Value": v["value"],
                 "Read from line": v["line"]}
                for k, v in result["fields"].items()])
            st.dataframe(table)
        else:
            st.warning("No known salary fields were found in this document.")
        with st.expander("Show raw OCR text"):
            st.text(result["text"])

    section_title("2. Compare with what you entered")
    comp = compare_salary(entered, salary_type, result)
    {"match": st.success, "differs": st.error,
     "needs_review": st.warning}[comp["status"]](comp["message"])
    for reason in comp["reasons"]:
        st.write("- " + reason)

    m1, m2, m3 = st.columns(3)
    m1.metric("You entered", "Rs. " + format(comp["entered"], ",.0f"))
    m2.metric("Document shows",
              "Not found" if comp["extracted"] is None
              else "Rs. " + format(comp["extracted"], ",.0f"))
    m3.metric("Difference",
              "-" if comp["difference"] is None else format(comp["difference"], "+,.0f"))

    section_title("3. Correct your entry")
    st.write("If there is a discrepancy, check your figure and your gross/net "
             "choice, then change the amount above. The comparison updates "
             "automatically.")
    st.info("A match only shows that the typed number agrees with the text on "
            "the image. Genuine verification needs trusted sources, not OCR.")


# ------------------------------------------------------------- navigation
PAGES = {
    "🏠  Dashboard": dashboard_page,
    "🎯  Credit Risk Prediction": borrower_page,
    "📊  Model Performance": model_performance_page,
    "🧠  Explainable AI (SHAP)": shap_page,
    "🔍  Local Explanations (LIME)": lime_page,
    "👥  Customer Analytics": customer_analytics_page,
    "📄  Document Check": document_page,
    "🧮  Financial Health Calculator": calculator_page,
    "📚  Financial Awareness": awareness_page,
    "ℹ️  About Project": about_page,
}

st.sidebar.markdown(brand_html(), unsafe_allow_html=True)
choice = st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed",
                          key="nav")
st.sidebar.markdown('<div class="glass side-note">Educational prototype.<br>'
                    'Not a credit decision system.</div>', unsafe_allow_html=True)
PAGES[choice]()