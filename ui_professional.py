import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

from explainability import DISCLAIMER
from predict import (MODEL_LABELS, feature_columns, predict_scores, band_array,
                     check_batch, MAX_ROWS, risk_band)
from ui_helpers import card, get_explainer
from ui_theme import (BAND_COLORS, COLORS, MODEL_COLORS, kpi_card, section_title,
                      show_fig, gauge_figure, contribution_chart)

METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}


# ------------------------------------------------------------ performance
@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    cv = None
    if os.path.exists("models/cv_results.json"):
        with open("models/cv_results.json") as f:
            cv = json.load(f)
    return metrics, cv


@st.cache_data
def roc_data():
    X = pd.read_csv("data/X_test.csv")
    y = pd.read_csv("data/y_test.csv").iloc[:, 0]
    out = {}
    for m in MODEL_LABELS:
        p = predict_scores(m, X)
        fpr, tpr, _ = roc_curve(y, p)
        out[m] = (fpr, tpr, roc_auc_score(y, p))
    return out


@st.cache_data
def baseline_accuracy():
    y = pd.read_csv("data/y_test.csv").iloc[:, 0]
    return 1 - float(y.mean())


def performance_tab():
    metrics, cv = load_metrics()
    model_name = st.selectbox("Model", list(MODEL_LABELS),
                              format_func=MODEL_LABELS.get, index=1, key="perf_model")
    m = metrics[model_name]
    spec = [("Accuracy", "accuracy", "target", COLORS["blue"]),
            ("Precision", "precision", "shield", COLORS["cyan"]),
            ("Recall", "recall", "activity", COLORS["violet"]),
            ("F1 score", "f1", "bar-chart", COLORS["green"]),
            ("ROC-AUC", "roc_auc", "trending-up", COLORS["amber"])]
    for col, (label, key, ic, accent) in zip(st.columns(5), spec):
        with col:
            sub = "Always-'no default' baseline " + format(baseline_accuracy(), ".1%") \
                if key == "accuracy" else ""
            kpi_card(ic, label, format(m[key], ".3f"), sub, accent)

    section_title("Model comparison")
    table = pd.DataFrame({
        MODEL_LABELS[k]: {METRIC_NAMES[n]: metrics[k][n] for n in METRIC_NAMES}
        for k in MODEL_LABELS}).T
    st.dataframe(table.round(3))
    long = (table.reset_index()
            .melt(id_vars="index", var_name="Metric", value_name="Value")
            .rename(columns={"index": "Model"}))
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group",
                 range_y=[0, 1.08], text_auto=".2f",
                 color_discrete_map={MODEL_LABELS[k]: MODEL_COLORS[k] for k in MODEL_LABELS})
    fig.update_traces(textposition="outside")
    fig.update_layout(title="Held-out test-set metrics")
    show_fig(fig, height=420)

    section_title("ROC curves")
    fig = go.Figure()
    for k, (fpr, tpr, auc) in roc_data().items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 line=dict(color=MODEL_COLORS[k], width=3),
                                 name=MODEL_LABELS[k] + " (AUC " + format(auc, ".3f") + ")"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="#64748B"),
                             name="Random guessing"))
    fig.update_layout(xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    show_fig(fig, height=420)

    section_title("Confusion matrices")
    for col, k in zip(st.columns(3), MODEL_LABELS):
        cm = np.array(metrics[k]["confusion_matrix"])
        f = px.imshow(cm, text_auto=True, title=MODEL_LABELS[k],
                      x=["Pred: no default", "Pred: default"],
                      y=["Actual: no default", "Actual: default"],
                      color_continuous_scale=[[0, "#111827"], [1, "#6366F1"]])
        f.update_coloraxes(showscale=False)
        with col:
            show_fig(f, height=330)

    if cv:
        section_title("5-fold cross-validation (ROC-AUC, training data)")
        st.dataframe(pd.DataFrame(cv).T.rename(
            index=MODEL_LABELS,
            columns={"mean_auc": "Mean AUC", "std_auc": "Std dev"}).round(3))
    st.caption("Metrics come from one 80/20 split of the UCI dataset. Small "
               "differences between models may not be meaningful.")


# ------------------------------------------------------------------ batch
def batch_tab():
    cols = feature_columns()
    st.write("Upload a CSV with one row per customer and the columns below.")
    st.download_button("Download CSV template",
                       data=pd.DataFrame(columns=cols).to_csv(index=False),
                       file_name="customer_template.csv", mime="text/csv")
    st.caption("Required columns: " + ", ".join(cols))
    st.caption("Up to " + format(MAX_ROWS, ",") + " rows. Extra columns are ignored, "
               "non-numeric entries count as missing, and rows with errors are "
               "skipped and listed. Files are processed in memory and not saved. "
               "Use synthetic or public data only.")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_csv")
    if uploaded is None:
        st.session_state.pop("batch", None)
        st.info("Upload a CSV to begin.")
        return

    key = uploaded.name + "-" + str(uploaded.size)
    batch = st.session_state.get("batch")
    if batch is None or batch["key"] != key:
        st.session_state.pop("batch", None)
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Could not read this file as CSV: " + str(e))
            return
        if len(df) > MAX_ROWS:
            st.error("This prototype accepts up to " + format(MAX_ROWS, ",") + " rows.")
            return
        with st.spinner("Checking and scoring rows..."):
            valid, problems, warning_rows, missing_cols = check_batch(df)
            if missing_cols:
                st.error("Missing required columns: " + ", ".join(missing_cols))
                return
            if valid.empty:
                st.error("No rows passed validation.")
                st.dataframe(problems.head(50))
                return
            scores = {m: predict_scores(m, valid) for m in MODEL_LABELS}
        batch = {"key": key, "uploaded_rows": len(df), "valid": valid,
                 "problems": problems, "warning_rows": warning_rows,
                 "scores": scores}
        st.session_state["batch"] = batch

    valid, problems = batch["valid"], batch["problems"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("users", "Rows uploaded", format(batch["uploaded_rows"], ","),
                 accent=COLORS["blue"])
    with c2:
        kpi_card("shield", "Rows scored", format(len(valid), ","), accent=COLORS["green"])
    with c3:
        kpi_card("alert", "Skipped (errors)", format(problems["row"].nunique(), ","),
                 accent=COLORS["red"])
    with c4:
        kpi_card("activity", "With warnings", format(batch["warning_rows"], ","),
                 accent=COLORS["amber"])
    if not problems.empty:
        with st.expander("Rows that were skipped, and why"):
            st.dataframe(problems.head(100))

    model_name = st.selectbox("Model for the charts below", list(MODEL_LABELS),
                              format_func=MODEL_LABELS.get, index=1, key="batch_model")
    scores = batch["scores"][model_name]
    bands = band_array(scores)
    counts = (pd.Series(bands).value_counts()
              .reindex(["Lower", "Moderate", "Higher"]).fillna(0))

    section_title("Risk distribution")
    left, right = st.columns(2)
    hist = px.histogram(x=scores * 100, nbins=30, labels={"x": "Model risk score (%)"})
    hist.update_traces(marker_color=COLORS["violet"])
    hist.update_layout(yaxis_title="Customers", bargap=0.05, title="Score distribution")
    with left:
        show_fig(hist, height=340)
    bar = go.Figure(go.Bar(x=list(counts.index), y=counts.values,
                           marker_color=[BAND_COLORS[b] for b in counts.index],
                           text=counts.values.astype(int), textposition="outside"))
    bar.update_layout(title="Risk bands (demo)", yaxis_title="Customers")
    with right:
        show_fig(bar, height=340)

    section_title("Model comparison on this file")
    comp = pd.DataFrame({
        MODEL_LABELS[m]: {
            "Average risk score (%)": s.mean() * 100,
            "In Higher band (%)": (s >= 0.60).mean() * 100,
            "In Lower band (%)": (s < 0.30).mean() * 100}
        for m, s in batch["scores"].items()}).T
    st.dataframe(comp.round(1))

    result = valid.copy()
    result["risk_score"] = scores.round(4)
    result["risk_band"] = bands
    result.index.name = "row"
    section_title("Highest-scoring rows")
    st.dataframe(result.sort_values("risk_score", ascending=False)
                 [["risk_score", "risk_band", "LIMIT_BAL", "AGE", "PAY_1",
                   "PAY_2", "BILL_AMT1", "PAY_AMT1"]].head(20))
    st.download_button("Download scored CSV",
                       data=result.reset_index().to_csv(index=False).encode("utf-8"),
                       file_name="scored_customers.csv", mime="text/csv")
    st.caption("Scores are model outputs, not calibrated probabilities, and bands "
               "are for demonstration. Not a lending decision.")


# ------------------------------------------------------------- individual
def individual_tab():
    batch = st.session_state.get("batch")
    if batch is None:
        st.info("Upload and score a CSV in the Batch tab first.")
        return
    valid = batch["valid"]

    model_name = st.selectbox("Model", list(MODEL_LABELS),
                              format_func=MODEL_LABELS.get, index=1,
                              key="single_model")
    scores = batch["scores"][model_name]
    top_row = int(valid.index[int(np.argmax(scores))])
    row_no = st.number_input(
        "Row number (first data row = 1)", min_value=1,
        max_value=int(valid.index.max()), value=int(valid.index.min()), step=1,
        key="single_row_" + batch["key"])
    st.caption("Highest-scoring row under this model: " + str(top_row))

    if row_no not in valid.index:
        st.error("Row " + str(row_no) + " was not scored because it had errors.")
        return

    score = float(scores[valid.index.get_loc(row_no)])
    band, _ = risk_band(score)
    color = BAND_COLORS[band]
    left, right = st.columns([1, 1.3])
    with left:
        show_fig(gauge_figure(score, color))
    with right:
        c1, c2 = st.columns(2)
        with c1:
            card("Risk band (demo)", band, color)
        with c2:
            card("Row", str(row_no))

    frame = valid.loc[[row_no]]
    with st.spinner("Computing explanations..."):
        explainer = get_explainer(model_name)
        shap_df = explainer.shap_local(frame)
        lime_df = explainer.lime_local(frame)

    t1, t2, t3 = st.tabs(["SHAP", "LIME", "Customer record"])
    with t1:
        show_fig(contribution_chart(shap_df, "label", "shap_value",
                                    "SHAP: what moved this estimate",
                                    "SHAP value (" + explainer.units + ")"))
    with t2:
        show_fig(contribution_chart(lime_df, "rule", "weight",
                                    "LIME: local approximation", "LIME weight"))
    with t3:
        st.dataframe(frame.T)
    st.caption(DISCLAIMER)


# ------------------------------------------------------------------- page
def professional_page():
    """Kept for compatibility. The new navigation shows these tabs as
    separate pages (Model Performance and Customer Analytics)."""
    st.title("Financial Professional Dashboard")
    st.caption(DISCLAIMER)
    t1, t2, t3 = st.tabs(["Model performance", "Batch prediction", "Individual analysis"])
    with t1:
        performance_tab()
    with t2:
        batch_tab()
    with t3:
        individual_tab()