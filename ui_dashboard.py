import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from explainability import DISCLAIMER
from predict import MODEL_LABELS, feature_columns, predict_scores, band_array, risk_band
from ui_helpers import get_explainer, get_global_importance
from ui_professional import load_metrics, performance_tab, batch_tab, individual_tab
from ui_theme import (BRAND_NAME, COLORS, BAND_COLORS, page_header, section_title,
                      kpi_card, glass_card, pill, show_fig, gauge_figure,
                      contribution_chart, importance_chart)


# ------------------------------------------------------------ data helpers
@st.cache_data
def test_data():
    X = pd.read_csv("data/X_test.csv")
    y = pd.read_csv("data/y_test.csv").iloc[:, 0]
    return X, y


@st.cache_data
def test_scores(model_name):
    X, _ = test_data()
    return predict_scores(model_name, X)


@st.cache_data
def dataset_size():
    X, _ = test_data()
    return len(pd.read_csv("data/X_train.csv", usecols=["AGE"])) + len(X)


def model_select(key, label="Model"):
    return st.selectbox(label, list(MODEL_LABELS), format_func=MODEL_LABELS.get,
                        index=1, key=key)


def pick_customer(model_name, key):
    """Choose one customer from the held-out test set (real data)."""
    X, y = test_data()
    scores = test_scores(model_name)
    mode = st.radio("Customer", ["Highest-risk customer", "Lowest-risk customer",
                                 "Choose a row"], horizontal=True, key=key + "_mode")
    if mode == "Highest-risk customer":
        idx = int(np.argmax(scores))
    elif mode == "Lowest-risk customer":
        idx = int(np.argmin(scores))
    else:
        idx = int(st.number_input("Row in the held-out test set (0 to " +
                                  str(len(X) - 1) + ")", min_value=0,
                                  max_value=len(X) - 1, value=0, step=1,
                                  key=key + "_row"))
    return X.iloc[[idx]], idx, float(scores[idx]), int(y.iloc[idx])


def customer_pills(band, idx, actual):
    outcome = "Default" if actual == 1 else "No default"
    st.markdown(pill("Model band: " + band) + pill("Actual outcome in data: " + outcome) +
                pill("Test row " + str(idx)), unsafe_allow_html=True)


# --------------------------------------------------------------- dashboard
def activity_panel():
    log = st.session_state.get("activity", [])
    if not log:
        st.markdown('<div class="glass panel muted">No predictions yet in this '
                    'session. Run one on the Credit Risk Prediction page. Activity '
                    'is kept in memory only and disappears when you close the tab.'
                    '</div>', unsafe_allow_html=True)
        return
    rows = ""
    for e in reversed(log[-6:]):
        rows += ('<div class="activity"><span>' + e["time"] + " · " + e["model"] +
                 '</span><span style="color:' + BAND_COLORS[e["band"]] +
                 ';font-weight:600">' + format(e["score"], ".0%") + " · " +
                 e["band"] + "</span></div>")
    st.markdown('<div class="glass panel">' + rows + "</div>", unsafe_allow_html=True)


def dashboard_page():
    page_header("Welcome to " + BRAND_NAME,
                "Estimate credit-card default risk, see which inputs the model "
                "weighed, and check supporting documents, all in one explainable "
                "workspace.", eyebrow="Credit risk intelligence")

    metrics, _ = load_metrics()
    sel, _spacer = st.columns([1, 3])
    with sel:
        model_name = model_select("dash_model", "Model powering this overview")
    X, y = test_data()
    scores = test_scores(model_name)
    n_test = len(X)
    high = int((scores >= 0.60).sum())
    low = int((scores < 0.30).sum())
    base = 1 - float(y.mean())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("users", "Total customers", format(dataset_size(), ","),
                 "UCI dataset records", COLORS["blue"])
    with k2:
        kpi_card("alert", "High risk customers", format(high, ","),
                 format(high / n_test, ".1%") + " of " + format(n_test, ",") +
                 " test customers", COLORS["red"])
    with k3:
        kpi_card("shield", "Low risk customers", format(low, ","),
                 format(low / n_test, ".1%") + " of " + format(n_test, ",") +
                 " test customers", COLORS["green"])
    with k4:
        kpi_card("target", "Model accuracy",
                 format(metrics[model_name]["accuracy"], ".1%"),
                 "ROC-AUC " + format(metrics[model_name]["roc_auc"], ".3f") +
                 " · baseline " + format(base, ".1%"), COLORS["violet"])

    section_title("Distributions (held-out test set)")
    c1, c2 = st.columns(2)
    donut = go.Figure(go.Pie(
        labels=["No default", "Default"],
        values=[int((y == 0).sum()), int((y == 1).sum())], hole=0.62,
        marker=dict(colors=[COLORS["blue"], COLORS["red"]],
                    line=dict(color="#0B1020", width=2)),
        textinfo="percent"))
    donut.add_annotation(text=format(len(y), ",") + "<br>customers",
                         showarrow=False, font=dict(size=15))
    donut.update_layout(title="Credit default distribution (actual outcomes)")
    with c1:
        show_fig(donut, height=360)

    counts = (pd.Series(band_array(scores)).value_counts()
              .reindex(["Lower", "Moderate", "Higher"]).fillna(0))
    bar = go.Figure(go.Bar(x=list(counts.index), y=counts.values,
                           marker_color=[BAND_COLORS[b] for b in counts.index],
                           text=[format(v / n_test, ".1%") for v in counts.values],
                           textposition="outside"))
    bar.update_layout(title="Risk category distribution (" + MODEL_LABELS[model_name] +
                      ")", yaxis_title="Customers")
    with c2:
        show_fig(bar, height=360)

    left, right = st.columns([3, 2])
    with left:
        section_title("Do scores separate the two outcomes?")
        hist = go.Figure()
        for label, val, color in (("No default", 0, COLORS["blue"]),
                                  ("Default", 1, COLORS["red"])):
            hist.add_trace(go.Histogram(x=scores[y.values == val] * 100, name=label,
                                        marker_color=color, opacity=0.65, nbinsx=30))
        hist.update_layout(barmode="overlay", xaxis_title="Model risk score (%)",
                           yaxis_title="Customers")
        show_fig(hist, height=340)
    with right:
        section_title("Recent prediction activity")
        activity_panel()

    section_title("How it works")
    s1, s2, s3 = st.columns(3)
    with s1:
        glass_card("1. Predict", "Enter a customer profile and six months of "
                   "repayment history. Three tree-based models produce a risk score.",
                   "target")
    with s2:
        glass_card("2. Explain", "SHAP and LIME show which inputs pushed the "
                   "estimate up or down. They describe the model, not causes.",
                   "search")
    with s3:
        glass_card("3. Verify", "Upload a sample salary slip. OCR extracts text and "
                   "compares it with your entry. It cannot prove authenticity.",
                   "file-text")
    st.caption(DISCLAIMER)


# ------------------------------------------------------ simple page wrappers
def model_performance_page():
    page_header("Model Performance",
                "Compare the Decision Tree, Random Forest and XGBoost on the "
                "held-out test set.", eyebrow="Evaluation")
    performance_tab()


def shap_page():
    page_header("Explainable AI (SHAP)",
                "SHAP splits one prediction into per-feature contributions and "
                "shows which inputs matter most overall.", eyebrow="Explainability")
    model_name = model_select("shap_model")
    explainer = get_explainer(model_name)

    section_title("Overall feature importance")
    show_fig(importance_chart(get_global_importance(model_name)))

    section_title("Explain one customer")
    frame, idx, score, actual = pick_customer(model_name, "shap")
    band, _ = risk_band(score)
    color = BAND_COLORS[band]
    customer_pills(band, idx, actual)
    with st.spinner("Computing SHAP values..."):
        shap_df = explainer.shap_local(frame)
    left, right = st.columns([1, 2])
    with left:
        show_fig(gauge_figure(score, color))
    with right:
        show_fig(contribution_chart(shap_df, "label", "shap_value",
                                    "SHAP contributions for this customer",
                                    "SHAP value (" + explainer.units + ")"))
    with st.expander("Show values"):
        st.dataframe(shap_df[["label", "shap_value"]].head(15))
    st.caption("Positive values push the estimate up, negative values push it "
               "down. XGBoost values are in log-odds and the other models in "
               "probability, so compare the ordering across models, not bar sizes. "
               + DISCLAIMER)


def lime_page():
    page_header("Local Explanations (LIME)",
                "LIME fits a small, simple model around one customer to "
                "approximate how the full model behaves nearby.",
                eyebrow="Explainability")
    model_name = model_select("lime_model")
    explainer = get_explainer(model_name)
    frame, idx, score, actual = pick_customer(model_name, "lime")
    band, _ = risk_band(score)
    color = BAND_COLORS[band]
    customer_pills(band, idx, actual)
    with st.spinner("Computing explanations..."):
        lime_df = explainer.lime_local(frame)
        shap_df = explainer.shap_local(frame)

    left, right = st.columns([1, 2])
    with left:
        show_fig(gauge_figure(score, color))
    with right:
        show_fig(contribution_chart(lime_df, "rule", "weight",
                                    "LIME feature contributions", "LIME weight"))
    with st.expander("Show LIME rules and weights"):
        st.dataframe(lime_df)

    section_title("Compare with SHAP for the same customer")
    show_fig(contribution_chart(shap_df, "label", "shap_value", "SHAP contributions",
                                "SHAP value (" + explainer.units + ")"))
    st.caption("LIME is an approximation built from random samples, so small "
               "differences between runs and between LIME and SHAP are normal. "
               "Neither shows causes. " + DISCLAIMER)


# -------------------------------------------------------- customer analytics
def insights_tab():
    X, y = test_data()
    df = X.copy()
    df["default"] = y.values
    st.caption("Actual default rates in the " + format(len(df), ",") + " held-out "
               "test customers. These are descriptive statistics, not causes, and "
               "small groups are unreliable (hover to see group sizes).")

    def rate_chart(frame, group_col, title, xlabel):
        g = (frame.groupby(group_col, observed=True)["default"]
             .agg(rate="mean", n="count").reset_index())
        g = g[g["n"] >= 30]
        pct = g["rate"] * 100
        fig = go.Figure(go.Bar(
            x=g[group_col].astype(str), y=pct, customdata=g["n"],
            marker=dict(color=pct, colorscale=[[0, COLORS["blue"]], [1, COLORS["red"]]]),
            hovertemplate=xlabel + " %{x}<br>Default rate %{y:.1f}%<br>"
                          "customers %{customdata}<extra></extra>"))
        fig.update_layout(title=title, xaxis_title=xlabel,
                          yaxis_title="Actual default rate (%)")
        return fig

    c1, c2 = st.columns(2)
    with c1:
        show_fig(rate_chart(df, "PAY_1", "Default rate by latest repayment status",
                            "PAY_1"), height=360)
    df["age_band"] = pd.cut(df["AGE"], [0, 29, 39, 49, 59, 120],
                            labels=["Under 30", "30-39", "40-49", "50-59", "60+"])
    with c2:
        show_fig(rate_chart(df, "age_band", "Default rate by age band", "Age band"),
                 height=360)
    df["limit_band"] = pd.qcut(df["LIMIT_BAL"], 5,
                               labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    show_fig(rate_chart(df, "limit_band", "Default rate by credit-limit quintile",
                        "Credit limit"), height=340)

    section_title("Are the scores probabilities? Calibration check")
    model_name = model_select("insight_model")
    scores = test_scores(model_name)
    bins = pd.qcut(pd.Series(scores), 10, duplicates="drop")
    g = (pd.DataFrame({"score": scores, "actual": y.values, "bin": bins})
         .groupby("bin", observed=True)
         .agg(score=("score", "mean"), actual=("actual", "mean"), n=("actual", "size"))
         .reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines",
                             line=dict(dash="dash", color="#64748B"),
                             name="Perfectly calibrated"))
    fig.add_trace(go.Scatter(x=g["score"] * 100, y=g["actual"] * 100,
                             mode="lines+markers", customdata=g["n"],
                             line=dict(color=COLORS["cyan"], width=3),
                             name=MODEL_LABELS[model_name],
                             hovertemplate="Average score %{x:.1f}%<br>Actual default "
                                           "rate %{y:.1f}%<br>customers %{customdata}"
                                           "<extra></extra>"))
    fig.update_layout(title="Average model score vs actual default rate (score groups)",
                      xaxis_title="Average model risk score (%)",
                      yaxis_title="Actual default rate (%)")
    show_fig(fig, height=420)
    st.caption("Points below the dashed line mean the score is higher than the "
               "real default rate in that group. That is why scores are shown as "
               "risk scores, not probabilities.")


def customer_analytics_page():
    page_header("Customer Analytics",
                "Score many customers from a CSV, inspect individuals, and explore "
                "patterns in the held-out test data.", eyebrow="Analytics")
    st.warning("Prototype: no login or access control is implemented, so anyone who "
               "can open this page can use it. A real system would need "
               "authentication, authorisation and audit logging. Do not upload real "
               "customer data.")
    t1, t2, t3 = st.tabs(["Batch upload and scoring", "Individual analysis",
                          "Portfolio insights"])
    with t1:
        batch_tab()
    with t2:
        individual_tab()
    with t3:
        insights_tab()


# ------------------------------------------------------------------- about
def about_page():
    page_header("About This Project",
                "AI-powered financial document verification and explainable credit "
                "default prediction using SHAP, LIME and tree-based models.",
                eyebrow="Project")
    X, y = test_data()
    a, b = st.columns(2)
    with a:
        glass_card("What it does",
                   "<ul><li>Estimates the risk of a credit-card customer missing a "
                   "payment next month.</li><li>Explains each estimate with SHAP and "
                   "LIME.</li><li>Reads a sample salary slip with OCR and compares it "
                   "with the entered value.</li><li>Scores CSV files of customers and "
                   "compares three models.</li></ul>", "layers")
    with b:
        glass_card("Data and models",
                   "<ul><li>UCI <i>Default of Credit Card Clients</i>: " +
                   format(dataset_size(), ",") + " customers (Taiwan, 2005, amounts in "
                   "NT$).</li><li>" + str(len(feature_columns())) + " model features; "
                   "about " + format(float(y.mean()), ".1%") + " of customers "
                   "defaulted.</li><li>Decision Tree, Random Forest and XGBoost; "
                   "SHAP and LIME for explanations.</li></ul>", "bar-chart")
    section_title("Stack")
    st.markdown("".join(pill(t) for t in [
        "Python", "Streamlit", "Pandas", "NumPy", "Scikit-learn", "XGBoost", "SHAP",
        "LIME", "Tesseract OCR", "Pillow", "OpenCV", "Plotly"]), unsafe_allow_html=True)
    section_title("Limitations")
    c, d = st.columns(2)
    with c:
        glass_card("Model",
                   "<ul><li>Predicts credit-card default, not general loan default."
                   "</li><li>No salary or employment information in the data.</li>"
                   "<li>Scores are not calibrated probabilities.</li><li>Not trained "
                   "on Indian customers or current data.</li></ul>", "alert")
    with d:
        glass_card("Explanations and documents",
                   "<ul><li>SHAP and LIME describe model behaviour, not causes."
                   "</li><li>OCR reads text. It does not prove a document is genuine."
                   "</li><li>Tested on synthetic documents only.</li><li>No login or "
                   "access control.</li></ul>", "shield")
    st.caption(DISCLAIMER)