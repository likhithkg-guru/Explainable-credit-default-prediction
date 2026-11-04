import pandas as pd
import streamlit as st

from explainability import Explainer


@st.cache_resource
def get_explainer(model_name):
    return Explainer(model_name)


@st.cache_data
def get_global_importance(model_name):
    X_test = pd.read_csv("data/X_test.csv")
    return get_explainer(model_name).shap_global(X_test)


def card(label, value, color=None):
    style = ' style="color:' + color + '"' if color else ""
    st.markdown(
        '<div class="card"><div class="label">' + label + "</div><h1" + style + ">" +
        str(value) + "</h1></div>", unsafe_allow_html=True)