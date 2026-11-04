"""Visual theme: CSS, icons and Plotly styling helpers.
Presentation only. No prediction or explanation logic lives here."""
import plotly.graph_objects as go
import streamlit as st

BRAND_NAME = "CreditLens"
BRAND_TAGLINE = "Explainable credit risk"

COLORS = {"blue": "#3B82F6", "cyan": "#22D3EE", "violet": "#8B5CF6",
          "green": "#34D399", "amber": "#FBBF24", "red": "#F87171",
          "muted": "#94A3B8"}
MODEL_COLORS = {"decision_tree": "#FBBF24", "random_forest": "#3B82F6",
                "xgboost": "#22D3EE"}
BAND_COLORS = {"Lower": COLORS["green"], "Moderate": COLORS["amber"],
               "Higher": COLORS["red"]}

_ICON_PATHS = {
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "bar-chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "layers": '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "file-text": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
}


def icon(name, size=22):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">%s</svg>'
            % (size, size, _ICON_PATHS[name]))


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--panel:rgba(255,255,255,0.045);--line:rgba(255,255,255,0.09);--text:#E5E7EB;--muted:#94A3B8;--blue:#3B82F6;--cyan:#22D3EE;--violet:#8B5CF6;--green:#34D399;--amber:#FBBF24;--red:#F87171;}
html,body,.stApp,.stMarkdown,.stMarkdown p,h1,h2,h3,h4,label,button,input,textarea{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;}
.stApp{background:radial-gradient(1100px 600px at 8% -8%,rgba(99,102,241,.22),transparent 60%),radial-gradient(900px 520px at 100% 0%,rgba(34,211,238,.13),transparent 55%),#0B1020 !important;color:var(--text);}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1280px;}
h1,h2,h3{letter-spacing:-.02em;}
[data-testid="stCaptionContainer"],.stCaption{color:var(--muted);}
section[data-testid="stSidebar"],section[data-testid="stSidebar"] > div{background:linear-gradient(180deg,#0E1530 0%,#090E1D 100%) !important;}
section[data-testid="stSidebar"]{border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:.25rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{padding:.55rem .8rem;border-radius:12px;border:1px solid transparent;transition:all .18s ease;cursor:pointer;width:100%;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:rgba(255,255,255,.06);}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{display:none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){background:linear-gradient(90deg,rgba(99,102,241,.30),rgba(34,211,238,.10));border-color:rgba(99,102,241,.50);}
section[data-testid="stSidebar"] div[role="radiogroup"] p{font-weight:500;font-size:.93rem;margin:0;}
.brand{display:flex;align-items:center;gap:.7rem;padding:.2rem .4rem 1rem;}
.brand-mark{width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#3B82F6,#8B5CF6 60%,#22D3EE);display:flex;align-items:center;justify-content:center;color:#fff;box-shadow:0 8px 24px rgba(99,102,241,.45);}
.brand-name{font-weight:800;letter-spacing:-.01em;font-size:1.15rem;line-height:1.1;}
.brand-sub{font-size:.72rem;color:var(--muted);}
.side-note{padding:12px 14px;font-size:.78rem;color:var(--muted);margin-top:1rem;}
.glass,.card,.hero,.kpi,[data-testid="stMetric"],[data-testid="stExpander"],[data-testid="stForm"]{background:var(--panel);border:1px solid var(--line);border-radius:16px;-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);box-shadow:0 10px 30px rgba(0,0,0,.30);transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;}
.glass:hover,.card:hover,.kpi:hover,[data-testid="stMetric"]:hover{transform:translateY(-2px);border-color:rgba(129,140,248,.55);box-shadow:0 14px 40px rgba(59,130,246,.18);}
[data-testid="stForm"]{padding:1.2rem 1.4rem;}
[data-testid="stMetric"]{padding:.9rem 1.1rem;}
[data-testid="stMetricLabel"] p{color:var(--muted);}
.card{padding:18px 22px;}
.card .label{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.card h1{margin:.2rem 0 0;font-size:2rem;font-weight:700;letter-spacing:-.02em;color:var(--text);}
.hero{padding:28px 32px;margin-bottom:1.2rem;background:linear-gradient(135deg,rgba(99,102,241,.18),rgba(34,211,238,.06) 60%,rgba(255,255,255,.03));}
.hero h1{margin:.25rem 0 .35rem;font-size:2rem;font-weight:800;letter-spacing:-.02em;background:linear-gradient(90deg,#fff,#A5B4FC 60%,#67E8F9);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:var(--muted);margin:0;max-width:760px;}
.eyebrow{font-size:.75rem;letter-spacing:.14em;text-transform:uppercase;color:var(--cyan);font-weight:600;}
.section-title{font-size:1.05rem;font-weight:600;margin:1.5rem 0 .7rem;display:flex;align-items:center;gap:.55rem;}
.section-title::before{content:"";width:4px;height:18px;border-radius:2px;background:linear-gradient(var(--blue),var(--violet));}
.kpi{display:flex;align-items:center;gap:14px;padding:18px 20px;position:relative;overflow:hidden;min-height:108px;}
.kpi::before{content:"";position:absolute;top:0;bottom:0;left:0;width:3px;background:var(--accent);}
.kpi-icon{width:44px;height:44px;flex:0 0 44px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:rgba(255,255,255,.06);}
.kpi-label{font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.kpi-value{font-size:1.75rem;font-weight:700;line-height:1.15;letter-spacing:-.02em;}
.kpi-sub{font-size:.76rem;color:var(--muted);margin-top:2px;}
.step{padding:20px;height:100%;}
.step-icon{color:var(--cyan);margin-bottom:.5rem;}
.step-title{font-weight:700;margin-bottom:.3rem;}
.step-text{color:var(--muted);font-size:.9rem;}
.step-text ul{margin:.3rem 0 0 1.1rem;padding:0;}
.step-text li{margin:.25rem 0;}
.panel{padding:14px 18px;}
.muted{color:var(--muted);}
.small{font-size:.8rem;margin:.6rem 0 .3rem;}
.activity{display:flex;justify-content:space-between;padding:.55rem 0;border-bottom:1px solid var(--line);font-size:.9rem;}
.activity:last-child{border-bottom:0;}
.pill{display:inline-block;padding:.22rem .65rem;border-radius:999px;font-size:.76rem;font-weight:600;border:1px solid var(--line);background:rgba(255,255,255,.05);margin:0 .35rem .35rem 0;}
.pill.up{color:var(--red);border-color:rgba(248,113,113,.4);background:rgba(248,113,113,.10);}
.pill.down{color:var(--green);border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.10);}
div.stButton > button,div[data-testid="stFormSubmitButton"] > button,div.stDownloadButton > button{background:linear-gradient(90deg,#3B82F6,#8B5CF6);color:#fff;border:0;border-radius:12px;padding:.55rem 1.3rem;font-weight:600;box-shadow:0 8px 20px rgba(99,102,241,.35);transition:transform .15s ease,box-shadow .15s ease;}
div.stButton > button:hover,div[data-testid="stFormSubmitButton"] > button:hover,div.stDownloadButton > button:hover{transform:translateY(-1px);box-shadow:0 12px 28px rgba(99,102,241,.5);color:#fff;border:0;}
button[data-baseweb="tab"]{font-weight:600;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:#fff;}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,#3B82F6,#22D3EE);height:3px;}
div[data-baseweb="input"],div[data-baseweb="select"] > div{background:rgba(255,255,255,.05) !important;border-radius:10px !important;border-color:var(--line) !important;}
div[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line);}
@media (max-width:768px){.block-container{padding-left:1rem;padding-right:1rem;}.hero{padding:20px;}.hero h1{font-size:1.5rem;}.kpi-value{font-size:1.45rem;}}
"""


def inject_css():
    st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def brand_html():
    return ('<div class="brand"><div class="brand-mark">' + icon("layers", 22) +
            '</div><div><div class="brand-name">' + BRAND_NAME +
            '</div><div class="brand-sub">' + BRAND_TAGLINE + '</div></div></div>')


def page_header(title, subtitle="", eyebrow=""):
    eb = '<div class="eyebrow">' + eyebrow + '</div>' if eyebrow else ""
    sub = "<p>" + subtitle + "</p>" if subtitle else ""
    st.markdown('<div class="hero">' + eb + "<h1>" + title + "</h1>" + sub + "</div>",
                unsafe_allow_html=True)


def section_title(text):
    st.markdown('<div class="section-title">' + text + "</div>", unsafe_allow_html=True)


def pill(text, kind=""):
    return '<span class="pill ' + kind + '">' + text + "</span>"


def kpi_card(icon_name, label, value, sub="", accent=COLORS["blue"]):
    sub_html = '<div class="kpi-sub">' + sub + "</div>" if sub else ""
    st.markdown(
        '<div class="kpi" style="--accent:' + accent + '">'
        '<div class="kpi-icon" style="color:' + accent + '">' + icon(icon_name) + "</div>"
        '<div><div class="kpi-label">' + label + "</div>"
        '<div class="kpi-value">' + str(value) + "</div>" + sub_html + "</div></div>",
        unsafe_allow_html=True)


def glass_card(title, body_html, icon_name=None):
    ic = '<div class="step-icon">' + icon(icon_name) + "</div>" if icon_name else ""
    st.markdown('<div class="glass step">' + ic + '<div class="step-title">' + title +
                '</div><div class="step-text">' + body_html + "</div></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- charts
def style_fig(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#CBD5E1", size=13),
        title=dict(font=dict(size=15, color="#E5E7EB"), x=0.0, xanchor="left"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#111827", font_color="#E5E7EB",
                        bordercolor="rgba(255,255,255,.15)"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)")
    if fig.layout.margin.t is None:
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    if height:
        fig.update_layout(height=height)
    return fig


def show_fig(fig, height=None):
    style_fig(fig, height)
    try:
        st.plotly_chart(fig, theme=None)
    except TypeError:
        st.plotly_chart(fig)


def gauge_figure(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number={"suffix": "%", "font": {"size": 46}},
        title={"text": "Model risk score", "font": {"size": 14, "color": "#94A3B8"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#64748B"},
               "bar": {"color": color, "thickness": 0.28},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 30], "color": "rgba(52,211,153,.18)"},
                         {"range": [30, 60], "color": "rgba(251,191,36,.18)"},
                         {"range": [60, 100], "color": "rgba(248,113,113,.18)"}]}))
    fig.update_layout(height=270, margin=dict(l=25, r=25, t=50, b=5))
    return fig


def contribution_chart(df, label_col, value_col, title, xaxis_title, top_n=10):
    d = df.head(top_n).iloc[::-1]
    colors = [COLORS["red"] if v > 0 else COLORS["green"] for v in d[value_col]]
    fig = go.Figure(go.Bar(
        x=d[value_col], y=d[label_col], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}<br>%{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color="rgba(255,255,255,.35)", line_width=1)
    fig.update_layout(
        title=title, height=max(320, 34 * len(d) + 120),
        xaxis_title=xaxis_title + "  (red raises the estimate, green lowers it)")
    fig.update_yaxes(automargin=True)
    return fig


def importance_chart(df, top_n=15, title="Overall feature importance (mean |SHAP|)"):
    d = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["mean_abs_shap"], y=d["feature"], orientation="h",
        marker=dict(color=d["mean_abs_shap"],
                    colorscale=[[0, COLORS["blue"]], [1, COLORS["violet"]]],
                    line=dict(width=0)),
        hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Average absolute SHAP value",
                      height=max(320, 30 * len(d) + 120))
    fig.update_yaxes(automargin=True)
    return fig