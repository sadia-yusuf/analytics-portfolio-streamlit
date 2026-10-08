"""
Analytics portfolio - Sadia Yusuf, Chartered Accountant.

One Streamlit app, six projects. Each project opens as a "working paper":
the question, the work performed, then findings you can filter and re-run.

Run locally:   streamlit run app.py
Deploy:        push this folder (app.py, requirements.txt, data/, .streamlit/) to GitHub,
               then create the app on share.streamlit.io and point it at app.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.stats import norm

DATA = Path(__file__).parent / "data"

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
INK = "#14213D"      # navy, the colour of audit-file ink
MUTED = "#5B677A"
RULE = "#D3DAE4"
PAPER = "#F3F5F8"
BLUE = "#33689E"
TEAL = "#0E8F6E"     # the tick mark: checked, recomputed
AMBER = "#C77A12"    # the exception: flagged, above baseline
PLUM = "#7B4B94"
SLATE = "#8896AB"
BRICK = "#B5452F"
PALETTE = [BLUE, TEAL, AMBER, PLUM, SLATE, BRICK, "#2B9BB8", "#8A7B2E"]
px.defaults.color_discrete_sequence = PALETTE

st.set_page_config(
    page_title="Analytics portfolio - Sadia Yusuf",
    page_icon="📒",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --rule:{RULE}; --paper:{PAPER}; --tick:{TEAL}; --flag:{AMBER}; --blue:{BLUE}; }}
.stApp, .stMarkdown, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp textarea, .stApp td, .stApp th {{ font-family: 'Public Sans', system-ui, sans-serif; }}
.stApp {{ background: var(--paper); color: var(--ink); }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ font-family: 'Newsreader', Georgia, serif !important; color: var(--ink); letter-spacing: -0.01em; }}
h1 {{ font-weight: 600; font-size: 2.35rem !important; line-height: 1.12 !important; }}
h2 {{ font-weight: 600; font-size: 1.55rem !important; }}
h3 {{ font-weight: 600; font-size: 1.25rem !important; }}
p, li {{ line-height: 1.6; }}
section[data-testid="stSidebar"] {{ background: #E6EAF0; border-right: 1px solid var(--rule); }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
.side-name {{ font-family: 'Newsreader', serif; font-size: 1.35rem; font-weight: 600; margin: 0; }}
.side-role {{ color: var(--muted); font-size: .88rem; margin: .1rem 0 1rem; }}

/* working-paper header */
.wp-head {{ display:flex; gap:1.1rem; align-items:flex-start; margin-bottom:.4rem; }}
.wp-ref {{ flex:none; white-space:nowrap; border:1.5px solid var(--ink); padding:.35rem .6rem; font-weight:600; font-size:.95rem;
          border-radius:3px; background:#fff; margin-top:.55rem; font-variant-numeric: tabular-nums; }}
.wp-head h1 {{ margin:0; padding:0 !important; }}
.wp-course {{ color: var(--muted); margin:.3rem 0 0; font-size:.95rem; }}
.wp-grid {{ display:grid; grid-template-columns: 1fr 1.6fr; gap:0; background:#fff; border:1px solid var(--rule);
           border-radius:4px; margin:1rem 0 1.4rem; }}
.wp-grid > div {{ padding:.9rem 1.1rem; border-right:1px solid var(--rule); }}
.wp-grid > div:nth-child(2) {{ border-right:none; }}
.wp-tools {{ grid-column: 1 / -1; border-top:1px solid var(--rule); border-right:none !important; padding:.55rem 1.1rem !important; font-size:.88rem; color: var(--muted); }}
.wp-tools b {{ color: var(--ink); font-weight:600; margin-right:.4rem; }}
.wp-grid h4 {{ margin:0 0 .35rem; font-size:1.02rem !important; }}
.wp-grid p, .wp-grid li {{ font-size:.92rem; margin:0; color:#27324a; }}
.wp-grid ul {{ margin:0; padding-left:1.05rem; }}
@media (max-width: 900px) {{ .wp-grid {{ grid-template-columns: 1fr; }} .wp-grid > div {{ border-right:none; border-bottom:1px solid var(--rule); }} }}

/* key figures */
.kpis {{ display:flex; flex-wrap:wrap; gap:.8rem; margin:.4rem 0 1.1rem; }}
.kpi {{ flex:1 1 150px; background:#fff; border:1px solid var(--rule); border-left:3px solid var(--blue);
        border-radius:3px; padding:.65rem .85rem; }}
.kpi b {{ display:block; font-family:'Newsreader',serif; font-size:1.65rem; font-weight:600; line-height:1.15;
         font-variant-numeric: tabular-nums; }}
.kpi span {{ color: var(--muted); font-size:.85rem; }}

/* findings with tick marks */
.finds {{ list-style:none; padding:0; margin:.2rem 0 0; }}
.finds li {{ display:flex; gap:.7rem; padding:.55rem 0; border-bottom:1px solid var(--rule); font-size:.97rem; }}
.finds li:last-child {{ border-bottom:none; }}
.mk {{ flex:none; width:1.25rem; text-align:center; font-weight:700; }}
.mk.v {{ color: var(--tick); }}
.mk.n {{ color: var(--flag); }}
.legend {{ font-size:.86rem; color: var(--muted); margin: .2rem 0 .9rem; }}
.legend .mk {{ display:inline-block; width:auto; margin: 0 .15rem 0 .5rem; }}
.note {{ color: var(--muted); font-size:.88rem; }}
.callout {{ background:#fff; border:1px solid var(--rule); border-left:3px solid var(--flag); padding:.7rem 1rem; border-radius:3px; margin:.6rem 0; }}

/* overview index */
.idx-ref {{ white-space:nowrap; font-weight:600; border:1.5px solid var(--ink); border-radius:3px; padding:.15rem .5rem; background:#fff; display:inline-block; }}
.idx-title {{ font-family:'Newsreader',serif; font-weight:600; font-size:1.2rem; margin:0; }}
.idx-q {{ color: var(--muted); font-size:.92rem; margin:.1rem 0 0; }}
.idx-res {{ font-size:.93rem; margin:0; }}
.idx-tools {{ color: var(--muted); font-size:.82rem; margin:.25rem 0 0; }}

/* controls */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{ padding: .55rem 0; font-weight:500; }}
.stTabs [aria-selected="true"] {{ color: var(--ink); }}
:focus-visible {{ outline: 2px solid {BLUE} !important; outline-offset: 2px; }}
.stButton button {{ border-radius: 3px; border:1.5px solid var(--ink); color: var(--ink); background:#fff; font-weight:500; }}
.stButton button:hover {{ background: var(--ink); color:#fff; border-color: var(--ink); }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
footer {{ visibility: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def show(fig: go.Figure, height: int = 380, title: str | None = None) -> None:
    """Apply the house chart style and render it."""
    fig.update_layout(
        height=height,
        font=dict(family="Public Sans, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        margin=dict(l=10, r=10, t=54 if title else 18, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family="Public Sans"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(family="Newsreader, serif", size=18)))
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickcolor=RULE, zeroline=False)
    fig.update_yaxes(gridcolor="#E3E8EF", linecolor="rgba(0,0,0,0)", zeroline=False)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kpis(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def finds(items: list[tuple[str, str]]) -> None:
    """items: ('v'|'n', text). v = recomputed here, n = taken from the project notebook or report."""
    rows = "".join(
        f'<li><span class="mk {k}">{"✓" if k == "v" else "△"}</span><span>{t}</span></li>' for k, t in items
    )
    st.markdown(f'<ul class="finds">{rows}</ul>', unsafe_allow_html=True)


def legend() -> None:
    st.markdown(
        '<p class="legend">Tick marks:<span class="mk v">✓</span>recomputed live from the data in this app'
        '<span class="mk n">△</span>taken from the project notebook or report (raw data not bundled)</p>',
        unsafe_allow_html=True,
    )


def workpaper(ref: str, title: str, course: str, objective: str, performed: list[str], tools: str) -> None:
    lis = "".join(f"<li>{p}</li>" for p in performed)
    st.markdown(
        f"""
<div class="wp-head"><div class="wp-ref">{ref}</div><div><h1>{title}</h1><p class="wp-course">{course}</p></div></div>
<div class="wp-grid">
  <div><h4>Objective</h4><p>{objective}</p></div>
  <div><h4>Work performed</h4><ul>{lis}</ul></div>
  <div class="wp-tools"><b>Tools</b>{tools}</div>
</div>""",
        unsafe_allow_html=True,
    )


def conclusion(text: str) -> None:
    st.markdown("### Conclusion")
    st.markdown(text)


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


# ----------------------------------------------------------------------------
# Data loaders (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_turtle() -> pd.DataFrame:
    d = pd.read_csv(DATA / "turtle_reviews.csv")
    d = d.rename(columns={"remuneration (k£)": "remuneration", "spending_score (1-100)": "spending_score"})
    return d.drop(columns=["language", "platform"])


@st.cache_data(show_spinner=False)
def load_ar() -> pd.DataFrame:
    return pd.read_csv(DATA / "nhs_ar.csv")


@st.cache_data(show_spinner=False)
def load_nc() -> pd.DataFrame:
    return pd.read_csv(DATA / "nhs_nc.csv")


@st.cache_data(show_spinner=False)
def load_ad() -> pd.DataFrame:
    return pd.read_csv(DATA / "nhs_ad.csv")


@st.cache_data(show_spinner=False)
def load_tags() -> pd.DataFrame:
    return pd.read_csv(DATA / "tweet_hashtags.csv")


@st.cache_data(show_spinner=False)
def load_marketing() -> pd.DataFrame:
    d = pd.read_csv(DATA / "marketing.csv")
    d = d[~d["Marital_Status"].isin(["Absurd", "YOLO"])].copy()
    d["Age band"] = pd.cut(d["Average Age"], [0, 34, 44, 54, 64, 120], labels=["Under 35", "35-44", "45-54", "55-64", "65+"])
    d["Income tier"] = pd.cut(
        d["Income"], [-1, 10_000, 25_000, 50_000, 75_000, 100_000, 1e9],
        labels=["Under 10K", "10-25K", "25-50K", "50-75K", "75-100K", "100K+"],
    )
    return d


@st.cache_data(show_spinner=False)
def load_bank() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(DATA / "transactions.csv", parse_dates=["datetime"])
    t["month"] = t["datetime"].dt.to_period("M").astype(str)
    a = pd.read_csv(DATA / "anomaly_flagged.csv")
    return t, a


# ----------------------------------------------------------------------------
# Pricing maths (option page + overview)
# ----------------------------------------------------------------------------
def bs_call(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def binomial_call(S, K, T, r, sig, N):
    dt = T / N
    u = np.exp(sig * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(N + 1)
    V = np.maximum(S * u ** (N - j) * d**j - K, 0.0)
    for _ in range(N):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
    return float(V[0])


def mc_payoffs(S, K, T, r, sig, n, seed):
    z = np.random.default_rng(seed).standard_normal(n)
    st_ = S * np.exp((r - 0.5 * sig**2) * T + sig * np.sqrt(T) * z)
    return np.exp(-r * T) * np.maximum(st_ - K, 0.0)


def mc_call(S, K, T, r, sig, n, seed):
    pay = mc_payoffs(S, K, T, r, sig, n, seed)
    return float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(n))


# ============================================================================
# PAGES
# ============================================================================
PAGES = {
    "overview": "Overview",
    "turtle": "Turtle Games: loyalty and segments",
    "churn": "ConnectTel: customer churn",
    "nhs": "NHS: appointment capacity",
    "market": "2Market: customers and channels",
    "bank": "Banking: transaction anomalies",
    "option": "Options: three-way price check",
}
REFS = {"turtle": "WP 1", "churn": "WP 2", "nhs": "WP 3", "market": "WP 4", "bank": "WP 5", "option": "WP 6"}


def go_to(key: str) -> None:
    st.session_state["page"] = key


# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
def page_overview() -> None:
    t = load_turtle()
    r2 = stats.linregress(t["spending_score"], t["loyalty_points"]).rvalue ** 2
    ar = load_ar()
    modes = ar.pivot_table(index="appointment_month", columns="appointment_mode", values="count_of_appointments", aggfunc="sum")
    tel = modes["Telephone"] / modes.sum(axis=1) * 100
    nc = load_nc()
    gp = nc.loc[nc.service_setting == "General Practice", "count_of_appointments"].sum() / nc["count_of_appointments"].sum() * 100
    m = load_marketing()
    top = m[m.RFM_segment.isin(["Champions", "Big Spenders"])]
    ig = m.loc[m.Instagram_ad == 1, "Total_Spending"].mean() / m.loc[m.Instagram_ad == 0, "Total_Spending"].mean()
    bt, ba = load_bank()
    done = bt[bt.status == "completed"]
    bs = bs_call(1400, 1500, 1, 0.065, 0.28)
    bn = binomial_call(1400, 1500, 1, 0.065, 0.28, 1000)
    mc, _ = mc_call(1400, 1500, 1, 0.065, 0.28, 1_000_000, 42)

    st.markdown(
        """
<h1>Six analytics projects, from raw data to a decision</h1>
<p style="font-size:1.08rem;max-width:46rem;margin-top:.6rem">
By Sadia Yusuf, Chartered Accountant moving into financial data analytics. Each project opens as a working paper:
the question it answers, the work performed, then findings you can filter and re-run yourself.</p>
""",
        unsafe_allow_html=True,
    )
    legend()
    st.markdown("---")

    rows = [
        ("turtle", "Turtle Games", "What drives customer loyalty, and which customers should marketing target?",
         f"Spending score alone explains {r2:.0%} of loyalty points. Five customer segments separate cleanly.",
         "Python, regression, decision trees, k-means, NLP"),
        ("churn", "ConnectTel", "Which telecom customers will leave, and what is it worth to keep them?",
         "26.5% of customers churned, about 30% of annual revenue. Contract type is the strongest driver.",
         "Python, statistical tests, ensemble models, cost-benefit"),
        ("nhs", "NHS appointments", "Is primary-care capacity adequate, and how is it used?",
         f"{gp:.0f}% of appointments are in General Practice. Telephone went from {tel.iloc[0]:.0f}% to {tel.max():.0f}% of the mix.",
         "Python, time series, 1.5M rows reduced to monthly views"),
        ("market", "2Market", "Who are the customers, which channels work, which products sell?",
         f"Champions and Big Spenders are {len(top)/len(m):.0%} of customers and {top.Total_Spending.sum()/m.Total_Spending.sum():.0%} of spend. "
         f"Instagram responders spend {ig:.1f}x non-responders.",
         "Excel, SQL, Tableau, RFM segmentation"),
        ("bank", "Banking transactions", "Which transactions need a human to look at them?",
         f"{int(ba.shape[0])} of {len(done):,} completed transactions flagged ({ba.shape[0]/len(done):.1%}) using two outlier tests, a velocity check and a dormant-account check.",
         "Python, z-score and IQR, rolling windows"),
        ("option", "Option pricing", "Can a derivative's fair value be re-derived independently?",
         f"Black-Scholes {bs:,.2f}, binomial {bn:,.2f}, Monte Carlo {mc:,.2f}. The three agree to within {max(abs(bn-bs),abs(mc-bs)):.2f}.",
         "Python, NumPy, SciPy"),
    ]
    for key, name, q, res, tools in rows:
        c1, c2, c3, c4 = st.columns([0.55, 2.7, 3.2, 0.9], vertical_alignment="center")
        c1.markdown(f'<span class="idx-ref">{REFS[key]}</span>', unsafe_allow_html=True)
        c2.markdown(f'<p class="idx-title">{name}</p><p class="idx-q">{q}</p>', unsafe_allow_html=True)
        c3.markdown(f'<p class="idx-res">{res}</p><p class="idx-tools">{tools}</p>', unsafe_allow_html=True)
        c4.button("Open", key=f"open_{key}", on_click=go_to, args=(key,))
        st.markdown('<hr style="margin:.5rem 0;border:none;border-top:1px solid #D3DAE4">', unsafe_allow_html=True)

    st.markdown(
        '<p class="note">Datasets for the banking and option-pricing projects are generated or illustrative; no client data appears anywhere in this app. '
        "The Turtle Games, NHS and 2Market projects use the public or course-provided datasets named on each page.</p>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# WP 1  Turtle Games
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def turtle_tree_curve() -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    t = load_turtle()
    X, y = t[["age", "remuneration", "spending_score"]], t["loyalty_points"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    out = []
    for d in range(1, 13):
        m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(Xtr, ytr)
        out.append((d, m.score(Xtr, ytr), m.score(Xte, yte)))
    return pd.DataFrame(out, columns=["depth", "Train", "Test"])


@st.cache_data(show_spinner=False)
def turtle_k_curve() -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Z = load_turtle()[["remuneration", "spending_score"]]
    rows = []
    for k in range(2, 11):
        km = KMeans(k, n_init=10, random_state=42).fit(Z)
        rows.append((k, km.inertia_, silhouette_score(Z, km.labels_)))
    return pd.DataFrame(rows, columns=["k", "Inertia", "Silhouette"])


@st.cache_data(show_spinner=False)
def turtle_sentiment() -> pd.DataFrame:
    from textblob import TextBlob

    d = load_turtle().drop_duplicates("review").copy()
    d["review_polarity"] = d["review"].map(lambda s: TextBlob(s).sentiment.polarity)
    d["summary_polarity"] = d["summary"].fillna("").map(lambda s: TextBlob(s).sentiment.polarity)
    return d


def level(x, lo, hi):
    return "low" if x < lo else "high" if x > hi else "mid"


def page_turtle() -> None:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    import re

    workpaper(
        REFS["turtle"], "Turtle Games: loyalty and customer segments",
        "LSE project 3 · Advanced Analytics for Organisational Impact",
        "Turtle Games sells games, toys and books worldwide. Marketing wants to know how customers earn loyalty points, "
        "which groups to target, and what reviews say about products.",
        ["Cleaned 2,000 customer records and fitted simple linear regressions",
         "Grew and pruned a decision tree, checking train against test fit",
         "Segmented customers with k-means, choosing k by elbow and silhouette",
         "Tokenised reviews and scored polarity to find common words and the most positive and negative reviews"],
        "Python: pandas, scikit-learn, SciPy, TextBlob",
    )
    legend()
    t = load_turtle()
    tabs = st.tabs(["How points are earned", "Decision tree", "Customer segments", "What reviews say"])

    # --- regression
    with tabs[0]:
        labels = {"spending_score": "Spending score (1-100)", "remuneration": "Annual income (£k)", "age": "Age (years)"}
        r2s = {k: stats.linregress(t[k], t["loyalty_points"]).rvalue ** 2 for k in labels}
        c1, c2 = st.columns([1, 2.2])
        with c1:
            x = st.radio("Compare loyalty points against", list(labels), format_func=labels.get)
            colour = st.selectbox("Colour the points by", ["Nothing", "gender", "education"])
            res = stats.linregress(t[x], t["loyalty_points"])
            kpis([("R² (share explained)", f"{res.rvalue**2:.1%}"),
                  ("Points per unit", f"{res.slope:+.1f}"),
                  ("p-value", "<0.001" if res.pvalue < 0.001 else f"{res.pvalue:.3f}")])
        with c2:
            fig = px.scatter(t, x=x, y="loyalty_points", color=None if colour == "Nothing" else colour,
                             opacity=0.55, labels={x: labels[x], "loyalty_points": "Loyalty points"})
            xs = np.array([t[x].min(), t[x].max()])
            fig.add_trace(go.Scatter(x=xs, y=res.intercept + res.slope * xs, mode="lines", name="Fitted line",
                                     line=dict(color=INK, width=2.5)))
            show(fig, 400)
        order = sorted(r2s, key=r2s.get, reverse=True)
        fig = go.Figure(go.Bar(x=[r2s[k] for k in order], y=[labels[k] for k in order], orientation="h",
                               marker_color=[TEAL if r2s[k] > 0.1 else SLATE for k in order],
                               text=[f"{r2s[k]:.1%}" for k in order], textposition="outside"))
        fig.update_xaxes(tickformat=".0%", range=[0, max(r2s.values()) * 1.25])
        fig.update_yaxes(autorange="reversed")
        show(fig, 190, "Share of loyalty-point variation each factor explains on its own")

    # --- decision tree
    with tabs[1]:
        c1, c2 = st.columns([1, 2])
        with c1:
            depth = st.slider("Maximum depth", 1, 12, 4)
            leaf = st.slider("Minimum customers per leaf", 1, 60, 5)
            X, y = t[["age", "remuneration", "spending_score"]], t["loyalty_points"]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=42).fit(Xtr, ytr)
            kpis([("Train R²", f"{mdl.score(Xtr, ytr):.3f}"), ("Test R²", f"{mdl.score(Xte, yte):.3f}")])
            st.markdown("**Predict a customer**")
            a = st.slider("Age", 18, 72, 40)
            inc = st.slider("Income (£k)", 12, 113, 50)
            sp = st.slider("Spending score", 1, 99, 50)
            pred = mdl.predict(pd.DataFrame([[a, inc, sp]], columns=X.columns))[0]
            st.markdown(f"Predicted loyalty points: **{pred:,.0f}**")
        with c2:
            curve = turtle_tree_curve().melt("depth", var_name="Data", value_name="R²")
            fig = px.line(curve, x="depth", y="R²", color="Data", markers=True, color_discrete_map={"Train": BLUE, "Test": AMBER})
            fig.add_vline(x=depth, line_dash="dot", line_color=INK)
            fig.update_xaxes(dtick=1, title="Tree depth")
            show(fig, 300, "Train and test fit as the tree grows")
            imp = pd.Series(mdl.feature_importances_, index=["Age", "Income", "Spending score"]).sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE,
                                   text=[f"{v:.0%}" for v in imp.values], textposition="outside"))
            fig.update_xaxes(tickformat=".0%", range=[0, 1.1])
            show(fig, 190, "What the tree actually splits on")

    # --- clustering
    with tabs[2]:
        kc = turtle_k_curve()
        k = st.slider("Number of segments (k)", 2, 10, 5)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(kc, x="k", y="Inertia", markers=True)
            fig.add_vline(x=k, line_dash="dot", line_color=INK)
            fig.update_xaxes(dtick=1)
            show(fig, 260, "Elbow: within-cluster spread")
        with c2:
            fig = px.line(kc, x="k", y="Silhouette", markers=True, color_discrete_sequence=[TEAL])
            fig.add_vline(x=k, line_dash="dot", line_color=INK)
            fig.update_xaxes(dtick=1)
            show(fig, 260, "Silhouette: separation (higher is better)")
        best = int(kc.loc[kc.Silhouette.idxmax(), "k"])
        st.markdown(f"Silhouette peaks at **k = {best}** ({kc.Silhouette.max():.2f}); you have chosen k = {k}.")
        km = KMeans(k, n_init=10, random_state=42).fit(t[["remuneration", "spending_score"]])
        seg = t.assign(cluster=km.labels_)
        prof = seg.groupby("cluster").agg(Customers=("age", "size"), Income=("remuneration", "mean"),
                                           Spending=("spending_score", "mean"), Points=("loyalty_points", "mean")).reset_index()
        prof["Segment"] = [f"{level(r.Income, 35, 60).capitalize()} income, {level(r.Spending, 35, 65)} spending" for r in prof.itertuples()]
        prof["Segment"] = prof["Segment"] + " (" + (prof["cluster"] + 1).astype(str) + ")"
        seg = seg.merge(prof[["cluster", "Segment"]], on="cluster")
        fig = px.scatter(seg, x="remuneration", y="spending_score", color="Segment", opacity=0.75,
                         labels={"remuneration": "Annual income (£k)", "spending_score": "Spending score"})
        show(fig, 430, "Customers by income and spending score")
        out = prof.sort_values("Points", ascending=False)[["Segment", "Customers", "Income", "Spending", "Points"]]
        out = out.rename(columns={"Income": "Avg income (£k)", "Spending": "Avg spending score", "Points": "Avg loyalty points"})
        table(out.round(1))

    # --- NLP
    with tabs[3]:
        d = turtle_sentiment()
        col = st.radio("Analyse", ["review", "summary"], horizontal=True, format_func=lambda s: "Full reviews" if s == "review" else "Review summaries")
        n = st.slider("Words to show", 5, 30, 15)
        words = pd.Series([w for s in d[col].fillna("").str.lower() for w in re.findall(r"[a-z]{3,}", s) if w not in ENGLISH_STOP_WORDS])
        top = words.value_counts().head(n).sort_values()
        pol = d[f"{col}_polarity"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=BLUE))
            show(fig, 80 + 24 * n, f"{n} most common words")
        with c2:
            fig = px.histogram(pol, nbins=15, color_discrete_sequence=[TEAL])
            fig.update_layout(showlegend=False, bargap=0.05)
            fig.update_xaxes(title="Polarity (-1 negative, +1 positive)")
            fig.update_yaxes(title="Reviews")
            show(fig, 80 + 24 * n if n > 12 else 380, "How positive the text is")
        st.markdown(f"**{(pol > 0.05).mean():.0%}** of {('reviews' if col == 'review' else 'summaries')} read as positive and **{(pol < -0.05).mean():.0%}** as negative "
                    f"(after removing {len(d.index)-len(load_turtle()):,} duplicates, {len(d):,} remain).")
        c1, c2 = st.columns(2)
        base = d[["summary", "review", f"{col}_polarity"]].rename(columns={f"{col}_polarity": "polarity"})
        with c1:
            st.markdown("**Most positive**")
            table(base.nlargest(8, "polarity")[["summary", "polarity"]].round(2))
        with c2:
            st.markdown("**Most negative**")
            table(base.nsmallest(8, "polarity")[["summary", "polarity"]].round(2))
        q = st.text_input("Search the reviews for a word (for example: puzzle, instructions, kids)")
        if q:
            hit = d[d["review"].str.contains(q, case=False, na=False)]
            st.markdown(f"{len(hit):,} reviews mention **{q}**; average polarity {hit['review_polarity'].mean():.2f}.")
            table(hit[["summary", "review_polarity"]].head(30).round(2))

    spend_r2, inc_r2, age_r2 = r2s["spending_score"], r2s["remuneration"], r2s["age"]
    kc = turtle_k_curve()
    conclusion("")
    finds([
        ("v", f"Spending score and income each explain a large share of loyalty points on their own ({spend_r2:.0%} and {inc_r2:.0%}). Age explains almost nothing ({age_r2:.1%})."),
        ("v", "A depth-4 tree reaches a test R² of about 0.94 using only income and spending score. Deeper trees add little and begin to memorise the training data."),
        ("v", f"Segmenting on income and spending score, silhouette is highest at k = {int(kc.loc[kc.Silhouette.idxmax(), 'k'])}. The most valuable group combines high income with high spending."),
        ("v", "Reviews are mostly positive, and the most common words (game, great, fun, play) show customers talk about the play experience more than price."),
    ])
    st.markdown("**Recommendation.** Target by income and spending behaviour, not age. Treat the high-income, low-spending group as the main growth opportunity: they have money and are not spending it with Turtle Games.")


# ----------------------------------------------------------------------------
# WP 2  ConnectTel churn (figures from the capstone notebook)
# ----------------------------------------------------------------------------
CHURN_BASE = 26.5
CHURN_FACTORS = {
    "Contract": {"Month-to-month": 42.7, "One year": 11.3, "Two year": 2.8},
    "Internet service": {"Fibre optic": 41.9, "DSL": 19.0, "No internet": 7.4},
    "Online security": {"No": 41.8, "Yes": 14.6, "No internet service": 7.4},
    "Payment method": {"Electronic check": 45.3, "Mailed check": 19.1, "Bank transfer (auto)": 16.7, "Credit card (auto)": 15.2},
    "Paperless billing": {"Yes": 33.6, "No": 16.3},
    "Fibre customers, stacked risk": {"All fibre": 41.9, "Fibre, no online security": 49.4, "Fibre, no security or tech support": 55.0},
}
CRAMER = [("Contract", .4101), ("OnlineSecurity", .3474), ("TechSupport", .3429), ("InternetService", .3225), ("PaymentMethod", .3034),
          ("OnlineBackup", .2923), ("DeviceProtection", .2816), ("StreamingMovies", .2310), ("StreamingTV", .2305),
          ("PaperlessBilling", .1915), ("Dependents", .1639), ("Partner", .1501), ("MultipleLines", .0401),
          ("PhoneService", .0114), ("Gender", .0083)]
STRATEGIES = {  # name: (customers contacted, churners caught) on the 1,409-customer test cohort with 374 churners
    "Random forest, threshold 0.50": (293, 192),
    "Random forest, threshold 0.35": (472, 266),
    "Ensemble, threshold 0.50": (392, 233),
    "Ensemble, threshold 0.34": (568, 295),
}
TEST_CHURNERS = 374


def page_churn() -> None:
    workpaper(
        REFS["churn"], "ConnectTel: who will leave, and what is it worth to keep them",
        "LSE project 4 · Capstone",
        "A UK telecom provider loses a quarter of its customers. Predict who is likely to leave and "
        "show what a retention programme is worth in pounds, not accuracy points.",
        ["Cleaned 7,043 customer records and tested every feature against churn (chi-square, Cramér's V)",
         "Compared logistic regression, decision tree, random forest and neural network",
         "Rebuilt the winner as a leakage-safe ensemble and scored customers into risk bands",
         "Chose the contact threshold by minimising expected cost, then stress-tested the assumptions"],
        "Python: scikit-learn pipelines, SciPy, pandas",
    )
    legend()
    st.caption("The customer file is not bundled with this app, so the figures below are carried over from the capstone notebook. The economics calculator re-computes live.")
    kpis([("Customers", "7,043"), ("Churned", "1,869 (26.5%)"), ("Annual revenue lost", "£1.67M"), ("Share of revenue", "30.5%")])
    tabs = st.tabs(["Why customers leave", "Who is worth keeping", "Model and lift", "Retention economics"])

    with tabs[0]:
        f = st.selectbox("Churn rate by", list(CHURN_FACTORS))
        d = pd.DataFrame({"Group": list(CHURN_FACTORS[f]), "Churn rate": list(CHURN_FACTORS[f].values())})
        fig = go.Figure(go.Bar(x=d["Group"], y=d["Churn rate"], marker_color=[AMBER if v > CHURN_BASE else TEAL for v in d["Churn rate"]],
                               text=[f"{v:.1f}%" for v in d["Churn rate"]], textposition="outside"))
        fig.add_hline(y=CHURN_BASE, line_dash="dash", line_color=INK, annotation_text="Overall 26.5%", annotation_position="top right")
        fig.update_yaxes(range=[0, max(60, d["Churn rate"].max() * 1.2)], ticksuffix="%")
        show(fig, 340, f"Churn rate by {f.lower()}  (amber: above average, green: below)")
        cv = pd.DataFrame(CRAMER, columns=["Feature", "V"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=cv["V"], y=cv["Feature"], orientation="h",
                               marker_color=[TEAL if v >= .3 else BLUE if v >= .15 else SLATE for v in cv["V"]]))
        fig.update_xaxes(title="Cramér's V (strength of association with churn)")
        show(fig, 460, "Which features matter most")
        finds([("n", "Contract type is the strongest driver (V = 0.41). Gender and phone service show no significant association with churn."),
               ("n", "68.7% of fibre customers are month-to-month, so part of fibre's high churn is really contract mix. The notebook flags this as a confound and tests it in the multivariate model.")])

    with tabs[1]:
        tiers = pd.DataFrame({"Tier": ["Bronze", "Silver", "Gold", "Platinum"], "Customers": [1761, 1773, 1751, 1758],
                              "Avg CLV (£)": [159.6, 323.0, 446.4, 1313.9], "Share of CLV": [7.1, 14.5, 19.8, 58.5], "Churn rate": [30, 40, 30, 10]})
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=tiers["Tier"], y=tiers["Share of CLV"], marker_color=BLUE, text=[f"{v}%" for v in tiers["Share of CLV"]], textposition="outside"))
            fig.update_yaxes(ticksuffix="%", range=[0, 70])
            show(fig, 320, "Share of lifetime value")
        with c2:
            fig = go.Figure(go.Bar(x=tiers["Tier"], y=tiers["Churn rate"], marker_color=[AMBER if v > CHURN_BASE else TEAL for v in tiers["Churn rate"]],
                                   text=[f"{v}%" for v in tiers["Churn rate"]], textposition="outside"))
            fig.update_yaxes(ticksuffix="%", range=[0, 55])
            show(fig, 320, "Churn rate by value tier")
        seg = pd.DataFrame({
            "Segment": ["New big spenders (at risk)", "Loyal high-value", "Low-value / dormant", "Developing", "Champions"],
            "Customers": [279, 1688, 1288, 1827, 1961], "Avg tenure (months)": [4.7, 31.6, 8.6, 28.2, 56.5],
            "Avg monthly spend (£)": [84.2, 74.6, 31.0, 44.8, 94.4], "Churn rate (%)": [74, 31, 28, 24, 18],
            "Monthly revenue (£)": [23484, 125852, 39911, 81822, 185047]}).sort_values("Churn rate (%)", ascending=False)
        st.markdown("**Customer segments by tenure, engagement and spend**")
        table(seg)
        finds([("n", "Platinum customers hold 58.5% of lifetime value and churn at 10%. Silver is the worst tier at 40%, worse than Bronze."),
               ("n", "The most urgent group is 279 new, high-spending customers who churn at 74%. They are few, but they are paying £84 a month.")])

    with tabs[2]:
        comp = pd.DataFrame({"Model": ["Logistic regression", "Decision tree", "Random forest", "Neural network"],
                             "Accuracy": [.798, .801, .799, .794], "Recall (churn)": [.551, .543, .513, .545], "ROC-AUC": [.839, .835, .840, .840]})
        c1, c2 = st.columns([1.1, 1])
        with c1:
            long = comp.melt("Model", var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", color_discrete_sequence=[SLATE, AMBER, BLUE])
            fig.update_yaxes(range=[0.4, 0.9])
            show(fig, 340, "Four models, one ceiling")
        with c2:
            bands = pd.DataFrame({"Risk band": ["Low", "Medium", "High"], "Customers": [790, 339, 280], "Predicted": [10.4, 44.9, 74.2], "Actual": [8.2, 36.0, 66.8]})
            fig = px.bar(bands.melt("Risk band", value_vars=["Predicted", "Actual"], var_name="Rate", value_name="Churn %"),
                         x="Risk band", y="Churn %", color="Rate", barmode="group", color_discrete_sequence=[SLATE, TEAL])
            show(fig, 340, "Risk bands separate real churners")
        dec = pd.DataFrame({"Decile": range(1, 11), "Churn rate": [75.9, 56.7, 39.0, 35.5, 21.3, 20.7, 8.5, 5.7, 1.4, 0.7],
                            "Captured": [28.6, 50.0, 64.7, 78.1, 86.1, 93.9, 97.1, 99.2, 99.7, 100.0]})
        n = st.select_slider("Contact the top n deciles of customers by risk", options=list(range(1, 11)), value=3)
        row = dec[dec.Decile == n].iloc[0]
        kpis([("Customers contacted", f"{n*10}%"), ("Churners reached", f"{row.Captured:.1f}%"), ("Churn rate in last decile contacted", f"{row['Churn rate']:.1f}%")])
        fig = go.Figure()
        fig.add_bar(x=dec["Decile"], y=dec["Churn rate"], name="Churn rate in decile",
                    marker_color=[BLUE if d <= n else "#C9D3E0" for d in dec["Decile"]])
        fig.add_scatter(x=dec["Decile"], y=dec["Captured"], name="Cumulative share of churners caught", mode="lines+markers", line=dict(color=AMBER, width=3))
        fig.update_xaxes(dtick=1, title="Risk decile (1 = highest risk)")
        fig.update_yaxes(ticksuffix="%")
        show(fig, 360, "Lift: the riskiest 30% of customers hold 65% of churners")
        finds([("n", "All four models land near 0.84 ROC-AUC. The notebook reads this as a data ceiling: there is no usage, complaint or outage data to learn from."),
               ("n", "The ensemble does not beat logistic regression on ROC-AUC (0.841 vs 0.842) but does on precision-recall (0.648 vs 0.634) and is more stable. It was chosen on expected cost, not accuracy.")])

    with tabs[3]:
        st.markdown("Change the assumptions and see which operating point saves the most. Defaults match the notebook.")
        c1, c2, c3 = st.columns(3)
        offer = c1.slider("Cost of a retention offer (£)", 10, 150, 50, step=5)
        loss = c2.slider("Value lost per churner (£)", 100, 1000, 500, step=25)
        succ = c3.slider("Offers that keep the customer (%)", 5, 80, 35, step=5) / 100
        base_cost = TEST_CHURNERS * loss
        rows = []
        for name, (contacted, caught) in STRATEGIES.items():
            cost = (TEST_CHURNERS - caught) * loss + contacted * offer + caught * loss * (1 - succ)
            rows.append((name, contacted, caught, caught / TEST_CHURNERS, cost, base_cost - cost))
        res = pd.DataFrame(rows, columns=["Strategy", "Contacted", "Churners caught", "Recall", "Expected cost (£)", "Saving vs doing nothing (£)"])
        best = res.loc[res["Saving vs doing nothing (£)"].idxmax()]
        fig = go.Figure(go.Bar(x=res["Saving vs doing nothing (£)"], y=res["Strategy"], orientation="h",
                               marker_color=[TEAL if s == best["Strategy"] else SLATE for s in res["Strategy"]],
                               text=[f"£{v:,.0f}" for v in res["Saving vs doing nothing (£)"]], textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[min(0, res["Saving vs doing nothing (£)"].min() * 1.3), max(1, res["Saving vs doing nothing (£)"].max() * 1.3)])
        show(fig, 280, "Saving on the 1,409-customer test cohort")
        fmt = res.copy()
        fmt["Recall"] = (fmt["Recall"] * 100).round(1).astype(str) + "%"
        fmt["Expected cost (£)"] = fmt["Expected cost (£)"].map("{:,.0f}".format)
        fmt["Saving vs doing nothing (£)"] = fmt["Saving vs doing nothing (£)"].map("{:,.0f}".format)
        table(fmt)
        worth = loss * succ > offer
        st.markdown(
            f'<div class="callout">Doing nothing costs <b>£{base_cost:,.0f}</b>. An offer pays for itself when the value lost × success rate '
            f'(£{loss*succ:,.0f}) exceeds its cost (£{offer}): <b>{"yes, so the programme is worth running" if worth else "no, so targeting should be narrower or the offer cheaper"}</b>. '
            f'Best operating point: <b>{best["Strategy"]}</b>.</div>',
            unsafe_allow_html=True,
        )
        st.caption("Cost = missed churners × value lost + offers sent × offer cost + churners reached × value lost × (1 − success rate). With the defaults this reproduces the notebook's £163,775 for the ensemble at 0.34.")

    conclusion("Deploy the soft-voting ensemble and contact customers above a risk score of 0.34, chosen by expected cost rather than accuracy. "
               "The decision to run a retention programme holds under every plausible valuation of a churner. How deep to target is sensitive to that valuation, "
               "which is why the calculator above exists. Priority groups: month-to-month fibre customers without security or support add-ons, and new high-spending customers. "
               "Limits: the data is a single snapshot with no usage or complaint history, and the cost assumptions are illustrative until Finance validates them.")


# ----------------------------------------------------------------------------
# WP 3  NHS
# ----------------------------------------------------------------------------
def page_nhs() -> None:
    workpaper(
        REFS["nhs"], "NHS: is primary-care capacity adequate?",
        "LSE project 2 · Data Analytics using Python",
        "NHS England must decide between adding capacity and using what it has better. "
        "Work out how appointments changed through COVID-19, where they happen, and whether staffing should grow.",
        ["Cleaned three NHS datasets (about 1.5M rows) and removed 21,604 duplicate rows",
         "Built monthly views of volume, mode, provider, attendance and booking lead time",
         "Set a statistical capacity benchmark (mean + 1 standard deviation) and counted breaches",
         "Counted hashtags on UK healthcare tweets as a signal of public concern"],
        "Python: pandas, Matplotlib and Seaborn in the original; data reduced to monthly totals for this app",
    )
    legend()
    ar, nc, ad, tags = load_ar(), load_nc(), load_ad(), load_tags()
    tabs = st.tabs(["Volume and COVID", "Capacity check", "Who and how", "Settings and categories", "Public conversation"])

    tot = ar.groupby("appointment_month")["count_of_appointments"].sum().reset_index()
    tot["date"] = pd.to_datetime(tot["appointment_month"])
    modes = ar.pivot_table(index="appointment_month", columns="appointment_mode", values="count_of_appointments", aggfunc="sum")
    share = modes.div(modes.sum(axis=1), axis=0) * 100

    with tabs[0]:
        view = st.radio("Show", ["Total appointments", "By mode (count)", "By mode (share of month)"], horizontal=True)
        if view == "Total appointments":
            fig = px.area(tot, x="date", y="count_of_appointments", color_discrete_sequence=[BLUE], labels={"count_of_appointments": "Appointments", "date": ""})
            fig.update_yaxes(rangemode="tozero")
        else:
            src = (share if "share" in view else modes).reset_index().melt("appointment_month", var_name="Mode", value_name="v")
            src["date"] = pd.to_datetime(src["appointment_month"])
            fig = px.line(src, x="date", y="v", color="Mode", labels={"v": "% of month" if "share" in view else "Appointments", "date": ""})
        fig.add_vline(x=pd.Timestamp("2020-03-01"), line_dash="dot", line_color=INK)
        fig.add_annotation(x=pd.Timestamp("2020-03-01"), y=1, yref="paper", text="First lockdown", showarrow=False, xanchor="left", yanchor="top")
        for a, b in [("2020-10-01", "2021-02-28"), ("2021-12-01", "2022-02-28")]:
            fig.add_vrect(x0=a, x1=b, fillcolor=AMBER, opacity=0.12, line_width=0)
        show(fig, 400, "Appointments per month, January 2020 to June 2022 (amber: restriction periods noted in the report)")
        tel0, telmax = share["Telephone"].iloc[0], share["Telephone"].max()
        f2f0, f2f_apr = share["Face-to-Face"].iloc[0], share.loc["2020-04", "Face-to-Face"]
        finds([("v", f"Telephone consultations were {tel0:.0f}% of appointments in January 2020 and reached {telmax:.0f}% at their peak. Face-to-face fell from {f2f0:.0f}% to {f2f_apr:.0f}% by April 2020."),
               ("v", f"By June 2022 face-to-face had recovered to {share['Face-to-Face'].iloc[-1]:.0f}%, still below the {f2f0:.0f}% before the pandemic."),
               ("v", "Volume dips line up with the first lockdown and the winter restriction periods.")])

    with tabs[1]:
        post = tot[tot["appointment_month"] >= "2021-08"].copy()
        c1, c2 = st.columns(2)
        k = c1.slider("Benchmark: mean plus this many standard deviations", 0.0, 2.0, 1.0, step=0.25)
        up = c2.slider("What if the benchmark were this much higher (%)", 0, 20, 0, step=1)
        thr = (post["count_of_appointments"].mean() + k * post["count_of_appointments"].std()) * (1 + up / 100)
        post["over"] = post["count_of_appointments"] > thr
        fig = go.Figure(go.Bar(x=post["date"], y=post["count_of_appointments"], marker_color=[AMBER if o else BLUE for o in post["over"]]))
        fig.add_hline(y=thr, line_dash="dash", line_color=INK, annotation_text=f"Benchmark {thr/1e6:.1f}M", annotation_position="top left")
        fig.update_yaxes(range=[0, post["count_of_appointments"].max() * 1.15])
        show(fig, 380, "Monthly appointments since August 2021 against the capacity benchmark")
        n_over = int(post["over"].sum())
        st.markdown(f'<div class="callout"><b>{n_over} of {len(post)} months</b> exceed the benchmark. At the default (mean + 1 SD, no uplift) that is 2 of 11, in October and November 2021.</div>', unsafe_allow_html=True)
        st.caption("The benchmark is statistical, not a measured staffing limit. It shows how close peak months run to the system's own typical ceiling.")
        finds([("v", "The system can reach about 30 million appointments a month (October and November 2021), then falls back sharply in winter. That pattern points to reactive rather than planned capacity."),
               ("v", "Raising the benchmark by 5 to 10% removes the peak-month breaches, which is the basis for the report's modest-uplift staffing recommendation.")])

    with tabs[2]:
        view = st.selectbox("View", ["Attendance (did not attend rate)", "Provider type", "Booking lead time", "Appointment length"])
        if view.startswith("Attendance"):
            s = ar.pivot_table(index="appointment_month", columns="appointment_status", values="count_of_appointments", aggfunc="sum")
            d = (s["DNA"] / s.sum(axis=1) * 100).reset_index(name="DNA %")
            d["date"] = pd.to_datetime(d["appointment_month"])
            fig = px.line(d, x="date", y="DNA %", markers=True, color_discrete_sequence=[AMBER])
            fig.update_yaxes(range=[0, max(8, d["DNA %"].max() * 1.2)], ticksuffix="%")
            show(fig, 340, "Share of booked appointments missed")
            st.markdown(f"Did-not-attend rate has stayed between **{d['DNA %'].min():.1f}%** and **{d['DNA %'].max():.1f}%** throughout, with no pandemic-sized shift.")
        elif view == "Provider type":
            h = ar.pivot_table(index="appointment_month", columns="hcp_type", values="count_of_appointments", aggfunc="sum")
            h = (h.div(h.sum(axis=1), axis=0) * 100).reset_index().melt("appointment_month", var_name="Provider", value_name="%")
            h["date"] = pd.to_datetime(h["appointment_month"])
            fig = px.area(h, x="date", y="%", color="Provider", color_discrete_sequence=[BLUE, TEAL, SLATE])
            fig.update_yaxes(ticksuffix="%")
            show(fig, 360, "Who delivers appointments (share of month)")
        elif view == "Booking lead time":
            since = st.checkbox("Since August 2021 only", value=True)
            a = ar[ar["appointment_month"] >= "2021-08"] if since else ar
            tb = a.pivot_table(index="appointment_month", columns="time_between_book_and_appointment", values="count_of_appointments", aggfunc="sum")
            order = ["Same Day", "1 Day", "2 to 7 Days", "8  to 14 Days", "15  to 21 Days", "22  to 28 Days", "More than 28 Days"]
            tb = tb[order]
            tb = tb.div(tb.sum(axis=1), axis=0) * 100
            long = tb.sum(axis=1) * 0 + tb[["15  to 21 Days", "22  to 28 Days", "More than 28 Days"]].sum(axis=1)
            long.index = pd.to_datetime(long.index)
            fig = go.Figure(go.Scatter(x=long.index, y=long.values, mode="lines+markers", line=dict(color=AMBER, width=3)))
            fig.update_yaxes(ticksuffix="%", rangemode="tozero")
            show(fig, 300, "Share of appointments booked more than 14 days ahead")
            m = tb.reset_index().melt("appointment_month", var_name="Lead time", value_name="%")
            m["date"] = pd.to_datetime(m["appointment_month"])
            fig = px.area(m, x="date", y="%", color="Lead time", category_orders={"Lead time": order})
            fig.update_yaxes(ticksuffix="%")
            show(fig, 340, "Full lead-time mix")
            st.markdown(f"Long waits were **{long.iloc[0]:.1f}%** in {long.index[0]:%b %Y} and **{long.iloc[-1]:.1f}%** in {long.index[-1]:%b %Y}. They dip in winter, then rise again.")
        else:
            d = ad.groupby("actual_duration")["count_of_appointments"].sum()
            order = ["1-5 Minutes", "6-10 Minutes", "11-15 Minutes", "16-20 Minutes", "21-30 Minutes", "31-60 Minutes", "Unknown / Data Quality"]
            d = (d.reindex(order) / d.sum() * 100)
            fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=[AMBER if i.startswith("Unknown") else BLUE for i in d.index],
                                   text=[f"{v:.0f}%" for v in d.values], textposition="outside"))
            fig.update_yaxes(ticksuffix="%", range=[0, 30])
            show(fig, 340, "How long appointments last (December 2021 to June 2022)")
            st.markdown("A quarter of appointments have no recorded duration. That data-quality gap limits capacity planning more than any modelling choice.")

    with tabs[3]:
        setting = nc.groupby("service_setting")["count_of_appointments"].sum().sort_values()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=setting.values / setting.sum() * 100, y=setting.index, orientation="h", marker_color=BLUE,
                                   text=[f"{v:.1f}%" for v in setting.values / setting.sum() * 100], textposition="outside"))
            fig.update_xaxes(ticksuffix="%", range=[0, 105])
            show(fig, 300, "Service setting (August 2021 to June 2022)")
        with c2:
            ctx = nc.groupby("context_type")["count_of_appointments"].sum()
            fig = go.Figure(go.Pie(labels=ctx.index, values=ctx.values, hole=0.55, marker_colors=[BLUE, AMBER, SLATE], textinfo="percent"))
            show(fig, 300, "Context type")
        nmax = st.slider("National categories to show", 5, 18, 10)
        cat = nc.groupby("national_category")["count_of_appointments"].sum().sort_values().tail(nmax)
        fig = go.Figure(go.Bar(x=cat.values / 1e6, y=cat.index, orientation="h", marker_color=TEAL))
        fig.update_xaxes(title="Appointments (millions)")
        show(fig, 90 + 26 * nmax, "National categories")
        ex = st.checkbox("Exclude General Practice to see the other settings")
        m = nc[nc.service_setting != "General Practice"] if ex else nc
        mm = m.groupby(["month", "service_setting"])["count_of_appointments"].sum().reset_index()
        fig = px.bar(mm, x="month", y="count_of_appointments", color="service_setting", labels={"count_of_appointments": "Appointments", "month": ""})
        show(fig, 340, "Monthly appointments by service setting")
        finds([("v", f"General Practice accounts for {setting.iloc[-1]/setting.sum():.1%} of appointments, so any staffing decision is mostly a General Practice decision."),
               ("v", "Outside General Practice, monthly volumes are small and uneven, which suggests those services are run reactively.")])

    with tabs[4]:
        n = st.slider("Hashtags to show", 5, 30, 15)
        d = tags.head(n).iloc[::-1]
        fig = go.Figure(go.Bar(x=d["count"], y=d["hashtag"], orientation="h", marker_color=PLUM))
        show(fig, 90 + 24 * n, "Most frequent hashtags on UK healthcare tweets")
        st.caption("Counting all tweets rather than only retweeted or favourited ones avoids over-weighting popular posts, at the cost of more noise.")

    conclusion("Yes, General Practice needs more capacity, but modestly: a 5 to 10% uplift would absorb winter peaks that currently push demand against the system's ceiling. "
               "Three supporting points: telephone and hybrid care became permanent features of the mix, the share of long waits is creeping up, and one quarter of appointments have no recorded duration. "
               "Fixing that data gap is the cheapest first step because every capacity estimate depends on it. "
               "Note: the benchmark here is statistical, so treat it as a screening tool, not a staffing model.")


# ----------------------------------------------------------------------------
# WP 4  2Market
# ----------------------------------------------------------------------------
COUNTRY = {"SP": "Spain", "SA": "Saudi Arabia", "CA": "Canada", "AUS": "Australia", "IND": "India", "GER": "Germany", "US": "United States", "ME": "ME"}
PRODUCTS = {"AmtNonVeg": "Meat items", "AmtLiq": "Alcoholic beverages", "AmtComm": "Commodities", "AmtPes": "Fish products", "AmtChocolates": "Chocolates", "AmtVege": "Vegetables"}
CHANNELS = {"Bulkmail_ad": "Bulk mail", "Twitter_ad": "Twitter", "Instagram_ad": "Instagram", "Facebook_ad": "Facebook", "Brochure_ad": "Brochure"}


def page_market() -> None:
    workpaper(
        REFS["market"], "2Market: who buys, what they buy, and which channels work",
        "LSE project 1 · Business analytics",
        "A multi-country retailer wants to know who its customers are, which advertising channels influence spend, and which products carry the business.",
        ["Cleaned 2,212 customers in Excel: bad birth years, an income outlier, 47 duplicates, wrong data types",
         "Engineered age bands, income tiers and spend metrics in PostgreSQL",
         "Scored every customer on recency, frequency and monetary value (quintiles) to form RFM segments",
         "Built a Tableau story with three questions per page, two contrasting colours and a colour-blind-safe palette"],
        "Excel, PostgreSQL, Tableau; this page re-runs the analysis in Python",
    )
    legend()
    m = load_marketing()
    with st.expander("Filter customers", expanded=False):
        c1, c2, c3 = st.columns(3)
        ctry = c1.multiselect("Country", sorted(m["Country"].unique()), default=sorted(m["Country"].unique()), format_func=lambda c: COUNTRY.get(c, c))
        edu = c2.multiselect("Education", sorted(m["Education"].unique()), default=sorted(m["Education"].unique()))
        mar = c3.multiselect("Marital status", sorted(m["Marital_Status"].unique()), default=sorted(m["Marital_Status"].unique()))
        st.caption("Marital status 'Absurd' and 'YOLO' (two records each) are left out, as in the report.")
    f = m[m["Country"].isin(ctry) & m["Education"].isin(edu) & m["Marital_Status"].isin(mar)]
    if f.empty:
        st.warning("No customers match these filters. Add a country, education level or marital status back.")
        return
    kpis([("Customers", f"{len(f):,}"), ("Total spend", f"${f.Total_Spending.sum():,.0f}"),
          ("Average spend", f"${f.Total_Spending.mean():,.0f}"), ("Average income", f"${f.Income.mean():,.0f}")])
    tabs = st.tabs(["Customers", "Products", "RFM segments", "Ad channels"])

    with tabs[0]:
        dim = st.selectbox("Average spend by", ["Age band", "Income tier", "Education", "Marital_Status", "Country"], format_func=lambda s: s.replace("_", " "))
        g = f.groupby(dim, observed=True).agg(Customers=("ID", "size"), Spend=("Total_Spending", "mean")).reset_index()
        if dim == "Country":
            g[dim] = g[dim].map(lambda c: COUNTRY.get(c, c))
        fig = go.Figure(go.Bar(x=g[dim].astype(str), y=g["Spend"], marker_color=BLUE, text=[f"${v:,.0f}" for v in g["Spend"]], textposition="outside",
                               customdata=g["Customers"], hovertemplate="%{x}<br>Average spend $%{y:,.0f}<br>%{customdata} customers<extra></extra>"))
        fig.update_yaxes(tickprefix="$", range=[0, g["Spend"].max() * 1.18])
        show(fig, 340, f"Average spend by {dim.replace('_', ' ').lower()}")
        colour = st.radio("Colour the scatter by", ["RFM_segment", "Country", "Education"], horizontal=True, format_func=lambda s: s.replace("_", " "))
        fig = px.scatter(f, x="Income", y="Total_Spending", color=colour, opacity=0.6, labels={"Total_Spending": "Total spend ($)", "Income": "Income ($)"})
        show(fig, 420, "Income against spend: spend rises with income, then flattens")

    with tabs[1]:
        c1, c2 = st.columns([1, 1])
        by = c1.selectbox("Split by", ["Country", "Education", "Marital_Status", "Age band", "Income tier"], format_func=lambda s: s.replace("_", " "))
        as_share = c2.radio("Show", ["Share of category spend", "Dollars"], horizontal=True)
        g = f.groupby(by, observed=True)[list(PRODUCTS)].sum().rename(columns=PRODUCTS)
        if by == "Country":
            g.index = [COUNTRY.get(c, c) for c in g.index]
        if as_share.startswith("Share"):
            g = g.div(g.sum(axis=1), axis=0) * 100
        long = g.reset_index(names=by).melt(by, var_name="Product", value_name="v")
        fig = px.bar(long, x=by, y="v", color="Product", barmode="stack", labels={"v": "% of spend" if as_share.startswith("Share") else "$"},
                     category_orders={"Product": list(PRODUCTS.values())})
        show(fig, 420, "Product mix")
        tot = f[list(PRODUCTS)].sum().rename(PRODUCTS).sort_values(ascending=False)
        st.markdown(f"**{tot.index[0]}** and **{tot.index[1].lower()}** are the two biggest categories here, together **{tot.iloc[:2].sum()/tot.sum():.0%}** of spend.")

    with tabs[2]:
        seg = f.groupby("RFM_segment").agg(Customers=("ID", "size"), Spend=("Total_Spending", "sum"), Avg=("Total_Spending", "mean")).reset_index()
        seg["Customers %"] = seg["Customers"] / seg["Customers"].sum() * 100
        seg["Spend %"] = seg["Spend"] / seg["Spend"].sum() * 100
        seg = seg.sort_values("Spend %", ascending=False)
        long = seg.melt("RFM_segment", value_vars=["Customers %", "Spend %"], var_name="Measure", value_name="%")
        fig = px.bar(long, x="RFM_segment", y="%", color="Measure", barmode="group", color_discrete_sequence=[SLATE, TEAL], labels={"RFM_segment": ""})
        fig.update_yaxes(ticksuffix="%")
        show(fig, 360, "Share of customers against share of spend")
        out = seg.rename(columns={"RFM_segment": "Segment", "Spend": "Total spend ($)", "Avg": "Average spend ($)"}).round(1)
        table(out[["Segment", "Customers", "Customers %", "Total spend ($)", "Spend %", "Average spend ($)"]])
        top = seg[seg.RFM_segment.isin(["Champions", "Big Spenders"])]
        finds([("v", f"Champions and Big Spenders are {top['Customers %'].sum():.0f}% of customers in this view and {top['Spend %'].sum():.0f}% of spend."),
               ("v", "Hibernating customers average a tiny fraction of that spend, which makes them a reactivation target rather than a loss-making group.")])

    with tabs[3]:
        rows = []
        for c, name in CHANNELS.items():
            r, n = f.loc[f[c] == 1, "Total_Spending"], f.loc[f[c] == 0, "Total_Spending"]
            rows.append((name, int((f[c] == 1).sum()), r.mean() if len(r) else np.nan, n.mean() if len(n) else np.nan))
        ch = pd.DataFrame(rows, columns=["Channel", "Customers who responded", "Responders", "Non-responders"])
        ch["Ratio"] = ch["Responders"] / ch["Non-responders"]
        long = ch.melt("Channel", value_vars=["Responders", "Non-responders"], var_name="Group", value_name="Average spend ($)")
        fig = px.bar(long, x="Channel", y="Average spend ($)", color="Group", barmode="group", color_discrete_sequence=[TEAL, SLATE])
        fig.update_yaxes(tickprefix="$")
        show(fig, 360, "Average spend: customers who responded to a channel against those who did not")
        t2 = ch.copy()
        t2["Responders"] = t2["Responders"].round(0)
        t2["Non-responders"] = t2["Non-responders"].round(0)
        t2["Ratio"] = t2["Ratio"].round(1).astype(str) + "x"
        table(t2)
        n = f.groupby("Total_Response")["Total_Spending"].agg(["mean", "size"]).reset_index()
        fig = go.Figure(go.Bar(x=n["Total_Response"].astype(str), y=n["mean"], marker_color=BLUE, text=[f"n={s}" for s in n["size"]], textposition="outside"))
        fig.update_xaxes(title="Number of channels the customer responded to")
        fig.update_yaxes(tickprefix="$", range=[0, n["mean"].max() * 1.2])
        show(fig, 320, "Average spend by number of channels responded to")
        st.markdown('<div class="callout">This is association, not proof of cause: customers who respond to ads may already be the keenest buyers. '
                    "A controlled test would be needed to claim a channel causes extra spend.</div>", unsafe_allow_html=True)

    conclusion("Concentrate effort on the segments that carry the revenue and use channels in combination. "
               "Customers aged 45 to 64 who are married or together, hold higher degrees and earn more spend the most, and meat and alcohol lead the product mix. "
               "Instagram and Facebook responders spend far more than non-responders, while brochures and bulk mail still work for older customers in specific countries, which argues for a hybrid plan. "
               "Hibernating and 'Others' customers are the untapped reactivation pool.")


# ----------------------------------------------------------------------------
# WP 5  Banking
# ----------------------------------------------------------------------------
def page_bank() -> None:
    workpaper(
        REFS["bank"], "Banking transactions: finding the ones worth a second look",
        "Portfolio project · built on audit experience with bank branch data",
        "Nobody can review every transaction. Flag the unusual ones with simple, explainable rules so a reviewer can start in the right place.",
        ["Generated 2,013 realistic transactions across 50 accounts and two years (no client data)",
         "Flagged amount outliers only when both z-score and IQR tests agree, to cut false positives",
         "Added a velocity check (more than 5 in an hour) and a dormant-account check (30-day gap, then 3 in a day)",
         "Excluded failed and reversed items from spend totals but kept them for pattern analysis"],
        "Python: pandas, NumPy, Matplotlib",
    )
    legend()
    t, a = load_bank()
    done = t[t.status == "completed"].copy()
    tabs = st.tabs(["Spending patterns", "Anomaly lab", "Method"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        ch = c1.multiselect("Channel", sorted(t.channel.unique()), default=sorted(t.channel.unique()))
        cat = c2.multiselect("Category", sorted(t.category.unique()), default=sorted(t.category.unique()))
        d = done[done.channel.isin(ch) & done.category.isin(cat)]
        if d.empty:
            st.warning("No transactions match. Add a channel or category back.")
        else:
            kpis([("Completed transactions", f"{len(d):,}"), ("Total spend", f"₹{d.amount.sum():,.0f}"),
                  ("Median transaction", f"₹{d.amount.median():,.0f}"), ("Accounts", f"{d.account_id.nunique()}")])
            mo = d.groupby("month")["amount"].sum().reset_index()
            fig = px.bar(mo, x="month", y="amount", color_discrete_sequence=[BLUE], labels={"amount": "Spend (₹)", "month": ""})
            show(fig, 320, "Monthly spend")
            c1, c2 = st.columns(2)
            with c1:
                cs = d.groupby("category")["amount"].sum().sort_values()
                fig = go.Figure(go.Bar(x=cs.values, y=cs.index, orientation="h", marker_color=TEAL))
                show(fig, 340, "Spend by category")
                top3 = cs.sort_values(ascending=False).head(3)
                st.markdown(f"Top three categories (**{', '.join(top3.index)}**) take **{top3.sum()/cs.sum():.1%}** of spend.")
            with c2:
                n = st.slider("Merchants to show", 5, 20, 10)
                ms = d.groupby("merchant")["amount"].sum().sort_values().tail(n)
                fig = go.Figure(go.Bar(x=ms.values, y=ms.index, orientation="h", marker_color=PLUM))
                show(fig, 340 if n <= 12 else 40 + 26 * n, "Top merchants")

    with tabs[1]:
        st.markdown("Move the thresholds and watch the flagged list change. The project's settings are z-score 3, IQR multiplier 3, both tests required.")
        c1, c2, c3 = st.columns(3)
        zt = c1.slider("Z-score threshold", 2.0, 5.0, 3.0, step=0.1)
        km = c2.slider("IQR multiplier", 1.0, 5.0, 3.0, step=0.25)
        mode = c3.radio("Amount rule", ["Both tests must trigger", "Either test"], horizontal=False)
        mu, sd = done.amount.mean(), done.amount.std()
        q1, q3 = done.amount.quantile([0.25, 0.75])
        z_hi = mu + zt * sd
        iqr_hi = q3 + km * (q3 - q1)
        done["z"] = (done.amount - mu) / sd
        zf, qf = done.amount > z_hi, done.amount > iqr_hi
        done["amount_flag"] = (zf & qf) if mode.startswith("Both") else (zf | qf)
        other = a[(a["_flag_velocity"] == True) | (a["_flag_dormant_spike"] == True)]["transaction_id"]  # noqa: E712
        done["other_flag"] = done.transaction_id.isin(other)
        done["flagged"] = done.amount_flag | done.other_flag
        nflag = int(done.flagged.sum())
        kpis([("Amount outliers", f"{int(done.amount_flag.sum())}"), ("Velocity or dormant flags", f"{int(done.other_flag.sum())}"),
              ("Total flagged", f"{nflag} ({nflag/len(done):.1%})"), ("Project setting", "58 (3.2%)")])
        fig = go.Figure()
        ok = done[~done.flagged]
        fig.add_scatter(x=ok.datetime, y=ok.amount, mode="markers", name="Not flagged", marker=dict(color=SLATE, size=5, opacity=0.45))
        fl = done[done.amount_flag]
        fig.add_scatter(x=fl.datetime, y=fl.amount, mode="markers", name="Amount outlier", marker=dict(color=AMBER, size=9, line=dict(color=INK, width=1)))
        ot = done[done.other_flag & ~done.amount_flag]
        fig.add_scatter(x=ot.datetime, y=ot.amount, mode="markers", name="Velocity or dormant", marker=dict(color=PLUM, size=9, symbol="diamond", line=dict(color=INK, width=1)))
        fig.add_hline(y=min(z_hi, iqr_hi) if not mode.startswith("Both") else max(z_hi, iqr_hi), line_dash="dash", line_color=INK,
                      annotation_text="Amount threshold", annotation_position="top left")
        fig.update_yaxes(title="Amount (₹)")
        show(fig, 420, "Every completed transaction, with flags")
        flagged = done[done.flagged].copy()
        flagged["Reason"] = np.where(flagged.amount_flag, "Amount outlier", "")
        flagged.loc[flagged.other_flag, "Reason"] = (flagged.loc[flagged.other_flag, "Reason"] + " Velocity or dormant").str.strip()
        out = flagged[["transaction_id", "account_id", "datetime", "merchant", "category", "amount", "channel", "Reason"]].sort_values("amount", ascending=False)
        st.markdown(f"**{len(out)} transactions to review**")
        table(out)
        st.download_button("Download flagged transactions (CSV)", out.to_csv(index=False).encode(), "flagged_transactions.csv", "text/csv")
        st.caption("Velocity and dormant-account flags are taken from the original pipeline run; only the amount rules are recomputed here.")

    with tabs[2]:
        st.markdown("""
**Why two outlier tests.** A z-score assumes a roughly bell-shaped distribution, but transaction amounts are skewed, so a few very large items inflate the standard deviation. The IQR test does not depend on the mean. Requiring both cuts false positives; z-score alone flags 57 completed transactions here, the pair flags 50.

**Why velocity and dormancy.** Amount alone misses patterns: ten small payments in an hour, or an account that sits idle for a month and then wakes up.

**What this is not.** A screen, not a verdict. Every flag is a prompt for a reviewer, and the thresholds are constants at the top of the script so they can be tuned to a bank's own risk appetite.

**Data.** Transactions are generated to resemble branch-level retail banking activity (UPI, debit card, net banking, NEFT). No client data is used.
""")

    conclusion(f"Using the project's settings, {a.shape[0]} of {len(done):,} completed transactions are flagged ({a.shape[0]/len(done):.1%}), and the top three categories (travel, retail, ATM withdrawals) account for about two-thirds of spend. "
               "The point for a reviewer is triage: the interactive thresholds show how sensitive the flagged list is to the rule, which is the first question an auditor should ask of any exception report.")


# ----------------------------------------------------------------------------
# WP 6  Options
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def bin_curve(S, K, T, r, sig):
    steps = np.unique(np.round(np.logspace(0, 3, 45)).astype(int))
    return pd.DataFrame({"steps": steps, "price": [binomial_call(S, K, T, r, sig, int(n)) for n in steps]})


@st.cache_data(show_spinner=False)
def mc_curve(S, K, T, r, sig, seed, nmax):
    pay = mc_payoffs(S, K, T, r, sig, nmax, seed)
    cs, cs2 = np.cumsum(pay), np.cumsum(pay**2)
    ns = np.unique(np.round(np.logspace(2, np.log10(nmax), 50)).astype(int))
    mean = cs[ns - 1] / ns
    var = np.maximum((cs2[ns - 1] - ns * mean**2) / (ns - 1), 0)
    return pd.DataFrame({"paths": ns, "price": mean, "se": np.sqrt(var / ns)})


def page_option() -> None:
    workpaper(
        REFS["option"], "Options: one price, three independent methods",
        "Portfolio project · independent price verification",
        "A fair value you cannot re-derive is not one you can defend. Price one European call three ways and check that the answers reconcile.",
        ["Priced the call with Black-Scholes (closed form), a Cox-Ross-Rubinstein binomial tree and Monte Carlo simulation",
         "Reconciled the three prices and tested whether Black-Scholes sits inside the Monte Carlo 95% interval",
         "Showed how each method converges, and where each one misleads when under-sampled",
         "Ranked inputs by price sensitivity to show which assumptions deserve the most challenge"],
        "Python: NumPy, SciPy",
    )
    legend()
    st.markdown("**Inputs** (defaults are the project's illustrative parameters)")
    c = st.columns(5)
    S = c[0].number_input("Spot (₹)", 100.0, 5000.0, 1400.0, step=50.0)
    K = c[1].number_input("Strike (₹)", 100.0, 5000.0, 1500.0, step=50.0)
    T = c[2].number_input("Years to expiry", 0.1, 5.0, 1.0, step=0.25)
    r = c[3].number_input("Risk-free rate (%)", 0.0, 15.0, 6.5, step=0.25) / 100
    sig = c[4].number_input("Volatility (%)", 5.0, 80.0, 28.0, step=1.0) / 100
    c = st.columns(3)
    N = c[0].select_slider("Binomial steps", [1, 5, 10, 20, 50, 100, 250, 500, 1000, 2000], value=1000)
    paths = c[1].select_slider("Monte Carlo paths", [1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000], value=1_000_000, format_func=lambda v: f"{v:,}")
    seed = c[2].number_input("Random seed", 0, 9999, 42)

    bs = bs_call(S, K, T, r, sig)
    bn = binomial_call(S, K, T, r, sig, int(N))
    mc, se = mc_call(S, K, T, r, sig, int(paths), int(seed))
    lo, hi = mc - 1.96 * se, mc + 1.96 * se
    inside = lo <= bs <= hi
    res = pd.DataFrame({"Method": ["Black-Scholes (closed form)", f"Binomial tree ({N:,} steps)", f"Monte Carlo ({paths:,} paths)"],
                        "Price (₹)": [bs, bn, mc], "Difference vs Black-Scholes": [0.0, bn - bs, mc - bs]})
    kpis([("Black-Scholes", f"₹{bs:,.4f}"), ("Binomial", f"₹{bn:,.4f}"), ("Monte Carlo", f"₹{mc:,.4f}"), ("Monte Carlo standard error", f"{se:.4f}")])
    fmt = res.copy()
    fmt["Price (₹)"] = fmt["Price (₹)"].map("{:,.4f}".format)
    fmt["Difference vs Black-Scholes"] = fmt["Difference vs Black-Scholes"].map(lambda v: "-" if v == 0 else f"{v:+.4f}")
    table(fmt)
    kind = "teal" if inside else "amber"
    st.markdown(f'<div class="callout" style="border-left-color:{TEAL if inside else AMBER}">Monte Carlo 95% interval: <b>₹{lo:,.4f} to ₹{hi:,.4f}</b>. '
                f'Black-Scholes is <b>{"inside" if inside else "outside"}</b> it. '
                f'{"The three methods reconcile." if inside and abs(bn - bs) < 0.5 else "Something does not reconcile: raise the steps or paths, or look for a modelling difference."}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Convergence", "Sensitivity", "Payoff"])
    with tabs[0]:
        bc = bin_curve(S, K, T, r, sig)
        fig = go.Figure()
        fig.add_scatter(x=bc.steps, y=bc.price, mode="lines+markers", name="Binomial", line=dict(color=BLUE, width=2.5), marker=dict(size=5))
        fig.add_hline(y=bs, line_dash="dash", line_color=INK, annotation_text="Black-Scholes", annotation_position="bottom right")
        fig.update_xaxes(type="log", title="Tree steps (log scale)")
        fig.update_yaxes(title="Price (₹)")
        show(fig, 340, "Binomial tree: wrong at low step counts, then it settles")
        s1, s20 = binomial_call(S, K, T, r, sig, 1), binomial_call(S, K, T, r, sig, 20)
        st.markdown(f"With one step the tree returns **₹{s1:,.2f}** (off by ₹{s1-bs:+,.2f}). With twenty it returns **₹{s20:,.2f}** (off by ₹{s20-bs:+,.2f}) while looking precise. It oscillates around the true value rather than approaching from one side.")
        mcc = mc_curve(S, K, T, r, sig, int(seed), int(paths))
        fig = go.Figure()
        fig.add_scatter(x=mcc.paths, y=mcc.price + 1.96 * mcc.se, line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=mcc.paths, y=mcc.price - 1.96 * mcc.se, fill="tonexty", fillcolor="rgba(14,143,110,0.18)", line=dict(width=0), name="95% interval", hoverinfo="skip")
        fig.add_scatter(x=mcc.paths, y=mcc.price, mode="lines", name="Monte Carlo", line=dict(color=TEAL, width=2.5))
        fig.add_hline(y=bs, line_dash="dash", line_color=INK, annotation_text="Black-Scholes", annotation_position="bottom right")
        fig.update_xaxes(type="log", title="Simulated paths (log scale)")
        fig.update_yaxes(title="Price (₹)")
        show(fig, 340, "Monte Carlo: the band narrows with the square root of paths")
        i10, i100 = mcc.iloc[(mcc.paths - 10_000).abs().argmin()], mcc.iloc[(mcc.paths - 100_000).abs().argmin()]
        st.markdown(f"Going from about {int(i10.paths):,} to {int(i100.paths):,} paths cuts the standard error from **{i10.se:.2f}** to **{i100.se:.2f}**, a factor of about {i10.se/i100.se:.1f}, not ten. Halving the error takes four times the simulations.")

    with tabs[1]:
        shocks = [("Volatility +5 points", dict(sig=sig + 0.05)), ("Volatility -5 points", dict(sig=max(sig - 0.05, 0.01))),
                  ("Risk-free rate +1.5 points", dict(r=r + 0.015)), ("Risk-free rate -1.5 points", dict(r=max(r - 0.015, 0))),
                  ("Expiry +1 year", dict(T=T + 1)), ("Spot +7%", dict(S=S * 1.07)), ("Spot -7%", dict(S=S * 0.93))]
        rows = []
        for lab, kw in shocks:
            a = dict(S=S, K=K, T=T, r=r, sig=sig)
            a.update(kw)
            rows.append((lab, bs_call(**a) - bs))
        sens = pd.DataFrame(rows, columns=["Change", "Impact (₹)"]).sort_values("Impact (₹)", key=abs)
        fig = go.Figure(go.Bar(x=sens["Impact (₹)"], y=sens["Change"], orientation="h", marker_color=[TEAL if v > 0 else BRICK for v in sens["Impact (₹)"]],
                               text=[f"{v:+,.1f}" for v in sens["Impact (₹)"]], textposition="outside"))
        fig.update_xaxes(title="Change in price (₹)")
        show(fig, 380, "How much the price moves when one input moves")
        st.markdown("Volatility moves the price most per unit of change, and it is the input most often accepted from a client without challenge. That gap between how much an input matters and how much it is tested is the audit point.")

    with tabs[2]:
        spots = np.linspace(S * 0.5, S * 1.5, 120)
        fig = go.Figure()
        fig.add_scatter(x=spots, y=np.maximum(spots - K, 0), name="Value at expiry", line=dict(color=SLATE, width=2, dash="dot"))
        fig.add_scatter(x=spots, y=bs_call(spots, K, T, r, sig), name="Value today (Black-Scholes)", line=dict(color=BLUE, width=3))
        fig.add_vline(x=S, line_dash="dash", line_color=INK, annotation_text="Spot")
        fig.update_xaxes(title="Underlying price (₹)")
        fig.update_yaxes(title="Option value (₹)")
        show(fig, 380, "Call option value against the underlying")

    conclusion("The three methods rest on identical assumptions, so they must agree, and here they do. The more useful lesson is where they fail: a binomial tree with too few steps and a Monte Carlo run with too few paths both produce a confident-looking wrong number, "
               "so step count and path count deserve the same scrutiny as any other model input. "
               "Limits: European exercise only, no dividends, and constant volatility and interest rate.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
ROUTES = {"overview": page_overview, "turtle": page_turtle, "churn": page_churn, "nhs": page_nhs,
          "market": page_market, "bank": page_bank, "option": page_option}

# Deep links: https://<your-app>.streamlit.app/?project=churn opens that project directly,
# so each button on a personal website can point at its own project.
ALIASES = {"lse3": "turtle", "lse4": "churn", "lse2": "nhs", "lse1": "market",
           "banking": "bank", "options": "option", "option-pricing": "option", "home": "overview"}
if "page" not in st.session_state:
    wanted = str(st.query_params.get("project", "overview")).lower()
    wanted = ALIASES.get(wanted, wanted)
    st.session_state["page"] = wanted if wanted in ROUTES else "overview"

with st.sidebar:
    st.markdown('<p class="side-name">Sadia Yusuf, CA</p><p class="side-role">Financial data analytics portfolio</p>', unsafe_allow_html=True)
    st.radio("Go to", list(PAGES), label_visibility="collapsed", format_func=lambda k: (f"{REFS[k]}  {PAGES[k]}" if k in REFS else PAGES[k]), key="page")
    st.markdown('<p class="note" style="margin-top:1.4rem">Built with Python and Streamlit. Charts are interactive: hover, zoom, and use the controls above each one.</p>', unsafe_allow_html=True)

if st.query_params.get("project") != st.session_state["page"]:
    st.query_params["project"] = st.session_state["page"]

ROUTES[st.session_state["page"]]()
