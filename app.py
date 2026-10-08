"""
Analytics portfolio - Sadia Yusuf, Chartered Accountant.

One Streamlit app, six projects. Every project opens from a button or a direct link:
    https://<your-app>.streamlit.app/?project=churn

Run locally:   streamlit run app.py
Deploy:        push this folder (app.py, requirements.txt, data/, .streamlit/) to GitHub,
               then create the app on share.streamlit.io and point it at app.py.
"""
from __future__ import annotations

import re
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
INK = "#0F1B3D"
MUTED = "#5B677A"
BLUE = "#2F6FDE"
GREEN = "#2FA66A"
SAFFRON = "#F2A33A"
CORAL = "#E4572E"
VIOLET = "#7C4DDB"
TEAL = "#14A3A3"
SLATE = "#8896AB"
ROSE = "#C2417A"
PALETTE = [BLUE, GREEN, SAFFRON, VIOLET, CORAL, TEAL, SLATE, ROSE]
px.defaults.color_discrete_sequence = PALETTE

ICONS = {
    "turtle": '<path d="M12 3l7 4v10l-7 4-7-4V7z"/><path d="M12 3v18M5 7l14 10M19 7L5 17"/>',
    "churn": '<path d="M9 4H5v16h4"/><path d="M16 8l4 4-4 4M20 12H9"/>',
    "nhs": '<path d="M10 3h4v7h7v4h-7v7h-4v-7H3v-4h7z"/>',
    "market": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "bank": '<path d="M3 10l9-6 9 6M5 10v8M10 10v8M14 10v8M19 10v8M3 20h18"/>',
    "option": '<path d="M3 4v15h18"/><path d="M6 17l5-5 3 3 6-8"/>',
}
PROJECTS = {
    "turtle": dict(name="Turtle Games", nav="Turtle Games", accent=GREEN,
                   tag="What makes customers earn loyalty points, and which customers should marketing target?",
                   tools=["Python", "scikit-learn", "NLP"]),
    "churn": dict(name="ConnectTel", nav="ConnectTel", accent=CORAL,
                  tag="Which telecom customers will leave, and what is it worth to keep them?",
                  tools=["Python", "Statistics", "Ensemble models"]),
    "nhs": dict(name="NHS Appointments", nav="NHS", accent=BLUE,
                tag="Is primary-care capacity adequate, and how is it used?",
                tools=["Python", "Time series", "1.5M rows"]),
    "market": dict(name="2Market", nav="2Market", accent=VIOLET,
                   tag="Who are the customers, which advertising channels work, and which products sell?",
                   tools=["Excel", "SQL", "Tableau", "RFM"]),
    "bank": dict(name="Banking Transactions", nav="Banking", accent=SAFFRON,
                 tag="Which transactions deserve a human look, and why?",
                 tools=["Python", "Z-score", "IQR"]),
    "option": dict(name="Option Pricing", nav="Options", accent=TEAL,
                   tag="Can a derivative's fair value be re-derived independently?",
                   tools=["Python", "NumPy", "SciPy"]),
}
ORDER = list(PROJECTS)

st.set_page_config(
    page_title="Sadia Yusuf | Analytics portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&display=swap');
:root { --ink:__INK__; --muted:__MUTED__; --rule:#DDE2EC; --paper:#F6F7FB; --accent:__BLUE__; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp textarea, .stApp td, .stApp th,
.stApp [data-baseweb] { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp { background: var(--paper); color: var(--ink); }
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 4rem; }
h1, h2, h3, h4, .display { font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif !important; letter-spacing: -0.02em; color: var(--ink); }
h2 { font-size: 1.65rem !important; line-height: 1.25 !important; padding-top: .6rem !important; }
h3 { font-size: 1.25rem !important; line-height: 1.3 !important; }
p, li { line-height: 1.62; }

@keyframes rise { from { opacity:0; transform: translateY(16px); } to { opacity:1; transform:none; } }
@keyframes fade { from { opacity:0; } to { opacity:1; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes drift { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-18px, 14px); } }

/* brand bar and navigation */
.brand { display:flex; justify-content:space-between; align-items:center; margin:.2rem 0 .7rem; }
.brand b { font-family:'Bricolage Grotesque',sans-serif; font-size:1.2rem; letter-spacing:-.01em; }
.brand span { color: var(--muted); font-size:.88rem; }
.st-key-navbar { background:#fff; border:1px solid var(--rule); border-radius:16px; padding:.45rem; margin-bottom:1.1rem; }
.st-key-navbar button { border-radius:11px; border:1px solid transparent; background:transparent; color:var(--ink);
    min-height:2.5rem; transition: background .2s, transform .15s; }
.st-key-navbar button p { color: inherit; font-weight:500; font-size:.95rem; }
.st-key-navbar button:hover { background:#EEF1F8; transform: translateY(-2px); border-color: transparent; color: var(--ink); }
.st-key-navbar button:active { transform: translateY(0) scale(.98); }
.st-key-navbar button[data-testid="stBaseButton-primary"] { background: var(--accent); color:#fff; }
.st-key-navbar button[data-testid="stBaseButton-primary"] p { color:#fff; }

/* hero */
.hero { position:relative; overflow:hidden; background: var(--ink); color:#fff; border-radius:22px; padding:1.9rem 2.1rem 1.7rem;
    margin: 0 0 1.2rem; animation: rise .7s cubic-bezier(.2,.7,.2,1) both; }
.hero::before { content:""; position:absolute; right:-70px; top:-80px; width:300px; height:300px; border-radius:50%;
    background: var(--accent); opacity:.28; animation: drift 14s ease-in-out infinite; }
.hero::after { content:""; position:absolute; right:110px; bottom:-120px; width:220px; height:220px; border-radius:50%;
    background: var(--accent); opacity:.14; animation: drift 18s ease-in-out infinite reverse; }
.hero > * { position:relative; z-index:1; }
.hero-top { display:flex; gap:.6rem; align-items:center; flex-wrap:wrap; }
.hero-ico { width:44px; height:44px; border-radius:13px; background: var(--accent); display:grid; place-items:center; }
.hero-ico svg { width:25px; height:25px; stroke:#fff; fill:none; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.chip { font-size:.8rem; padding:.22rem .7rem; border-radius:999px; border:1px solid rgba(255,255,255,.26); color:#E8ECF7; }
.chip.live { background: rgba(47,166,106,.35); border-color: transparent; }
.chip.note { background: rgba(242,163,58,.32); border-color: transparent; }
.hero-title { font-family:'Bricolage Grotesque',sans-serif; font-size: clamp(2.1rem, 4.6vw, 3.3rem); line-height:1.15;
    font-weight:700; letter-spacing:-.025em; margin:.85rem 0 .35rem; color:#fff !important; padding:.05em 0; }
.hero-sub { color:#C9D2EA !important; font-size:1.07rem; max-width:44rem; margin:0; }
.hero-kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:.8rem; margin-top:1.4rem; }
.hk { background: rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); border-radius:15px; padding:.75rem .95rem;
    animation: rise .6s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 90ms + 220ms); }
.hk b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:1.75rem; font-weight:700; color:#fff;
    font-variant-numeric: tabular-nums; line-height:1.2; }
.hk span { color:#B9C4E2; font-size:.86rem; }

/* story cards */
.story { display:grid; grid-template-columns: repeat(3, 1fr); gap:.9rem; margin:.1rem 0 1.1rem; }
.sc { background:#fff; border:1px solid var(--rule); border-radius:18px; padding:1.05rem 1.2rem;
    animation: rise .6s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 100ms + 350ms); }
.sc h4 { margin:0 0 .45rem; font-size:1.08rem !important; color: var(--accent) !important; }
.sc p, .sc li { font-size:.94rem; margin:0; color:#27324a; }
.sc ul { margin:0; padding-left:1.05rem; }
@media (max-width: 900px) { .story { grid-template-columns: 1fr; } }

/* overview tiles */
.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:1.05rem; margin: .4rem 0 1.4rem; }
@media (max-width: 980px) { .grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
a.tile { display:block; text-decoration:none !important; color: var(--ink) !important; background:#fff; border:1px solid var(--rule);
    border-radius:20px; padding:1.3rem 1.3rem 1.1rem; position:relative; overflow:hidden;
    transition: transform .28s cubic-bezier(.2,.7,.2,1), box-shadow .28s, border-color .28s;
    animation: rise .65s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 90ms + 250ms); }
a.tile:hover { transform: translateY(-7px); box-shadow: 0 22px 44px -20px rgba(15,27,61,.4); border-color: var(--c); }
a.tile:focus-visible { outline: 3px solid var(--c); outline-offset: 3px; }
a.tile .bar { position:absolute; left:0; top:0; height:6px; width:100%; background: var(--c); transform-origin:left;
    animation: grow .9s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(var(--i) * 90ms + 450ms); }
a.tile .ico { width:46px; height:46px; border-radius:14px; background: color-mix(in srgb, var(--c) 14%, white); display:grid; place-items:center; margin-bottom:.8rem; transition: transform .3s; }
a.tile:hover .ico { transform: rotate(-6deg) scale(1.08); }
a.tile .ico svg { width:25px; height:25px; stroke: var(--c); fill:none; stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round; }
a.tile h3 { margin:0 0 .25rem; font-size:1.4rem !important; }
a.tile .q { color: var(--muted); font-size:.92rem; margin:0 0 .75rem; }
a.tile .res { font-size:.95rem; margin:0 0 .6rem; font-weight:500; }
a.tile .tags { display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.85rem; }
a.tile .tags span { font-size:.76rem; background:#EEF1F8; border-radius:999px; padding:.15rem .6rem; color:#3A4560; }
a.tile .go { font-weight:700; color: var(--c); display:inline-flex; align-items:center; gap:.4rem; transition: gap .22s; }
a.tile:hover .go { gap:.8rem; }
a.tile .go svg { width:18px; height:18px; stroke: var(--c); fill:none; stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }

/* body components */
.kpis { display:flex; flex-wrap:wrap; gap:.8rem; margin:.3rem 0 1rem; }
.kpi { flex:1 1 150px; background:#fff; border:1px solid var(--rule); border-left:4px solid var(--accent); border-radius:14px; padding:.7rem .95rem;
    animation: rise .5s cubic-bezier(.2,.7,.2,1) both; }
.kpi b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:1.55rem; font-weight:700; line-height:1.2; font-variant-numeric: tabular-nums; }
.kpi span { color: var(--muted); font-size:.86rem; }
.lead { font-size:1.05rem; color:#27324a; max-width:50rem; margin:.2rem 0 .8rem; }
.callout { background:#fff; border:1px solid var(--rule); border-left:4px solid var(--accent); padding:.8rem 1.05rem; border-radius:12px; margin:.6rem 0; animation: fade .5s both; }
.finds { list-style:none; padding:0; margin:.3rem 0 0; }
.finds li { display:flex; gap:.7rem; padding:.6rem 0; border-bottom:1px solid var(--rule); font-size:.97rem; }
.finds li:last-child { border-bottom:none; }
.mk { flex:none; width:1.3rem; text-align:center; font-weight:700; }
.mk.v { color: __GREEN__; }
.mk.n { color: __SAFFRON__; }
.legend { font-size:.88rem; color: var(--muted); }
.legend .mk { display:inline-block; width:auto; margin:0 .2rem 0 .6rem; }
.note { color: var(--muted); font-size:.88rem; }
.persona { background:#fff; border:1px solid var(--rule); border-top:5px solid var(--c); border-radius:16px; padding:.9rem 1rem; height:100%;
    transition: transform .25s, box-shadow .25s; animation: rise .5s both; }
.persona:hover { transform: translateY(-4px); box-shadow: 0 16px 30px -18px rgba(15,27,61,.4); }
.persona h4 { margin:0 0 .2rem; font-size:1.02rem !important; }
.persona p { margin:.15rem 0; font-size:.88rem; color:#27324a; }
.persona .big { font-family:'Bricolage Grotesque',sans-serif; font-size:1.5rem; font-weight:700; }
.conclusion { background: var(--ink); color:#fff; border-radius:20px; padding:1.4rem 1.6rem; margin:1.4rem 0 1rem; animation: rise .6s both; }
.conclusion h3 { color:#fff !important; margin:0 0 .5rem; }
.conclusion p { color:#DDE4F7 !important; margin:.3rem 0; }

/* controls */
.stTabs [data-baseweb="tab-list"] { gap:.35rem; background:#fff; border:1px solid var(--rule); border-radius:14px; padding:.3rem; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { height:2.4rem; padding:0 1rem; border-radius:10px; transition: background .2s; }
.stTabs [data-baseweb="tab"]:hover { background:#EEF1F8; }
.stTabs [data-baseweb="tab"] p { font-weight:500; }
.stTabs [aria-selected="true"] { background: var(--accent); }
.stTabs [aria-selected="true"] p { color:#fff; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1rem; animation: fade .4s both; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid var(--rule); border-radius:18px; padding:.5rem .5rem .1rem; animation: fade .5s both; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--rule) !important; border-radius:14px; }
[data-testid="stExpander"] summary { font-weight:500; }
[data-testid="stDataFrame"] { border:1px solid var(--rule); border-radius:12px; overflow:hidden; }
.stButton button { border-radius:11px; border:1.5px solid var(--ink); color: var(--ink); background:#fff; font-weight:500; transition: transform .15s, background .2s; }
.stButton button:hover { background: var(--ink); color:#fff; transform: translateY(-2px); }
.stSlider [role="slider"] { background: var(--accent); }
:focus-visible { outline: 3px solid var(--accent) !important; outline-offset: 2px; }
footer { visibility: hidden; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""
st.markdown(CSS.replace("__INK__", INK).replace("__MUTED__", MUTED).replace("__BLUE__", BLUE)
            .replace("__GREEN__", GREEN).replace("__SAFFRON__", SAFFRON), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def show(fig: go.Figure, height: int = 380, title: str | None = None) -> None:
    fig.update_layout(
        height=height,
        font=dict(family="DM Sans, sans-serif", size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        margin=dict(l=12, r=12, t=58 if title else 20, b=64 if len(fig.data) > 1 else 12),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(font_family="DM Sans", bgcolor="white"),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(family="Bricolage Grotesque, sans-serif", size=18)))
    fig.update_xaxes(showgrid=False, linecolor="#DDE2EC", tickcolor="#DDE2EC", zeroline=False)
    fig.update_yaxes(gridcolor="#E9EDF4", linecolor="rgba(0,0,0,0)", zeroline=False)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df: pd.DataFrame, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def sbutton(label: str, **kw) -> bool:
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:
        return st.button(label, use_container_width=True, **kw)


def kpis(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="kpi" style="animation-delay:{i*70}ms"><b>{v}</b><span>{k}</span></div>' for i, (k, v) in enumerate(items))
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def finds(items: list[tuple[str, str]]) -> None:
    rows = "".join(f'<li><span class="mk {k}">{"✓" if k == "v" else "△"}</span><span>{t}</span></li>' for k, t in items)
    st.markdown(f'<ul class="finds">{rows}</ul>', unsafe_allow_html=True)


def legend() -> None:
    st.markdown('<p class="legend">Marks:<span class="mk v">✓</span>recomputed live from the data in this app'
                f'<span class="mk n">△</span>carried over from the project notebook or report</p>', unsafe_allow_html=True)


def lead(text: str) -> None:
    st.markdown(f'<p class="lead">{text}</p>', unsafe_allow_html=True)


def how(text: str, title: str = "How to read this") -> None:
    with st.expander(title):
        st.markdown(text)


def callout(html: str) -> None:
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def conclusion(title: str, paragraphs: list[str]) -> None:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    st.markdown(f'<div class="conclusion"><h3>{title}</h3>{body}</div>', unsafe_allow_html=True)


def go_to(key: str) -> None:
    st.session_state["page"] = key


def navbar(active: str) -> None:
    st.markdown('<div class="brand"><b>Sadia Yusuf, CA</b><span>Financial data analytics portfolio</span></div>', unsafe_allow_html=True)
    with st.container(key="navbar"):
        cols = st.columns(len(ORDER) + 1)
        with cols[0]:
            sbutton("Overview", key="nav_overview", type="primary" if active == "overview" else "secondary", on_click=go_to, args=("overview",))
        for c, k in zip(cols[1:], ORDER):
            with c:
                sbutton(PROJECTS[k]["nav"], key=f"nav_{k}", type="primary" if active == k else "secondary", on_click=go_to, args=(k,))


def hero(key: str, kpi_items: list[tuple[str, str]], live: bool, subtitle: str | None = None) -> None:
    p = PROJECTS[key]
    chips = "".join(f'<span class="chip">{t}</span>' for t in p["tools"])
    src = '<span class="chip live">Live data</span>' if live else '<span class="chip note">Figures from notebook</span>'
    ks = "".join(f'<div class="hk" style="--i:{i}"><b>{v}</b><span>{k}</span></div>' for i, (k, v) in enumerate(kpi_items))
    st.markdown(
        f"""<style>:root {{ --accent: {p['accent']}; }}</style>
<section class="hero">
  <div class="hero-top"><span class="hero-ico"><svg viewBox="0 0 24 24">{ICONS[key]}</svg></span>{src}{chips}</div>
  <div class="hero-title">{p['name']}</div>
  <p class="hero-sub">{subtitle or p['tag']}</p>
  <div class="hero-kpis">{ks}</div>
</section>""", unsafe_allow_html=True)


def story(question: str, did: list[str], found: str) -> None:
    lis = "".join(f"<li>{d}</li>" for d in did)
    st.markdown(
        f"""<div class="story">
<div class="sc" style="--i:0"><h4>The question</h4><p>{question}</p></div>
<div class="sc" style="--i:1"><h4>What I did</h4><ul>{lis}</ul></div>
<div class="sc" style="--i:2"><h4>What I found</h4><p>{found}</p></div></div>""", unsafe_allow_html=True)


def next_project(key: str) -> None:
    i = ORDER.index(key)
    nxt = ORDER[(i + 1) % len(ORDER)]
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sbutton("Back to overview", key="bk_overview", on_click=go_to, args=("overview",))
    with c3:
        sbutton(f"Next project: {PROJECTS[nxt]['nav']}", key="bk_next", on_click=go_to, args=(nxt,))


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


# ----------------------------------------------------------------------------
# Data loaders
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
    d["Income tier"] = pd.cut(d["Income"], [-1, 10_000, 25_000, 50_000, 75_000, 100_000, 1e9],
                              labels=["Under 10K", "10-25K", "25-50K", "50-75K", "75-100K", "100K+"])
    return d


@st.cache_data(show_spinner=False)
def load_bank() -> tuple[pd.DataFrame, pd.DataFrame]:
    t = pd.read_csv(DATA / "transactions.csv", parse_dates=["datetime"])
    t["month"] = t["datetime"].dt.to_period("M").astype(str)
    a = pd.read_csv(DATA / "anomaly_flagged.csv")
    return t, a


# ----------------------------------------------------------------------------
# Pricing maths
# ----------------------------------------------------------------------------
def bs_call(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_greeks(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return dict(
        delta=norm.cdf(d1),
        gamma=norm.pdf(d1) / (S * sig * np.sqrt(T)),
        vega=S * norm.pdf(d1) * np.sqrt(T) / 100,
        theta=(-S * norm.pdf(d1) * sig / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365,
        rho=K * T * np.exp(-r * T) * norm.cdf(d2) / 100,
    )


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
        "Advanced Analytics for Organisational Impact",
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
        tab_note("turtle", 0)
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
        tab_note("turtle", 1)
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
            fig = px.line(curve, x="depth", y="R²", color="Data", markers=True, color_discrete_map={"Train": BLUE, "Test": SAFFRON})
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
        tab_note("turtle", 2)
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
        tab_note("turtle", 3)
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
        "Capstone",
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
    tabs = st.tabs(["Why customers leave", "Who is worth keeping", "Model and lift", "Retention economics"])

    with tabs[0]:

        tab_note("churn", 0)
        f = st.selectbox("Churn rate by", list(CHURN_FACTORS))
        d = pd.DataFrame({"Group": list(CHURN_FACTORS[f]), "Churn rate": list(CHURN_FACTORS[f].values())})
        fig = go.Figure(go.Bar(x=d["Group"], y=d["Churn rate"], marker_color=[SAFFRON if v > CHURN_BASE else TEAL for v in d["Churn rate"]],
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

        tab_note("churn", 1)
        tiers = pd.DataFrame({"Tier": ["Bronze", "Silver", "Gold", "Platinum"], "Customers": [1761, 1773, 1751, 1758],
                              "Avg CLV (£)": [159.6, 323.0, 446.4, 1313.9], "Share of CLV": [7.1, 14.5, 19.8, 58.5], "Churn rate": [30, 40, 30, 10]})
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=tiers["Tier"], y=tiers["Share of CLV"], marker_color=BLUE, text=[f"{v}%" for v in tiers["Share of CLV"]], textposition="outside"))
            fig.update_yaxes(ticksuffix="%", range=[0, 70])
            show(fig, 320, "Share of lifetime value")
        with c2:
            fig = go.Figure(go.Bar(x=tiers["Tier"], y=tiers["Churn rate"], marker_color=[SAFFRON if v > CHURN_BASE else TEAL for v in tiers["Churn rate"]],
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

        tab_note("churn", 2)
        comp = pd.DataFrame({"Model": ["Logistic regression", "Decision tree", "Random forest", "Neural network"],
                             "Accuracy": [.798, .801, .799, .794], "Recall (churn)": [.551, .543, .513, .545], "ROC-AUC": [.839, .835, .840, .840]})
        c1, c2 = st.columns([1.1, 1])
        with c1:
            long = comp.melt("Model", var_name="Metric", value_name="Score")
            fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", color_discrete_sequence=[SLATE, SAFFRON, BLUE])
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
        fig.add_scatter(x=dec["Decile"], y=dec["Captured"], name="Cumulative share of churners caught", mode="lines+markers", line=dict(color=SAFFRON, width=3))
        fig.update_xaxes(dtick=1, title="Risk decile (1 = highest risk)")
        fig.update_yaxes(ticksuffix="%")
        show(fig, 360, "Lift: the riskiest 30% of customers hold 65% of churners")
        finds([("n", "All four models land near 0.84 ROC-AUC. The notebook reads this as a data ceiling: there is no usage, complaint or outage data to learn from."),
               ("n", "The ensemble does not beat logistic regression on ROC-AUC (0.841 vs 0.842) but does on precision-recall (0.648 vs 0.634) and is more stable. It was chosen on expected cost, not accuracy.")])

    with tabs[3]:

        tab_note("churn", 3)
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
        "Data Analytics using Python",
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

        tab_note("nhs", 0)
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
            fig.add_vrect(x0=a, x1=b, fillcolor=SAFFRON, opacity=0.12, line_width=0)
        show(fig, 400, "Appointments per month, January 2020 to June 2022 (amber: restriction periods noted in the report)")
        tel0, telmax = share["Telephone"].iloc[0], share["Telephone"].max()
        f2f0, f2f_apr = share["Face-to-Face"].iloc[0], share.loc["2020-04", "Face-to-Face"]
        finds([("v", f"Telephone consultations were {tel0:.0f}% of appointments in January 2020 and reached {telmax:.0f}% at their peak. Face-to-face fell from {f2f0:.0f}% to {f2f_apr:.0f}% by April 2020."),
               ("v", f"By June 2022 face-to-face had recovered to {share['Face-to-Face'].iloc[-1]:.0f}%, still below the {f2f0:.0f}% before the pandemic."),
               ("v", "Volume dips line up with the first lockdown and the winter restriction periods.")])

    with tabs[1]:

        tab_note("nhs", 1)
        post = tot[tot["appointment_month"] >= "2021-08"].copy()
        c1, c2 = st.columns(2)
        k = c1.slider("Benchmark: mean plus this many standard deviations", 0.0, 2.0, 1.0, step=0.25)
        up = c2.slider("What if the benchmark were this much higher (%)", 0, 20, 0, step=1)
        thr = (post["count_of_appointments"].mean() + k * post["count_of_appointments"].std()) * (1 + up / 100)
        post["over"] = post["count_of_appointments"] > thr
        fig = go.Figure(go.Bar(x=post["date"], y=post["count_of_appointments"], marker_color=[SAFFRON if o else BLUE for o in post["over"]]))
        fig.add_hline(y=thr, line_dash="dash", line_color=INK, annotation_text=f"Benchmark {thr/1e6:.1f}M", annotation_position="top left")
        fig.update_yaxes(range=[0, post["count_of_appointments"].max() * 1.15])
        show(fig, 380, "Monthly appointments since August 2021 against the capacity benchmark")
        n_over = int(post["over"].sum())
        st.markdown(f'<div class="callout"><b>{n_over} of {len(post)} months</b> exceed the benchmark. At the default (mean + 1 SD, no uplift) that is 2 of 11, in October and November 2021.</div>', unsafe_allow_html=True)
        st.caption("The benchmark is statistical, not a measured staffing limit. It shows how close peak months run to the system's own typical ceiling.")
        finds([("v", "The system can reach about 30 million appointments a month (October and November 2021), then falls back sharply in winter. That pattern points to reactive rather than planned capacity."),
               ("v", "Raising the benchmark by 5 to 10% removes the peak-month breaches, which is the basis for the report's modest-uplift staffing recommendation.")])

    with tabs[2]:

        tab_note("nhs", 2)
        view = st.selectbox("View", ["Attendance (did not attend rate)", "Provider type", "Booking lead time", "Appointment length"])
        if view.startswith("Attendance"):
            s = ar.pivot_table(index="appointment_month", columns="appointment_status", values="count_of_appointments", aggfunc="sum")
            d = (s["DNA"] / s.sum(axis=1) * 100).reset_index(name="DNA %")
            d["date"] = pd.to_datetime(d["appointment_month"])
            fig = px.line(d, x="date", y="DNA %", markers=True, color_discrete_sequence=[SAFFRON])
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
            fig = go.Figure(go.Scatter(x=long.index, y=long.values, mode="lines+markers", line=dict(color=SAFFRON, width=3)))
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
            fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=[SAFFRON if i.startswith("Unknown") else BLUE for i in d.index],
                                   text=[f"{v:.0f}%" for v in d.values], textposition="outside"))
            fig.update_yaxes(ticksuffix="%", range=[0, 30])
            show(fig, 340, "How long appointments last (December 2021 to June 2022)")
            st.markdown("A quarter of appointments have no recorded duration. That data-quality gap limits capacity planning more than any modelling choice.")

    with tabs[3]:

        tab_note("nhs", 3)
        setting = nc.groupby("service_setting")["count_of_appointments"].sum().sort_values()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=setting.values / setting.sum() * 100, y=setting.index, orientation="h", marker_color=BLUE,
                                   text=[f"{v:.1f}%" for v in setting.values / setting.sum() * 100], textposition="outside"))
            fig.update_xaxes(ticksuffix="%", range=[0, 105])
            show(fig, 300, "Service setting (August 2021 to June 2022)")
        with c2:
            ctx = nc.groupby("context_type")["count_of_appointments"].sum()
            fig = go.Figure(go.Pie(labels=ctx.index, values=ctx.values, hole=0.55, marker_colors=[BLUE, SAFFRON, SLATE], textinfo="percent"))
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

        tab_note("nhs", 4)
        n = st.slider("Hashtags to show", 5, 30, 15)
        d = tags.head(n).iloc[::-1]
        fig = go.Figure(go.Bar(x=d["count"], y=d["hashtag"], orientation="h", marker_color=VIOLET))
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
        "Business analytics",
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

        tab_note("market", 0)
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

        tab_note("market", 1)
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

        tab_note("market", 2)
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

        tab_note("market", 3)
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

        tab_note("bank", 0)
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
                fig = go.Figure(go.Bar(x=ms.values, y=ms.index, orientation="h", marker_color=VIOLET))
                show(fig, 340 if n <= 12 else 40 + 26 * n, "Top merchants")

    with tabs[1]:

        tab_note("bank", 1)
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
        fig.add_scatter(x=fl.datetime, y=fl.amount, mode="markers", name="Amount outlier", marker=dict(color=SAFFRON, size=9, line=dict(color=INK, width=1)))
        ot = done[done.other_flag & ~done.amount_flag]
        fig.add_scatter(x=ot.datetime, y=ot.amount, mode="markers", name="Velocity or dormant", marker=dict(color=VIOLET, size=9, symbol="diamond", line=dict(color=INK, width=1)))
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

        tab_note("bank", 2)
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
    st.markdown(f'<div class="callout" style="border-left-color:{TEAL if inside else SAFFRON}">Monte Carlo 95% interval: <b>₹{lo:,.4f} to ₹{hi:,.4f}</b>. '
                f'Black-Scholes is <b>{"inside" if inside else "outside"}</b> it. '
                f'{"The three methods reconcile." if inside and abs(bn - bs) < 0.5 else "Something does not reconcile: raise the steps or paths, or look for a modelling difference."}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Convergence", "Sensitivity", "Payoff"])
    with tabs[0]:
        tab_note("option", 0)
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

        tab_note("option", 1)
        shocks = [("Volatility +5 points", dict(sig=sig + 0.05)), ("Volatility -5 points", dict(sig=max(sig - 0.05, 0.01))),
                  ("Risk-free rate +1.5 points", dict(r=r + 0.015)), ("Risk-free rate -1.5 points", dict(r=max(r - 0.015, 0))),
                  ("Expiry +1 year", dict(T=T + 1)), ("Spot +7%", dict(S=S * 1.07)), ("Spot -7%", dict(S=S * 0.93))]
        rows = []
        for lab, kw in shocks:
            a = dict(S=S, K=K, T=T, r=r, sig=sig)
            a.update(kw)
            rows.append((lab, bs_call(**a) - bs))
        sens = pd.DataFrame(rows, columns=["Change", "Impact (₹)"]).sort_values("Impact (₹)", key=abs)
        fig = go.Figure(go.Bar(x=sens["Impact (₹)"], y=sens["Change"], orientation="h", marker_color=[TEAL if v > 0 else CORAL for v in sens["Impact (₹)"]],
                               text=[f"{v:+,.1f}" for v in sens["Impact (₹)"]], textposition="outside"))
        fig.update_xaxes(title="Change in price (₹)")
        show(fig, 380, "How much the price moves when one input moves")
        st.markdown("Volatility moves the price most per unit of change, and it is the input most often accepted from a client without challenge. That gap between how much an input matters and how much it is tested is the audit point.")

    with tabs[2]:

        tab_note("option", 2)
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
# Page-level meta (hero KPIs and headline finding, all computed from bundled data)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def meta() -> dict:
    t = load_turtle()
    r2 = stats.linregress(t["spending_score"], t["loyalty_points"]).rvalue ** 2
    r2i = stats.linregress(t["remuneration"], t["loyalty_points"]).rvalue ** 2
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
    gap = max(abs(bn - bs), abs(mc - bs))
    return {
        "turtle": dict(
            k=[("Customers", f"{len(t):,}"), ("Spending score explains", f"{r2:.0%}"), ("Income explains", f"{r2i:.0%}"), ("Customer segments", "5")],
            found=f"Spending score alone explains {r2:.0%} of loyalty points and income {r2i:.0%}; age explains almost nothing. Five customer segments separate cleanly.",
            short=f"Spending score alone explains {r2:.0%} of loyalty points. Five customer segments separate cleanly."),
        "churn": dict(
            k=[("Customers", "7,043"), ("Churned", "26.5%"), ("Annual revenue lost", "£1.67M"), ("Share of revenue", "30.5%")],
            found="A quarter of customers leave, taking about 30% of revenue with them. Contract type is the strongest driver, and targeting by expected cost beats targeting by accuracy.",
            short="26.5% of customers churned, about 30% of annual revenue. Contract type is the strongest driver."),
        "nhs": dict(
            k=[("Rows analysed", "1.5M"), ("In General Practice", f"{gp:.0f}%"),
               ("Telephone, first month", f"{tel.iloc[0]:.0f}%"), ("Telephone, peak", f"{tel.max():.0f}%")],
            found=f"General Practice carries {gp:.0f}% of appointments. Telephone consultations grew from {tel.iloc[0]:.0f}% to a peak of {tel.max():.0f}% of the mix after the first lockdown.",
            short=f"{gp:.0f}% of appointments are in General Practice. Telephone went from {tel.iloc[0]:.0f}% to {tel.max():.0f}% of the mix."),
        "market": dict(
            k=[("Customers", f"{len(m):,}"), ("Champions + Big Spenders", f"{len(top)/len(m):.0%}"), ("Their share of spend", f"{top.Total_Spending.sum()/m.Total_Spending.sum():.0%}"), ("Instagram responders spend", f"{ig:.1f}x")],
            found=f"Champions and Big Spenders are {len(top)/len(m):.0%} of customers but {top.Total_Spending.sum()/m.Total_Spending.sum():.0%} of spend, and Instagram responders spend {ig:.1f}x non-responders.",
            short=f"Champions and Big Spenders are {len(top)/len(m):.0%} of customers and {top.Total_Spending.sum()/m.Total_Spending.sum():.0%} of spend. Instagram responders spend {ig:.1f}x."),
        "bank": dict(
            k=[("Completed transactions", f"{len(done):,}"), ("Flagged for review", f"{ba.shape[0]}"), ("Flag rate", f"{ba.shape[0]/len(done):.1%}"), ("Rules combined", "4")],
            found=f"{ba.shape[0]} of {len(done):,} completed transactions ({ba.shape[0]/len(done):.1%}) are flagged using two outlier tests, a velocity check and a dormant-account check.",
            short=f"{ba.shape[0]} of {len(done):,} completed transactions flagged ({ba.shape[0]/len(done):.1%}) by four rules."),
        "option": dict(
            k=[("Black-Scholes", f"{bs:,.2f}"), ("Binomial tree", f"{bn:,.2f}"), ("Monte Carlo", f"{mc:,.2f}"), ("Largest gap", f"{gap:.2f}")],
            found=f"Three independent methods give {bs:,.2f}, {bn:,.2f} and {mc:,.2f} for the same option, so the price is verified, not just calculated.",
            short=f"Black-Scholes {bs:,.2f}, binomial {bn:,.2f}, Monte Carlo {mc:,.2f}. They agree to within {gap:.2f}."),
    }


SLUG = {"WP 1": "turtle", "WP 2": "churn", "WP 3": "nhs", "WP 4": "market", "WP 5": "bank", "WP 6": "option"}
REFS = {v: k for k, v in SLUG.items()}


def workpaper(ref: str, title: str, course: str, objective: str, performed: list[str], tools: str) -> None:
    """Page header: hero banner, then question / what I did / what I found."""
    key = SLUG[ref]
    info = meta()[key]
    hero(key, info["k"], live=key != "churn")
    story(objective, performed, info["found"])
    legend()


def conclusion_page(text: str) -> None:
    if text:
        _conclusion_card("Conclusion", [text])
    else:
        st.markdown("### What it adds up to")


# the page bodies below call conclusion(text); keep their signature
_conclusion_card = conclusion


def conclusion(text: str) -> None:  # noqa: F811
    conclusion_page(text)


TAB_NOTES = {
    ("turtle", 0): "Each dot is one customer. Pick a factor and see how tightly loyalty points follow it: the closer the dots hug the line, the more that factor explains. R² is the share of the variation it accounts for.",
    ("turtle", 1): "A decision tree predicts points by asking a chain of yes/no questions. Too shallow misses patterns; too deep memorises noise. Watch the test line to find the sweet spot, then try a customer of your own.",
    ("turtle", 2): "Segmentation groups similar customers. The two curves help choose how many groups; the table then says who each group is and how many points they earn.",
    ("turtle", 3): "Every review is scored from -1 (negative) to +1 (positive). See the words customers use most, the happiest and unhappiest reviews, and search for any topic.",
    ("churn", 0): "Each bar is the share of customers in a group who left. Bars above the dashed line churn faster than the company average.",
    ("churn", 1): "Not every customer is worth the same. These views show how much lifetime value sits in each tier, and where the risk is concentrated.",
    ("churn", 2): "A good model ranks customers by risk. Compare the models, then see how much of the churn is caught by contacting only the riskiest customers.",
    ("churn", 3): "Set what a lost customer costs and what a retention offer costs. The chart shows which targeting strategy saves the most money, not which is most accurate.",
    ("nhs", 0): "Monthly appointment volume, with the first lockdown marked. Switch the view to see how the mix of consultation modes shifted.",
    ("nhs", 1): "Does demand ever exceed a planning benchmark? Move the threshold to see which months go over.",
    ("nhs", 2): "Who is seen, how, and by whom: missed appointments, healthcare professionals and appointment timing.",
    ("nhs", 3): "Where care is delivered and for what: service settings and clinical categories, with and without General Practice.",
    ("nhs", 4): "What the public said on Twitter about the NHS: the most common hashtags in the sampled posts.",
    ("market", 0): "Who the customers are. Open the filter to slice by country, education and marital status; every chart updates.",
    ("market", 1): "Which product categories carry spend, split by any customer attribute.",
    ("market", 2): "RFM scores every customer on recency, frequency and monetary value. Compare each segment's share of customers with its share of spend.",
    ("market", 3): "Did customers who responded to an ad channel spend more? Compare responders with non-responders, channel by channel.",
    ("bank", 0): "Where the money goes: spend by month, category and merchant.",
    ("bank", 1): "Change the thresholds and rules and watch the flagged transactions change. Download the flagged list as a CSV.",
    ("bank", 2): "Why each rule exists, and what it can and cannot catch.",
    ("option", 0): "As the binomial tree gets more steps and the Monte Carlo more paths, both should settle on the Black-Scholes price. Change the inputs on the left to test it.",
    ("option", 1): "Which input moves the option price most? See the effect of a change in each, plus the Greeks.",
    ("option", 2): "The payoff at expiry against the value today, as the spot price moves.",
}


def tab_note(key: str, i: int) -> None:
    t = TAB_NOTES.get((key, i))
    if t:
        lead(t)


# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
def page_overview() -> None:
    info = meta()
    tiles = ""
    for i, k in enumerate(ORDER):
        p = PROJECTS[k]
        tags = "".join(f"<span>{x}</span>" for x in p["tools"])
        tiles += f"""<a class="tile" href="?project={k}" target="_self" style="--c:{p['accent']};--i:{i}">
<div class="bar"></div><div class="ico"><svg viewBox="0 0 24 24">{ICONS[k]}</svg></div>
<h3>{p['name']}</h3><p class="q">{p['tag']}</p><p class="res">{info[k]['short']}</p>
<div class="tags">{tags}</div>
<span class="go">Open project <svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg></span></a>"""
    st.markdown(
        f"""<style>:root {{ --accent: {BLUE}; }}</style>
<section class="hero">
  <div class="hero-top"><span class="chip">Chartered Accountant</span><span class="chip">New Delhi</span><span class="chip live">6 projects</span></div>
  <div class="hero-title">From raw data to a decision</div>
  <p class="hero-sub">Six analytics projects across retail, telecom, healthcare, marketing, banking and derivatives. Pick one to explore: every chart is interactive and every number is traceable to its data.</p>
</section>
<div class="grid">{tiles}</div>""", unsafe_allow_html=True)

    legend()
    st.markdown("## The portfolio at a glance")
    lead("A first look at the data behind three of the projects. Open any project for the full story.")
    c1, c2, c3 = st.columns(3)
    t = load_turtle()
    with c1:
        fig = px.scatter(t, x="spending_score", y="loyalty_points", color="loyalty_points", color_continuous_scale=["#CFE9DB", GREEN, INK], opacity=.7)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title="Spending score"); fig.update_yaxes(title="Loyalty points")
        show(fig, 300, "Turtle Games: spend drives points")
    with c2:
        ar = load_ar()
        modes = ar.pivot_table(index="appointment_month", columns="appointment_mode", values="count_of_appointments", aggfunc="sum")
        share = (modes.div(modes.sum(axis=1), axis=0) * 100).drop(columns=[c for c in ["Unknown"] if c in modes.columns])
        d = share.reset_index().melt("appointment_month", var_name="Mode", value_name="%")
        fig = px.area(d, x="appointment_month", y="%", color="Mode")
        fig.update_xaxes(title="", nticks=4, tickangle=0); fig.update_yaxes(title="% of month")
        show(fig, 300, "NHS: how care is delivered")
    with c3:
        Ks = np.linspace(1000, 1800, 60)
        fig = go.Figure()
        fig.add_scatter(x=Ks, y=np.maximum(1400 - Ks, 0) * 0 + bs_call(1400, Ks, 1, 0.065, 0.28), name="Call price today", line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(20,163,163,.12)")
        fig.update_xaxes(title="Strike"); fig.update_yaxes(title="Option price")
        show(fig, 300, "Options: price falls as strike rises")

    st.markdown("## What I bring")
    st.markdown(
        """<div class="story">
<div class="sc" style="--i:0"><h4>Finance first</h4><p>Chartered Accountant training means I check numbers before I trust them: reconciling, testing assumptions and asking what a result is worth in money.</p></div>
<div class="sc" style="--i:1"><h4>Analysis end to end</h4><ul><li>Cleaning and SQL</li><li>Statistics and modelling</li><li>Dashboards and storytelling</li></ul></div>
<div class="sc" style="--i:2"><h4>Honest about evidence</h4><p>Every finding is marked: ✓ recomputed live here, or △ carried over from the original notebook. Datasets for banking and options are generated or illustrative.</p></div></div>""",
        unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
ROUTES = {"overview": page_overview, "turtle": page_turtle, "churn": page_churn, "nhs": page_nhs,
          "market": page_market, "bank": page_bank, "option": page_option}
ALIASES = {"lse3": "turtle", "lse4": "churn", "lse2": "nhs", "lse1": "market",
           "banking": "bank", "options": "option", "option-pricing": "option", "home": "overview"}
if "page" not in st.session_state:
    wanted = str(st.query_params.get("project", "overview")).lower()
    wanted = ALIASES.get(wanted, wanted)
    st.session_state["page"] = wanted if wanted in ROUTES else "overview"
if st.query_params.get("project") != st.session_state["page"]:
    st.query_params["project"] = st.session_state["page"]

navbar(st.session_state["page"])
ROUTES[st.session_state["page"]]()
if st.session_state["page"] != "overview":
    next_project(st.session_state["page"])
