import os
import base64
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

st.set_page_config(page_title="DBSCAN Explorer", layout="wide", page_icon="🔵")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Design tokens ────────────────────────────────────────────── */
:root {
  --bg-base:         #04060d;
  --bg-surface:      #080c16;
  --bg-elevated:     #0c1220;

  --accent-cyan:     #00c8ff;
  --accent-cyan-dim: #0094c0;
  --accent-amber:    #f5a623;

  --text-primary:    #dce8f5;
  --text-secondary:  #7a8fa8;
  --text-muted:      #3a4a5e;
  --text-faint:      #1e2a38;

  --border-subtle:   rgba(0,200,255,0.07);
  --border-normal:   rgba(0,200,255,0.13);
  --border-active:   rgba(0,200,255,0.32);

  --success: #3ecf8e;
  --warn:    #f5a623;
  --error:   #f26c6c;

  --font-display: 'DM Serif Display', Georgia, serif;
  --font-mono:    'JetBrains Mono', 'Courier New', monospace;
  --font-body:    'DM Sans', system-ui, sans-serif;

  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;

  --shadow-glow-cyan:  0 0 28px rgba(0,200,255,0.08);
  --shadow-card:       0 4px 32px rgba(0,0,0,0.4);
}

/* ── Reset & base ─────────────────────────────────────────────── */
html, body { background: var(--bg-base) !important; margin: 0 !important; padding: 0 !important; }

#root, .stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
[data-testid="stHeader"],
section.main, .main, .main > div,
.stApp > div, .stApp > div > div,
.stApp > div > div > div,
.stApp > div > div > section,
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewContainer"] > section {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
}

[data-testid="stHeader"] { display: none !important; }

.block-container {
  padding: 2.25rem 3rem !important;
  max-width: 1480px !important;
  background: var(--bg-base) !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
  color: var(--text-secondary) !important;
  font-size: 11px !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; margin: 1.1rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
  background: rgba(0,200,255,0.03) !important;
  border: 1px solid var(--border-normal) !important;
  border-radius: var(--radius-sm) !important;
  transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div:hover {
  border-color: var(--border-active) !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div { background: rgba(0,200,255,0.1) !important; }
[data-testid="stSlider"] > div > div > div > div {
  background: linear-gradient(135deg, var(--accent-cyan), var(--accent-cyan-dim)) !important;
  box-shadow: 0 0 8px rgba(0,200,255,0.4) !important;
}

/* Sidebar section label */
.sidebar-section {
  font-family: var(--font-mono) !important;
  font-size: 8.5px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.26em !important;
  color: var(--accent-cyan) !important;
  margin: 1.6rem 0 0.7rem 0 !important;
  display: flex !important; align-items: center !important; gap: 10px !important;
}
.sidebar-section::before {
  content: '' !important;
  display: inline-block !important;
  width: 3px !important; height: 12px !important;
  background: linear-gradient(180deg, var(--accent-cyan), transparent) !important;
  border-radius: 2px !important; flex-shrink: 0 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-testid="stTab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--text-muted) !important;
  font-size: 11px !important; font-weight: 500 !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.07em !important;
  padding: 0.65rem 1.5rem !important;
  text-transform: uppercase !important;
  transition: color 0.2s, border-color 0.2s !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  background: transparent !important;
  color: var(--accent-cyan) !important;
  border-bottom: 2px solid var(--accent-cyan) !important;
}
[data-testid="stTabs"] [data-testid="stTab"]:hover { color: var(--text-secondary) !important; }
[data-testid="stTabPanel"] {
  background: transparent !important;
  border: none !important;
  border-top: 1px solid var(--border-subtle) !important;
  padding: 1.8rem 0 !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
  background: transparent !important;
  color: var(--accent-cyan) !important;
  border: 1px solid var(--border-active) !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.55rem 1.4rem !important;
  font-weight: 500 !important; font-size: 11px !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  width: 100% !important;
  transition: background 0.2s, box-shadow 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
  background: rgba(0,200,255,0.06) !important;
  border-color: var(--accent-cyan) !important;
  box-shadow: 0 0 24px rgba(0,200,255,0.12), inset 0 0 12px rgba(0,200,255,0.04) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Metric cards ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(0,200,255,0.05) 0%, rgba(0,200,255,0.01) 60%, transparent 100%) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  padding: 1.1rem 1.25rem !important;
  position: relative !important;
  overflow: hidden !important;
  transition: border-color 0.3s, box-shadow 0.3s !important;
  box-shadow: var(--shadow-card) !important;
}
[data-testid="stMetric"]:hover {
  border-color: var(--border-normal) !important;
  box-shadow: var(--shadow-glow-cyan), var(--shadow-card) !important;
}
[data-testid="stMetric"]::before {
  content: '' !important;
  position: absolute !important;
  top: 0 !important; left: -100% !important;
  width: 100% !important; height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent) !important;
  animation: shimmer 3.5s ease-in-out infinite !important;
  opacity: 0.55 !important;
}
@keyframes shimmer {
  0%   { left: -100%; opacity: 0; }
  20%  { opacity: 0.55; }
  80%  { opacity: 0.55; }
  100% { left: 100%; opacity: 0; }
}
[data-testid="stMetric"]::after {
  content: '' !important;
  position: absolute !important;
  top: 0 !important; right: 0 !important;
  width: 48px !important; height: 48px !important;
  background: radial-gradient(circle at top right, rgba(0,200,255,0.07), transparent 70%) !important;
  pointer-events: none !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 8.5px !important;
  color: var(--text-muted) !important;
  font-weight: 700 !important; text-transform: uppercase !important;
  letter-spacing: 0.2em !important;
  font-family: var(--font-mono) !important;
}
[data-testid="stMetricValue"] {
  font-size: 30px !important; font-weight: 400 !important;
  color: var(--text-primary) !important;
  font-family: var(--font-display) !important;
  font-style: italic !important;
  line-height: 1.15 !important;
  letter-spacing: -0.01em !important;
}

/* ── Status badges ────────────────────────────────────────────── */
.status-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 16px; border-radius: var(--radius-sm);
  font-size: 10.5px; font-weight: 500; margin-top: 0.9rem;
  font-family: var(--font-mono); letter-spacing: 0.06em;
  text-transform: uppercase;
}
.status-badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.badge-good {
  background: rgba(62,207,142,0.05); color: var(--success);
  border: 1px solid rgba(62,207,142,0.18);
}
.badge-good::before {
  background: var(--success);
  animation: pulse-dot 2.2s ease-out infinite;
}
@keyframes pulse-dot {
  0%   { box-shadow: 0 0 0 0 rgba(62,207,142,0.45); }
  60%  { box-shadow: 0 0 0 6px rgba(62,207,142,0); }
  100% { box-shadow: 0 0 0 0 rgba(62,207,142,0); }
}
.badge-warn {
  background: rgba(245,166,35,0.05); color: var(--warn);
  border: 1px solid rgba(245,166,35,0.18);
}
.badge-warn::before { background: var(--warn); box-shadow: 0 0 7px rgba(245,166,35,0.5); }
.badge-bad {
  background: rgba(242,108,108,0.05); color: var(--error);
  border: 1px solid rgba(242,108,108,0.18);
}
.badge-bad::before { background: var(--error); box-shadow: 0 0 7px rgba(242,108,108,0.5); }

/* ── Tour card ────────────────────────────────────────────────── */
.tour-card {
  background: linear-gradient(135deg, rgba(0,200,255,0.04) 0%, rgba(0,0,0,0) 70%);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 1.5rem 1.8rem;
  margin-bottom: 1.1rem;
  position: relative; overflow: hidden;
}
.tour-card::before {
  content: '';
  position: absolute; top: 0; left: 0;
  width: 2px; height: 100%;
  background: linear-gradient(180deg, var(--accent-cyan), var(--accent-amber) 50%, transparent);
  animation: border-flow 4s ease-in-out infinite alternate;
}
@keyframes border-flow {
  0%   { opacity: 0.6; background: linear-gradient(180deg, var(--accent-cyan) 0%, transparent 100%); }
  100% { opacity: 1; background: linear-gradient(180deg, transparent 0%, var(--accent-cyan) 40%, var(--accent-amber) 100%); }
}
.tour-card::after {
  content: '';
  position: absolute; top: -60px; right: -60px;
  width: 160px; height: 160px;
  background: radial-gradient(circle, rgba(0,200,255,0.04) 0%, transparent 70%);
  pointer-events: none;
}
.tour-step-label {
  font-family: var(--font-mono);
  font-size: 8.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.26em; color: var(--accent-amber); margin-bottom: 10px;
}
.tour-title {
  font-size: 19px; font-weight: 400; color: var(--text-primary);
  margin-bottom: 10px; font-family: var(--font-display);
  font-style: italic; line-height: 1.28;
}
.tour-desc { font-size: 13px; color: var(--text-secondary); line-height: 1.75; }

/* ── Info box ─────────────────────────────────────────────────── */
.info-box {
  background: linear-gradient(135deg, rgba(245,166,35,0.04) 0%, transparent 100%);
  border: 1px solid rgba(245,166,35,0.1);
  border-left: 2px solid rgba(245,166,35,0.5);
  border-radius: var(--radius-md); padding: 13px 18px;
  margin-bottom: 20px;
  display: flex; align-items: center; gap: 14px;
}
.info-box-icon { font-size: 15px; flex-shrink: 0; }
.info-box-text { font-size: 12.5px; color: var(--text-secondary); }
.info-box-link {
  font-family: var(--font-mono);
  font-size: 10px; color: var(--accent-amber); margin-left: 8px;
  letter-spacing: 0.04em;
}

/* ── Expander ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: rgba(0,200,255,0.02) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--radius-md) !important;
  transition: border-color 0.2s !important;
}
[data-testid="stExpander"]:hover { border-color: var(--border-normal) !important; }
[data-testid="stExpander"] summary {
  color: var(--text-muted) !important;
  font-size: 11px !important;
  font-family: var(--font-mono) !important;
  letter-spacing: 0.05em !important;
}
[data-testid="stExpander"] p,
[data-testid="stExpander"] li {
  color: var(--text-secondary) !important;
  font-size: 13px !important;
  line-height: 1.75 !important;
}
[data-testid="stExpander"] strong { color: var(--text-primary) !important; }

hr { border-color: var(--border-subtle) !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

/* ── Explainer frame ──────────────────────────────────────────── */
.explainer-frame {
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: var(--bg-surface);
  box-shadow: var(--shadow-card);
}

/* ── Section divider ──────────────────────────────────────────── */
.section-divider {
  height: 1px;
  background: linear-gradient(90deg, var(--accent-cyan), rgba(0,200,255,0.06) 40%, transparent);
  margin: 0 0 30px 0;
  opacity: 0.25;
}

/* ═══════════════════════════════════════════
   LANDING PAGE
═══════════════════════════════════════════ */
.landing-bg {
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 60% 50% at 15% 15%,  rgba(0,200,255,0.05) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 85% 80%,  rgba(245,166,35,0.04) 0%, transparent 55%),
    radial-gradient(ellipse 70% 60% at 50% 50%,  rgba(0,100,180,0.03) 0%, transparent 70%);
}
.landing-bg::before {
  content: '';
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(0,200,255,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,200,255,0.04) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse 85% 85% at 50% 50%, black 0%, transparent 100%);
}
.landing-bg::after {
  content: '';
  position: absolute;
  width: 500px; height: 500px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,200,255,0.03) 0%, transparent 70%);
  top: 50%; left: 50%; transform: translate(-50%, -50%);
  animation: orb-breathe 8s ease-in-out infinite alternate;
}
@keyframes orb-breathe {
  0%   { transform: translate(-50%, -50%) scale(0.85); opacity: 0.5; }
  100% { transform: translate(-50%, -52%) scale(1.15); opacity: 1; }
}

.landing-hero {
  text-align: center;
  padding: 80px 20px 60px;
  position: relative; z-index: 1;
}
.landing-eyebrow {
  font-family: var(--font-mono);
  font-size: 9.5px; font-weight: 500; letter-spacing: 0.3em;
  text-transform: uppercase; color: var(--accent-amber);
  margin-bottom: 22px;
  display: inline-flex; align-items: center; gap: 14px;
  opacity: 0;
  animation: fade-up 0.8s 0.1s ease-out forwards;
}
.landing-eyebrow::before, .landing-eyebrow::after {
  content: ''; display: inline-block;
  width: 28px; height: 1px; background: rgba(245,166,35,0.35);
}
@keyframes fade-up {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
.landing-title {
  font-family: var(--font-display);
  font-size: 88px; font-weight: 400; color: var(--text-primary);
  letter-spacing: -0.025em; line-height: 0.9;
  margin-bottom: 6px;
  opacity: 0;
  animation: fade-up 0.8s 0.25s ease-out forwards;
}
.landing-title .accent {
  color: var(--accent-cyan);
  font-style: italic;
  position: relative;
}
.landing-title .accent::after {
  content: '';
  position: absolute; bottom: -6px; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, var(--accent-cyan), transparent);
  opacity: 0.45;
}
.landing-tagline {
  font-family: var(--font-mono);
  font-size: 10px; color: var(--text-muted);
  letter-spacing: 0.15em; text-transform: uppercase;
  margin-bottom: 20px; margin-top: 16px;
  opacity: 0;
  animation: fade-up 0.8s 0.4s ease-out forwards;
}
.landing-sub {
  font-size: 15.5px; color: var(--text-secondary); line-height: 1.72;
  max-width: 480px; margin: 0 auto 68px;
  font-weight: 300;
  opacity: 0;
  animation: fade-up 0.8s 0.55s ease-out forwards;
}

/* Mode cards */
.mode-card {
  background: linear-gradient(160deg, rgba(255,255,255,0.02) 0%, rgba(0,0,0,0) 100%);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: var(--radius-lg);
  padding: 44px 38px 34px;
  text-align: left;
  transition: border-color 0.35s, transform 0.35s, box-shadow 0.35s;
  cursor: pointer; height: 100%;
  position: relative; overflow: hidden;
  opacity: 0;
  animation: fade-up 0.8s 0.7s ease-out forwards;
}
.mode-card::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(160deg, rgba(0,200,255,0.04) 0%, transparent 60%);
  opacity: 0; transition: opacity 0.35s;
}
.mode-card:hover {
  border-color: rgba(0,200,255,0.22);
  transform: translateY(-4px);
  box-shadow: 0 24px 64px rgba(0,0,0,0.5), 0 0 40px rgba(0,200,255,0.05);
}
.mode-card:hover::before { opacity: 1; }

.mode-card-amber::before {
  background: linear-gradient(160deg, rgba(245,166,35,0.04) 0%, transparent 60%) !important;
}
.mode-card-amber:hover {
  border-color: rgba(245,166,35,0.22) !important;
  box-shadow: 0 24px 64px rgba(0,0,0,0.5), 0 0 40px rgba(245,166,35,0.05) !important;
}

.mode-card-number {
  font-family: var(--font-mono);
  font-size: 9px; font-weight: 500;
  color: var(--text-faint); letter-spacing: 0.18em;
  margin-bottom: 22px; display: block;
}
.mode-card-icon { font-size: 26px; margin-bottom: 20px; display: block; }
.mode-card-title {
  font-family: var(--font-display);
  font-size: 24px; font-weight: 400; color: var(--text-primary);
  margin-bottom: 14px; line-height: 1.18;
}
.mode-card-desc {
  font-size: 13.5px; color: var(--text-secondary); line-height: 1.72;
  margin-bottom: 26px;
}
.mode-card-tag {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--font-mono);
  font-size: 9px; font-weight: 500; letter-spacing: 0.16em;
  text-transform: uppercase; padding: 5px 12px;
  border-radius: 2px; color: var(--accent-cyan);
  background: rgba(0,200,255,0.05); border: 1px solid rgba(0,200,255,0.15);
}
.mode-card-tag::before { content: ''; width: 4px; height: 4px; border-radius: 50%; background: var(--accent-cyan); opacity: 0.7; }
.mode-card-tag-amber {
  color: var(--accent-amber) !important;
  background: rgba(245,166,35,0.05) !important;
  border-color: rgba(245,166,35,0.15) !important;
}
.mode-card-tag-amber::before { background: var(--accent-amber) !important; }

.landing-footer {
  text-align: center; margin-top: 68px;
  font-family: var(--font-mono);
  font-size: 9.5px; color: var(--text-faint); letter-spacing: 0.14em;
  text-transform: uppercase;
  display: flex; align-items: center; justify-content: center; gap: 18px;
  opacity: 0;
  animation: fade-up 0.8s 0.9s ease-out forwards;
}
.landing-footer::before, .landing-footer::after {
  content: ''; flex: 1; max-width: 70px; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,200,255,0.07));
}
.landing-footer::after { transform: scaleX(-1); }

/* ═══════════════════════════════════════════
   PAGE HEADER
═══════════════════════════════════════════ */
.page-header {
  display: flex; align-items: baseline; gap: 0; margin-bottom: 0;
}
.page-header-title {
  font-family: var(--font-display);
  font-size: 24px; font-weight: 400; color: var(--text-primary);
  letter-spacing: -0.01em; font-style: italic;
}
.page-header-badge {
  font-family: var(--font-mono);
  font-size: 8.5px; font-weight: 500;
  color: var(--accent-amber); background: rgba(245,166,35,0.06);
  border: 1px solid rgba(245,166,35,0.18); padding: 3px 10px;
  border-radius: 2px; letter-spacing: 0.14em; text-transform: uppercase;
  margin-left: 16px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE     = ["#00c8ff","#f5a623","#3ecf8e","#f26c6c","#a78bfa","#06b6d4","#fbbf24","#f472b6","#2dd4bf","#818cf8"]
NOISE_COLOR = "#1e2a3a"
PLOT_BG     = "#04060d"
GRID_COL    = "#0a0e1a"
TICK_COL    = "#2a3a50"
FONT_COL    = "#dce8f5"

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {"page": "landing", "show_clustered": False, "tour_step": 0}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def get_data(dataset, n, noise):
    if dataset == "Moons":
        X, _ = make_moons(n_samples=n, noise=noise * 2.5, random_state=42)
    elif dataset == "Blobs":
        X, _ = make_blobs(n_samples=n, centers=4, cluster_std=noise * 5, random_state=42)
    elif dataset == "Concentric Circles":
        X, _ = make_circles(n_samples=n, noise=noise, factor=0.4, random_state=42)
    else:
        rng = np.random.default_rng(42)
        X = rng.uniform(-2, 2, size=(n, 2))
    return StandardScaler().fit_transform(X)

def run_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    core_mask = np.zeros_like(labels, dtype=bool)
    core_mask[db.core_sample_indices_] = True
    return labels, core_mask

def run_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    return km.labels_, km.cluster_centers_

def scatter_traces(X, labels, core_mask, point_size, palette=PALETTE):
    traces = []
    for label in sorted(set(labels)):
        if label == -1:
            mask = labels == -1
            traces.append(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers",
                marker=dict(size=point_size - 1, color=NOISE_COLOR, opacity=0.55,
                            symbol="x", line=dict(width=1.5, color="#2a3a50")),
                name="Noise",
                hovertemplate="Noise — x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
            ))
        else:
            color = palette[label % len(palette)]
            cmask  = labels == label
            border = cmask & ~core_mask
            core   = cmask & core_mask
            if border.any():
                traces.append(go.Scatter(
                    x=X[border, 0], y=X[border, 1], mode="markers",
                    marker=dict(size=point_size - 1, color=color, opacity=0.2,
                                line=dict(width=1, color=color)),
                    name=f"Cluster {label + 1} (border)",
                    legendgroup=f"c{label}", showlegend=False,
                    hovertemplate=f"Cluster {label+1} border<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
                ))
            if core.any():
                traces.append(go.Scatter(
                    x=X[core, 0], y=X[core, 1], mode="markers",
                    marker=dict(size=point_size + 1, color=color, opacity=0.92,
                                line=dict(width=0.5, color="rgba(255,255,255,0.12)")),
                    name=f"Cluster {label + 1}",
                    legendgroup=f"c{label}",
                    hovertemplate=f"Cluster {label+1} core<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
                ))
    return traces

def base_layout(title, height=490):
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(family="'JetBrains Mono', monospace", color=FONT_COL, size=10),
        margin=dict(l=44, r=24, t=50, b=38),
        height=height,
        title=dict(text=title, font=dict(size=10, color="#3a4a5e", family="'JetBrains Mono', monospace"),
                   x=0.012, xanchor="left"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(color=TICK_COL, size=9),
                   title=dict(text="Feature 1", font=dict(color=TICK_COL, size=9))),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(color=TICK_COL, size=9),
                   title=dict(text="Feature 2", font=dict(color=TICK_COL, size=9))),
        legend=dict(bgcolor="rgba(4,6,13,0.92)", bordercolor="rgba(0,200,255,0.1)", borderwidth=1,
                    font=dict(size=10, color=FONT_COL), itemsizing="constant",
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hoverlabel=dict(bgcolor="#080c16", bordercolor="rgba(0,200,255,0.18)",
                        font=dict(size=11, color="#dce8f5", family="'DM Sans', sans-serif"))
    )

# ── HTML explainer loader ──────────────────────────────────────────────────────
@st.cache_data
def load_explainer_html():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "dbscan_explainer.html")
    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), "dbscan_explainer.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<p style='color:#f26c6c;padding:2rem'>dbscan_explainer.html not found</p>"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LANDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    st.markdown('<div class="landing-bg"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-hero">
      <div class="landing-eyebrow">Density · Clustering · Machine Learning</div>
      <div class="landing-title">DB<span class="accent">SCAN</span></div>
      <div class="landing-tagline">Density-Based Spatial Clustering of Applications with Noise</div>
      <div class="landing-sub">
        An interactive environment for understanding how DBSCAN discovers clusters,
        classifies noise, and outperforms centroid-based methods on complex shapes.
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="mode-card">
          <span class="mode-card-number">01 / 02</span>
          <span class="mode-card-icon">⚡</span>
          <div class="mode-card-title">Interactive<br>Dashboard</div>
          <div class="mode-card-desc">
            Tune ε and MinPts live on real datasets.
            Compare DBSCAN against K-Means side by side.
            Follow a guided tour to build parameter intuition.
          </div>
          <span class="mode-card-tag">Explore · Compare · Tune</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if st.button("Open Dashboard →", key="go_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()

    with right:
        st.markdown("""
        <div class="mode-card mode-card-amber">
          <span class="mode-card-number">02 / 02</span>
          <span class="mode-card-icon">🔍</span>
          <div class="mode-card-title">Step-by-Step<br>Explainer</div>
          <div class="mode-card-desc">
            Walk through how DBSCAN classifies every point —
            core, border, noise — one step at a time.
            Animated diagrams and Python code included.
          </div>
          <span class="mode-card-tag mode-card-tag-amber">Learn · Visualize · Understand</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if st.button("Open Explainer →", key="go_explainer"):
            st.session_state.page = "explainer"
            st.rerun()

    st.markdown("""
    <div class="landing-footer">Ester, Kriegel, Sander, Xu · KDD 1996</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "explainer":

    hcol1, hcol2 = st.columns([1, 6])
    with hcol1:
        if st.button("← Back", key="back_explainer"):
            st.session_state.page = "landing"
            st.rerun()
    with hcol2:
        st.markdown("""
        <div class="page-header">
          <span class="page-header-title">Step-by-Step Explainer</span>
          <span class="page-header-badge">15 steps</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider' style='margin-top:16px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <p style="color:var(--text-secondary,#7a8fa8);font-size:13.5px;line-height:1.78;margin-bottom:20px;max-width:720px;font-family:'DM Sans',sans-serif;">
      Walk through how DBSCAN classifies every point — step by step. Use
      <strong style="color:#dce8f5">Next / Prev</strong> or
      <strong style="color:#dce8f5">← → arrow keys</strong> to advance.
      The final three steps cover direct density-reachability, density-reachability, and density-connectivity.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <span class="info-box-icon">⚡</span>
      <div>
        <span class="info-box-text">Want to tune parameters and see DBSCAN run on real datasets?</span>
        <span class="info-box-link">→ Switch to the Interactive Dashboard</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    xcol, _ = st.columns([1, 5])
    with xcol:
        if st.button("Open Dashboard →", key="explainer_to_dash"):
            st.session_state.page = "dashboard"
            st.rerun()

    html_content = load_explainer_html()

    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DBSCAN.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(os.getcwd(), "DBSCAN.mp4")
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        html_content = html_content.replace('src="DBSCAN.mp4"', f'src="data:video/mp4;base64,{video_b64}"')

    st.markdown("<div class='explainer-frame'>", unsafe_allow_html=True)
    components.html(html_content, height=4000, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sidebar-section">Dataset</p>', unsafe_allow_html=True)
        dataset     = st.selectbox("Dataset", ["Moons","Blobs","Concentric Circles","Random Noise"],
                                   label_visibility="collapsed")
        n_points    = st.slider("Points", 100, 600, 300, step=50)
        noise_level = st.slider("Noise level", 0.01, 0.15, 0.08, step=0.01)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">DBSCAN</p>', unsafe_allow_html=True)
        eps         = st.slider("ε  neighborhood radius", 0.05, 1.5, 0.35, step=0.01)
        min_samples = st.slider("MinPts  core threshold", 1, 20, 5, step=1)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">K-Means</p>', unsafe_allow_html=True)
        k_clusters  = st.slider("K  cluster count", 2, 8, 2, step=1)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">Display</p>', unsafe_allow_html=True)
        show_epsilon_circles = st.checkbox("Show ε circles", value=False)
        point_size  = st.slider("Point size", 4, 16, 7)

    # ── Page header ───────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([1, 6])
    with hcol1:
        if st.button("← Back", key="back_dashboard"):
            st.session_state.page = "landing"
            st.rerun()
    with hcol2:
        st.markdown("""
        <div class="page-header">
          <span class="page-header-title">Interactive Dashboard</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider' style='margin-top:16px'></div>", unsafe_allow_html=True)

    # ── Tour steps ────────────────────────────────────────────────────────────
    TOUR_STEPS = [
        {"title": "Start: Raw Data",
         "desc": "Your dataset with no clustering applied. Two interleaved moons — how would you separate them by hand?",
         "dataset": "Moons", "eps": 0.35, "min_samples": 5, "clustered": False, "k": 2},
        {"title": "ε Too Small — Everything is Noise",
         "desc": "With ε = 0.05, almost no point has enough neighbors. The algorithm sees nothing but outliers.",
         "dataset": "Moons", "eps": 0.05, "min_samples": 5, "clustered": True, "k": 2},
        {"title": "ε Too Large — Everything Merges",
         "desc": "With ε = 1.2, every neighborhood swallows the entire dataset. Both moons collapse into one cluster.",
         "dataset": "Moons", "eps": 1.2, "min_samples": 5, "clustered": True, "k": 2},
        {"title": "Sweet Spot — Clean Separation",
         "desc": "ε = 0.35, MinPts = 5 finds exactly two clusters following the natural moon shapes.",
         "dataset": "Moons", "eps": 0.35, "min_samples": 5, "clustered": True, "k": 2},
        {"title": "MinPts Too High — Sparse Points Become Noise",
         "desc": "Raising MinPts to 15 makes the algorithm stricter. Border regions get classified as noise.",
         "dataset": "Moons", "eps": 0.35, "min_samples": 15, "clustered": True, "k": 2},
        {"title": "Where K-Means Fails",
         "desc": "K-Means assumes spherical clusters and cuts with straight boundaries — it cannot follow moon shapes.",
         "dataset": "Moons", "eps": 0.35, "min_samples": 5, "clustered": True, "k": 2},
        {"title": "Concentric Circles — Another K-Means Failure",
         "desc": "K-Means splits circles vertically. DBSCAN correctly identifies inner and outer rings.",
         "dataset": "Concentric Circles", "eps": 0.25, "min_samples": 5, "clustered": True, "k": 2},
    ]

    tab1, tab2, tab3 = st.tabs(["DBSCAN Explorer", "DBSCAN vs K-Means", "Guided Tour"])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        X = get_data(dataset, n_points, noise_level)
        labels, core_mask = run_dbscan(X, eps, min_samples)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int(np.sum(labels == -1))
        n_core     = int(np.sum(core_mask))
        n_border   = int(np.sum((labels != -1) & ~core_mask))

        btn_col, xlink_col, _ = st.columns([1, 2, 4])
        with btn_col:
            btn_label = "▶  Run DBSCAN" if not st.session_state.show_clustered else "◀  Show Raw"
            if st.button(btn_label, key="toggle_btn"):
                st.session_state.show_clustered = not st.session_state.show_clustered
        with xlink_col:
            st.markdown("""
            <div style="padding-top:8px">
              <span style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#2a3a50;letter-spacing:0.04em;">
                New to DBSCAN?
              </span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("→ Step-by-Step Explainer", key="tab1_to_explainer"):
                st.session_state.page = "explainer"
                st.rerun()

        show_clustered = st.session_state.show_clustered

        if show_clustered:
            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clusters",      n_clusters)
            c2.metric("Core points",   n_core)
            c3.metric("Border points", n_border)
            c4.metric("Noise points",  n_noise)

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

        fig = go.Figure()
        if not show_clustered:
            fig.add_trace(go.Scatter(
                x=X[:, 0], y=X[:, 1], mode="markers",
                marker=dict(size=point_size, color="#12192a", opacity=0.82,
                            line=dict(width=0.5, color="#1e2a3a")),
                name="Raw data",
                hovertemplate="x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
            ))
        else:
            for t in scatter_traces(X, labels, core_mask, point_size):
                fig.add_trace(t)
            if show_epsilon_circles:
                core_indices = np.where(core_mask)[0]
                sample = core_indices[::max(1, len(core_indices) // 8)][:8]
                theta = np.linspace(0, 2 * np.pi, 60)
                for idx in sample:
                    cx, cy = X[idx, 0], X[idx, 1]
                    lbl    = labels[idx]
                    col    = PALETTE[lbl % len(PALETTE)] if lbl >= 0 else NOISE_COLOR
                    fig.add_trace(go.Scatter(
                        x=cx + eps * np.cos(theta), y=cy + eps * np.sin(theta),
                        mode="lines", line=dict(color=col, width=1, dash="dot"),
                        opacity=0.14, showlegend=False, hoverinfo="skip"
                    ))

        subtitle = (
            f"  ·  <b>{n_clusters} cluster{'s' if n_clusters!=1 else ''}</b>  ·  {n_noise} noise"
            if show_clustered else "  ·  raw"
        )
        layout = base_layout(f"{dataset}   ε={eps:.2f}   MinPts={min_samples}" + subtitle)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        if show_clustered:
            if n_clusters == 0:
                cls, txt = "badge-bad",  "All noise — ε too small or MinPts too high"
            elif n_clusters == 1 and n_noise < 5:
                cls, txt = "badge-warn", "Everything merged — ε is too large"
            elif n_noise > n_points * 0.5:
                cls, txt = "badge-warn", f"{n_noise} noise points ({n_noise/n_points*100:.0f}%) — raise ε or lower MinPts"
            else:
                cls, txt = "badge-good", f"{n_clusters} cluster{'s' if n_clusters!=1 else ''}  ·  {n_noise} noise ({n_noise/n_points*100:.1f}%)"
            st.markdown(f'<span class="status-badge {cls}">{txt}</span>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        with st.expander("How do these parameters work?"):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("""
**ε (epsilon) — neighborhood radius**
- Too small → everything is noise
- Too large → everything merges
- Sweet spot → natural dense regions emerge
                """)
            with cb:
                st.markdown("""
**MinPts — core point threshold**
- Low → sparse regions form clusters
- High → only dense regions qualify
- Rule of thumb: MinPts ≥ dimensions + 1
                """)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        X2 = get_data(dataset, n_points, noise_level)
        db_labels, db_core = run_dbscan(X2, eps, min_samples)
        km_labels, km_centers = run_kmeans(X2, k_clusters)

        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["DBSCAN", f"K-Means  (k={k_clusters})"],
            horizontal_spacing=0.06
        )
        for t in scatter_traces(X2, db_labels, db_core, point_size):
            t.showlegend = False
            fig2.add_trace(t, row=1, col=1)

        km_pal = ["#f5a623","#00c8ff","#3ecf8e","#f26c6c","#a78bfa","#06b6d4","#fbbf24","#f472b6"]
        for k in range(k_clusters):
            mask  = km_labels == k
            color = km_pal[k % len(km_pal)]
            fig2.add_trace(go.Scatter(
                x=X2[mask, 0], y=X2[mask, 1], mode="markers",
                marker=dict(size=point_size, color=color, opacity=0.68),
                name=f"K-Means cluster {k+1}", showlegend=False,
                hovertemplate=f"K-Means cluster {k+1}<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
            ), row=1, col=2)
        fig2.add_trace(go.Scatter(
            x=km_centers[:, 0], y=km_centers[:, 1], mode="markers",
            marker=dict(size=14, color="white", symbol="star",
                        line=dict(width=1.5, color="#374151")),
            name="Centroids", showlegend=False,
            hovertemplate="Centroid<br>x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
        ), row=1, col=2)

        fig2.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            font=dict(family="'JetBrains Mono', monospace", color=FONT_COL, size=10),
            margin=dict(l=44, r=24, t=55, b=38), height=490,
            hoverlabel=dict(bgcolor="#080c16", bordercolor="rgba(0,200,255,0.18)",
                            font=dict(size=11, color="#dce8f5")),
        )
        for ann in fig2.layout.annotations:
            ann.font.color = "#3a4a5e"; ann.font.size = 11
        for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
            fig2.update_layout(**{axis: dict(showgrid=True, gridcolor=GRID_COL,
                                             zeroline=False, tickfont=dict(color=TICK_COL, size=9))})
        st.plotly_chart(fig2, use_container_width=True, key="tab2_fig2")

        if dataset in ["Moons", "Concentric Circles"]:
            st.markdown(
                '<span class="status-badge badge-bad">K-Means fails — assumes convex, equally-sized clusters</span>'
                '&nbsp;&nbsp;'
                '<span class="status-badge badge-good">DBSCAN follows density — shape does not matter</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge badge-warn">Both work on blobs — try Moons or Concentric Circles to see K-Means fail</span>',
                unsafe_allow_html=True
            )

        # Comparison table
        st.markdown("<div style='margin-top:2.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'JetBrains Mono\',monospace;font-size:8.5px;font-weight:700;'
            'text-transform:uppercase;letter-spacing:0.24em;color:#00c8ff;margin-bottom:18px;">'
            '── Algorithm Comparison</p>',
            unsafe_allow_html=True
        )

        COMPARISON_ROWS = [
            ("Clustering Method",        "Density-Based",  "Centroid-Based", None),
            ("Needs Number of Clusters", "No",             "Yes ↑ manual",   "dbscan"),
            ("Outlier Detection",        "Excellent",      "Poor",           "dbscan"),
            ("Arbitrary Shapes",         "Yes",            "No",             "dbscan"),
            ("High-Dimensional Data",    "Poor",           "Better",         "kmeans"),
            ("Noise Sensitivity",        "Low",            "High",           "dbscan"),
            ("Speed",                    "Moderate",       "Fast",           "kmeans"),
        ]

        def cell_style(winner, col):
            if winner == "dbscan" and col == "dbscan": return "color:#3ecf8e;font-weight:600;"
            elif winner == "dbscan" and col == "kmeans": return "color:#f26c6c;font-weight:600;"
            elif winner == "kmeans" and col == "kmeans": return "color:#3ecf8e;font-weight:600;"
            elif winner == "kmeans" and col == "dbscan": return "color:#f26c6c;font-weight:600;"
            return "color:#5a6a7e;"

        rows_html = ""
        for i, (feature, dbscan_val, kmeans_val, winner) in enumerate(COMPARISON_ROWS):
            row_bg = "rgba(0,200,255,0.018)" if i % 2 == 0 else "transparent"
            ds_style = cell_style(winner, "dbscan")
            km_style = cell_style(winner, "kmeans")
            rows_html += (
                f'<tr style="background:{row_bg};border-bottom:1px solid rgba(0,200,255,0.045);">'
                f'<td style="padding:12px 18px;font-size:12.5px;color:#8a9aae;font-family:\'DM Sans\',sans-serif;white-space:nowrap;">{feature}</td>'
                f'<td style="padding:12px 22px;font-size:11px;{ds_style}font-family:\'JetBrains Mono\',monospace;text-align:center;">{dbscan_val}</td>'
                f'<td style="padding:12px 22px;font-size:11px;{km_style}font-family:\'JetBrains Mono\',monospace;text-align:center;">{kmeans_val}</td>'
                f'</tr>'
            )

        table_html = (
            '<div style="border:1px solid rgba(0,200,255,0.07);border-radius:10px;overflow:hidden;">'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead>'
            '<tr style="background:rgba(0,200,255,0.03);border-bottom:1px solid rgba(0,200,255,0.09);">'
            '<th style="padding:13px 18px;text-align:left;font-family:\'JetBrains Mono\',monospace;font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.2em;color:#3a4a5e;">Feature</th>'
            '<th style="padding:13px 22px;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.2em;color:#00c8ff;">DBSCAN</th>'
            '<th style="padding:13px 22px;text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:8.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.2em;color:#f5a623;">K-Means</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
            '<p style="font-size:10px;color:#1e2a38;margin-top:9px;font-family:\'JetBrains Mono\',monospace;letter-spacing:0.06em;">'
            'Green = advantage · Red = disadvantage</p>'
        )
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        with st.expander("Why does K-Means fail on non-convex shapes?"):
            st.markdown("""
K-Means assigns each point to the **nearest centroid** using Euclidean distance —
it can only create straight-line Voronoi boundaries and cannot follow curved or interleaved shapes.

DBSCAN asks: *"are there enough nearby points here?"* — it finds clusters of **any shape**
as long as they are dense enough.

**Key insight:** K-Means defines clusters by *geometry*. DBSCAN defines them by *density*.
            """)

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        step = st.session_state.tour_step
        s    = TOUR_STEPS[step]

        st.markdown("""
        <div class="info-box">
          <span class="info-box-icon">🔍</span>
          <div>
            <span class="info-box-text">Want the step-by-step algorithmic walkthrough?</span>
            <span class="info-box-link">→ Open the Step-by-Step Explainer</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        xcol2, _ = st.columns([1, 5])
        with xcol2:
            if st.button("Open Explainer →", key="tour_to_explainer"):
                st.session_state.page = "explainer"
                st.rerun()

        st.markdown(f"""
        <div class="tour-card">
          <p class="tour-step-label">Step {step+1} of {len(TOUR_STEPS)}</p>
          <p class="tour-title">{s['title']}</p>
          <p class="tour-desc">{s['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        n1, n2, n3 = st.columns([1, 1, 4])
        with n1:
            if st.button("← Prev", key="prev", disabled=(step == 0)):
                st.session_state.tour_step = max(0, step - 1)
                st.rerun()
        with n2:
            if st.button("Next →", key="next", disabled=(step == len(TOUR_STEPS) - 1)):
                st.session_state.tour_step = min(len(TOUR_STEPS) - 1, step + 1)
                st.rerun()

        progress = step / (len(TOUR_STEPS) - 1)
        filled = int(progress * (len(TOUR_STEPS) - 1))
        dots_html = "".join([
            f'<span style="display:inline-block;width:{"8" if i == filled else "6"}px;'
            f'height:{"8" if i == filled else "6"}px;border-radius:50%;'
            f'background:{"#00c8ff" if i <= filled else "rgba(0,200,255,0.1)"};'
            f'box-shadow:{"0 0 8px #00c8ff" if i == filled else "none"};'
            f'margin:0 3px;transition:all 0.35s;vertical-align:middle;"></span>'
            for i in range(len(TOUR_STEPS))
        ])
        st.markdown(f"""
        <div style="margin:0.9rem 0 1.3rem 0;display:flex;align-items:center;gap:12px;">
          {dots_html}
          <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#1e2a38;letter-spacing:0.12em;">
            {step+1} / {len(TOUR_STEPS)}
          </span>
        </div>
        """, unsafe_allow_html=True)

        Xt = get_data(s["dataset"], 300, 0.08)
        tour_labels, tour_core = run_dbscan(Xt, s["eps"], s["min_samples"])
        tour_km_labels, tour_km_centers = run_kmeans(Xt, s["k"])
        tour_n_clusters = len(set(tour_labels)) - (1 if -1 in tour_labels else 0)
        tour_n_noise    = int(np.sum(tour_labels == -1))

        if s["clustered"]:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clusters",   tour_n_clusters)
            m2.metric("Core pts",   int(np.sum(tour_core)))
            m3.metric("Border pts", int(np.sum((tour_labels != -1) & ~tour_core)))
            m4.metric("Noise pts",  tour_n_noise)
            st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

        if step >= 5:
            fig3 = make_subplots(rows=1, cols=2,
                subplot_titles=["DBSCAN", f"K-Means (k={s['k']})"],
                horizontal_spacing=0.06)
            for t in scatter_traces(Xt, tour_labels, tour_core, point_size):
                t.showlegend = False
                fig3.add_trace(t, row=1, col=1)
            for k in range(s["k"]):
                mask  = tour_km_labels == k
                color = ["#f5a623","#00c8ff","#3ecf8e","#f26c6c"][k % 4]
                fig3.add_trace(go.Scatter(
                    x=Xt[mask, 0], y=Xt[mask, 1], mode="markers",
                    marker=dict(size=point_size, color=color, opacity=0.68),
                    showlegend=False,
                    hovertemplate=f"K-Means cluster {k+1}<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
                ), row=1, col=2)
            fig3.add_trace(go.Scatter(
                x=tour_km_centers[:, 0], y=tour_km_centers[:, 1], mode="markers",
                marker=dict(size=14, color="white", symbol="star",
                            line=dict(width=1.5, color="#374151")),
                showlegend=False, hovertemplate="Centroid<extra></extra>"
            ), row=1, col=2)
            fig3.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                font=dict(family="'JetBrains Mono', monospace", color=FONT_COL, size=10),
                margin=dict(l=44, r=24, t=50, b=38), height=440,
                hoverlabel=dict(bgcolor="#080c16", font=dict(size=11, color="#dce8f5"))
            )
            for ann in fig3.layout.annotations:
                ann.font.color = "#3a4a5e"; ann.font.size = 11
            for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
                fig3.update_layout(**{axis: dict(showgrid=True, gridcolor=GRID_COL,
                                                 zeroline=False, tickfont=dict(color=TICK_COL, size=9))})
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3 = go.Figure()
            if not s["clustered"]:
                fig3.add_trace(go.Scatter(
                    x=Xt[:, 0], y=Xt[:, 1], mode="markers",
                    marker=dict(size=point_size, color="#12192a", opacity=0.82,
                                line=dict(width=0.5, color="#1e2a3a")),
                    name="Raw data",
                    hovertemplate="x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
                ))
            else:
                for t in scatter_traces(Xt, tour_labels, tour_core, point_size):
                    fig3.add_trace(t)
            layout3 = base_layout(f"{s['dataset']}   ε={s['eps']:.2f}   MinPts={s['min_samples']}", height=440)
            fig3.update_layout(**layout3)
            st.plotly_chart(fig3, use_container_width=True)

        if s["clustered"]:
            if tour_n_clusters == 0:
                cls, txt = "badge-bad",  "All noise"
            elif tour_n_clusters == 1 and tour_n_noise < 5:
                cls, txt = "badge-warn", "Everything merged"
            else:
                cls, txt = "badge-good", f"{tour_n_clusters} cluster{'s' if tour_n_clusters!=1 else ''}  ·  {tour_n_noise} noise"
            st.markdown(f'<span class="status-badge {cls}">{txt}</span>', unsafe_allow_html=True)