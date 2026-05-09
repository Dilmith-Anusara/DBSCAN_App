import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

st.set_page_config(page_title="DBSCAN Explorer", layout="wide", page_icon="🔵")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body {
  background-color: #0f1117 !important;
  margin: 0 !important;
  padding: 0 !important;
}

#root, .stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
[data-testid="stHeader"],
section.main, .main, .main > div,
.stApp > div,
.stApp > div > div,
.stApp > div > div > div,
.stApp > div > div > section,
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewContainer"] > section {
  background-color: #0f1117 !important;
  color: #e5e7eb !important;
  font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
  display: none !important;
}

.block-container {
  padding: 2rem 2.5rem !important;
  max-width: 1400px !important;
  background-color: #0f1117 !important;
  padding-top: 2rem !important;
}

[data-testid="stSidebar"] {
  background-color: #0a0c12 !important;
  border-right: 1px solid #1e2130 !important;
}
[data-testid="stSidebar"] * { color: #9ca3af !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #9ca3af !important; font-size: 13px !important; }
[data-testid="stSidebar"] hr { border-color: #1e2130 !important; margin: 1rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
  background: #1a1f2e !important;
  border: 1px solid #2d3348 !important;
  border-radius: 8px !important;
}
[data-testid="stSlider"] > div > div > div > div { background: #4f8ef7 !important; }
[data-testid="stCheckbox"] label { color: #9ca3af !important; font-size: 13px !important; }

.sidebar-section {
  font-size: 11px !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: 0.1em !important;
  color: #4f8ef7 !important; margin: 1.25rem 0 0.5rem 0 !important;
}

.app-title {
  font-size: 26px; font-weight: 700; color: #f9fafb;
  margin: 0 0 4px 0; letter-spacing: -0.02em;
}
.app-subtitle { font-size: 14px; color: #6b7280; margin: 0 0 1.5rem 0; }

[data-testid="stTabs"] [data-testid="stTab"] {
  background: #161b27 !important;
  border: 1px solid #1e2130 !important;
  border-radius: 8px 8px 0 0 !important;
  color: #9ca3af !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 0.5rem 1.25rem !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  background: #4f8ef7 !important;
  color: #ffffff !important;
  border-color: #4f8ef7 !important;
}
[data-testid="stTabPanel"] {
  background: #0f1117 !important;
  border: 1px solid #1e2130 !important;
  border-radius: 0 8px 8px 8px !important;
  padding: 1.25rem !important;
}

.stButton > button {
  background: #4f8ef7 !important; color: #ffffff !important;
  border: none !important; border-radius: 8px !important;
  padding: 0.5rem 1.4rem !important; font-weight: 600 !important;
  font-size: 14px !important; width: 100% !important;
}
.stButton > button:hover { background: #3b7de8 !important; }

[data-testid="stMetric"] {
  background: #161b27 !important; border: 1px solid #1e2130 !important;
  border-radius: 12px !important; padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 11px !important; color: #6b7280 !important;
  font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
  font-size: 30px !important; font-weight: 700 !important; color: #f9fafb !important;
}

.status-badge {
  display: inline-block; padding: 7px 16px; border-radius: 999px;
  font-size: 13px; font-weight: 500; margin-top: 0.75rem;
}
.badge-good { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-warn { background: #1c1408; color: #fbbf24; border: 1px solid #854d0e; }
.badge-bad  { background: #1c0a0a; color: #f87171; border: 1px solid #991b1b; }

.tour-card {
  background: #161b27; border: 1px solid #1e2130; border-radius: 12px;
  padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.tour-step-label {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: #4f8ef7; margin-bottom: 6px;
}
.tour-title { font-size: 17px; font-weight: 600; color: #f9fafb; margin-bottom: 6px; }
.tour-desc { font-size: 14px; color: #9ca3af; line-height: 1.6; }

[data-testid="stExpander"] {
  background: #161b27 !important; border: 1px solid #1e2130 !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: #9ca3af !important; font-size: 13px !important; }
[data-testid="stExpander"] p, [data-testid="stExpander"] li { color: #9ca3af !important; font-size: 13px !important; }
[data-testid="stExpander"] strong { color: #d1d5db !important; }

hr { border-color: #1e2130 !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PALETTE = ["#4f8ef7","#f97316","#10b981","#f43f5e","#8b5cf6","#06b6d4","#eab308","#ec4899","#14b8a6","#6366f1"]
NOISE_COLOR = "#4b5563"
PLOT_BG  = "#0f1117"
GRID_COL = "#161b27"
TICK_COL = "#374151"
FONT_COL = "#e5e7eb"

# ── Step-by-step HTML component ────────────────────────────────────────────────
DBSCAN_STEPTHROUGH_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: transparent;
  color: #c9d1d9;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0 16px 0;
}
.main-container {
  display: flex;
  gap: 20px;
  align-items: flex-start;
  flex-wrap: wrap;
  justify-content: center;
  max-width: 1180px;
  width: 100%;
  margin: 0 auto;
}
.viz-container {
  background: #161b22;
  border-radius: 14px;
  padding: 14px;
  border: 1px solid #30363d;
  flex-shrink: 0;
}
svg {
  border-radius: 10px;
  display: block;
  background: #0d1117;
  max-width: 100%;
}
.controls {
  display: flex;
  gap: 12px;
  margin-top: 16px;
  justify-content: center;
}
button {
  padding: 10px 24px;
  font-size: 14px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 600;
}
.btn-prev {
  background: #21262d;
  color: #c9d1d9;
  border: 1px solid #30363d;
}
.btn-prev:hover:not(:disabled) { background: #30363d; }
.btn-next { background: #238636; color: #fff; }
.btn-next:hover:not(:disabled) { background: #2ea043; }
button:disabled { opacity: 0.35; cursor: not-allowed; }
.info-panel {
  background: #161b22;
  border-radius: 14px;
  padding: 20px;
  border: 1px solid #30363d;
  min-width: 260px;
  max-width: 300px;
  flex-shrink: 0;
  flex-grow: 1;
}
.step-counter {
  font-size: 11px; color: #58a6ff; text-transform: uppercase;
  letter-spacing: 1.5px; margin-bottom: 8px; font-weight: 700;
}
.step-title { font-size: 17px; font-weight: 700; margin-bottom: 10px; color: #f0f6fc; }
.step-description { color: #8b949e; line-height: 1.7; font-size: 13px; margin-bottom: 16px; }
.params {
  padding: 12px; background: rgba(88,166,255,0.06);
  border-radius: 10px; border: 1px solid rgba(88,166,255,0.15);
}
.param-row { display: flex; justify-content: space-between; margin-bottom: 7px; font-size: 12px; }
.param-row:last-child { margin-bottom: 0; }
.param-label { color: #8b949e; }
.param-value { color: #58a6ff; font-weight: 700; }
.legend { margin-top: 16px; padding-top: 16px; border-top: 1px solid #30363d; }
.legend-title {
  font-size: 10px; font-weight: 700; color: #8b949e;
  margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1.5px;
}
.legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }
.legend-dot {
  width: 11px; height: 11px; border-radius: 50%;
  border: 2px solid transparent; flex-shrink: 0;
}
.legend-dot.unvisited { background: #21262d; border-color: #484f58; }
.legend-dot.active    { background: #00d4ff; border-color: #00d4ff; box-shadow: 0 0 6px #00d4ff; }
.legend-dot.frontier  { background: #3fb950; border-color: #3fb950; }
.legend-dot.core      { background: #da3633; border-color: #da3633; }
.legend-dot.border    { background: #58a6ff; border-color: #58a6ff; }
.legend-dot.c1        { background: #f59e0b; border-color: #f59e0b; }
.legend-dot.c2        { background: #22d3ee; border-color: #22d3ee; }
.legend-dot.noise     { background: #6b7280; border-color: #6b7280; }
.legend-dot.epsilon   { background: transparent; border-color: #00d4ff; border-style: dashed; }

.point { transition: all 0.35s ease; cursor: pointer; }
.point-circle { transition: all 0.35s ease; fill: #21262d; stroke: #484f58; stroke-width: 2; }
.epsilon-circle {
  fill: none; stroke: #00d4ff; stroke-width: 2;
  stroke-dasharray: 8, 5; opacity: 0; transition: all 0.35s ease; pointer-events: none;
}
.epsilon-circle.visible { opacity: 0.5; }
.connection-line { stroke: #8b949e; stroke-width: 1; opacity: 0; transition: all 0.35s ease; }
.connection-line.visible { opacity: 0.25; }
.density-line { stroke: #fff; stroke-width: 3; opacity: 0; transition: all 0.5s ease; }
.density-line.visible { opacity: 0.9; }
.label-text {
  font-size: 10px; fill: #f0f6fc; opacity: 0; transition: all 0.35s ease;
  pointer-events: none; font-weight: 700;
}
.label-text.visible { opacity: 1; }
.point.active .point-circle { animation: pulse 1.2s infinite ease-in-out; stroke: #fff; stroke-width: 3; }
.point.highlight .point-circle { stroke: #fff; stroke-width: 3; filter: drop-shadow(0 0 6px rgba(255,255,255,0.8)); }
.point.dim .point-circle { opacity: 0.25; }
@keyframes pulse { 0%, 100% { r: 7; } 50% { r: 11; } }
</style>
</head>
<body>
<div class="main-container">
  <div class="viz-container">
    <svg id="viz" width="740" height="400" viewBox="0 0 780 420">
      <!-- Connection lines Cluster 1 -->
      <line class="connection-line" id="conn-0-1" x1="220" y1="230" x2="250" y2="210"/>
      <line class="connection-line" id="conn-0-2" x1="220" y1="230" x2="250" y2="250"/>
      <line class="connection-line" id="conn-0-3" x1="220" y1="230" x2="220" y2="270"/>
      <line class="connection-line" id="conn-0-4" x1="220" y1="230" x2="190" y2="250"/>
      <line class="connection-line" id="conn-0-5" x1="220" y1="230" x2="190" y2="210"/>
      <line class="connection-line" id="conn-0-11" x1="220" y1="230" x2="220" y2="180"/>
      <line class="connection-line" id="conn-1-2" x1="250" y1="210" x2="250" y2="250"/>
      <line class="connection-line" id="conn-1-5" x1="250" y1="210" x2="190" y2="210"/>
      <line class="connection-line" id="conn-1-6" x1="250" y1="210" x2="290" y2="190"/>
      <line class="connection-line" id="conn-2-3" x1="250" y1="250" x2="220" y2="270"/>
      <line class="connection-line" id="conn-2-7" x1="250" y1="250" x2="290" y2="270"/>
      <line class="connection-line" id="conn-3-4" x1="220" y1="270" x2="190" y2="250"/>
      <line class="connection-line" id="conn-3-8" x1="220" y1="270" x2="220" y2="310"/>
      <line class="connection-line" id="conn-4-5" x1="190" y1="250" x2="190" y2="210"/>
      <line class="connection-line" id="conn-4-9" x1="190" y1="250" x2="150" y2="270"/>
      <line class="connection-line" id="conn-5-10" x1="190" y1="210" x2="150" y2="190"/>
      <line class="connection-line" id="conn-5-11" x1="190" y1="210" x2="220" y2="180"/>
      <!-- Connection lines Cluster 2 -->
      <line class="connection-line" id="conn-12-13" x1="560" y1="230" x2="590" y2="210"/>
      <line class="connection-line" id="conn-12-14" x1="560" y1="230" x2="590" y2="250"/>
      <line class="connection-line" id="conn-12-15" x1="560" y1="230" x2="560" y2="270"/>
      <line class="connection-line" id="conn-12-16" x1="560" y1="230" x2="530" y2="250"/>
      <line class="connection-line" id="conn-12-17" x1="560" y1="230" x2="530" y2="210"/>
      <line class="connection-line" id="conn-12-23" x1="560" y1="230" x2="560" y2="180"/>
      <line class="connection-line" id="conn-13-14" x1="590" y1="210" x2="590" y2="250"/>
      <line class="connection-line" id="conn-13-17" x1="590" y1="210" x2="530" y2="210"/>
      <line class="connection-line" id="conn-13-18" x1="590" y1="210" x2="630" y2="190"/>
      <line class="connection-line" id="conn-14-15" x1="590" y1="250" x2="560" y2="270"/>
      <line class="connection-line" id="conn-14-19" x1="590" y1="250" x2="630" y2="270"/>
      <line class="connection-line" id="conn-15-16" x1="560" y1="270" x2="530" y2="250"/>
      <line class="connection-line" id="conn-15-20" x1="560" y1="270" x2="560" y2="310"/>
      <line class="connection-line" id="conn-16-17" x1="530" y1="250" x2="530" y2="210"/>
      <line class="connection-line" id="conn-16-21" x1="530" y1="250" x2="490" y2="270"/>
      <line class="connection-line" id="conn-17-22" x1="530" y1="210" x2="490" y2="190"/>
      <line class="connection-line" id="conn-17-23" x1="530" y1="210" x2="560" y2="180"/>
      <!-- Density lines -->
      <line class="density-line" id="dense-6-1"  x1="290" y1="190" x2="250" y2="210"/>
      <line class="density-line" id="dense-0-1"  x1="220" y1="230" x2="250" y2="210"/>
      <line class="density-line" id="dense-0-5"  x1="220" y1="230" x2="190" y2="210"/>
      <line class="density-line" id="dense-5-10" x1="190" y1="210" x2="150" y2="190"/>
      <line class="density-line" id="dense-10-5" x1="150" y1="190" x2="190" y2="210"/>
      <line class="density-line" id="dense-1-0"  x1="250" y1="210" x2="220" y2="230"/>
      <line class="density-line" id="dense-1-2"  x1="250" y1="210" x2="250" y2="250"/>
      <line class="density-line" id="dense-2-0"  x1="250" y1="250" x2="220" y2="230"/>
      <!-- Epsilon circles -->
      <circle class="epsilon-circle" id="eps-0"  cx="220" cy="230" r="45"/>
      <circle class="epsilon-circle" id="eps-1"  cx="250" cy="210" r="45"/>
      <circle class="epsilon-circle" id="eps-2"  cx="250" cy="250" r="45"/>
      <circle class="epsilon-circle" id="eps-3"  cx="220" cy="270" r="45"/>
      <circle class="epsilon-circle" id="eps-4"  cx="190" cy="250" r="45"/>
      <circle class="epsilon-circle" id="eps-5"  cx="190" cy="210" r="45"/>
      <circle class="epsilon-circle" id="eps-6"  cx="290" cy="190" r="45"/>
      <circle class="epsilon-circle" id="eps-7"  cx="290" cy="270" r="45"/>
      <circle class="epsilon-circle" id="eps-8"  cx="220" cy="310" r="45"/>
      <circle class="epsilon-circle" id="eps-9"  cx="150" cy="270" r="45"/>
      <circle class="epsilon-circle" id="eps-10" cx="150" cy="190" r="45"/>
      <circle class="epsilon-circle" id="eps-11" cx="220" cy="180" r="45"/>
      <circle class="epsilon-circle" id="eps-12" cx="560" cy="230" r="45"/>
      <circle class="epsilon-circle" id="eps-13" cx="590" cy="210" r="45"/>
      <circle class="epsilon-circle" id="eps-14" cx="590" cy="250" r="45"/>
      <circle class="epsilon-circle" id="eps-15" cx="560" cy="270" r="45"/>
      <circle class="epsilon-circle" id="eps-16" cx="530" cy="250" r="45"/>
      <circle class="epsilon-circle" id="eps-17" cx="530" cy="210" r="45"/>
      <circle class="epsilon-circle" id="eps-18" cx="630" cy="190" r="45"/>
      <circle class="epsilon-circle" id="eps-19" cx="630" cy="270" r="45"/>
      <circle class="epsilon-circle" id="eps-20" cx="560" cy="310" r="45"/>
      <circle class="epsilon-circle" id="eps-21" cx="490" cy="270" r="45"/>
      <circle class="epsilon-circle" id="eps-22" cx="490" cy="190" r="45"/>
      <circle class="epsilon-circle" id="eps-23" cx="560" cy="180" r="45"/>
      <circle class="epsilon-circle" id="eps-24" cx="390" cy="100" r="45"/>
      <circle class="epsilon-circle" id="eps-25" cx="390" cy="360" r="45"/>
      <circle class="epsilon-circle" id="eps-26" cx="80"  cy="150" r="45"/>
      <circle class="epsilon-circle" id="eps-27" cx="80"  cy="310" r="45"/>
      <circle class="epsilon-circle" id="eps-28" cx="700" cy="150" r="45"/>
      <circle class="epsilon-circle" id="eps-29" cx="700" cy="310" r="45"/>
      <!-- Labels -->
      <text class="label-text" id="label-0"  x="220" y="220" text-anchor="middle">Core</text>
      <text class="label-text" id="label-1"  x="250" y="200" text-anchor="middle">Core</text>
      <text class="label-text" id="label-2"  x="250" y="240" text-anchor="middle">Core</text>
      <text class="label-text" id="label-3"  x="220" y="260" text-anchor="middle">Core</text>
      <text class="label-text" id="label-4"  x="190" y="240" text-anchor="middle">Core</text>
      <text class="label-text" id="label-5"  x="190" y="200" text-anchor="middle">Core</text>
      <text class="label-text" id="label-6"  x="290" y="180" text-anchor="middle">Border</text>
      <text class="label-text" id="label-7"  x="290" y="260" text-anchor="middle">Border</text>
      <text class="label-text" id="label-8"  x="220" y="300" text-anchor="middle">Border</text>
      <text class="label-text" id="label-9"  x="150" y="260" text-anchor="middle">Border</text>
      <text class="label-text" id="label-10" x="150" y="180" text-anchor="middle">Border</text>
      <text class="label-text" id="label-11" x="220" y="170" text-anchor="middle">Border</text>
      <text class="label-text" id="label-12" x="560" y="220" text-anchor="middle">Core</text>
      <text class="label-text" id="label-13" x="590" y="200" text-anchor="middle">Core</text>
      <text class="label-text" id="label-14" x="590" y="240" text-anchor="middle">Core</text>
      <text class="label-text" id="label-15" x="560" y="260" text-anchor="middle">Core</text>
      <text class="label-text" id="label-16" x="530" y="240" text-anchor="middle">Core</text>
      <text class="label-text" id="label-17" x="530" y="200" text-anchor="middle">Core</text>
      <text class="label-text" id="label-18" x="630" y="180" text-anchor="middle">Border</text>
      <text class="label-text" id="label-19" x="630" y="260" text-anchor="middle">Border</text>
      <text class="label-text" id="label-20" x="560" y="300" text-anchor="middle">Border</text>
      <text class="label-text" id="label-21" x="490" y="260" text-anchor="middle">Border</text>
      <text class="label-text" id="label-22" x="490" y="180" text-anchor="middle">Border</text>
      <text class="label-text" id="label-23" x="560" y="170" text-anchor="middle">Border</text>
      <text class="label-text" id="label-24" x="390" y="90"  text-anchor="middle">Noise</text>
      <text class="label-text" id="label-25" x="390" y="350" text-anchor="middle">Noise</text>
      <text class="label-text" id="label-26" x="80"  y="140" text-anchor="middle">Noise</text>
      <text class="label-text" id="label-27" x="80"  y="300" text-anchor="middle">Noise</text>
      <text class="label-text" id="label-28" x="700" y="140" text-anchor="middle">Noise</text>
      <text class="label-text" id="label-29" x="700" y="300" text-anchor="middle">Noise</text>
      <!-- Points -->
      <g class="point" id="point-0"  transform="translate(220,230)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-1"  transform="translate(250,210)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-2"  transform="translate(250,250)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-3"  transform="translate(220,270)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-4"  transform="translate(190,250)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-5"  transform="translate(190,210)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-6"  transform="translate(290,190)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-7"  transform="translate(290,270)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-8"  transform="translate(220,310)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-9"  transform="translate(150,270)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-10" transform="translate(150,190)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-11" transform="translate(220,180)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-12" transform="translate(560,230)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-13" transform="translate(590,210)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-14" transform="translate(590,250)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-15" transform="translate(560,270)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-16" transform="translate(530,250)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-17" transform="translate(530,210)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-18" transform="translate(630,190)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-19" transform="translate(630,270)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-20" transform="translate(560,310)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-21" transform="translate(490,270)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-22" transform="translate(490,190)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-23" transform="translate(560,180)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-24" transform="translate(390,100)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-25" transform="translate(390,360)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-26" transform="translate(80,150)"><circle class="point-circle"  r="7"/></g>
      <g class="point" id="point-27" transform="translate(80,310)"><circle class="point-circle"  r="7"/></g>
      <g class="point" id="point-28" transform="translate(700,150)"><circle class="point-circle" r="7"/></g>
      <g class="point" id="point-29" transform="translate(700,310)"><circle class="point-circle" r="7"/></g>
    </svg>
    <div class="controls">
      <button class="btn-prev" id="btnPrev" onclick="prevStep()">← Previous</button>
      <button class="btn-next" id="btnNext" onclick="nextStep()">Next →</button>
    </div>
  </div>

  <div class="info-panel">
    <div class="step-counter" id="stepCounter">Step 1 of 25</div>
    <div class="step-title" id="stepTitle">Initialization</div>
    <div class="step-description" id="stepDesc">
      30 points in 2D space — two clusters with dense centers and sparse edges, plus scattered noise.
      DBSCAN will discover clusters based on density.
    </div>
    <div class="params">
      <div class="param-row"><span class="param-label">Epsilon (ε)</span><span class="param-value">45 px</span></div>
      <div class="param-row"><span class="param-label">Min Points</span><span class="param-value">4</span></div>
      <div class="param-row"><span class="param-label">Cluster</span><span class="param-value" id="currentCluster">—</span></div>
    </div>
    <div class="legend">
      <div class="legend-title">Legend</div>
      <div class="legend-item"><div class="legend-dot unvisited"></div><span>Unvisited</span></div>
      <div class="legend-item"><div class="legend-dot active"></div><span>Examining (ε shown)</span></div>
      <div class="legend-item"><div class="legend-dot frontier"></div><span>Frontier (to visit)</span></div>
      <div class="legend-item"><div class="legend-dot core"></div><span>Core Point</span></div>
      <div class="legend-item"><div class="legend-dot border"></div><span>Border Point</span></div>
      <div class="legend-item"><div class="legend-dot c1"></div><span>Cluster 1</span></div>
      <div class="legend-item"><div class="legend-dot c2"></div><span>Cluster 2</span></div>
      <div class="legend-item"><div class="legend-dot noise"></div><span>Noise Point</span></div>
      <div class="legend-item"><div class="legend-dot epsilon"></div><span>Epsilon Radius</span></div>
    </div>
  </div>
</div>

<script>
const neighbors = {
  0:[1,2,3,4,5], 1:[0,2,5,6], 2:[0,1,3,7], 3:[0,2,4,8],
  4:[0,3,5,9], 5:[0,1,4,10,11], 6:[1], 7:[2], 8:[3], 9:[4], 10:[5], 11:[5],
  12:[13,14,15,16,17], 13:[12,14,17,18], 14:[12,13,15,19], 15:[12,14,16,20],
  16:[12,15,17,21], 17:[12,13,16,22,23], 18:[13], 19:[14], 20:[15], 21:[16], 22:[17], 23:[17],
  24:[], 25:[], 26:[], 27:[], 28:[], 29:[]
};
const minPts = 4;
function classify(i) {
  const n = neighbors[i];
  if (n.length >= minPts - 1) return 'core';
  for (let j of n) { if (neighbors[j].length >= minPts - 1) return 'border'; }
  return 'noise';
}
const classification = {};
for (let i = 0; i < 30; i++) classification[i] = classify(i);

const cluster1 = [0,1,2,3,4,5,6,7,8,9,10,11];
const cluster2 = [12,13,14,15,16,17,18,19,20,21,22,23];
const allNoise = [24,25,26,27,28,29];

const c1conns = [
  'conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-0-11',
  'conn-1-2','conn-1-5','conn-1-6',
  'conn-2-3','conn-2-7',
  'conn-3-4','conn-3-8',
  'conn-4-5','conn-4-9',
  'conn-5-10','conn-5-11'
];
const c2conns = [
  'conn-12-13','conn-12-14','conn-12-15','conn-12-16','conn-12-17','conn-12-23',
  'conn-13-14','conn-13-17','conn-13-18',
  'conn-14-15','conn-14-19',
  'conn-15-16','conn-15-20',
  'conn-16-17','conn-16-21',
  'conn-17-22','conn-17-23'
];
const allConns = [...c1conns, ...c2conns];

const steps = [
  { title:"Initialization", desc:"30 points in 2D space — two clusters with dense centers and sparse edges, plus scattered noise. DBSCAN discovers clusters based on local density without knowing the cluster count upfront.", cluster:"—", action:()=>{ resetAll(); } },
  { title:"Pick a Starting Point", desc:"We pick an unvisited point at random and draw its epsilon (ε) radius circle. We will count how many other points fall inside this circle.", cluster:"—", action:()=>{ resetAll(); activatePoint(0); showEpsilon(0); } },
  { title:"Core Point Found!", desc:"The ε-circle contains <strong>5 neighbors</strong>. With the point itself, that's 6 ≥ MinPts=4. This is a <strong>CORE POINT</strong> — colored <span style='color:#da3633'>red</span>. All unvisited neighbors within ε are added to the frontier (green).", cluster:"Cluster 1", action:()=>{ resetAll(); setPointType(0,'core'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5']); setFrontier([1,2,3,4,5]); showLabels([0]); } },
  { title:"Visit a Frontier Neighbor", desc:"We pick the next point from the frontier (green). It becomes the active point (cyan pulse). We draw its ε-circle to check if it can expand the cluster further.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5']); setFrontier([2,3,4,5]); activatePoint(1); showEpsilon(1); showLabels([0]); } },
  { title:"Another Core Point", desc:"This point has <strong>4 neighbors</strong> within ε — it's another <strong>CORE POINT</strong> (<span style='color:#da3633'>red</span>)! Its unvisited neighbors join the frontier (green).", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6']); setFrontier([2,3,4,5,6]); showLabels([0,1]); } },
  { title:"Visit Another Unvisited Neighbor", desc:"We continue by picking another point from the frontier. This point is density-reachable from the previous core points through the chain. Let's check if it's also a core point.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6']); setFrontier([3,4,5,6]); activatePoint(2); showEpsilon(2); showLabels([0,1]); } },
  { title:"Third Core Point", desc:"This point has <strong>4 neighbors</strong> within ε — it's a <strong>CORE POINT</strong> (<span style='color:#da3633'>red</span>)! DBSCAN follows the density chain — points only need to be density-reachable through core points.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); setPointType(2,'core'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6','conn-2-3','conn-2-7']); setFrontier([3,4,5,6,7]); showLabels([0,1,2]); } },
  { title:"Visit a Frontier Point — Border?", desc:"Now we pick a frontier point sitting on the outer edge of the cluster. We draw its ε-circle to check if it's a core point or just a border point.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); setPointType(2,'core'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6','conn-2-3','conn-2-7']); setFrontier([3,4,5,7]); activatePoint(6); showEpsilon(6); showLabels([0,1,2]); } },
  { title:"Border Point #1 Discovered", desc:"The ε-circle contains only <strong>1 neighbor</strong> (p1). Since MinPts−1 = 3, this is <strong>NOT</strong> enough to be a core point. It's a <strong>BORDER POINT</strong> (<span style='color:#58a6ff'>blue</span>). It belongs to Cluster 1 but cannot expand the cluster.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); setPointType(2,'core'); setPointType(6,'border'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6','conn-2-3','conn-2-7']); setFrontier([3,4,5,7]); showLabels([0,1,2,6]); } },
  { title:"Border Point #2 Discovered", desc:"Again, the ε-circle contains only <strong>1 neighbor</strong> (p2). Also a <strong>BORDER POINT</strong> (<span style='color:#58a6ff'>blue</span>). Border points sit on the edge of the cluster, within ε of a core point, but with too few neighbors to expand.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); setPointType(2,'core'); setPointType(6,'border'); setPointType(7,'border'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6','conn-2-3','conn-2-7']); setFrontier([3,4,5]); showLabels([0,1,2,6,7]); } },
  { title:"Another Core Point Found!", desc:"This point has <strong>5 neighbors</strong> within ε — it's another <strong>CORE POINT</strong> (<span style='color:#da3633'>red</span>)! Its unvisited neighbors (p10, p11) are added to the frontier (green).", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); setPointType(2,'core'); setPointType(5,'core'); setPointType(6,'border'); setPointType(7,'border'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6','conn-2-3','conn-2-7','conn-5-10','conn-5-11']); setFrontier([3,4,10,11]); showLabels([0,1,2,5,6,7]); } },
  { title:"Border Point #3 Discovered", desc:"The ε-circle contains only <strong>1 neighbor</strong> (p5). It's a <strong>BORDER POINT</strong> (<span style='color:#58a6ff'>blue</span>). It belongs to Cluster 1 because it's within ε of core point p5, but it cannot expand the cluster.", cluster:"Cluster 1", action:()=>{ setPointType(0,'core'); setPointType(1,'core'); setPointType(2,'core'); setPointType(5,'core'); setPointType(6,'border'); setPointType(7,'border'); setPointType(11,'border'); showConnections(['conn-0-1','conn-0-2','conn-0-3','conn-0-4','conn-0-5','conn-1-2','conn-1-5','conn-1-6','conn-2-3','conn-2-7','conn-5-10','conn-5-11']); setFrontier([3,4,10]); showLabels([0,1,2,5,6,7,11]); } },
  { title:"All Core Points of Cluster 1", desc:"Here are all the <strong>CORE POINTS</strong> of Cluster 1 (<span style='color:#da3633'>red</span>). These 6 points form the dense backbone — each has ≥3 neighbors within ε, all density-reachable from one another.", cluster:"Cluster 1", action:()=>{ for(let i of cluster1){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'unvisited'); } showConnections(c1conns); clearFrontier(); showLabels([0,1,2,3,4,5]); } },
  { title:"All Border Points of Cluster 1", desc:"And here are all the <strong>BORDER POINTS</strong> of Cluster 1 (<span style='color:#58a6ff'>blue</span>). These 6 points are within ε of at least one core point. They define the cluster's outer boundary.", cluster:"Cluster 1", action:()=>{ for(let i of cluster1){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } showConnections(c1conns); clearFrontier(); showLabels(cluster1); } },
  { title:"Noise Points", desc:"These points have <strong>zero neighbors</strong> within their ε-radius. They're classified as <strong>NOISE</strong> (<span style='color:#6b7280'>gray</span>) — outliers that don't belong to any cluster.", cluster:"—", action:()=>{ for(let i of cluster1){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } showConnections(c1conns); clearFrontier(); showLabels(cluster1); for(let i of [24,25]) setPointType(i,'noise'); activatePoint(24); showEpsilon(24); showLabels([...cluster1,24,25]); } },
  { title:"More Noise Points", desc:"All 6 noise points are classified as outliers. DBSCAN correctly identifies them without any parameter tuning for outlier detection.", cluster:"—", action:()=>{ for(let i of cluster1){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } showConnections(c1conns); clearFrontier(); showLabels(cluster1); for(let i of allNoise) setPointType(i,'noise'); showLabels([...cluster1,24,25,26,27,28,29]); } },
  { title:"Cluster 2 Fully Revealed", desc:"Finally, we discover <strong>Cluster 2</strong> — 6 core points (<span style='color:#da3633'>red</span>) and 6 border points (<span style='color:#58a6ff'>blue</span>), found completely independently. DBSCAN found both clusters without specifying cluster count!", cluster:"Cluster 1 & 2", action:()=>{ for(let i=0;i<30;i++){ if(cluster1.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else if(cluster2.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else { setPointType(i,'noise'); } } showConnections(allConns); clearFrontier(); showLabels([...Array(30).keys()]); } },
  { title:"Directly Density-Reachable", desc:"A point q is <strong>directly density-reachable</strong> from p if q is within the ε-radius of p and p is a core point. Here, border point p6 is directly density-reachable from core point p1. The bold white line highlights this direct connection.", cluster:"Concept", action:()=>{ for(let i=0;i<30;i++){ if(cluster1.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else if(cluster2.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else { setPointType(i,'noise'); } } showConnections(allConns); clearFrontier(); dimAllPoints(); highlightPoints([1,6]); showLabels([1,6]); showDensityLines(['dense-6-1']); } },
  { title:"Density-Reachable", desc:"A point q is <strong>density-reachable</strong> from p if there's a chain of core points connecting them. Here, p0→p1→p2 forms a chain — each a core point within ε of the next. The bold white lines trace this chain.", cluster:"Concept", action:()=>{ for(let i=0;i<30;i++){ if(cluster1.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else if(cluster2.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else { setPointType(i,'noise'); } } showConnections(allConns); clearFrontier(); dimAllPoints(); highlightPoints([0,1,2]); showLabels([0,1,2]); showDensityLines(['dense-0-1','dense-1-2']); } },
  { title:"Density-Connected", desc:"Two points are <strong>density-connected</strong> if there exists a core point that both are density-reachable from. Border points p6 and p10 are both density-reachable from p5 — so they belong to the same cluster even though they're far apart.", cluster:"Concept", action:()=>{ for(let i=0;i<30;i++){ if(cluster1.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else if(cluster2.includes(i)){ if(classification[i]==='core') setPointType(i,'core'); else setPointType(i,'border'); } else { setPointType(i,'noise'); } } showConnections(allConns); clearFrontier(); dimAllPoints(); highlightPoints([6,1,0,5,10]); showLabels([6,10]); showDensityLines(['dense-6-1','dense-1-0','dense-0-5','dense-5-10']); } },
  { title:"NOT Density-Connected", desc:"Points from different clusters are <strong>NOT density-connected</strong> — no chain of core points links them. The gap between Cluster 1 and Cluster 2 is too wide. This is why DBSCAN correctly separates them into distinct clusters.", cluster:"Concept", action:()=>{ for(let i=0;i<30;i++){ if(cluster1.includes(i)) setPointType(i,'c1'); else if(cluster2.includes(i)) setPointType(i,'c2'); else setPointType(i,'noise'); } showConnections(allConns); clearFrontier(); undimAllPoints(); unhighlightAll(); showLabels([]); hideDensityLines(); } }
];

let currentStep = 0;

function resetAll() {
  for (let i = 0; i < 30; i++) {
    const point = document.getElementById('point-' + i);
    const circle = point.querySelector('.point-circle');
    circle.style.fill = '#21262d'; circle.style.stroke = '#484f58';
    point.classList.remove('active');
    document.getElementById('label-' + i).classList.remove('visible');
    document.getElementById('eps-' + i).classList.remove('visible');
  }
  document.querySelectorAll('.connection-line').forEach(el => el.classList.remove('visible'));
  document.querySelectorAll('.density-line').forEach(el => el.classList.remove('visible'));
}

function setPointType(index, type) {
  const point = document.getElementById('point-' + index);
  const circle = point.querySelector('.point-circle');
  point.classList.remove('active');
  const map = { core:'#da3633', border:'#58a6ff', noise:'#6b7280', c1:'#f59e0b', c2:'#22d3ee', frontier:'#3fb950', unvisited:'#21262d' };
  const strokeMap = { unvisited:'#484f58' };
  circle.style.fill = map[type] || '#21262d';
  circle.style.stroke = strokeMap[type] || map[type] || '#484f58';
}

function activatePoint(index) {
  document.querySelectorAll('.point').forEach(p => p.classList.remove('active'));
  document.getElementById('point-' + index).classList.add('active');
}

function showEpsilon(index) {
  document.querySelectorAll('.epsilon-circle').forEach(el => el.classList.remove('visible'));
  document.getElementById('eps-' + index).classList.add('visible');
}

function showConnections(ids) {
  document.querySelectorAll('.connection-line').forEach(el => el.classList.remove('visible'));
  ids.forEach(id => { const el = document.getElementById(id); if(el) el.classList.add('visible'); });
}

function showDensityLines(ids) {
  document.querySelectorAll('.density-line').forEach(el => el.classList.remove('visible'));
  ids.forEach(id => { const el = document.getElementById(id); if(el) el.classList.add('visible'); });
}

function hideDensityLines() {
  document.querySelectorAll('.density-line').forEach(el => el.classList.remove('visible'));
}

function setFrontier(indices) {
  for (let i = 0; i < 30; i++) {
    const circle = document.getElementById('point-' + i).querySelector('.point-circle');
    const fill = circle.style.fill;
    if (!fill || fill === '#21262d' || fill === 'rgb(33, 38, 45)') {
      if (indices.includes(i)) { circle.style.fill = '#3fb950'; circle.style.stroke = '#3fb950'; }
      else { circle.style.fill = '#21262d'; circle.style.stroke = '#484f58'; }
    }
  }
}

function clearFrontier() {
  for (let i = 0; i < 30; i++) {
    const circle = document.getElementById('point-' + i).querySelector('.point-circle');
    const fill = circle.style.fill;
    if (fill === '#3fb950' || fill === 'rgb(63, 185, 80)') {
      circle.style.fill = '#21262d'; circle.style.stroke = '#484f58';
    }
  }
}

function showLabels(indices) {
  for (let i = 0; i < 30; i++) document.getElementById('label-' + i).classList.remove('visible');
  indices.forEach(i => document.getElementById('label-' + i).classList.add('visible'));
}

function dimAllPoints()    { for(let i=0;i<30;i++) document.getElementById('point-'+i).classList.add('dim'); }
function undimAllPoints()  { for(let i=0;i<30;i++) document.getElementById('point-'+i).classList.remove('dim'); }
function highlightPoints(indices) { for(let i of indices){ document.getElementById('point-'+i).classList.remove('dim'); document.getElementById('point-'+i).classList.add('highlight'); } }
function unhighlightAll()  { for(let i=0;i<30;i++) document.getElementById('point-'+i).classList.remove('highlight'); }

function updateUI() {
  const step = steps[currentStep];
  document.getElementById('stepCounter').textContent = 'Step ' + (currentStep+1) + ' of ' + steps.length;
  document.getElementById('stepTitle').textContent = step.title;
  document.getElementById('stepDesc').innerHTML = step.desc;
  document.getElementById('currentCluster').textContent = step.cluster;
  document.getElementById('btnPrev').disabled = currentStep === 0;
  document.getElementById('btnNext').disabled = currentStep === steps.length - 1;
}

function renderStep() { steps[currentStep].action(); updateUI(); }
function nextStep() { if(currentStep < steps.length-1){ currentStep++; renderStep(); } }
function prevStep() { if(currentStep > 0){ currentStep--; renderStep(); } }

document.addEventListener('keydown', e => {
  if(e.key==='ArrowRight') nextStep();
  if(e.key==='ArrowLeft') prevStep();
});

renderStep();
</script>
</body>
</html>
"""

# ── Guided tour steps ──────────────────────────────────────────────────────────
TOUR_STEPS = [
    {"title":"Start: Raw Data","desc":"This is your dataset — no clustering applied yet. Every point is neutral. Notice the shape: two interleaved moons. Ask yourself — how would you separate these by hand?","dataset":"Moons","eps":0.35,"min_samples":5,"clustered":False,"k":2},
    {"title":"ε Too Small — Everything is Noise","desc":"With ε = 0.05, almost no point has enough neighbors. The algorithm sees nothing but outliers. This is what happens when your neighborhood radius is too tight.","dataset":"Moons","eps":0.05,"min_samples":5,"clustered":True,"k":2},
    {"title":"ε Too Large — Everything Merges","desc":"With ε = 1.2, every point's neighborhood swallows the entire dataset. Both moons collapse into one cluster. DBSCAN loses all discrimination.","dataset":"Moons","eps":1.2,"min_samples":5,"clustered":True,"k":2},
    {"title":"Sweet Spot — Clean Separation","desc":"With ε = 0.35 and MinPts = 5, DBSCAN finds exactly two clusters following the natural moon shapes. Notice the core points (bright) vs border points (faded) at the edges.","dataset":"Moons","eps":0.35,"min_samples":5,"clustered":True,"k":2},
    {"title":"MinPts Too High — Sparse Points Become Noise","desc":"Raising MinPts to 15 makes the algorithm stricter. Only very dense regions qualify as core points. Border regions get classified as noise — shown as ✕ marks.","dataset":"Moons","eps":0.35,"min_samples":15,"clustered":True,"k":2},
    {"title":"Where K-Means Fails","desc":"Switch to the comparison tab. K-Means assumes spherical clusters and cuts space with straight boundaries — it cannot follow the moon shapes. DBSCAN follows density, not geometry.","dataset":"Moons","eps":0.35,"min_samples":5,"clustered":True,"k":2},
    {"title":"Concentric Circles — Another K-Means Failure","desc":"K-Means splits the circles vertically — completely wrong. DBSCAN correctly identifies the inner and outer ring as separate clusters because they have different densities and shapes.","dataset":"Concentric Circles","eps":0.25,"min_samples":5,"clustered":True,"k":2},
]

# ── Session state ──────────────────────────────────────────────────────────────
if "show_clustered" not in st.session_state:
    st.session_state.show_clustered = False
if "tour_step" not in st.session_state:
    st.session_state.tour_step = 0

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-section">Dataset</p>', unsafe_allow_html=True)
    dataset = st.selectbox("Dataset", ["Moons","Blobs","Concentric Circles","Random Noise"], label_visibility="collapsed")
    n_points = st.slider("Number of points", 100, 600, 300, step=50)
    noise_level = st.slider("Dataset noise", 0.01, 0.15, 0.08, step=0.01)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section">DBSCAN Parameters</p>', unsafe_allow_html=True)
    eps = st.slider("ε — neighborhood radius", 0.05, 1.5, 0.35, step=0.01)
    min_samples = st.slider("MinPts — core point threshold", 1, 20, 5, step=1)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section">K-Means Parameters</p>', unsafe_allow_html=True)
    k_clusters = st.slider("K — number of clusters", 2, 8, 2, step=1)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section">Display</p>', unsafe_allow_html=True)
    show_epsilon_circles = st.checkbox("Show ε neighborhood circles", value=False)
    point_size = st.slider("Point size", 4, 16, 7)

# ── Data helpers ───────────────────────────────────────────────────────────────
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
            traces.append(go.Scatter(x=X[mask,0], y=X[mask,1], mode="markers",
                marker=dict(size=point_size-1, color=NOISE_COLOR, opacity=0.6, symbol="x",
                            line=dict(width=1.5, color=NOISE_COLOR)),
                name="Noise", hovertemplate="Noise — x: %{x:.2f}  y: %{y:.2f}<extra></extra>"))
        else:
            color = palette[label % len(palette)]
            cmask = labels == label
            border = cmask & ~core_mask
            core   = cmask & core_mask
            if border.any():
                traces.append(go.Scatter(x=X[border,0], y=X[border,1], mode="markers",
                    marker=dict(size=point_size-1, color=color, opacity=0.3, line=dict(width=1, color=color)),
                    name=f"Cluster {label+1} (border)", legendgroup=f"c{label}", showlegend=False,
                    hovertemplate=f"Cluster {label+1} border<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"))
            if core.any():
                traces.append(go.Scatter(x=X[core,0], y=X[core,1], mode="markers",
                    marker=dict(size=point_size+1, color=color, opacity=0.95, line=dict(width=0)),
                    name=f"Cluster {label+1}", legendgroup=f"c{label}",
                    hovertemplate=f"Cluster {label+1} core<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"))
    return traces

def base_layout(title):
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COL, size=11),
        margin=dict(l=40, r=20, t=45, b=35),
        title=dict(text=title, font=dict(size=12, color=FONT_COL), x=0.01, xanchor="left"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(color=TICK_COL, size=9),
                   title=dict(text="Feature 1", font=dict(color=TICK_COL, size=10))),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(color=TICK_COL, size=9),
                   title=dict(text="Feature 2", font=dict(color=TICK_COL, size=10))),
        legend=dict(bgcolor="rgba(15,17,23,0.85)", bordercolor="#1e2130", borderwidth=1,
                    font=dict(size=10, color=FONT_COL), itemsizing="constant",
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hoverlabel=dict(bgcolor="#1e2130", bordercolor="#2d3348", font=dict(size=11, color="white"))
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">DBSCAN Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Explore density-based clustering — compare with K-Means, follow the guided tour, or step through the algorithm.</p>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔵  DBSCAN Explorer",
    "⚔️  DBSCAN vs K-Means",
    "🎓  Guided Tour",
    "🔍  Step-by-Step Algorithm"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DBSCAN Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    X = get_data(dataset, n_points, noise_level)
    labels, core_mask = run_dbscan(X, eps, min_samples)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    n_core     = int(np.sum(core_mask))
    n_border   = int(np.sum((labels != -1) & ~core_mask))

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        btn_label = "▶  Run DBSCAN" if not st.session_state.show_clustered else "◀  Show Raw"
        if st.button(btn_label, key="toggle_btn"):
            st.session_state.show_clustered = not st.session_state.show_clustered

    show_clustered = st.session_state.show_clustered

    if show_clustered:
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters", n_clusters)
        c2.metric("Core points", n_core)
        c3.metric("Border points", n_border)
        c4.metric("Noise points", n_noise)

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

    fig = go.Figure()
    if not show_clustered:
        fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers",
            marker=dict(size=point_size, color="#2d3348", opacity=0.9, line=dict(width=0.5, color="#374151")),
            name="Raw data", hovertemplate="x: %{x:.2f}  y: %{y:.2f}<extra></extra>"))
    else:
        for t in scatter_traces(X, labels, core_mask, point_size):
            fig.add_trace(t)
        if show_epsilon_circles:
            core_indices = np.where(core_mask)[0]
            sample = core_indices[::max(1, len(core_indices)//8)][:8]
            theta = np.linspace(0, 2*np.pi, 60)
            for idx in sample:
                cx, cy = X[idx,0], X[idx,1]
                lbl = labels[idx]
                col = PALETTE[lbl % len(PALETTE)] if lbl >= 0 else NOISE_COLOR
                fig.add_trace(go.Scatter(x=cx + eps*np.cos(theta), y=cy + eps*np.sin(theta),
                    mode="lines", line=dict(color=col, width=1, dash="dot"),
                    opacity=0.25, showlegend=False, hoverinfo="skip"))

    subtitle = (f"  →  <b>{n_clusters} cluster{'s' if n_clusters!=1 else ''}</b>, {n_noise} noise"
                if show_clustered else "  →  <i>raw data</i>")
    layout = base_layout(f"<b>{dataset}</b>   ε={eps:.2f}   MinPts={min_samples}" + subtitle)
    layout["height"] = 490
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    if show_clustered:
        if n_clusters == 0:
            cls, txt = "badge-bad",  "⚠  All noise — ε is too small or MinPts too high"
        elif n_clusters == 1 and n_noise < 5:
            cls, txt = "badge-warn", "⚠  Everything merged — ε is too large"
        elif n_noise > n_points * 0.5:
            cls, txt = "badge-warn", f"⚠  {n_noise} noise points ({n_noise/n_points*100:.0f}%) — try raising ε or lowering MinPts"
        else:
            cls, txt = "badge-good", f"✓  {n_clusters} cluster{'s' if n_clusters!=1 else ''}  ·  {n_noise} noise ({n_noise/n_points*100:.1f}%)"
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DBSCAN vs K-Means
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    X2 = get_data(dataset, n_points, noise_level)
    db_labels, db_core = run_dbscan(X2, eps, min_samples)
    km_labels, km_centers = run_kmeans(X2, k_clusters)

    st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    fig2 = make_subplots(rows=1, cols=2,
        subplot_titles=["DBSCAN", f"K-Means  (k={k_clusters})"],
        horizontal_spacing=0.06)

    for t in scatter_traces(X2, db_labels, db_core, point_size):
        t.showlegend = False
        fig2.add_trace(t, row=1, col=1)

    km_palette = ["#f97316","#4f8ef7","#10b981","#f43f5e","#8b5cf6","#06b6d4","#eab308","#ec4899"]
    for k in range(k_clusters):
        mask = km_labels == k
        color = km_palette[k % len(km_palette)]
        fig2.add_trace(go.Scatter(x=X2[mask,0], y=X2[mask,1], mode="markers",
            marker=dict(size=point_size, color=color, opacity=0.75, line=dict(width=0)),
            name=f"K-Means cluster {k+1}", showlegend=False,
            hovertemplate=f"K-Means cluster {k+1}<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"),
            row=1, col=2)

    fig2.add_trace(go.Scatter(x=km_centers[:,0], y=km_centers[:,1], mode="markers",
        marker=dict(size=14, color="white", symbol="star", line=dict(width=1.5, color="#374151")),
        name="Centroids", showlegend=False,
        hovertemplate="Centroid<br>x: %{x:.2f}  y: %{y:.2f}<extra></extra>"), row=1, col=2)

    fig2.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COL, size=11),
        margin=dict(l=40, r=20, t=55, b=35), height=490,
        hoverlabel=dict(bgcolor="#1e2130", bordercolor="#2d3348", font=dict(size=11, color="white")))
    for ann in fig2.layout.annotations:
        ann.font.color = FONT_COL; ann.font.size = 13
    for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
        fig2.update_layout(**{axis: dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                                         tickfont=dict(color=TICK_COL, size=9))})

    st.plotly_chart(fig2, use_container_width=True)

    if dataset in ["Moons", "Concentric Circles"]:
        st.markdown(
            '<span class="status-badge badge-bad">✕  K-Means fails here — it assumes convex, equally-sized clusters</span>'
            '&nbsp;&nbsp;'
            '<span class="status-badge badge-good">✓  DBSCAN follows density — shape does not matter</span>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-badge badge-warn">Both algorithms work on blob data — try Moons or Concentric Circles to see K-Means fail</span>',
            unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
    with st.expander("Why does K-Means fail on non-convex shapes?"):
        st.markdown("""
K-Means assigns each point to the **nearest centroid** using Euclidean distance.
This means it can only create boundaries that are straight lines (Voronoi regions) — it cannot follow curved or interleaved shapes.

DBSCAN doesn't use centroids at all. It asks: *"are there enough nearby points here?"*
That means it can find clusters of **any shape**, as long as they are dense enough.

**The key insight:** K-Means defines clusters by *geometry*. DBSCAN defines them by *density*.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Guided Tour
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    step = st.session_state.tour_step
    s = TOUR_STEPS[step]

    st.markdown(f"""
    <div class="tour-card">
      <p class="tour-step-label">Step {step+1} of {len(TOUR_STEPS)}</p>
      <p class="tour-title">{s['title']}</p>
      <p class="tour-desc">{s['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    n1, n2, n3 = st.columns([1, 1, 4])
    with n1:
        if st.button("← Previous", key="prev", disabled=(step == 0)):
            st.session_state.tour_step = max(0, step - 1)
            st.rerun()
    with n2:
        if st.button("Next →", key="next", disabled=(step == len(TOUR_STEPS)-1)):
            st.session_state.tour_step = min(len(TOUR_STEPS)-1, step + 1)
            st.rerun()

    progress = step / (len(TOUR_STEPS) - 1)
    st.markdown(f"""
    <div style="background:#1e2130;border-radius:999px;height:4px;margin:0.75rem 0 1rem 0;">
      <div style="background:#4f8ef7;height:4px;border-radius:999px;width:{progress*100:.0f}%;transition:width 0.3s;"></div>
    </div>
    """, unsafe_allow_html=True)

    Xt = get_data(s["dataset"], 300, 0.08)
    tour_labels, tour_core = run_dbscan(Xt, s["eps"], s["min_samples"])
    tour_km_labels, tour_km_centers = run_kmeans(Xt, s["k"])

    tour_n_clusters = len(set(tour_labels)) - (1 if -1 in tour_labels else 0)
    tour_n_noise    = int(np.sum(tour_labels == -1))

    if s["clustered"]:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters", tour_n_clusters)
        m2.metric("Core pts",  int(np.sum(tour_core)))
        m3.metric("Border pts", int(np.sum((tour_labels != -1) & ~tour_core)))
        m4.metric("Noise pts", tour_n_noise)
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

    if step >= 5:
        fig3 = make_subplots(rows=1, cols=2,
            subplot_titles=["DBSCAN", f"K-Means (k={s['k']})"],
            horizontal_spacing=0.06)

        for t in scatter_traces(Xt, tour_labels, tour_core, point_size):
            t.showlegend = False
            fig3.add_trace(t, row=1, col=1)

        for k in range(s["k"]):
            mask = tour_km_labels == k
            color = ["#f97316","#4f8ef7","#10b981","#f43f5e"][k % 4]
            fig3.add_trace(go.Scatter(x=Xt[mask,0], y=Xt[mask,1], mode="markers",
                marker=dict(size=point_size, color=color, opacity=0.75), showlegend=False,
                hovertemplate=f"K-Means cluster {k+1}<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"), row=1, col=2)

        fig3.add_trace(go.Scatter(x=tour_km_centers[:,0], y=tour_km_centers[:,1], mode="markers",
            marker=dict(size=14, color="white", symbol="star", line=dict(width=1.5, color="#374151")),
            showlegend=False, hovertemplate="Centroid<extra></extra>"), row=1, col=2)

        fig3.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            font=dict(family="Inter, sans-serif", color=FONT_COL, size=11),
            margin=dict(l=40, r=20, t=50, b=35), height=440,
            hoverlabel=dict(bgcolor="#1e2130", font=dict(size=11, color="white")))
        for ann in fig3.layout.annotations:
            ann.font.color = FONT_COL; ann.font.size = 12
        for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
            fig3.update_layout(**{axis: dict(showgrid=True, gridcolor=GRID_COL,
                                             zeroline=False, tickfont=dict(color=TICK_COL, size=9))})
        st.plotly_chart(fig3, use_container_width=True)

    else:
        fig3 = go.Figure()
        if not s["clustered"]:
            fig3.add_trace(go.Scatter(x=Xt[:,0], y=Xt[:,1], mode="markers",
                marker=dict(size=point_size, color="#2d3348", opacity=0.9,
                            line=dict(width=0.5, color="#374151")),
                name="Raw data", hovertemplate="x: %{x:.2f}  y: %{y:.2f}<extra></extra>"))
        else:
            for t in scatter_traces(Xt, tour_labels, tour_core, point_size):
                fig3.add_trace(t)

        layout3 = base_layout(f"<b>{s['dataset']}</b>   ε={s['eps']:.2f}   MinPts={s['min_samples']}")
        layout3["height"] = 440
        fig3.update_layout(**layout3)
        st.plotly_chart(fig3, use_container_width=True)

    if s["clustered"]:
        if tour_n_clusters == 0:
            cls, txt = "badge-bad",  "⚠  All noise"
        elif tour_n_clusters == 1 and tour_n_noise < 5:
            cls, txt = "badge-warn", "⚠  Everything merged"
        else:
            cls, txt = "badge-good", f"✓  {tour_n_clusters} cluster{'s' if tour_n_clusters!=1 else ''}  ·  {tour_n_noise} noise"
        st.markdown(f'<span class="status-badge {cls}">{txt}</span>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Step-by-Step Algorithm
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="background:#161b27;border:1px solid #1e2130;border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;">
      <p style="font-size:13px;color:#9ca3af;margin:0;line-height:1.6;">
        Walk through the DBSCAN algorithm step-by-step on a fixed dataset.
        Use <strong style="color:#f0f6fc">← → arrow keys</strong> or the buttons to navigate.
        Watch how core points, border points, and noise are discovered one step at a time.
      </p>
    </div>
    """, unsafe_allow_html=True)

    components.html(DBSCAN_STEPTHROUGH_HTML, height=620, scrolling=False)