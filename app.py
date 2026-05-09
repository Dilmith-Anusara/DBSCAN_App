import os
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

st.set_page_config(page_title="DBSCAN Explorer", layout="wide", page_icon="🔵")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body {
  background-color: #080b10 !important;
  margin: 0 !important; padding: 0 !important;
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
  background-color: #080b10 !important;
  color: #dde4ee !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}

[data-testid="stHeader"] { display: none !important; }

.block-container {
  padding: 2rem 2.5rem !important;
  max-width: 1440px !important;
  background-color: #080b10 !important;
  padding-top: 2rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: #060810 !important;
  border-right: 1px solid #141824 !important;
}
[data-testid="stSidebar"] * { color: #8a95a8 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #8a95a8 !important; font-size: 13px !important; }
[data-testid="stSidebar"] hr { border-color: #141824 !important; margin: 1rem 0 !important; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div {
  background: #0f1420 !important;
  border: 1px solid #1e2538 !important;
  border-radius: 6px !important;
}
[data-testid="stSlider"] > div > div > div > div { background: #00b8d4 !important; }

.sidebar-section {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: 0.14em !important;
  color: #00b8d4 !important; margin: 1.25rem 0 0.5rem 0 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-testid="stTab"] {
  background: #0d1220 !important;
  border: 1px solid #141824 !important;
  border-radius: 6px 6px 0 0 !important;
  color: #8a95a8 !important;
  font-size: 13px !important; font-weight: 500 !important;
  padding: 0.45rem 1.2rem !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  background: #00b8d4 !important;
  color: #080b10 !important;
  border-color: #00b8d4 !important;
  font-weight: 600 !important;
}
[data-testid="stTabPanel"] {
  background: #080b10 !important;
  border: 1px solid #141824 !important;
  border-radius: 0 6px 6px 6px !important;
  padding: 1.25rem !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  color: #00b8d4 !important;
  border: 1px solid #00b8d4 !important;
  border-radius: 6px !important;
  padding: 0.45rem 1.2rem !important;
  font-weight: 600 !important; font-size: 13px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  letter-spacing: 0.04em !important;
  width: 100% !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: #00b8d4 !important;
  color: #080b10 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: #0d1220 !important; border: 1px solid #141824 !important;
  border-radius: 8px !important; padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 10px !important; color: #5a6474 !important;
  font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
  font-size: 28px !important; font-weight: 700 !important;
  color: #eaf0f8 !important;
  font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Status badges ── */
.status-badge {
  display: inline-block; padding: 6px 14px; border-radius: 4px;
  font-size: 12px; font-weight: 600; margin-top: 0.75rem;
  font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.03em;
}
.badge-good { background: #011a0e; color: #34d399; border: 1px solid #065f38; }
.badge-warn { background: #1a1200; color: #fbbf24; border: 1px solid #854d0e; }
.badge-bad  { background: #1a0606; color: #f87171; border: 1px solid #991b1b; }

/* ── Tour card ── */
.tour-card {
  background: #0d1220; border: 1px solid #141824; border-left: 3px solid #00b8d4;
  border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.tour-step-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.14em; color: #00b8d4; margin-bottom: 6px;
}
.tour-title { font-size: 16px; font-weight: 600; color: #eaf0f8; margin-bottom: 6px; }
.tour-desc { font-size: 13.5px; color: #8a95a8; line-height: 1.65; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: #0d1220 !important; border: 1px solid #141824 !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: #8a95a8 !important; font-size: 13px !important; }
[data-testid="stExpander"] p, [data-testid="stExpander"] li { color: #8a95a8 !important; font-size: 13px !important; }
[data-testid="stExpander"] strong { color: #dde4ee !important; }

hr { border-color: #141824 !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

/* ── Landing page ── */
.landing-hero {
  text-align: center;
  padding: 60px 20px 40px;
}
.landing-eyebrow {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px; font-weight: 600; letter-spacing: 0.2em;
  text-transform: uppercase; color: #00b8d4;
  margin-bottom: 16px;
}
.landing-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 48px; font-weight: 700; color: #eaf0f8;
  letter-spacing: -0.02em; line-height: 1.1;
  margin-bottom: 16px;
}
.landing-title span { color: #00b8d4; }
.landing-sub {
  font-size: 16px; color: #5a6474; line-height: 1.6;
  max-width: 520px; margin: 0 auto 52px;
}

.mode-card {
  background: #0d1220;
  border: 1px solid #1e2538;
  border-radius: 12px;
  padding: 36px 32px 28px;
  text-align: left;
  transition: border-color 0.2s, transform 0.2s;
  cursor: pointer;
  height: 100%;
}
.mode-card:hover {
  border-color: #00b8d4;
  transform: translateY(-2px);
}
.mode-card-icon {
  font-size: 32px; margin-bottom: 16px; display: block;
}
.mode-card-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 18px; font-weight: 700; color: #eaf0f8;
  margin-bottom: 10px;
}
.mode-card-desc {
  font-size: 14px; color: #5a6474; line-height: 1.6;
  margin-bottom: 20px;
}
.mode-card-tag {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 4px 10px;
  border-radius: 3px; color: #00b8d4;
  background: rgba(0,184,212,0.08); border: 1px solid rgba(0,184,212,0.25);
}

/* Section headers */
.section-header {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 24px;
}
.section-back-btn {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px; font-weight: 600;
  color: #00b8d4; letter-spacing: 0.06em;
  text-decoration: none; cursor: pointer;
  padding: 5px 12px; border: 1px solid #00b8d4;
  border-radius: 4px; background: transparent;
  transition: all 0.2s;
}
.section-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 20px; font-weight: 700; color: #eaf0f8;
}
.section-divider {
  height: 1px; background: #141824; margin: 0 0 28px 0;
}

/* Explainer wrapper */
.explainer-frame {
  border: 1px solid #141824;
  border-radius: 10px;
  overflow: hidden;
  background: #090c10;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE    = ["#00b8d4","#f97316","#10b981","#f43f5e","#8b5cf6","#06b6d4","#eab308","#ec4899","#14b8a6","#6366f1"]
NOISE_COLOR = "#2e3748"
PLOT_BG    = "#080b10"
GRID_COL   = "#0d1220"
TICK_COL   = "#2e3748"
FONT_COL   = "#dde4ee"

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "page":           "landing",   # "landing" | "dashboard" | "explainer"
    "show_clustered": False,
    "tour_step":      0,
}
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
                x=X[mask,0], y=X[mask,1], mode="markers",
                marker=dict(size=point_size-1, color=NOISE_COLOR, opacity=0.7,
                            symbol="x", line=dict(width=1.5, color=NOISE_COLOR)),
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
                    x=X[border,0], y=X[border,1], mode="markers",
                    marker=dict(size=point_size-1, color=color, opacity=0.28,
                                line=dict(width=1, color=color)),
                    name=f"Cluster {label+1} (border)",
                    legendgroup=f"c{label}", showlegend=False,
                    hovertemplate=f"Cluster {label+1} border<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
                ))
            if core.any():
                traces.append(go.Scatter(
                    x=X[core,0], y=X[core,1], mode="markers",
                    marker=dict(size=point_size+1, color=color, opacity=0.95),
                    name=f"Cluster {label+1}",
                    legendgroup=f"c{label}",
                    hovertemplate=f"Cluster {label+1} core<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
                ))
    return traces

def base_layout(title, height=490):
    return dict(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(family="'IBM Plex Mono', monospace", color=FONT_COL, size=11),
        margin=dict(l=40, r=20, t=45, b=35),
        height=height,
        title=dict(text=title, font=dict(size=11, color=FONT_COL), x=0.01, xanchor="left"),
        xaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(color=TICK_COL, size=9),
                   title=dict(text="Feature 1", font=dict(color=TICK_COL, size=10))),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False,
                   tickfont=dict(color=TICK_COL, size=9),
                   title=dict(text="Feature 2", font=dict(color=TICK_COL, size=10))),
        legend=dict(bgcolor="rgba(8,11,16,0.9)", bordercolor="#1e2538", borderwidth=1,
                    font=dict(size=10, color=FONT_COL), itemsizing="constant",
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        hoverlabel=dict(bgcolor="#0d1220", bordercolor="#1e2538",
                        font=dict(size=11, color="white"))
    )

# ── Read HTML explainer ───────────────────────────────────────────────────────
@st.cache_data
def load_explainer_html():
    """Load the DBSCAN step-by-step HTML. Looks next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "dbscan_explainer.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<p style='color:#f87171;padding:2rem'>dbscan_explainer.html not found next to app.py</p>"

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LANDING
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    st.markdown("""
    <div class="landing-hero">
      <div class="landing-eyebrow">Clustering · Machine Learning · Visualization</div>
      <div class="landing-title">DBSCAN<span>.</span><br>Explorer</div>
      <div class="landing-sub">
        Understand density-based clustering through interactive exploration
        and a step-by-step visual walkthrough.
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="mode-card">
          <span class="mode-card-icon">⚡</span>
          <div class="mode-card-title">Interactive Dashboard</div>
          <div class="mode-card-desc">
            Tune ε and MinPts live on real datasets. Compare DBSCAN against
            K-Means side by side. Follow a guided parameter tour to build intuition.
          </div>
          <span class="mode-card-tag">Explore · Compare · Tune</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        if st.button("Open Dashboard →", key="go_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()

    with right:
        st.markdown("""
        <div class="mode-card">
          <span class="mode-card-icon">🔍</span>
          <div class="mode-card-title">Step-by-Step Explainer</div>
          <div class="mode-card-desc">
            Walk through how DBSCAN classifies every point — core, border, noise —
            one step at a time. Understand density-reachability and density-connectivity
            through animated diagrams.
          </div>
          <span class="mode-card-tag">Learn · Visualize · Understand</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        if st.button("Open Explainer →", key="go_explainer"):
            st.session_state.page = "explainer"
            st.rerun()

    st.markdown("""
    <div style="text-align:center;margin-top:60px;color:#1e2538;
                font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.1em;">
      DBSCAN · Density-Based Spatial Clustering of Applications with Noise · Ester et al. 1996
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: STEP-BY-STEP EXPLAINER
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "explainer":
    # Header row
    hcol1, hcol2 = st.columns([1, 6])
    with hcol1:
        if st.button("← Back to Home", key="back_explainer"):
            st.session_state.page = "landing"
            st.rerun()
    with hcol2:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;padding-top:4px">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                       font-weight:700;color:#eaf0f8;">Step-by-Step Explainer</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                       color:#00b8d4;background:rgba(0,184,212,0.08);
                       border:1px solid rgba(0,184,212,0.25);padding:3px 10px;
                       border-radius:3px;letter-spacing:0.1em;text-transform:uppercase;">
            15 steps
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider' style='margin-top:14px'></div>", unsafe_allow_html=True)

    # Short intro
    st.markdown("""
    <p style="color:#5a6474;font-size:13.5px;line-height:1.7;margin-bottom:20px;max-width:780px;">
      This visualizer walks through how DBSCAN classifies every point — step by step.
      Use <strong style="color:#dde4ee">Next / Prev</strong> or your <strong style="color:#dde4ee">← →</strong>
      arrow keys to advance. The final three steps cover the key theoretical concepts:
      direct density-reachability, density-reachability, and density-connectivity.
    </p>
    """, unsafe_allow_html=True)

    # Crosslink to dashboard
    st.markdown("""
    <div style="background:#0d1220;border:1px solid #141824;border-left:3px solid #00b8d4;
                border-radius:8px;padding:14px 18px;margin-bottom:22px;
                display:flex;align-items:center;gap:16px;">
      <span style="font-size:18px;">⚡</span>
      <div>
        <span style="font-size:13px;color:#8a95a8;">
          Want to tune parameters and see DBSCAN run on real datasets?
        </span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                     color:#00b8d4;margin-left:8px;">
          → Switch to the Interactive Dashboard
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    xcol, _ = st.columns([1, 5])
    with xcol:
        if st.button("Open Dashboard →", key="explainer_to_dash"):
            st.session_state.page = "dashboard"
            st.rerun()

    # Embed the HTML visualizer
    html_content = load_explainer_html()
    st.markdown("<div class='explainer-frame'>", unsafe_allow_html=True)
    components.html(html_content, height=720, scrolling=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INTERACTIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "dashboard":

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p class="sidebar-section">Dataset</p>', unsafe_allow_html=True)
        dataset     = st.selectbox("Dataset", ["Moons","Blobs","Concentric Circles","Random Noise"],
                                   label_visibility="collapsed")
        n_points    = st.slider("Number of points", 100, 600, 300, step=50)
        noise_level = st.slider("Dataset noise", 0.01, 0.15, 0.08, step=0.01)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">DBSCAN Parameters</p>', unsafe_allow_html=True)
        eps         = st.slider("ε — neighborhood radius", 0.05, 1.5, 0.35, step=0.01)
        min_samples = st.slider("MinPts — core point threshold", 1, 20, 5, step=1)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">K-Means Parameters</p>', unsafe_allow_html=True)
        k_clusters  = st.slider("K — number of clusters", 2, 8, 2, step=1)

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-section">Display</p>', unsafe_allow_html=True)
        show_epsilon_circles = st.checkbox("Show ε neighborhood circles", value=False)
        point_size  = st.slider("Point size", 4, 16, 7)

    # ── Page header ──────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([1, 6])
    with hcol1:
        if st.button("← Back to Home", key="back_dashboard"):
            st.session_state.page = "landing"
            st.rerun()
    with hcol2:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;padding-top:4px">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                       font-weight:700;color:#eaf0f8;">Interactive Dashboard</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider' style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    TOUR_STEPS = [
        {
            "title": "Start: Raw Data",
            "desc": "This is your dataset — no clustering applied yet. Every point is neutral. Notice the shape: two interleaved moons. Ask yourself — how would you separate these by hand?",
            "dataset": "Moons", "eps": 0.35, "min_samples": 5, "clustered": False, "k": 2
        },
        {
            "title": "ε Too Small — Everything is Noise",
            "desc": "With ε = 0.05, almost no point has enough neighbors. The algorithm sees nothing but outliers. This is what happens when your neighborhood radius is too tight.",
            "dataset": "Moons", "eps": 0.05, "min_samples": 5, "clustered": True, "k": 2
        },
        {
            "title": "ε Too Large — Everything Merges",
            "desc": "With ε = 1.2, every point's neighborhood swallows the entire dataset. Both moons collapse into one cluster. DBSCAN loses all discrimination.",
            "dataset": "Moons", "eps": 1.2, "min_samples": 5, "clustered": True, "k": 2
        },
        {
            "title": "Sweet Spot — Clean Separation",
            "desc": "With ε = 0.35 and MinPts = 5, DBSCAN finds exactly two clusters following the natural moon shapes. Notice the core points (bright) vs border points (faded) at the edges.",
            "dataset": "Moons", "eps": 0.35, "min_samples": 5, "clustered": True, "k": 2
        },
        {
            "title": "MinPts Too High — Sparse Points Become Noise",
            "desc": "Raising MinPts to 15 makes the algorithm stricter. Only very dense regions qualify as core points. Border regions get classified as noise — shown as ✕ marks.",
            "dataset": "Moons", "eps": 0.35, "min_samples": 15, "clustered": True, "k": 2
        },
        {
            "title": "Where K-Means Fails",
            "desc": "K-Means assumes spherical clusters and cuts space with straight boundaries — it cannot follow the moon shapes. DBSCAN follows density, not geometry.",
            "dataset": "Moons", "eps": 0.35, "min_samples": 5, "clustered": True, "k": 2
        },
        {
            "title": "Concentric Circles — Another K-Means Failure",
            "desc": "K-Means splits the circles vertically — completely wrong. DBSCAN correctly identifies the inner and outer ring as separate clusters because they have different densities and shapes.",
            "dataset": "Concentric Circles", "eps": 0.25, "min_samples": 5, "clustered": True, "k": 2
        },
    ]

    tab1, tab2, tab3 = st.tabs(["🔵  DBSCAN Explorer", "⚔️  DBSCAN vs K-Means", "🎓  Guided Tour"])

    # ── TAB 1 — DBSCAN Explorer ───────────────────────────────────────────────
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
            <div style="padding-top:6px">
              <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#5a6474;">
                New to DBSCAN?
              </span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("→ Step-by-Step Explainer", key="tab1_to_explainer"):
                st.session_state.page = "explainer"
                st.rerun()

        show_clustered = st.session_state.show_clustered

        if show_clustered:
            st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clusters",      n_clusters)
            c2.metric("Core points",   n_core)
            c3.metric("Border points", n_border)
            c4.metric("Noise points",  n_noise)

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

        fig = go.Figure()
        if not show_clustered:
            fig.add_trace(go.Scatter(
                x=X[:,0], y=X[:,1], mode="markers",
                marker=dict(size=point_size, color="#1e2538", opacity=0.9,
                            line=dict(width=0.5, color="#2e3748")),
                name="Raw data",
                hovertemplate="x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
            ))
        else:
            for t in scatter_traces(X, labels, core_mask, point_size):
                fig.add_trace(t)
            if show_epsilon_circles:
                core_indices = np.where(core_mask)[0]
                sample = core_indices[::max(1, len(core_indices)//8)][:8]
                theta = np.linspace(0, 2*np.pi, 60)
                for idx in sample:
                    cx, cy = X[idx,0], X[idx,1]
                    lbl    = labels[idx]
                    col    = PALETTE[lbl % len(PALETTE)] if lbl >= 0 else NOISE_COLOR
                    fig.add_trace(go.Scatter(
                        x=cx + eps*np.cos(theta), y=cy + eps*np.sin(theta),
                        mode="lines", line=dict(color=col, width=1, dash="dot"),
                        opacity=0.2, showlegend=False, hoverinfo="skip"
                    ))

        subtitle = (
            f"  →  <b>{n_clusters} cluster{'s' if n_clusters!=1 else ''}</b>, {n_noise} noise"
            if show_clustered else "  →  <i>raw data</i>"
        )
        layout = base_layout(f"<b>{dataset}</b>   ε={eps:.2f}   MinPts={min_samples}" + subtitle)
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

    # ── TAB 2 — DBSCAN vs K-Means ─────────────────────────────────────────────
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

        km_pal = ["#f97316","#00b8d4","#10b981","#f43f5e","#8b5cf6","#06b6d4","#eab308","#ec4899"]
        for k in range(k_clusters):
            mask  = km_labels == k
            color = km_pal[k % len(km_pal)]
            fig2.add_trace(go.Scatter(
                x=X2[mask,0], y=X2[mask,1], mode="markers",
                marker=dict(size=point_size, color=color, opacity=0.75),
                name=f"K-Means cluster {k+1}", showlegend=False,
                hovertemplate=f"K-Means cluster {k+1}<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
            ), row=1, col=2)
        fig2.add_trace(go.Scatter(
            x=km_centers[:,0], y=km_centers[:,1], mode="markers",
            marker=dict(size=14, color="white", symbol="star",
                        line=dict(width=1.5, color="#374151")),
            name="Centroids", showlegend=False,
            hovertemplate="Centroid<br>x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
        ), row=1, col=2)

        fig2.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
            font=dict(family="'IBM Plex Mono', monospace", color=FONT_COL, size=11),
            margin=dict(l=40, r=20, t=55, b=35), height=490,
            hoverlabel=dict(bgcolor="#0d1220", bordercolor="#1e2538",
                            font=dict(size=11, color="white")),
        )
        for ann in fig2.layout.annotations:
            ann.font.color = FONT_COL; ann.font.size = 12
        for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
            fig2.update_layout(**{axis: dict(showgrid=True, gridcolor=GRID_COL,
                                             zeroline=False, tickfont=dict(color=TICK_COL, size=9))})
        st.plotly_chart(fig2, use_container_width=True)

        if dataset in ["Moons", "Concentric Circles"]:
            st.markdown(
                '<span class="status-badge badge-bad">✕  K-Means fails here — it assumes convex, equally-sized clusters</span>'
                '&nbsp;&nbsp;'
                '<span class="status-badge badge-good">✓  DBSCAN follows density — shape does not matter</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge badge-warn">Both algorithms work on blob data — try Moons or Concentric Circles to see K-Means fail</span>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
        with st.expander("Why does K-Means fail on non-convex shapes?"):
            st.markdown("""
K-Means assigns each point to the **nearest centroid** using Euclidean distance.
This means it can only create boundaries that are straight lines (Voronoi regions) — it cannot follow curved or interleaved shapes.

DBSCAN doesn't use centroids at all. It asks: *"are there enough nearby points here?"*
That means it can find clusters of **any shape**, as long as they are dense enough.

**The key insight:** K-Means defines clusters by *geometry*. DBSCAN defines them by *density*.
            """)

    # ── TAB 3 — Guided Tour ───────────────────────────────────────────────────
    with tab3:
        step = st.session_state.tour_step
        s    = TOUR_STEPS[step]

        # Crosslink to explainer at top of tour
        st.markdown("""
        <div style="background:#0d1220;border:1px solid #141824;border-left:3px solid #00b8d4;
                    border-radius:8px;padding:12px 16px;margin-bottom:16px;">
          <span style="font-size:13px;color:#5a6474;">
            Want the step-by-step algorithmic walkthrough?
          </span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#00b8d4;margin-left:8px;">
            → Open the Step-by-Step Explainer
          </span>
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
            if st.button("← Previous", key="prev", disabled=(step == 0)):
                st.session_state.tour_step = max(0, step - 1)
                st.rerun()
        with n2:
            if st.button("Next →", key="next", disabled=(step == len(TOUR_STEPS)-1)):
                st.session_state.tour_step = min(len(TOUR_STEPS)-1, step + 1)
                st.rerun()

        progress = step / (len(TOUR_STEPS) - 1)
        st.markdown(f"""
        <div style="background:#141824;border-radius:999px;height:3px;margin:0.75rem 0 1rem 0;">
          <div style="background:#00b8d4;height:3px;border-radius:999px;
                      width:{progress*100:.0f}%;transition:width 0.3s;"></div>
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
                color = ["#f97316","#00b8d4","#10b981","#f43f5e"][k % 4]
                fig3.add_trace(go.Scatter(
                    x=Xt[mask,0], y=Xt[mask,1], mode="markers",
                    marker=dict(size=point_size, color=color, opacity=0.75),
                    showlegend=False,
                    hovertemplate=f"K-Means cluster {k+1}<br>x: %{{x:.2f}}  y: %{{y:.2f}}<extra></extra>"
                ), row=1, col=2)
            fig3.add_trace(go.Scatter(
                x=tour_km_centers[:,0], y=tour_km_centers[:,1], mode="markers",
                marker=dict(size=14, color="white", symbol="star",
                            line=dict(width=1.5, color="#374151")),
                showlegend=False, hovertemplate="Centroid<extra></extra>"
            ), row=1, col=2)
            fig3.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
                font=dict(family="'IBM Plex Mono', monospace", color=FONT_COL, size=11),
                margin=dict(l=40, r=20, t=50, b=35), height=440,
                hoverlabel=dict(bgcolor="#0d1220", font=dict(size=11, color="white"))
            )
            for ann in fig3.layout.annotations:
                ann.font.color = FONT_COL; ann.font.size = 12
            for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
                fig3.update_layout(**{axis: dict(showgrid=True, gridcolor=GRID_COL,
                                                 zeroline=False, tickfont=dict(color=TICK_COL, size=9))})
            st.plotly_chart(fig3, use_container_width=True)
        else:
            fig3 = go.Figure()
            if not s["clustered"]:
                fig3.add_trace(go.Scatter(
                    x=Xt[:,0], y=Xt[:,1], mode="markers",
                    marker=dict(size=point_size, color="#1e2538", opacity=0.9,
                                line=dict(width=0.5, color="#2e3748")),
                    name="Raw data",
                    hovertemplate="x: %{x:.2f}  y: %{y:.2f}<extra></extra>"
                ))
            else:
                for t in scatter_traces(Xt, tour_labels, tour_core, point_size):
                    fig3.add_trace(t)
            layout3 = base_layout(f"<b>{s['dataset']}</b>   ε={s['eps']:.2f}   MinPts={s['min_samples']}", height=440)
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