# ── stdlib / ML ──────────────────────────────────────────────
import io, os, copy, math, warnings, tempfile
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── matplotlib (non-interactive) ─────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Streamlit + Plotly ────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG 
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title=" ChronicML Condition Intelligence",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --c-primary: #4F8BFF;
    --c-secondary: #F472B6;
    --c-accent: #FB923C;
    --c-good: #10B981;
    --c-bad: #F87171;
    --c-warn: #F59E0B;
    --c-bg: #080C14;
    --c-panel: #0D1321;
    --c-text-1: #F1F5F9;
    --c-text-2: #E2E8F0;
    --c-text-3: #CBD5E1;
    --c-text-dim: #94A3B8;
    --c-text-faint: #64748B;
    --c-border: rgba(255,255,255,0.06);
    --c-border-accent: rgba(79,139,255,0.18);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Base with radial gradient texture ── */
.stApp {
    background:
      radial-gradient(ellipse 80% 60% at 12% 0%, rgba(79,139,255,0.06), transparent 60%),
      radial-gradient(ellipse 70% 50% at 92% 100%, rgba(244,114,182,0.045), transparent 55%),
      #080C14;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0D1321 0%, #080C14 100%);
    border-right: 1px solid rgba(79,139,255,0.12);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--c-text-dim); }

/* ── Top status chip (sticky) ── */
.top-banner {
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(180deg, rgba(8,12,20,0.97) 60%, rgba(8,12,20,0.7) 100%);
    backdrop-filter: blur(8px);
    padding: 14px 4px 12px;
    margin: -2px -2px 18px;
    border-bottom: 1px solid var(--c-border);
}
.top-banner-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid;
}
.status-chip .dot {
    width: 8px; height: 8px; border-radius: 50%;
    box-shadow: 0 0 10px currentColor;
}
.chip-idle   { color: var(--c-warn);    border-color: rgba(245,158,11,0.35);  background: rgba(245,158,11,0.08); }
.chip-run    { color: var(--c-primary); border-color: rgba(79,139,255,0.35);  background: rgba(79,139,255,0.08); }
.chip-done   { color: var(--c-good);    border-color: rgba(16,185,129,0.35);  background: rgba(16,185,129,0.08); }
.chip-err    { color: var(--c-bad);     border-color: rgba(248,113,113,0.35); background: rgba(248,113,113,0.08); }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(79,139,255,0.07) 0%, rgba(244,114,182,0.05) 100%);
    border: 1px solid var(--c-border-accent);
    border-radius: 16px;
    padding: 20px 22px 18px;
    margin: 4px 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
    min-width: 160px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--c-primary), var(--c-secondary));
    opacity: 0.75;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(79,139,255,0.15);
    border-color: rgba(79,139,255,0.35);
}
.kpi-icon {
    font-size: 1.15em;
    margin-bottom: 6px;
    opacity: 0.82;
}
.kpi-val {
    color: var(--c-text-1);    /* fallback if gradient clipping fails */
    font-size: 2.05em;
    font-weight: 700;
    background: linear-gradient(120deg, var(--c-primary) 20%, var(--c-secondary) 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.12;
    letter-spacing: -0.5px;
}
.kpi-label {
    font-size: 0.7em;
    color: var(--c-text-faint);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-top: 10px;
    font-weight: 500;
}
.kpi-delta-good { color: var(--c-good); font-size: 0.82em; margin-top: 4px; font-weight: 500; }
.kpi-delta-bad  { color: var(--c-bad);  font-size: 0.82em; margin-top: 4px; font-weight: 500; }

/* ── Unified Header System ── */
.h-page {
    font-size: 1.55em;
    font-weight: 700;
    color: var(--c-text-1);
    padding: 6px 0 10px;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, var(--c-primary), var(--c-secondary), transparent) 1;
    margin: 4px 0 22px 0;
    letter-spacing: -0.3px;
}
.h-section {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--c-text-2);
    border-left: 3px solid var(--c-primary);
    padding-left: 12px;
    margin: 22px 0 12px 0;
    letter-spacing: 0.1px;
}
.h-sub {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--c-text-3);
    margin: 14px 0 8px 0;
    letter-spacing: 0.2px;
    text-transform: uppercase;
}
/* Back-compat aliases */
.sec-head      { font-size:1.55em; font-weight:700; color:var(--c-text-1);
                 padding:6px 0 10px;
                 border-bottom:2px solid transparent;
                 border-image: linear-gradient(90deg, var(--c-primary), var(--c-secondary), transparent) 1;
                 margin:4px 0 22px 0; letter-spacing:-0.3px; }
.section-hdr   { font-size:1.05rem; font-weight:600; color:var(--c-text-2);
                 border-left:3px solid var(--c-primary); padding-left:12px;
                 margin:22px 0 12px 0; letter-spacing:0.1px; }

/* ── Glass Cards ── */
.glass {
    background: rgba(13,19,33,0.92);
    border: 1px solid var(--c-border);
    border-radius: 16px;
    padding: 22px 26px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
}
.glass-accent {
    background: linear-gradient(135deg, rgba(79,139,255,0.06), rgba(244,114,182,0.04));
    border: 1px solid rgba(79,139,255,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
}

/* ── Pipeline Stepper (home page) ── */
.stepper {
    display: flex;
    gap: 0;
    margin: 8px 0 4px;
    flex-wrap: wrap;
}
.stepper-node {
    flex: 1;
    min-width: 140px;
    padding: 14px 14px 14px 18px;
    position: relative;
}
.stepper-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--c-primary), var(--c-secondary));
    color: white;
    font-weight: 700;
    font-size: 0.8em;
    margin-bottom: 8px;
}
.stepper-title {
    color: var(--c-text-2);
    font-size: 0.92em;
    font-weight: 600;
    margin-bottom: 4px;
}
.stepper-desc {
    color: var(--c-text-dim);
    font-size: 0.82em;
    line-height: 1.5;
}
.stepper-node:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 28px; right: 0;
    width: 40%;
    height: 1px;
    background: linear-gradient(90deg, rgba(79,139,255,0.3), transparent);
}

/* ── Insight Boxes with auto icons ── */
.insight {
    background: linear-gradient(135deg, rgba(79,139,255,0.08), rgba(244,114,182,0.04));
    border-left: 3px solid var(--c-primary);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px 14px 14px;
    margin: 12px 0;
    color: var(--c-text-3);
    font-size: 0.9em;
    line-height: 1.65;
    position: relative;
}
.insight::before {
    content: 'ⓘ';
    color: var(--c-primary);
    font-weight: 700;
    margin-right: 8px;
    font-size: 1.05em;
}
.insight-critical {
    border-left-color: var(--c-bad);
    background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(239,68,68,0.04));
}
.insight-critical::before { content: '⚠'; color: var(--c-bad); }
.insight-success {
    border-left-color: var(--c-good);
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.04));
}
.insight-success::before { content: '✓'; color: var(--c-good); }
.insight-warning {
    border-left-color: var(--c-warn);
    background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(245,158,11,0.04));
}
.insight-warning::before { content: '!'; color: var(--c-warn); }

/* ── Info / Status Boxes ── */
.info-box {
    background: rgba(79,139,255,0.07);
    border-left: 3px solid var(--c-primary);
    padding: 10px 14px;
    border-radius: 4px;
    margin: 6px 0;
    color: var(--c-text-3);
    font-size: 0.88em;
}
.success-box {
    background: rgba(16,185,129,0.07);
    border-left: 3px solid var(--c-good);
    padding: 10px 14px;
    border-radius: 4px;
    margin: 6px 0;
    color: var(--c-text-3);
    font-size: 0.88em;
}
.warn-box {
    background: rgba(248,113,113,0.07);
    border-left: 3px solid var(--c-bad);
    padding: 10px 14px;
    border-radius: 4px;
    margin: 6px 0;
    color: var(--c-text-3);
    font-size: 0.88em;
}
.diag-box {
    background: rgba(244,114,182,0.07);
    border-left: 3px solid var(--c-secondary);
    padding: 10px 14px;
    border-radius: 4px;
    margin: 6px 0;
    color: var(--c-text-3);
    font-size: 0.88em;
}

/* ── File checklist (sidebar) ── */
.file-check { display:flex; justify-content:space-between; align-items:center;
              padding:4px 10px; margin:2px 0; border-radius:6px; font-size:0.78em;
              font-family:'JetBrains Mono', monospace; }
.file-check.ok       { background: rgba(16,185,129,0.07);  color: #CBD5E1; }
.file-check.ok b     { color: var(--c-good); }
.file-check.missing  { background: rgba(248,113,113,0.06); color: #94A3B8; }
.file-check.missing b{ color: var(--c-bad); }
.file-check.opt      { background: rgba(79,139,255,0.05);  color: #94A3B8; }
.file-check.opt b    { color: var(--c-primary); }

/* ── Progress Checklist (during pipeline run) ── */
.progress-list { margin: 4px 0; }
.progress-step {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 14px; margin: 4px 0;
    background: rgba(13,19,33,0.7);
    border: 1px solid var(--c-border);
    border-radius: 8px;
    font-size: 0.88em;
    color: var(--c-text-dim);
    transition: all 0.3s ease;
}
.progress-step.active {
    background: rgba(79,139,255,0.1);
    border-color: rgba(79,139,255,0.35);
    color: var(--c-text-2);
}
.progress-step.done {
    background: rgba(16,185,129,0.06);
    border-color: rgba(16,185,129,0.2);
    color: var(--c-text-3);
}
.progress-icon {
    width: 22px; height: 22px;
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.7em; font-weight: 700;
    flex-shrink: 0;
}
.progress-step .progress-icon { background: rgba(100,116,139,0.15); color: var(--c-text-faint); }
.progress-step.active .progress-icon { background: var(--c-primary); color: white; animation: pulse 1.5s ease-in-out infinite; }
.progress-step.done .progress-icon { background: var(--c-good); color: white; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(79,139,255,0.45); }
    50%      { box-shadow: 0 0 0 8px rgba(79,139,255,0); }
}

/* ── Badges ── */
.badge-low  { background:#0D2B1D; color:#10B981; border:1px solid #10B981; border-radius:20px; padding:2px 12px; font-size:0.8em; font-weight:600; }
.badge-mod  { background:#2D2007; color:#F59E0B; border:1px solid #F59E0B; border-radius:20px; padding:2px 12px; font-size:0.8em; font-weight:600; }
.badge-high { background:#2D1111; color:#F87171; border:1px solid #F87171; border-radius:20px; padding:2px 12px; font-size:0.8em; font-weight:600; }
.badge-crit { background:#200A0A; color:#FCA5A5; border:1px solid #FCA5A5; border-radius:20px; padding:2px 12px; font-size:0.8em; font-weight:600; }

/* ── Tabs (tightened for 11 tabs to fit) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid var(--c-border);
    background: rgba(13,19,33,0.6);
    border-radius: 10px 10px 0 0;
    padding: 4px 4px 0;
    overflow-x: auto;
    scrollbar-width: thin;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--c-text-faint);
    padding: 8px 13px;
    border-radius: 8px 8px 0 0;
    font-size: 0.8em;
    font-weight: 500;
    letter-spacing: 0.15px;
    transition: color 0.2s, background 0.2s;
    white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--c-text-3); background: rgba(79,139,255,0.04); }
.stTabs [aria-selected="true"] {
    background: rgba(79,139,255,0.1) !important;
    color: var(--c-primary) !important;
    border-bottom: 2px solid var(--c-primary) !important;
    font-weight: 600;
}

/* ── Metric cards (Streamlit native) ── */
[data-testid="metric-container"] {
    background: rgba(13,19,33,0.8);
    border: 1px solid rgba(79,139,255,0.12);
    border-radius: 12px;
    padding: 14px 18px;
}

/* ── Dataframes ── */
.stDataFrame { border-radius: 12px; overflow: hidden;
               border: 1px solid var(--c-border); }
[data-testid="stDataFrame"] { border-radius: 12px;
                              border: 1px solid var(--c-border); }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--c-primary), var(--c-secondary)) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s, transform 0.2s;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.92;
    transform: translateY(-1px);
}

/* ── Download buttons (tinted) ── */
[data-testid="stDownloadButton"] button {
    background: rgba(79,139,255,0.08) !important;
    border: 1px solid rgba(79,139,255,0.25) !important;
    color: var(--c-text-2) !important;
    border-radius: 8px !important;
    font-size: 0.82em !important;
    font-weight: 500 !important;
    transition: all 0.2s;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(79,139,255,0.15) !important;
    border-color: var(--c-primary) !important;
    transform: translateY(-1px);
}

/* ── Expanders (cleaner look) ── */
[data-testid="stExpander"] {
    border: 1px solid var(--c-border) !important;
    border-radius: 10px !important;
    background: rgba(13,19,33,0.6) !important;
}

/* ── Dividers ── */
hr { border-color: var(--c-border) !important; }

/* ── Hide streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--c-primary), var(--c-secondary)); }

/* ── Code blocks ── */
code { font-family: 'JetBrains Mono', monospace; font-size: 0.88em; }
</style>
""", unsafe_allow_html=True)
# TEAM INFO
TEAM_NUMBER  = "27"    
TEAM_MEMBERS = [             
    "Harshal Shah (2023A7PS0055H)",
    "Marmik Sapovadia (2023A7PS0057H)",
    "Riya Doshi (2023AAPS0210H)",
    "Archit Khandelwal (2023AAPS0184H)"
]
# CONSTANTS
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,20,0.7)",
    font=dict(color="#94A3B8", family="Space Grotesk, sans-serif", size=12),
    title_font=dict(color="#E2E8F0", size=14, family="Space Grotesk, sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)",
               tickfont=dict(color="#64748B"), title_font=dict(color="#94A3B8")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)",
               tickfont=dict(color="#64748B"), title_font=dict(color="#94A3B8")),
    margin=dict(l=40, r=20, t=55, b=40),
    hoverlabel=dict(bgcolor="#0D1321", font_color="#E2E8F0", bordercolor="rgba(79,139,255,0.3)"),
)
C_D1 = "#4F8BFF"; C_D2 = "#FB923C"; C_POS = "#F472B6"
C_NEG = "#4F8BFF"; C_GREEN = "#10B981"; C_AMBER = "#F59E0B"
# Canonical multi-series palette for comparison charts (ROC, PR, per-model plots)
MULTI_SERIES_COLORS = ["#4F8BFF", "#F472B6", "#FB923C", "#34D399", "#FBBF24",
                       "#A855F7", "#F87171", "#22D3EE", "#F59E0B", "#EC4899"]
TEMPORAL_SPLIT_DATE = "2015-01-01"
TEST_SIZE = 0.2
SVM_MAX_SAMPLES = 15000

ADV_PLT_PARAMS = {
    "figure.facecolor": "#080C14", "axes.facecolor": "#0D1321",
    "axes.labelcolor": "#94A3B8", "xtick.color": "#64748B", "ytick.color": "#64748B",
    "text.color": "#CBD5E1", "grid.color": "#1E293B", "grid.linewidth": 0.5,
    "axes.titlecolor": "#E2E8F0", "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#1E293B", "axes.spines.top": False, "axes.spines.right": False,
}

OBS_FEATURE_MAP = {
    "Body Height": "body_height", "Body Weight": "body_weight",
    "Body mass index (BMI) [Ratio]": "bmi", "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp", "Heart rate": "heart_rate",
    "Respiratory rate": "respiratory_rate",
    "Pain severity - 0-10 verbal numeric rating [Score] - Reported": "pain_severity",
    "Glucose [Mass/volume] in Blood": "glucose",
    "Creatinine [Mass/volume] in Blood": "creatinine",
    "Calcium [Mass/volume] in Blood": "calcium",
    "Sodium [Moles/volume] in Blood": "sodium",
    "Potassium [Moles/volume] in Blood": "potassium",
    "Chloride [Moles/volume] in Blood": "chloride",
    "Carbon Dioxide": "co2", "QALY": "qaly", "DALY": "daly",
}

CHRONIC_KEYWORDS = [
    "hypertension", "diabetes", "prediabetes", "heart disease",
    "heart failure", "kidney disease", "renal disease", "obesity",
    "metabolic syndrome", "hyperlipidemia", "anemia", "chronic pain",
    "osteoarthritis", "sleep apnea", "epilepsy", "alzheimer",
    "atrial fibrillation", "chronic sinusitis", "hypothyroidism",
    "chronic low back pain", "chronic neck pain", "chronic kidney",
    "ischemic heart", "coronary", "chronic congestive",
    "hypertriglyceridemia", "osteoporosis", "chronic intractable migraine",
    "neoplasm", "carcinoma", "cancer", "copd",
]

REQUIRED_CSVS = {"patients", "encounters", "observations", "conditions",
                 "medications", "allergies", "procedures"}
OPTIONAL_CSVS = {"careplans", "immunizations", "devices", "payer_transitions"}
# ═══════════════════════════════════════════════════════════════
# HELPERS — shared
# ═══════════════════════════════════════════════════════════════
def is_chronic(description):
    if pd.isna(description):
        return False
    return any(kw in str(description).lower() for kw in CHRONIC_KEYWORDS)


def _save_fig(fig, plots_dir: str, fname: str):
    """Save a matplotlib figure to plots_dir/fname."""
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, fname), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGE 1 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def _read_csv(csv_dict: dict, name: str, **kwargs) -> pd.DataFrame:
    """Read a named CSV from the upload dict (BytesIO objects)."""
    buf = csv_dict.get(name)
    if buf is None:
        return pd.DataFrame()
    buf.seek(0)
    return pd.read_csv(buf, on_bad_lines="skip", **kwargs)

def run_preprocessing(csv_dict: dict, plots_dir: str, status_fn=None):
    """
    Full preprocessing pipeline (01_data_preprocessing.py logic).
    csv_dict: {stem: BytesIO}  e.g. {"patients": <BytesIO>, ...}
    Returns: d1, d2, splits, scaler, feature_names, eda_stats
    """
    def _log(msg):
        if status_fn:
            status_fn(msg)

    # ── Load CSVs ────────────────────────────────────────────
    _log("Loading patients.csv …")
    patients = _read_csv(csv_dict, "patients",
        usecols=["Id","BIRTHDATE","DEATHDATE","GENDER","RACE","ETHNICITY",
                 "INCOME","HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","MARITAL","COUNTY"])
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")

    _log("Loading encounters.csv …")
    encounters = _read_csv(csv_dict, "encounters", low_memory=False)
    enc_cols = ["Id","START","STOP","PATIENT","ENCOUNTERCLASS","CODE","DESCRIPTION",
                "BASE_ENCOUNTER_COST","TOTAL_CLAIM_COST","PAYER_COVERAGE",
                "REASONCODE","REASONDESCRIPTION"]
    encounters = encounters[[c for c in enc_cols if c in encounters.columns]]
    encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce", utc=True).dt.tz_localize(None)
    encounters["STOP"]  = pd.to_datetime(encounters["STOP"],  errors="coerce", utc=True).dt.tz_localize(None)

    _log("Loading observations.csv (chunked filter) …")
    obs_buf = csv_dict.get("observations")
    obs_chunks = []
    target_descriptions = set(OBS_FEATURE_MAP.keys())
    if obs_buf:
        obs_buf.seek(0)
        for chunk in pd.read_csv(obs_buf, on_bad_lines="skip", chunksize=100_000,
                                  usecols=["ENCOUNTER","DESCRIPTION","VALUE","TYPE"],
                                  low_memory=False):
            mask = (chunk["TYPE"] == "numeric") & (chunk["DESCRIPTION"].isin(target_descriptions))
            gfr_mask = chunk["DESCRIPTION"].str.startswith("Glomerular filtration rate", na=False)
            obs_chunks.append(chunk[mask | gfr_mask].copy())
    observations = pd.concat(obs_chunks, ignore_index=True) if obs_chunks else pd.DataFrame()
    if not observations.empty:
        observations.loc[observations["DESCRIPTION"].str.startswith("Glomerular", na=False), "DESCRIPTION"] = "GFR"
        OBS_FEATURE_MAP["GFR"] = "gfr"
        observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")
        observations = observations.dropna(subset=["VALUE"])

    _log("Loading conditions, medications, allergies, procedures …")
    conditions = _read_csv(csv_dict, "conditions",
        usecols=["START","STOP","PATIENT","ENCOUNTER","DESCRIPTION"])
    conditions["START"] = pd.to_datetime(conditions["START"], format="%d-%m-%Y", errors="coerce")
    conditions["STOP"]  = pd.to_datetime(conditions["STOP"],  format="%d-%m-%Y", errors="coerce")

    medications = _read_csv(csv_dict, "medications",
        usecols=["PATIENT","ENCOUNTER","DESCRIPTION"])
    allergies   = _read_csv(csv_dict, "allergies",
        usecols=["PATIENT","DESCRIPTION"])
    procedures  = _read_csv(csv_dict, "procedures",
        usecols=["PATIENT","ENCOUNTER"])

    # ── Feature Engineering ──────────────────────────────────
    _log("Pivoting observation features …")
    obs_pivot = pd.DataFrame()
    if not observations.empty:
        observations["FEATURE"] = observations["DESCRIPTION"].map(OBS_FEATURE_MAP)
        observations = observations.dropna(subset=["FEATURE"])
        obs_pivot = observations.groupby(["ENCOUNTER","FEATURE"])["VALUE"].mean().unstack().reset_index()
        obs_pivot.columns.name = None

    _log("Creating target variable …")
    encounters["target_from_reason"] = encounters["REASONDESCRIPTION"].apply(is_chronic).astype(int)
    if not conditions.empty:
        conditions["is_chronic"] = conditions["DESCRIPTION"].apply(is_chronic).astype(int)
        chronic_enc = conditions[conditions["is_chronic"] == 1]["ENCOUNTER"].unique()
        encounters["target_from_conditions"] = encounters["Id"].isin(chronic_enc).astype(int)
    else:
        encounters["target_from_conditions"] = 0
    encounters["target"] = ((encounters["target_from_reason"] == 1) |
                             (encounters["target_from_conditions"] == 1)).astype(int)

    _log("Computing history features …")
    encounters = encounters.sort_values(["PATIENT","START"])
    encounters["num_prior_encounters"] = encounters.groupby("PATIENT").cumcount()
    if not conditions.empty:
        cond_cnt = conditions.groupby("PATIENT").size().reset_index(name="num_conditions")
        encounters = encounters.merge(cond_cnt, on="PATIENT", how="left")
        encounters["num_conditions"] = encounters["num_conditions"].fillna(0)
    else:
        encounters["num_conditions"] = 0
    for df_hist, col in [(medications, "num_medications"),
                          (allergies,   "num_allergies"),
                          (procedures,  "num_procedures")]:
        if not df_hist.empty and "PATIENT" in df_hist.columns:
            cnt = df_hist.groupby("PATIENT").size().reset_index(name=col)
            encounters = encounters.merge(cnt, on="PATIENT", how="left")
        else:
            encounters[col] = 0
        encounters[col] = encounters[col].fillna(0)
    encounters["encounter_duration_min"] = (
        (encounters["STOP"] - encounters["START"]).dt.total_seconds() / 60.0)

    _log("Building feature matrix …")
    demo = patients[["Id","BIRTHDATE","GENDER","RACE","ETHNICITY","INCOME",
                      "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","MARITAL"]].copy()
    df = encounters.merge(demo, left_on="PATIENT", right_on="Id", how="left", suffixes=("","_patient"))
    df.drop(columns=["Id_patient"], errors="ignore", inplace=True)
    df["age_at_encounter"] = ((df["START"] - df["BIRTHDATE"]).dt.days / 365.25)
    df["gender_M"] = (df["GENDER"] == "M").astype(int)
    race_dummies = pd.get_dummies(df["RACE"].fillna("unknown"), prefix="race")
    df = pd.concat([df, race_dummies], axis=1)
    df["ethnicity_hispanic"] = (df["ETHNICITY"] == "hispanic").astype(int)
    marital_dummies = pd.get_dummies(df["MARITAL"].fillna("unknown"), prefix="marital")
    df = pd.concat([df, marital_dummies], axis=1)
    enc_dummies = pd.get_dummies(df["ENCOUNTERCLASS"].fillna("unknown"), prefix="enc_class")
    df = pd.concat([df, enc_dummies], axis=1)
    for col in ["INCOME","HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE",
                "BASE_ENCOUNTER_COST","TOTAL_CLAIM_COST","PAYER_COVERAGE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if not obs_pivot.empty:
        df = df.merge(obs_pivot, left_on="Id", right_on="ENCOUNTER", how="left")
        df.drop(columns=["ENCOUNTER"], errors="ignore", inplace=True)

    # Patient aggregates
    agg_feats = ["systolic_bp","diastolic_bp","heart_rate","bmi","glucose","creatinine","body_weight"]
    agg_dict = {f: ["mean","std","min","max"] for f in agg_feats if f in df.columns}
    if agg_dict:
        patient_aggs = df.groupby("PATIENT").agg(agg_dict)
        patient_aggs.columns = [f"patient_{f}_{s}" for f, s in patient_aggs.columns]
        patient_aggs = patient_aggs.reset_index()
        df = df.merge(patient_aggs, on="PATIENT", how="left")

    # ── Temporal Split ────────────────────────────────────────
    _log("Applying temporal split at 2015-01-01 …")
    split_dt = pd.to_datetime(TEMPORAL_SPLIT_DATE)
    d1 = df[df["START"] < split_dt].copy()
    d2 = df[df["START"] >= split_dt].copy()

    # ── Preprocessing ─────────────────────────────────────────
    _log("Preprocessing: imputation + scaling …")
    exclude_cols = {"Id","START","STOP","PATIENT","BIRTHDATE","GENDER","RACE","ETHNICITY",
                    "MARITAL","ENCOUNTERCLASS","CODE","DESCRIPTION","REASONCODE",
                    "REASONDESCRIPTION","target","target_from_reason","target_from_conditions",
                    "DEATHDATE","Id_patient","ENCOUNTER","COUNTY"}
    feature_cols_raw = [c for c in d1.columns if c not in exclude_cols]
    feature_cols = d1[feature_cols_raw].select_dtypes(include=[np.number]).columns.tolist()

    impute_vals = d1[feature_cols].median()
    d1[feature_cols] = d1[feature_cols].fillna(impute_vals).fillna(0)
    d2[feature_cols] = d2[feature_cols].fillna(impute_vals).fillna(0)
    d1[feature_cols] = d1[feature_cols].replace([np.inf, -np.inf], 0)
    d2[feature_cols] = d2[feature_cols].replace([np.inf, -np.inf], 0)

    zero_var = [c for c in feature_cols if d1[c].std() == 0]
    feature_cols = [c for c in feature_cols if c not in zero_var]
    d1.drop(columns=zero_var, inplace=True, errors="ignore")
    d2.drop(columns=zero_var, inplace=True, errors="ignore")

    scaler = StandardScaler()
    scaler.fit(d1[feature_cols])

    # ── Train/Test Splits ─────────────────────────────────────
    X1 = d1[feature_cols].values; y1 = d1["target"].values
    X2 = d2[feature_cols].values; y2 = d2["target"].values
    # Robustness: stratify requires ≥2 classes. Fallback to unstratified otherwise.
    strat1 = y1 if len(np.unique(y1)) > 1 else None
    strat2 = y2 if len(np.unique(y2)) > 1 else None
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=TEST_SIZE,
                                                    random_state=42, stratify=strat1)
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=TEST_SIZE,
                                                    random_state=42, stratify=strat2)
    splits = {"X1_train": X1_tr, "X1_test": X1_te, "y1_train": y1_tr, "y1_test": y1_te,
              "X2_train": X2_tr, "X2_test": X2_te, "y2_train": y2_tr, "y2_test": y2_te}

    # ── EDA Plots ─────────────────────────────────────────────
    _log("Generating EDA plots …")
    eda_stats = _run_eda(d1, d2, feature_cols, plots_dir)

    return d1, d2, splits, scaler, feature_cols, eda_stats

def _run_eda(d1, d2, feature_cols, eda_dir):
    """Generates all EDA plots from 01_data_preprocessing.py into eda_dir."""
    os.makedirs(eda_dir, exist_ok=True)
    eda_stats = {}
    plt.style.use("seaborn-v0_8-darkgrid")

    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    d1["target"].value_counts().plot(kind="bar", ax=axes[0], color=["#2196F3","#F44336"])
    axes[0].set_title("Dataset 1 (Historical) — Class Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(["Non-Chronic (0)","Chronic (1)"], rotation=0)
    d2["target"].value_counts().plot(kind="bar", ax=axes[1], color=["#2196F3","#F44336"])
    axes[1].set_title("Dataset 2 (Current) — Class Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(["Non-Chronic (0)","Chronic (1)"], rotation=0)
    plt.tight_layout()
    _save_fig(fig, eda_dir, "class_distribution.png")

    eda_stats["d1_class_dist"] = d1["target"].value_counts().to_dict()
    eda_stats["d2_class_dist"] = d2["target"].value_counts().to_dict()
    eda_stats["d1_stats"] = d1[feature_cols].describe().T
    eda_stats["d2_stats"] = d2[feature_cols].describe().T

    # Data availability
    key_features = [f for f in ["systolic_bp","diastolic_bp","heart_rate","bmi","glucose",
                                  "creatinine","body_weight","body_height","age_at_encounter","INCOME"]
                    if f in feature_cols]
    if key_features:
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(key_features)); w = 0.35
        n1 = d1[key_features].shape[0]; n2 = d2[key_features].shape[0]
        ax.bar(x_pos - w/2, d1[key_features].describe().loc["count"] / n1 * 100,
               w, label="D1", color="#2196F3", alpha=0.8)
        ax.bar(x_pos + w/2, d2[key_features].describe().loc["count"] / n2 * 100,
               w, label="D2", color="#F44336", alpha=0.8)
        ax.set_xlabel("Feature"); ax.set_ylabel("Data Availability (%)")
        ax.set_title("Feature Data Availability: D1 vs D2", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos); ax.set_xticklabels(key_features, rotation=45, ha="right")
        ax.legend(); plt.tight_layout()
        _save_fig(fig, eda_dir, "data_availability.png")

    # Correlation heatmap
    corr_feats = [f for f in key_features if f in d1.columns]
    if len(corr_feats) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sns.heatmap(d1[corr_feats].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=axes[0], vmin=-1, vmax=1)
        axes[0].set_title("D1 Correlation Heatmap", fontsize=12, fontweight="bold")
        sns.heatmap(d2[corr_feats].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=axes[1], vmin=-1, vmax=1)
        axes[1].set_title("D2 Correlation Heatmap", fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, eda_dir, "correlation_heatmap.png")

    # Feature distributions
    dist_feats = [f for f in ["age_at_encounter","systolic_bp","diastolic_bp","heart_rate",
                               "bmi","glucose","creatinine","body_weight","INCOME"]
                  if f in feature_cols]
    if dist_feats:
        ncols = 3; nrows = (len(dist_feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes_flat = axes.flatten() if len(dist_feats) > 1 else [axes]
        for i, feat in enumerate(dist_feats):
            axes_flat[i].hist(d1[feat].dropna(), bins=50, alpha=0.6, label="D1", color="#4F8BFF", density=True)
            axes_flat[i].hist(d2[feat].dropna(), bins=50, alpha=0.6, label="D2", color="#FB923C", density=True)
            axes_flat[i].set_title(feat, fontsize=11, fontweight="bold"); axes_flat[i].legend(fontsize=8)
        for j in range(i + 1, len(axes_flat)): axes_flat[j].set_visible(False)
        plt.suptitle("Feature Distributions: D1 vs D2", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save_fig(fig, eda_dir, "feature_distributions.png")

    # Box plots
    box_feats = [f for f in ["systolic_bp","diastolic_bp","heart_rate","bmi","glucose"]
                 if f in feature_cols]
    if box_feats:
        fig, axes = plt.subplots(1, len(box_feats), figsize=(4 * len(box_feats), 5))
        if len(box_feats) == 1: axes = [axes]
        for i, feat in enumerate(box_feats):
            bp = axes[i].boxplot([d1[feat].dropna().values, d2[feat].dropna().values],
                                  labels=["D1","D2"], patch_artist=True)
            bp["boxes"][0].set_facecolor("#4F8BFF"); bp["boxes"][1].set_facecolor("#FB923C")
            axes[i].set_title(feat, fontsize=11, fontweight="bold")
        plt.suptitle("Feature Distributions: Box Plots", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, eda_dir, "box_plots.png")

    # Drift analysis
    drift_results = []
    for feat in feature_cols:
        v1 = d1[feat].dropna(); v2 = d2[feat].dropna()
        if len(v1) < 10 or len(v2) < 10: continue
        ks_stat, ks_pval = stats.ks_2samp(v1, v2)
        try:
            bins = np.histogram_bin_edges(np.concatenate([v1, v2]), bins=50)
            h1, _ = np.histogram(v1, bins=bins, density=True)
            h2, _ = np.histogram(v2, bins=bins, density=True)
            eps = 1e-10; h1 += eps; h2 += eps
            kl_div = stats.entropy(h1, h2)
        except Exception: kl_div = np.nan
        try: psi = np.sum((h2 - h1) * np.log(h2 / h1))
        except Exception: psi = np.nan
        drift_results.append({
            "feature": feat, "ks_statistic": ks_stat, "ks_pvalue": ks_pval,
            "kl_divergence": kl_div, "psi": psi, "mean_shift": abs(v2.mean() - v1.mean()),
            "d1_mean": v1.mean(), "d2_mean": v2.mean(), "d1_std": v1.std(), "d2_std": v2.std(),
            "drift_detected": ks_pval < 0.05,
        })
    drift_df = pd.DataFrame(drift_results).sort_values("ks_statistic", ascending=False)
    eda_stats["drift_df"] = drift_df

    if len(drift_df) > 0:
        top = drift_df.head(15)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].barh(range(len(top)), top["ks_statistic"].values, color="#F87171")
        axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels(top["feature"].values, fontsize=8)
        axes[0].set_xlabel("KS Statistic"); axes[0].set_title("Top Features by KS Statistic", fontsize=12, fontweight="bold")
        axes[0].invert_yaxis()
        kl_vals = top["kl_divergence"].fillna(0).values
        axes[1].barh(range(len(top)), kl_vals, color="#4F8BFF")
        axes[1].set_yticks(range(len(top))); axes[1].set_yticklabels(top["feature"].values, fontsize=8)
        axes[1].set_xlabel("KL Divergence"); axes[1].set_title("Top Features by KL Divergence", fontsize=12, fontweight="bold")
        axes[1].invert_yaxis()
        plt.tight_layout()
        _save_fig(fig, eda_dir, "data_drift.png")

    # Dataset overview
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    d1_yr = d1.groupby(d1["START"].dt.year).size()
    d2_yr = d2.groupby(d2["START"].dt.year).size()
    all_yr = pd.concat([d1_yr, d2_yr])
    colors_bar = ["#4F8BFF" if y < 2015 else "#FB923C" for y in all_yr.index]
    axes[0].bar(all_yr.index, all_yr.values, color=colors_bar)
    axes[0].axvline(x=2019.5, color="#F59E0B", linestyle="--", alpha=0.7, label="Split")
    axes[0].set_title("Encounters Per Year", fontsize=12, fontweight="bold"); axes[0].legend()
    if "ENCOUNTERCLASS" in d1.columns:
        ec1 = d1["ENCOUNTERCLASS"].value_counts(); ec2 = d2["ENCOUNTERCLASS"].value_counts()
        all_cls = sorted(set(ec1.index) | set(ec2.index))
        xp = np.arange(len(all_cls)); w = 0.35
        axes[1].bar(xp-w/2, [ec1.get(c,0) for c in all_cls], w, label="D1", color="#4F8BFF")
        axes[1].bar(xp+w/2, [ec2.get(c,0) for c in all_cls], w, label="D2", color="#FB923C")
        axes[1].set_xticks(xp); axes[1].set_xticklabels(all_cls, rotation=45, ha="right", fontsize=8)
        axes[1].set_title("Encounter Classes", fontsize=12, fontweight="bold"); axes[1].legend()
    all_df_tmp = pd.concat([d1, d2])
    ann = all_df_tmp.groupby(all_df_tmp["START"].dt.year)["target"].mean()
    axes[2].plot(ann.index, ann.values * 100, "o-", color="#F472B6", linewidth=2)
    axes[2].axvline(x=2019.5, color="black", linestyle="--", alpha=0.7)
    axes[2].set_title("Chronic Condition Rate by Year", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Year"); axes[2].set_ylabel("% Chronic Encounters")
    plt.tight_layout()
    _save_fig(fig, eda_dir, "dataset_overview.png")

    eda_stats["d1_shape"] = d1.shape; eda_stats["d2_shape"] = d2.shape
    eda_stats["d1_target_rate"] = d1["target"].mean()
    eda_stats["d2_target_rate"] = d2["target"].mean()
    eda_stats["feature_cols"] = feature_cols
    return eda_stats
# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGE 2 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════
def _get_model_configs():
    configs = []
    for depth in [5, 10, 15, 20]:
        configs.append({"name": f"DT_depth{depth}", "model_type": "DecisionTree",
                         "model": DecisionTreeClassifier(max_depth=depth, random_state=42,
                             class_weight="balanced", min_samples_split=5, min_samples_leaf=2),
                         "needs_scaling": False, "hyperparams": {"max_depth": depth}})
    for C in [0.1, 1, 10, 100]:
        configs.append({"name": f"SVM_C{C}", "model_type": "SVM",
                         "model": SVC(kernel="rbf", C=C, random_state=42,
                             class_weight="balanced", probability=True, max_iter=5000),
                         "needs_scaling": True, "hyperparams": {"C": C, "kernel": "rbf"}})
    for arch_name, layers in [("64_32",(64,32)), ("128_64_32",(128,64,32))]:
        configs.append({"name": f"MLP_{arch_name}", "model_type": "MLP",
                         "model": MLPClassifier(hidden_layer_sizes=layers, activation="relu",
                             solver="adam", alpha=0.001, learning_rate="adaptive",
                             learning_rate_init=0.001, max_iter=300, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=15,
                             random_state=42, batch_size=256),
                         "needs_scaling": True, "hyperparams": {"hidden_layers": str(layers)}})
    return configs
# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGE 2b — HYPERPARAMETER TUNING (GridSearchCV + k-fold CV)
# ═══════════════════════════════════════════════════════════════.

HPTUNE_CV_FOLDS       = 3
HPTUNE_SVM_MAX_SAMPLES = 5000   # subsample to keep SVM-RBF tuning tractable
HPTUNE_MLP_MAX_SAMPLES = 8000   # MLP tuning subsample (max_iter reduced too)
HPTUNE_SCORING        = "f1_macro"
HPTUNE_RANDOM_STATE   = 42

HPTUNE_GRIDS = {
    "DecisionTree": {
        "max_depth":         [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 5],
    },
    "SVM-RBF": {
        "C":     [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1],
    },
    "MLP": {
        "hidden_layer_sizes": [(64, 32), (128, 64, 32)],
        "alpha":              [0.0001, 0.001],
    },
}

def _subsample(X, y, n_max, seed=HPTUNE_RANDOM_STATE):
    """Stratified subsample helper — preserves class balance."""
    if len(X) <= n_max:
        return X, y
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_rate = len(pos_idx) / len(y)
    n_pos = max(1, int(n_max * pos_rate))
    n_neg = n_max - n_pos
    rng = np.random.RandomState(seed)
    pos_keep = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    neg_keep = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    return X[keep], y[keep]

def run_hyperparameter_tuning(splits, scaler, feature_names, status_fn=None):
    """
    Formal hyperparameter tuning using GridSearchCV with k-fold cross-validation
    on D1's training set only (test sets are never touched during tuning).

    Returns a dict:
      {
        "summary_df":    DataFrame — one row per model family (best params, CV score, etc.)
        "cv_results":    dict — full GridSearchCV.cv_results_ per family
        "tuned_models":  dict — the re-fitted best_estimator_ for each family
        "tuned_metrics": DataFrame — evaluation of tuned models on D1_test + D2_test
        "tuned_roc":     dict — ROC curves for the tuned models
        "tuned_cm":      dict — confusion matrices for the tuned models
      }

    Fails gracefully: if any stage errors, it's logged and skipped.
    """
    def _log(msg):
        if status_fn: status_fn(msg)

    X1_tr = splits["X1_train"]; y1_tr = splits["y1_train"]
    X1_te = splits["X1_test"];  y1_te = splits["y1_test"]
    X2_te = splits["X2_test"];  y2_te = splits["y2_test"]
    X1_tr_s = scaler.transform(X1_tr); X1_te_s = scaler.transform(X1_te)
    X2_te_s = scaler.transform(X2_te)

    skf = StratifiedKFold(n_splits=HPTUNE_CV_FOLDS, shuffle=True,
                          random_state=HPTUNE_RANDOM_STATE)

    summary_rows = []
    cv_results_dict = {}
    tuned_models = {}

    # ── 1) Decision Tree ───────────────────────────────────────
    try:
        _log("Tuning Decision Tree (GridSearchCV, 3-fold)…")
        base_dt = DecisionTreeClassifier(random_state=HPTUNE_RANDOM_STATE,
                                          class_weight="balanced")
        gs_dt = GridSearchCV(
            base_dt, param_grid=HPTUNE_GRIDS["DecisionTree"],
            cv=skf, scoring=HPTUNE_SCORING, n_jobs=-1, return_train_score=True)
        gs_dt.fit(X1_tr, y1_tr)  # DT does not need scaling
        cv_results_dict["DecisionTree"] = gs_dt.cv_results_
        tuned_models["DecisionTree"] = gs_dt.best_estimator_
        summary_rows.append({
            "Model Family":      "DecisionTree",
            "Best Params":       gs_dt.best_params_,
            "Best CV F1 (macro)": round(gs_dt.best_score_, 4),
            "CV Std":            round(gs_dt.cv_results_["std_test_score"][gs_dt.best_index_], 4),
            "Configs Tested":    len(gs_dt.cv_results_["params"]),
            "Total Fits":        len(gs_dt.cv_results_["params"]) * HPTUNE_CV_FOLDS,
        })
        _log(f"  ✓ DT best CV F1 = {gs_dt.best_score_:.4f}")
    except Exception as e:
        _log(f"  ✗ DT tuning failed: {e}")

    # ── 2) SVM-RBF (subsampled for tractability) ───────────────
    try:
        _log(f"Tuning SVM-RBF (GridSearchCV, subsampled to {HPTUNE_SVM_MAX_SAMPLES})…")
        Xs, ys = _subsample(X1_tr_s, y1_tr, HPTUNE_SVM_MAX_SAMPLES)
        base_svm = SVC(kernel="rbf", random_state=HPTUNE_RANDOM_STATE,
                        class_weight="balanced", max_iter=3000)
        gs_svm = GridSearchCV(
            base_svm, param_grid=HPTUNE_GRIDS["SVM-RBF"],
            cv=skf, scoring=HPTUNE_SCORING, n_jobs=-1, return_train_score=True)
        gs_svm.fit(Xs, ys)
        cv_results_dict["SVM-RBF"] = gs_svm.cv_results_
        # Re-fit on full training data with best params + enable probability
        # (probability=True adds cost but is needed for ROC-AUC on test).
        best_svm = SVC(kernel="rbf", random_state=HPTUNE_RANDOM_STATE,
                        class_weight="balanced", probability=True,
                        max_iter=5000, **gs_svm.best_params_)
        # Refit on a reasonable subsample to keep final fit fast
        Xs2, ys2 = _subsample(X1_tr_s, y1_tr, SVM_MAX_SAMPLES)
        best_svm.fit(Xs2, ys2)
        tuned_models["SVM-RBF"] = best_svm
        summary_rows.append({
            "Model Family":      "SVM-RBF",
            "Best Params":       gs_svm.best_params_,
            "Best CV F1 (macro)": round(gs_svm.best_score_, 4),
            "CV Std":            round(gs_svm.cv_results_["std_test_score"][gs_svm.best_index_], 4),
            "Configs Tested":    len(gs_svm.cv_results_["params"]),
            "Total Fits":        len(gs_svm.cv_results_["params"]) * HPTUNE_CV_FOLDS,
        })
        _log(f"  ✓ SVM best CV F1 = {gs_svm.best_score_:.4f}")
    except Exception as e:
        _log(f"  ✗ SVM tuning failed: {e}")

    # ── 3) MLP (subsampled, reduced max_iter for tuning) ───────
    try:
        _log(f"Tuning MLP (GridSearchCV, subsampled to {HPTUNE_MLP_MAX_SAMPLES})…")
        Xs, ys = _subsample(X1_tr_s, y1_tr, HPTUNE_MLP_MAX_SAMPLES)
        base_mlp = MLPClassifier(activation="relu", solver="adam",
                                  learning_rate="adaptive",
                                  max_iter=100,  # reduced during tuning
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=10, batch_size=256,
                                  random_state=HPTUNE_RANDOM_STATE)
        gs_mlp = GridSearchCV(
            base_mlp, param_grid=HPTUNE_GRIDS["MLP"],
            cv=skf, scoring=HPTUNE_SCORING, n_jobs=-1, return_train_score=True)
        gs_mlp.fit(Xs, ys)
        cv_results_dict["MLP"] = gs_mlp.cv_results_
        # Re-fit on full training data with best params + full max_iter
        best_mlp = MLPClassifier(activation="relu", solver="adam",
                                  learning_rate="adaptive",
                                  max_iter=300, early_stopping=True,
                                  validation_fraction=0.1, n_iter_no_change=15,
                                  batch_size=256,
                                  random_state=HPTUNE_RANDOM_STATE,
                                  **gs_mlp.best_params_)
        best_mlp.fit(X1_tr_s, y1_tr)
        tuned_models["MLP"] = best_mlp
        summary_rows.append({
            "Model Family":      "MLP",
            "Best Params":       gs_mlp.best_params_,
            "Best CV F1 (macro)": round(gs_mlp.best_score_, 4),
            "CV Std":            round(gs_mlp.cv_results_["std_test_score"][gs_mlp.best_index_], 4),
            "Configs Tested":    len(gs_mlp.cv_results_["params"]),
            "Total Fits":        len(gs_mlp.cv_results_["params"]) * HPTUNE_CV_FOLDS,
        })
        _log(f"  ✓ MLP best CV F1 = {gs_mlp.best_score_:.4f}")
    except Exception as e:
        _log(f"  ✗ MLP tuning failed: {e}")

    # ── 4) Evaluate tuned models on D1_test and D2_test ─────────
    tuned_metrics_rows = []
    tuned_roc = {}
    tuned_cm = {}
    _log("Evaluating tuned models on D1_test and D2_test…")
    for family, model in tuned_models.items():
        # DT uses raw X, SVM/MLP use scaled X
        needs_scaling = family != "DecisionTree"
        for ds_label, X_te_raw, X_te_sc, y_te in [
            ("D1_test", X1_te, X1_te_s, y1_te),
            ("D2_test", X2_te, X2_te_s, y2_te),
        ]:
            X_use = X_te_sc if needs_scaling else X_te_raw
            mname = f"{family}_tuned"
            m, cm, roc = _eval_model(model, X_use, y_te, ds_label, mname, family)
            tuned_metrics_rows.append(m)
            tuned_roc.setdefault(mname, {})[ds_label] = roc
            tuned_cm.setdefault(mname, {})[ds_label] = cm

    summary_df = pd.DataFrame(summary_rows)
    tuned_metrics_df = pd.DataFrame(tuned_metrics_rows)

    _log(f"Hyperparameter tuning complete — {len(summary_rows)} model families tuned.")

    return {
        "summary_df":    summary_df,
        "cv_results":    cv_results_dict,
        "tuned_models":  tuned_models,
        "tuned_metrics": tuned_metrics_df,
        "tuned_roc":     tuned_roc,
        "tuned_cm":      tuned_cm,
    }

def _eval_model(model, X, y, dataset_name, model_name, model_type, extras=None):
    y_pred = model.predict(X)
    y_prob = (model.predict_proba(X)[:,1] if hasattr(model,"predict_proba")
              else (model.decision_function(X) if hasattr(model,"decision_function")
                    else y_pred.astype(float)))
    # Robustness: roc_curve raises when only one class is present in y.
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
    else:
        fpr, tpr, auc_val = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.0
    cm = confusion_matrix(y, y_pred)
    m = {"dataset": dataset_name, "model": model_name, "model_type": model_type,
         "accuracy":  accuracy_score(y, y_pred),
         "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
         "recall":    recall_score(y, y_pred, average="weighted", zero_division=0),
         "f1_score":  f1_score(y, y_pred, average="weighted", zero_division=0),
         "roc_auc":   auc_val}
    if extras: m.update(extras)
    return m, cm, (fpr, tpr)

def run_model_training(splits, scaler, feature_names, plots_dir, status_fn=None):
    def _log(msg):
        if status_fn: status_fn(msg)

    os.makedirs(plots_dir, exist_ok=True)
    configs = _get_model_configs()
    X1_tr = splits["X1_train"]; X1_te = splits["X1_test"]
    y1_tr = splits["y1_train"]; y1_te = splits["y1_test"]
    X2_te = splits["X2_test"]; y2_te = splits["y2_test"]
    X1_tr_s = scaler.transform(X1_tr); X1_te_s = scaler.transform(X1_te)
    X2_te_s = scaler.transform(X2_te)

    all_results = []; all_roc = {}; all_cm = {}
    trained_models = {}; complexity = []
    importance_results = {}

    for i, cfg in enumerate(configs):
        name = cfg["name"]; model = cfg["model"]; ns = cfg["needs_scaling"]
        _log(f"Training [{i+1}/{len(configs)}] {name} …")
        X_tr = X1_tr_s if ns else X1_tr
        X_te1 = X1_te_s if ns else X1_te
        X_te2 = X2_te_s if ns else X2_te

        X_sub, y_sub = X_tr, y1_tr
        if cfg["model_type"] == "SVM" and len(X_tr) > SVM_MAX_SAMPLES:
            idx = np.random.choice(len(X_tr), SVM_MAX_SAMPLES, replace=False)
            X_sub, y_sub = X_tr[idx], y1_tr[idx]

        model.fit(X_sub, y_sub)
        trained_models[name] = model

        tr_m, _, _ = _eval_model(model, X_sub, y_sub, "D1_train", name, cfg["model_type"], cfg["hyperparams"])
        d1_m, d1_cm, d1_roc = _eval_model(model, X_te1, y1_te, "D1_test", name, cfg["model_type"], cfg["hyperparams"])
        d2_m, d2_cm, d2_roc = _eval_model(model, X_te2, y2_te, "D2_test", name, cfg["model_type"], cfg["hyperparams"])
        d2_m["performance_gap_acc"] = d1_m["accuracy"] - d2_m["accuracy"]
        d2_m["performance_gap_f1"]  = d1_m["f1_score"]  - d2_m["f1_score"]

        all_results += [tr_m, d1_m, d2_m]
        all_roc[name] = {"D1_test": d1_roc, "D2_test": d2_roc}
        all_cm[name]  = {"D1_test": d1_cm,  "D2_test": d2_cm}
        complexity.append({"model": name, "type": cfg["model_type"],
                            "train_acc": tr_m["accuracy"], "test_d1_acc": d1_m["accuracy"],
                            "test_d2_acc": d2_m["accuracy"], "train_f1": tr_m["f1_score"],
                            "test_d1_f1": d1_m["f1_score"], "test_d2_f1": d2_m["f1_score"],
                            "hyperparams": str(cfg["hyperparams"])})
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({"feature": feature_names,
                                   "importance": model.feature_importances_}
                                  ).sort_values("importance", ascending=False)
            importance_results[name] = imp_df

    metrics_df = pd.DataFrame(all_results)
    complexity_df = pd.DataFrame(complexity)
    feat_imp = list(importance_results.values())[0] if importance_results else pd.DataFrame()

    _log("Generating model performance plots …")
    _plot_model_results(all_results, all_roc, all_cm, complexity, importance_results, plots_dir)

    # ─── Task 3a: Save trained model configurations to disk ───────────
    _log("Saving trained model configurations to disk …")
    models_dir = os.path.join(plots_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    model_pkl_paths = {}
    for model_name, model_obj in trained_models.items():
        pkl_path = os.path.join(models_dir, f"{model_name}.pkl")
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(model_obj, f)
            model_pkl_paths[model_name] = pkl_path
        except Exception:
            pass  # non-fatal; model remains in session_state
    # Also save the scaler for standalone reuse
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    try:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        model_pkl_paths["scaler"] = scaler_path
    except Exception:
        pass

    return metrics_df, all_roc, all_cm, trained_models, complexity_df, feat_imp, model_pkl_paths

def _plot_model_results(all_results, all_roc, all_cm, complexity, importance_results, plots_dir):
    plt.style.use("seaborn-v0_8-darkgrid")
    results_df = pd.DataFrame(all_results)

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_roc)))
    for idx, (mn, rd) in enumerate(all_roc.items()):
        for j, (ds, ax) in enumerate([("D1_test", axes[0]),("D2_test", axes[1])]):
            fpr, tpr = rd[ds]
            auc = results_df[(results_df["model"]==mn)&(results_df["dataset"]==ds)]["roc_auc"].values[0]
            ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                    label=f"{mn} ({auc:.3f})" if j == 0 else f"{mn} ({auc:.3f})")
    for ax, title in zip(axes, ["ROC Curves — D1 Test","ROC Curves — D2 Test"]):
        ax.plot([0,1],[0,1],"k--",alpha=0.5); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(title, fontsize=13, fontweight="bold"); ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    _save_fig(fig, plots_dir, "roc_curves.png")

    # Confusion matrices
    n_models = len(all_cm)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
    if n_models == 1: axes = axes[np.newaxis, :]
    for idx, (mn, cmd) in enumerate(all_cm.items()):
        for j, (ds, cm) in enumerate(cmd.items()):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx,j],
                        xticklabels=["Non-Chronic","Chronic"], yticklabels=["Non-Chronic","Chronic"])
            axes[idx,j].set_title(f"{mn} — {ds}", fontsize=10, fontweight="bold")
            axes[idx,j].set_xlabel("Predicted"); axes[idx,j].set_ylabel("Actual")
    plt.tight_layout()
    _save_fig(fig, plots_dir, "confusion_matrices.png")

    # Complexity analysis
    cdf = pd.DataFrame(complexity)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    dt = cdf[cdf["type"]=="DecisionTree"]
    if len(dt):
        depths = [5,10,15,20]
        axes[0].plot(depths, dt["train_acc"].values, "o-", label="D1 Train", color="#2196F3", markersize=8)
        axes[0].plot(depths, dt["test_d1_acc"].values, "s-", label="D1 Test", color="#4CAF50", markersize=8)
        axes[0].plot(depths, dt["test_d2_acc"].values, "^-", label="D2 Test", color="#F44336", markersize=8)
        axes[0].set_xlabel("Max Depth"); axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Decision Tree Complexity", fontsize=12, fontweight="bold")
        axes[0].legend(); axes[0].set_xticks(depths)
    sv = cdf[cdf["type"]=="SVM"]
    if len(sv):
        Cs = [0.1,1,10,100]
        axes[1].plot(range(len(Cs)), sv["train_acc"].values, "o-", label="D1 Train", color="#2196F3", markersize=8)
        axes[1].plot(range(len(Cs)), sv["test_d1_acc"].values, "s-", label="D1 Test", color="#4CAF50", markersize=8)
        axes[1].plot(range(len(Cs)), sv["test_d2_acc"].values, "^-", label="D2 Test", color="#F44336", markersize=8)
        axes[1].set_xlabel("C"); axes[1].set_title("SVM Complexity", fontsize=12, fontweight="bold")
        axes[1].legend(); axes[1].set_xticks(range(len(Cs))); axes[1].set_xticklabels([str(c) for c in Cs])
    ml = cdf[cdf["type"]=="MLP"]
    if len(ml):
        xp = range(len(ml))
        axes[2].bar([x-0.2 for x in xp], ml["train_acc"].values, 0.2, label="D1 Train", color="#2196F3")
        axes[2].bar([x     for x in xp], ml["test_d1_acc"].values, 0.2, label="D1 Test", color="#4CAF50")
        axes[2].bar([x+0.2 for x in xp], ml["test_d2_acc"].values, 0.2, label="D2 Test", color="#F44336")
        axes[2].set_title("MLP Architecture", fontsize=12, fontweight="bold")
        axes[2].legend(); axes[2].set_xticks(list(xp)); axes[2].set_xticklabels(ml["model"].values, rotation=15)
    plt.suptitle("Model Complexity & Generalization Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, plots_dir, "complexity_analysis.png")

    # Temporal shift
    d1t = results_df[results_df["dataset"]=="D1_test"]
    d2t = results_df[results_df["dataset"]=="D2_test"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    models = d1t["model"].values; xp = np.arange(len(models)); w = 0.35
    axes[0].bar(xp-w/2, d1t["accuracy"].values, w, label="D1 Test", color="#2196F3", alpha=0.8)
    axes[0].bar(xp+w/2, d2t["accuracy"].values, w, label="D2 Test", color="#F44336", alpha=0.8)
    axes[0].set_title("Accuracy: D1 vs D2 (Temporal Shift)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(xp); axes[0].set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    axes[0].legend()
    axes[1].bar(xp-w/2, d1t["f1_score"].values, w, label="D1 Test", color="#4CAF50", alpha=0.8)
    axes[1].bar(xp+w/2, d2t["f1_score"].values, w, label="D2 Test", color="#FF9800", alpha=0.8)
    axes[1].set_title("F1 Score: D1 vs D2 (Temporal Shift)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(xp); axes[1].set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    axes[1].legend()
    plt.tight_layout()
    _save_fig(fig, plots_dir, "temporal_shift.png")

    # Feature importance
    if importance_results:
        best_dt = max(importance_results, key=lambda n: results_df[
            (results_df["model"]==n)&(results_df["dataset"]=="D1_test")]["f1_score"].values[0]
            if len(results_df[(results_df["model"]==n)&(results_df["dataset"]=="D1_test")]) > 0 else 0)
        top = importance_results[best_dt].head(15)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
        ax.barh(range(len(top)), top["importance"].values, color=colors_imp)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top["feature"].values)
        ax.invert_yaxis(); ax.set_xlabel("Feature Importance (Gini)")
        ax.set_title(f"Top 15 Feature Importance ({best_dt})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, plots_dir, "feature_importance.png")

    # Performance heatmap
    pivot_acc = results_df.pivot_table(index="model", columns="dataset", values="accuracy", aggfunc="first")
    pivot_f1  = results_df.pivot_table(index="model", columns="dataset", values="f1_score",  aggfunc="first")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.heatmap(pivot_acc, annot=True, fmt=".4f", cmap="YlGn", ax=axes[0], vmin=0.5, vmax=1.0)
    axes[0].set_title("Accuracy Across Datasets", fontsize=12, fontweight="bold")
    sns.heatmap(pivot_f1,  annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[1], vmin=0.5, vmax=1.0)
    axes[1].set_title("F1 Score Across Datasets",  fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, plots_dir, "performance_heatmap.png")
# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGE 3 — CONTINUAL LEARNING
# ═══════════════════════════════════════════════════════════════
def run_continual_learning(splits, scaler, feature_names, metrics_df, trained_models,
                           plots_dir, status_fn=None):
    def _log(msg):
        if status_fn: status_fn(msg)

    def _eval(model, X, y):
        y_pred = model.predict(X)
        y_prob = (model.predict_proba(X)[:,1] if hasattr(model,"predict_proba")
                  else (model.decision_function(X) if hasattr(model,"decision_function")
                        else y_pred.astype(float)))
        return {"accuracy":  accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
                "recall":    recall_score(y, y_pred, average="weighted", zero_division=0),
                "f1_score":  f1_score(y, y_pred, average="weighted", zero_division=0),
                "roc_auc":   roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.0}

    def _best_name(model_type):
        sub = metrics_df[(metrics_df["model_type"]==model_type)&(metrics_df["dataset"]=="D1_test")]
        if sub.empty: return None
        return sub.loc[sub["f1_score"].idxmax(), "model"]

    results = []

    # Decision Tree
    _log("Continual learning: Decision Tree …")
    dt_name = _best_name("DecisionTree")
    if dt_name and dt_name in trained_models:
        orig = trained_models[dt_name]
        X2te = splits["X2_test"]; y2te = splits["y2_test"]
        before = _eval(orig, X2te, y2te)
        X_c = np.vstack([splits["X1_train"], splits["X2_train"]])
        y_c = np.concatenate([splits["y1_train"], splits["y2_train"]])
        sw = np.ones(len(y_c)); sw[len(splits["y1_train"]):] = 2.0
        new_dt = DecisionTreeClassifier(max_depth=orig.max_depth, random_state=42,
                                         class_weight="balanced", min_samples_split=5, min_samples_leaf=2)
        new_dt.fit(X_c, y_c, sample_weight=sw)
        after = _eval(new_dt, X2te, y2te)
        imp = (after["accuracy"] - before["accuracy"]) / before["accuracy"] * 100
        trained_models[f"{dt_name}_continual"] = new_dt
        results.append({"model": dt_name, "model_type": "DecisionTree",
                         "before_accuracy": before["accuracy"], "before_f1": before["f1_score"],
                         "before_precision": before["precision"], "before_recall": before["recall"],
                         "before_roc_auc": before["roc_auc"],
                         "after_accuracy": after["accuracy"], "after_f1": after["f1_score"],
                         "after_precision": after["precision"], "after_recall": after["recall"],
                         "after_roc_auc": after["roc_auc"], "improvement_pct": imp})

    # SVM
    _log("Continual learning: SVM …")
    svm_name = _best_name("SVM")
    if svm_name and svm_name in trained_models:
        orig = trained_models[svm_name]
        X2tr_s = scaler.transform(splits["X2_train"])
        X2te_s = scaler.transform(splits["X2_test"]); y2te = splits["y2_test"]
        X1tr_s = scaler.transform(splits["X1_train"])
        before = _eval(orig, X2te_s, y2te)
        n1 = min(len(X1tr_s), SVM_MAX_SAMPLES//2); n2 = min(len(X2tr_s), SVM_MAX_SAMPLES//2)
        i1 = np.random.choice(len(X1tr_s), n1, replace=False)
        i2 = np.random.choice(len(X2tr_s), n2, replace=False)
        X_c = np.vstack([X1tr_s[i1], X2tr_s[i2]])
        y_c = np.concatenate([splits["y1_train"][i1], splits["y2_train"][i2]])
        sw = np.ones(len(y_c)); sw[n1:] = 2.0
        new_svm = SVC(kernel="rbf", C=orig.C, random_state=42,
                      class_weight="balanced", probability=True, max_iter=5000)
        new_svm.fit(X_c, y_c, sample_weight=sw)
        after = _eval(new_svm, X2te_s, y2te)
        imp = (after["accuracy"] - before["accuracy"]) / before["accuracy"] * 100
        trained_models[f"{svm_name}_continual"] = new_svm
        results.append({"model": svm_name, "model_type": "SVM",
                         "before_accuracy": before["accuracy"], "before_f1": before["f1_score"],
                         "before_precision": before["precision"], "before_recall": before["recall"],
                         "before_roc_auc": before["roc_auc"],
                         "after_accuracy": after["accuracy"], "after_f1": after["f1_score"],
                         "after_precision": after["precision"], "after_recall": after["recall"],
                         "after_roc_auc": after["roc_auc"], "improvement_pct": imp})

    # MLP
    _log("Continual learning: MLP (fine-tuning) …")
    mlp_name = _best_name("MLP")
    if mlp_name and mlp_name in trained_models:
        orig = trained_models[mlp_name]
        X2tr_s = scaler.transform(splits["X2_train"])
        X2te_s = scaler.transform(splits["X2_test"]); y2te = splits["y2_test"]
        before = _eval(orig, X2te_s, y2te)
        new_mlp = copy.deepcopy(orig)
        new_mlp.learning_rate_init = 0.0001; new_mlp.max_iter = 100
        new_mlp.warm_start = True; new_mlp.early_stopping = True
        new_mlp.n_iter_no_change = 10; new_mlp.validation_fraction = 0.1
        new_mlp.fit(X2tr_s, splits["y2_train"])
        after = _eval(new_mlp, X2te_s, y2te)
        imp = (after["accuracy"] - before["accuracy"]) / before["accuracy"] * 100
        trained_models[f"{mlp_name}_continual"] = new_mlp
        results.append({"model": mlp_name, "model_type": "MLP",
                         "before_accuracy": before["accuracy"], "before_f1": before["f1_score"],
                         "before_precision": before["precision"], "before_recall": before["recall"],
                         "before_roc_auc": before["roc_auc"],
                         "after_accuracy": after["accuracy"], "after_f1": after["f1_score"],
                         "after_precision": after["precision"], "after_recall": after["recall"],
                         "after_roc_auc": after["roc_auc"], "improvement_pct": imp})

    cl_df = pd.DataFrame(results)

    # Plot
    if not cl_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        mods = cl_df["model_type"].values; xp = np.arange(len(mods)); w = 0.35
        axes[0].bar(xp-w/2, cl_df["before_accuracy"].values, w, label="Before", color="#F44336", alpha=0.8)
        axes[0].bar(xp+w/2, cl_df["after_accuracy"].values,  w, label="After",  color="#4CAF50", alpha=0.8)
        axes[0].set_title("Accuracy: Before vs After", fontsize=12, fontweight="bold")
        axes[0].set_xticks(xp); axes[0].set_xticklabels(mods); axes[0].legend()
        axes[1].bar(xp-w/2, cl_df["before_f1"].values, w, label="Before", color="#F44336", alpha=0.8)
        axes[1].bar(xp+w/2, cl_df["after_f1"].values,  w, label="After",  color="#4CAF50", alpha=0.8)
        axes[1].set_title("F1: Before vs After", fontsize=12, fontweight="bold")
        axes[1].set_xticks(xp); axes[1].set_xticklabels(mods); axes[1].legend()
        imp_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in cl_df["improvement_pct"].values]
        axes[2].bar(xp, cl_df["improvement_pct"].values, color=imp_colors, alpha=0.8)
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].set_title("Accuracy Improvement (%)", fontsize=12, fontweight="bold")
        axes[2].set_xticks(xp); axes[2].set_xticklabels(mods)
        plt.suptitle("Continual Learning: Performance Comparison on D2 Test Set",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save_fig(fig, plots_dir, "continual_learning.png")

    return cl_df
# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGE 4 — ADVANCED EDA
# ═══════════════════════════════════════════════════════════════
def run_advanced_eda(d1, d2, feat, scaler, adv_dir, extra_csv_dict=None, status_fn=None):
    def _log(msg):
        if status_fn: status_fn(msg)

    os.makedirs(adv_dir, exist_ok=True)
    plt.rcParams.update(ADV_PLT_PARAMS)
    P_D1 = "#3498DB"; P_D2 = "#E74C3C"

    # ── PCA ──────────────────────────────────────────────────
    _log("Running PCA …")
    X_d1 = scaler.transform(d1[feat].fillna(0))
    X_d2 = scaler.transform(d2[feat].fillna(0))
    y_d1 = d1["target"].values; y_d2 = d2["target"].values
    N = 3000
    i1 = np.random.choice(len(X_d1), min(N,len(X_d1)), replace=False)
    i2 = np.random.choice(len(X_d2), min(N,len(X_d2)), replace=False)
    pca = PCA(n_components=3); pca.fit(np.vstack([X_d1[i1], X_d2[i2]]))
    P1 = pca.transform(X_d1[i1]); P2 = pca.transform(X_d2[i2]); ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20,6)); fig.patch.set_facecolor("#0D1117")
    axes[0].scatter(P1[:,0],P1[:,1],c=P_D1,alpha=0.35,s=12,label="D1 Historical")
    axes[0].scatter(P2[:,0],P2[:,1],c=P_D2,alpha=0.35,s=12,label="D2 Current")
    axes[0].set_xlabel(f"PC1 ({ev[0]:.1%})"); axes[0].set_ylabel(f"PC2 ({ev[1]:.1%})")
    axes[0].set_title("PCA: Historical vs Current"); axes[0].legend(fontsize=9)
    for ax, P, y, lbl in [(axes[1],P1,y_d1[i1],"D1"),(axes[2],P2,y_d2[i2],"D2")]:
        ax.scatter(P[y==0,0],P[y==0,1],c="#58A6FF",alpha=0.3,s=12,label="Non-Chronic")
        ax.scatter(P[y==1,0],P[y==1,1],c="#FF7B72",alpha=0.4,s=16,label="Chronic")
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(f"PCA: Class Separation ({lbl})"); ax.legend(fontsize=9)
    plt.suptitle("PCA Visualisation – Clinical Feature Space", fontsize=15, y=1.02, color="#E6EDF3")
    plt.tight_layout()
    _save_fig(fig, adv_dir, "pca_2d.png")

    # Scree + loadings
    pca_full = PCA(n_components=min(15,len(feat))); pca_full.fit(np.vstack([X_d1, X_d2]))
    fig, axes = plt.subplots(1, 2, figsize=(16,6)); fig.patch.set_facecolor("#0D1117")
    cumvar = np.cumsum(pca_full.explained_variance_ratio_); xr = range(1,len(cumvar)+1)
    axes[0].bar(xr, pca_full.explained_variance_ratio_, color="#58A6FF", alpha=0.7)
    axes[0].plot(xr, cumvar, "o-", color="#FF7B72", linewidth=2, label="Cumulative")
    axes[0].axhline(0.8, color="white", linestyle="--", alpha=0.5)
    axes[0].set_title("PCA Scree Plot"); axes[0].legend(fontsize=9)
    loadings = pd.Series(abs(pca_full.components_[0]), index=feat).sort_values(ascending=False)
    top10 = loadings.head(10)
    axes[1].barh(range(len(top10)), top10.values, color=plt.cm.viridis(np.linspace(0.3,0.9,len(top10))))
    axes[1].set_yticks(range(len(top10))); axes[1].set_yticklabels(top10.index, fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_xlabel("|Loading|"); axes[1].set_title("PC1 Feature Loadings (Top 10)")
    plt.suptitle("PCA Diagnostics", fontsize=14, color="#E6EDF3"); plt.tight_layout()
    _save_fig(fig, adv_dir, "pca_scree_loadings.png")

    pca_ev_df = pd.DataFrame({"pc": range(1,len(cumvar)+1),
                               "ev": pca_full.explained_variance_ratio_, "cum_ev": cumvar})

    # ── t-SNE ────────────────────────────────────────────────
    _log("Running t-SNE (this may take a while) …")
    N2 = 2000
    i1t = np.random.choice(len(X_d1), min(N2,len(X_d1)), replace=False)
    i2t = np.random.choice(len(X_d2), min(N2,len(X_d2)), replace=False)
    X_all = np.vstack([X_d1[i1t], X_d2[i2t]])
    y_all = np.concatenate([y_d1[i1t], y_d2[i2t]])
    ds_all = np.array([0]*len(i1t)+[1]*len(i2t))
    tsne = TSNE(n_components=2, perplexity=40, max_iter=700, random_state=42)
    E = tsne.fit_transform(X_all)
    fig, axes = plt.subplots(1, 2, figsize=(16,7)); fig.patch.set_facecolor("#0D1117")
    clr_ds = [P_D1 if d==0 else P_D2 for d in ds_all]
    axes[0].scatter(E[:,0],E[:,1],c=clr_ds,alpha=0.45,s=14)
    axes[0].set_title("t-SNE: Historical vs Current"); axes[0].set_xlabel("t-SNE 1"); axes[0].set_ylabel("t-SNE 2")
    p1=mpatches.Patch(color=P_D1,label="D1"); p2=mpatches.Patch(color=P_D2,label="D2")
    axes[0].legend(handles=[p1,p2],fontsize=9)
    clr_cls = ["#FF7B72" if y==1 else "#58A6FF" for y in y_all]
    axes[1].scatter(E[:,0],E[:,1],c=clr_cls,alpha=0.45,s=14)
    axes[1].set_title("t-SNE: Non-Chronic vs Chronic"); axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")
    p3=mpatches.Patch(color="#FF7B72",label="Chronic"); p4=mpatches.Patch(color="#58A6FF",label="Non-Chronic")
    axes[1].legend(handles=[p3,p4],fontsize=9)
    plt.suptitle("t-SNE Embedding", fontsize=14, color="#E6EDF3"); plt.tight_layout()
    _save_fig(fig, adv_dir, "tsne.png")

    # ── Correlation network ───────────────────────────────────
    _log("Building feature correlation network …")
    corr = d1[feat].corr().abs(); cv = corr.values.copy(); np.fill_diagonal(cv,0)
    corr = pd.DataFrame(cv, index=corr.index, columns=corr.columns)
    THRESH = 0.35
    edges = [(f1,f2,corr.loc[f1,f2]) for i,f1 in enumerate(feat)
             for j,f2 in enumerate(feat) if j>i and corr.loc[f1,f2]>=THRESH]
    nodes = list(feat); deg = {n:0 for n in nodes}
    for e in edges: deg[e[0]]+=1; deg[e[1]]+=1
    N_n = len(nodes); angle_step = 2*math.pi/N_n if N_n else 1
    pos = {n:(( 1.0+deg[n]*0.04)*math.cos(k*angle_step),
               (1.0+deg[n]*0.04)*math.sin(k*angle_step))
           for k,n in enumerate(nodes)}

    def _feat_color(name):
        if any(x in name for x in ["systolic","diastolic","heart"]): return "#FF7B72"
        elif any(x in name for x in ["bmi","weight","height"]): return "#79C0FF"
        elif any(x in name for x in ["glucose","creatinine","gfr","calcium","sodium"]): return "#FFA657"
        elif any(x in name for x in ["age","gender","race","income","marital"]): return "#D2A8FF"
        elif any(x in name for x in ["num_","prior","encounter"]): return "#56D364"
        elif "patient_" in name: return "#E3B341"
        else: return "#8B949E"

    fig, ax = plt.subplots(figsize=(18,14)); fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117")
    max_w = max((e[2] for e in edges),default=1)
    for f1,f2,w in edges:
        x0,y0=pos[f1]; x1,y1=pos[f2]
        ax.plot([x0,x1],[y0,y1],color="#58A6FF" if w>0.6 else "#21262D",
                alpha=0.15+0.5*(w/max_w), linewidth=0.5+2.0*(w/max_w), zorder=1)
    for n in nodes:
        x,y=pos[n]; sz=80+deg[n]*60
        ax.scatter(x,y,s=sz,c=_feat_color(n),zorder=3,edgecolors="white",linewidths=0.4,alpha=0.9)
        short=n.replace("patient_","p_").replace("enc_class_","cl_")[:18]
        ax.text(x*1.13,y*1.13,short,fontsize=6.5 if len(short)>12 else 7.5,
                ha="center",va="center",color="#C9D1D9",zorder=4)
    legend_items=[mpatches.Patch(color=c,label=l) for c,l in [
        ("#FF7B72","Cardiovascular"),("#79C0FF","Anthropometry"),
        ("#FFA657","Labs/Metabolic"),("#D2A8FF","Demographics"),
        ("#56D364","History Counts"),("#E3B341","Patient Aggregates"),("#8B949E","Other")]]
    ax.legend(handles=legend_items,loc="lower right",fontsize=9,
              facecolor="#161B22",edgecolor="#30363D",labelcolor="#C9D1D9")
    ax.axis("off"); ax.set_title(f"Feature Correlation Network (|r|>={THRESH})",
                                  pad=20, color="#E6EDF3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, adv_dir, "correlation_network.png")

    # ── Anomaly detection ─────────────────────────────────────
    _log("Running Isolation Forest anomaly detection …")
    iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_d1)
    s1=iso.decision_function(X_d1); s2=iso.decision_function(X_d2)
    a1=iso.predict(X_d1)==-1; a2=iso.predict(X_d2)==-1
    d1=d1.copy(); d2=d2.copy()
    d1["anomaly_score"]=s1; d1["is_anomaly"]=a1.astype(int)
    d2["anomaly_score"]=s2; d2["is_anomaly"]=a2.astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(20,6)); fig.patch.set_facecolor("#0D1117")
    axes[0].hist(s1,bins=60,alpha=0.6,density=True,color=P_D1,label="D1")
    axes[0].hist(s2,bins=60,alpha=0.6,density=True,color=P_D2,label="D2")
    axes[0].axvline(0,color="yellow",linestyle="--",linewidth=1.5)
    axes[0].set_title("Anomaly Score Distribution"); axes[0].legend(fontsize=9)
    all_df_a = pd.concat([d1.assign(dataset="D1"),d2.assign(dataset="D2")])
    if "START" in all_df_a.columns:
        yr_anom = all_df_a.groupby([all_df_a["START"].dt.year,"dataset"])["is_anomaly"].mean().unstack(fill_value=0)
        for col,c in [("D1",P_D1),("D2",P_D2)]:
            if col in yr_anom.columns:
                axes[1].plot(yr_anom.index,yr_anom[col]*100,"o-",color=c,linewidth=2,label=col,markersize=5)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Anomaly Rate (%)"); axes[1].set_title("Anomaly Rate Over Time")
    axes[1].legend(fontsize=9); axes[1].axvline(2015,color="white",linestyle="--",alpha=0.5)
    groups=["D1 Normal","D1 Anomaly","D2 Normal","D2 Anomaly"]
    rates=[d1[d1["is_anomaly"]==0]["target"].mean()*100, d1[d1["is_anomaly"]==1]["target"].mean()*100,
           d2[d2["is_anomaly"]==0]["target"].mean()*100, d2[d2["is_anomaly"]==1]["target"].mean()*100]
    bars=axes[2].bar(groups,rates,color=[P_D1,"#FF6B6B",P_D2,"#FF9F43"],alpha=0.85,edgecolor="white",linewidth=0.5)
    for bar,val in zip(bars,rates):
        axes[2].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"{val:.1f}%",
                     ha="center",va="bottom",fontsize=9,color="#E6EDF3")
    axes[2].set_title("Chronic Rate: Normal vs Anomalous"); axes[2].tick_params(axis="x",rotation=20)
    plt.suptitle("Anomaly Detection — Isolation Forest", fontsize=14, color="#E6EDF3")
    plt.tight_layout()
    _save_fig(fig, adv_dir, "anomaly_detection.png")

    n1a=a1.sum(); n2a=a2.sum()
    anomaly_stats = pd.DataFrame({"Dataset":["D1","D2"],"Total":[len(d1),len(d2)],
        "Anomalies":[n1a,n2a],
        "Anomaly_Rate_pct":[n1a/len(d1)*100,n2a/len(d2)*100],
        "Chronic_Rate_Normal_pct":[d1[d1["is_anomaly"]==0]["target"].mean()*100,
                                    d2[d2["is_anomaly"]==0]["target"].mean()*100],
        "Chronic_Rate_Anomaly_pct":[d1[d1["is_anomaly"]==1]["target"].mean()*100,
                                     d2[d2["is_anomaly"]==1]["target"].mean()*100]})

    # ── PSI drift ─────────────────────────────────────────────
    _log("Computing PSI drift analysis …")
    psi_df = _compute_psi(d1, d2, feat, adv_dir)

    # ── Population pyramid ────────────────────────────────────
    _log("Plotting population pyramid …")
    _plot_population_pyramid(d1, d2, adv_dir)

    # ── Patient trajectories ──────────────────────────────────
    _log("Plotting patient trajectories …")
    _plot_patient_trajectories(d1, d2, adv_dir)

    # ── Cohort analysis ───────────────────────────────────────
    _log("Plotting cohort analysis …")
    _plot_cohort_analysis(d1, d2, adv_dir)

    # ── Healthcare utilisation ────────────────────────────────
    _log("Plotting healthcare utilisation …")
    _plot_utilisation(d1, d2, adv_dir)

    # ── Clinical summary ──────────────────────────────────────
    clin_df = _clinical_summary(d1, d2, psi_df)

    plt.rcParams.update(plt.rcParamsDefault)
    return d1, d2, psi_df, anomaly_stats, clin_df, pca_ev_df

def _risk_tiers(df, name):
    if "risk_composite" not in df.columns: return pd.DataFrame()
    dc = df.copy()
    dc["risk_tier"] = pd.cut(dc["risk_composite"], bins=[-0.01,0.25,0.5,0.75,1.0],
                              labels=["Low","Moderate","High","Very High"])
    return dc.groupby("risk_tier").agg(n=("target","count"), chronic_rate=("target","mean")
                                       ).reset_index().assign(chronic_rate_pct=lambda x: x["chronic_rate"]*100)

def _compute_psi(d1, d2, feat, adv_dir):
    def psi_fn(exp, act, bins=20):
        mn=min(exp.min(),act.min()); mx=max(exp.max(),act.max())
        edges=np.linspace(mn,mx,bins+1); eps=1e-8
        eh,_=np.histogram(exp,bins=edges); ah,_=np.histogram(act,bins=edges)
        ep=(eh+eps)/(len(exp)+eps*bins); ap=(ah+eps)/(len(act)+eps*bins)
        return np.sum((ap-ep)*np.log(ap/ep))
    rows=[]
    for f in feat:
        if f not in d1.columns or f not in d2.columns: continue
        v1=d1[f].dropna(); v2=d2[f].dropna()
        if len(v1)<20 or len(v2)<20: continue
        try:
            val=psi_fn(v1.values,v2.values)
            lv="No Drift" if val<0.1 else ("Slight Drift" if val<0.2 else "Significant Drift")
            rows.append({"feature":f,"psi":val,"level":lv})
        except Exception: pass
    psi_df=pd.DataFrame(rows).sort_values("psi",ascending=False)
    if psi_df.empty: return psi_df
    nn=(psi_df["level"]=="No Drift").sum()
    ns=(psi_df["level"]=="Slight Drift").sum()
    nsg=(psi_df["level"]=="Significant Drift").sum()
    cmap={"No Drift":"#56D364","Slight Drift":"#E3B341","Significant Drift":"#FF7B72"}
    top20=psi_df.head(20)
    fig,axes=plt.subplots(1,2,figsize=(20,8)); fig.patch.set_facecolor("#0D1117")
    colors_b=[cmap[l] for l in top20["level"]]
    axes[0].barh(range(len(top20)),top20["psi"].values,color=colors_b,alpha=0.85)
    axes[0].set_yticks(range(len(top20))); axes[0].set_yticklabels(top20["feature"].values,fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0.1,color="#E3B341",linestyle="--",linewidth=1.5,label="Slight (0.1)")
    axes[0].axvline(0.2,color="#FF7B72",linestyle="--",linewidth=1.5,label="Significant (0.2)")
    axes[0].set_title("PSI — Top 20 Features"); axes[0].legend(fontsize=9)
    axes[1].pie([nn,ns,nsg],labels=["No Drift","Slight","Significant"],
                colors=[cmap[k] for k in["No Drift","Slight Drift","Significant Drift"]],
                autopct="%1.0f%%",startangle=140,textprops={"color":"#C9D1D9","fontsize":11})
    axes[1].set_title("Drift Classification")
    plt.suptitle("PSI Drift Analysis",fontsize=14,color="#E6EDF3"); plt.tight_layout()
    _save_fig(fig,adv_dir,"psi_drift.png")
    return psi_df

def _plot_population_pyramid(d1, d2, adv_dir):
    if "age_at_encounter" not in d1.columns or "gender_M" not in d1.columns: return
    bins=[0,10,20,30,40,50,60,70,80,90,110]
    labels=["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90+"]
    fig,axes=plt.subplots(1,2,figsize=(18,8),sharey=True); fig.patch.set_facecolor("#0D1117")
    for ax,df,title in[(axes[0],d1,"D1 Historical"),(axes[1],d2,"D2 Current")]:
        dc=df.copy()
        dc["age_group"]=pd.cut(dc["age_at_encounter"],bins=bins,labels=labels,right=False)
        male=dc[dc["gender_M"]==1].groupby("age_group",observed=True).size()
        female=dc[dc["gender_M"]==0].groupby("age_group",observed=True).size()
        ai=pd.CategoricalIndex(labels)
        male=male.reindex(ai,fill_value=0); female=female.reindex(ai,fill_value=0)
        yp=range(len(labels))
        ax.barh(yp,-male.values,color="#58A6FF",alpha=0.8,label="Male")
        ax.barh(yp,female.values,color="#FF7B72",alpha=0.8,label="Female")
        ax.set_yticks(yp); ax.set_yticklabels(labels)
        mv=max(male.max(),female.max()); ax.set_xlim(-mv*1.15,mv*1.15)
        ax.axvline(0,color="white",linewidth=0.8); ax.set_xlabel("Encounter Count")
        ax.set_title(f"{title} — Population Pyramid"); ax.legend(fontsize=9)
        tks=ax.get_xticks(); ax.set_xticklabels([str(abs(int(t))) for t in tks])
    plt.suptitle("Clinical Population Age-Gender Pyramid",fontsize=14,color="#E6EDF3")
    plt.tight_layout(); _save_fig(fig,adv_dir,"population_pyramid.png")

def _plot_patient_trajectories(d1, d2, adv_dir):
    all_df=pd.concat([d1.assign(dataset="D1"),d2.assign(dataset="D2")],ignore_index=True)
    if "START" not in all_df.columns: return
    all_df["year"]=all_df["START"].dt.year
    fig,axes=plt.subplots(2,3,figsize=(22,10)); fig.patch.set_facecolor("#0D1117")
    monthly=all_df.groupby(all_df["START"].dt.to_period("M")).size()
    xp=range(len(monthly))
    split_idx=next((i for i,p in enumerate(monthly.index) if int(str(p).split("-")[0])>=2015),None)
    axes[0,0].fill_between(xp,monthly.values,alpha=0.5,color="#58A6FF")
    axes[0,0].plot(xp,monthly.values,linewidth=1.2,color="#58A6FF")
    if split_idx: axes[0,0].axvline(split_idx,color="yellow",linestyle="--",linewidth=1.5,label="2015 Split"); axes[0,0].legend(fontsize=8)
    tick_pos=[i for i,p in enumerate(monthly.index) if str(p).endswith("-01") and int(str(p).split("-")[0])%2==0]
    tick_lbl=[str(p).split("-")[0] for p in monthly.index if str(p).endswith("-01") and int(str(p).split("-")[0])%2==0]
    axes[0,0].set_xticks(tick_pos); axes[0,0].set_xticklabels(tick_lbl,rotation=45)
    axes[0,0].set_title("Monthly Encounter Volume"); axes[0,0].set_xlabel("Month"); axes[0,0].set_ylabel("Encounters")
    monthly_chr=all_df.groupby(all_df["START"].dt.to_period("M"))["target"].mean()*100
    axes[0,1].plot(range(len(monthly_chr)),monthly_chr.values,linewidth=1.2,color="#FF7B72",alpha=0.9)
    axes[0,1].fill_between(range(len(monthly_chr)),monthly_chr.values,alpha=0.25,color="#FF7B72")
    if split_idx: axes[0,1].axvline(split_idx,color="yellow",linestyle="--",linewidth=1.5)
    axes[0,1].set_xticks(tick_pos); axes[0,1].set_xticklabels(tick_lbl,rotation=45)
    axes[0,1].set_title("Monthly Chronic Rate (%)"); axes[0,1].set_xlabel("Month"); axes[0,1].set_ylabel("%")
    if "age_at_encounter" in all_df.columns:
        yr_age=all_df.groupby("year")["age_at_encounter"].mean()
        axes[0,2].plot(yr_age.index,yr_age.values,"o-",color="#FFA657",linewidth=2,markersize=5)
        axes[0,2].axvline(2015,color="yellow",linestyle="--",linewidth=1.5)
        axes[0,2].set_title("Mean Age at Encounter by Year"); axes[0,2].set_xlabel("Year"); axes[0,2].set_ylabel("Age")
    if "ENCOUNTERCLASS" in all_df.columns:
        enc_yr=(all_df.groupby(["year","ENCOUNTERCLASS"]).size().unstack(fill_value=0))
        enc_pct=enc_yr.div(enc_yr.sum(axis=1),axis=0)*100
        top_cls=enc_pct.mean().nlargest(6).index
        cmap2=plt.cm.get_cmap("tab10"); bottom=np.zeros(len(enc_pct))
        for ii,cls in enumerate(top_cls):
            if cls in enc_pct.columns:
                axes[1,0].bar(enc_pct.index,enc_pct[cls].values,bottom=bottom,label=cls,color=cmap2(ii),alpha=0.85)
                bottom+=enc_pct[cls].values
        axes[1,0].axvline(2015,color="yellow",linestyle="--",linewidth=1.5)
        axes[1,0].set_title("Encounter Class Mix (%)"); axes[1,0].set_xlabel("Year"); axes[1,0].set_ylabel("%"); axes[1,0].legend(fontsize=7,loc="upper left")
    for f_name,lbl,c in [("systolic_bp","Systolic BP","#FF7B72"),("diastolic_bp","Diastolic BP","#58A6FF"),("heart_rate","Heart Rate","#FFA657")]:
        if f_name in all_df.columns:
            yr=all_df.groupby("year")[f_name].mean()
            axes[1,1].plot(yr.index,yr.values,"o-",label=lbl,color=c,linewidth=2,markersize=5)
    axes[1,1].axvline(2015,color="yellow",linestyle="--",linewidth=1.5); axes[1,1].set_title("Mean Vital Signs Over Time")
    axes[1,1].set_xlabel("Year"); axes[1,1].set_ylabel("Value"); axes[1,1].legend(fontsize=8)
    for f_name,lbl,c in [("bmi","BMI","#D2A8FF"),("glucose","Glucose","#FFA657")]:
        if f_name in all_df.columns:
            yr=all_df.groupby("year")[f_name].mean()
            axes[1,2].plot(yr.index,yr.values,"o-",label=lbl,color=c,linewidth=2,markersize=5)
    axes[1,2].axvline(2015,color="yellow",linestyle="--",linewidth=1.5)
    axes[1,2].set_title("Metabolic Markers Over Time"); axes[1,2].set_xlabel("Year"); axes[1,2].set_ylabel("Value"); axes[1,2].legend(fontsize=8)
    plt.suptitle("Patient Population Trajectories Over Time",fontsize=15,color="#E6EDF3",y=1.02)
    plt.tight_layout(); _save_fig(fig,adv_dir,"patient_trajectories.png")

def _plot_cohort_analysis(d1, d2, adv_dir):
    key_features=[f for f in["age_at_encounter","systolic_bp","diastolic_bp","heart_rate",
                               "bmi","glucose","creatinine","INCOME","num_conditions","num_medications",
                               "num_prior_encounters","encounter_duration_min"] if f in d1.columns]
    if not key_features: return
    ncols=4; nrows=(len(key_features)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*5,nrows*4)); fig.patch.set_facecolor("#0D1117")
    axes_flat=axes.flatten() if len(key_features)>1 else [axes]
    i=0
    for i,fn in enumerate(key_features):
        ax=axes_flat[i]; data=[]; tick_labels=[]; colors_box=[]
        for df,prefix,c0,c1 in[(d1,"D1","#3498DB","#1060A0"),(d2,"D2","#E74C3C","#901C10")]:
            if fn not in df.columns: continue
            non=df[df["target"]==0][fn].dropna(); chr_=df[df["target"]==1][fn].dropna()
            data.extend([non.values,chr_.values]); tick_labels.extend([f"{prefix} Non-Chr",f"{prefix} Chronic"])
            colors_box.extend([c0,c1])
        if not data: ax.set_visible(False); continue
        bp=ax.boxplot(data,patch_artist=True,notch=False,medianprops=dict(color="white",linewidth=2.0),
                      whiskerprops=dict(color="#8B949E"),capprops=dict(color="#8B949E"),
                      flierprops=dict(marker=".",color="#8B949E",alpha=0.3,markersize=3))
        for patch,c in zip(bp["boxes"],colors_box): patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticklabels(tick_labels,rotation=30,ha="right",fontsize=7)
        ax.set_title(fn.replace("_"," ").title(),fontsize=10); ax.set_ylabel("Value",fontsize=8)
    for j in range(i+1,len(axes_flat)): axes_flat[j].set_visible(False)
    plt.suptitle("Cohort Analysis: Chronic vs Non-Chronic",fontsize=14,color="#E6EDF3",y=1.01)
    plt.tight_layout(); _save_fig(fig,adv_dir,"cohort_analysis.png")

def _plot_utilisation(d1, d2, adv_dir):
    cost_cols=[c for c in["BASE_ENCOUNTER_COST","TOTAL_CLAIM_COST","PAYER_COVERAGE"] if c in d1.columns]
    if not cost_cols: return
    P_D1="#3498DB"; P_D2="#E74C3C"
    fig,axes=plt.subplots(2,len(cost_cols),figsize=(len(cost_cols)*6,12)); fig.patch.set_facecolor("#0D1117")
    if len(cost_cols)==1: axes=axes[:,np.newaxis]
    for ci,col in enumerate(cost_cols):
        v1=d1[col].dropna().clip(0,d1[col].quantile(0.99)); v2=d2[col].dropna().clip(0,d2[col].quantile(0.99))
        axes[0,ci].hist(v1,bins=50,alpha=0.55,density=True,color=P_D1,label="D1")
        axes[0,ci].hist(v2,bins=50,alpha=0.55,density=True,color=P_D2,label="D2")
        axes[0,ci].set_title(col); axes[0,ci].set_xlabel("Cost ($)"); axes[0,ci].legend(fontsize=8)
        data_cls=[d1[d1["target"]==0][col].dropna(),d1[d1["target"]==1][col].dropna(),
                   d2[d2["target"]==0][col].dropna(),d2[d2["target"]==1][col].dropna()]
        lbls=["D1 Non-Chr","D1 Chronic","D2 Non-Chr","D2 Chronic"]
        bp2=axes[1,ci].boxplot(data_cls,patch_artist=True,medianprops=dict(color="white",linewidth=1.5))
        for patch,c in zip(bp2["boxes"],[P_D1,"#1060A0",P_D2,"#901C10"]): patch.set_facecolor(c); patch.set_alpha(0.75)
        axes[1,ci].set_xticklabels(lbls,rotation=20,ha="right",fontsize=8)
        axes[1,ci].set_title(f"{col}: by Class"); axes[1,ci].set_ylabel("Cost ($)")
    plt.suptitle("Healthcare Utilisation & Cost Analysis",fontsize=14,color="#E6EDF3")
    plt.tight_layout(); _save_fig(fig,adv_dir,"utilisation_costs.png")

def _clinical_summary(d1, d2, psi_df):
    d1_chr=d1[d1["target"]==1]; d2_chr=d2[d2["target"]==1]
    d1_non=d1[d1["target"]==0]; d2_non=d2[d2["target"]==0]
    rows=[]
    for f in["age_at_encounter","systolic_bp","diastolic_bp","heart_rate","bmi",
              "glucose","creatinine","INCOME","num_conditions","num_medications"]:
        if f not in d1.columns: continue
        pv=psi_df.loc[psi_df["feature"]==f,"psi"].values if not psi_df.empty else []
        rows.append({"Feature":f,
                      "D1_Chronic_Mean":d1_chr[f].mean(),"D1_NonChronic_Mean":d1_non[f].mean(),
                      "D2_Chronic_Mean":d2_chr[f].mean(),"D2_NonChronic_Mean":d2_non[f].mean(),
                      "PSI":pv[0] if len(pv)>0 else None})
    return pd.DataFrame(rows)
# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGE 5 — PRECISION-RECALL & ADDITIONAL ANALYSIS
# ═══════════════════════════════════════════════════════════════
def run_pr_analysis(splits, scaler, trained_models, metrics_df, plots_dir, status_fn=None):
    def _log(msg):
        if status_fn: status_fn(msg)

    # Filter to base models only
    order=["DT_depth5","DT_depth10","DT_depth15","DT_depth20",
           "SVM_C0.1","SVM_C1","SVM_C10","SVM_C100","MLP_64_32","MLP_128_64_32"]
    models={k:trained_models[k] for k in order if k in trained_models}

    def _needs_scaling(name): return name.startswith("SVM") or name.startswith("MLP")
    def _get_proba(model, X):
        if hasattr(model,"predict_proba"): return model.predict_proba(X)[:,1]
        elif hasattr(model,"decision_function"):
            df=model.decision_function(X); return (df-df.min())/(df.max()-df.min()+1e-8)
        return model.predict(X).astype(float)

    X1te=splits["X1_test"]; X2te=splits["X2_test"]
    y1te=splits["y1_test"]; y2te=splits["y2_test"]
    X1te_s=scaler.transform(X1te); X2te_s=scaler.transform(X2te)

    colors=["#FF7B72","#FFA657","#E3B341","#56D364","#79C0FF","#A5D6FF","#58A6FF","#1F6FEB","#D2A8FF","#B08FFF"]

    # PR curves
    _log("Generating Precision-Recall curves …")
    pr_summary={}
    fig,axes=plt.subplots(1,2,figsize=(18,8)); fig.patch.set_facecolor("#0D1117")
    for ax,y,ds,tag in[(axes[0],y1te,"D1_test","D1 Test"),(axes[1],y2te,"D2_test","D2 Test")]:
        ax.plot([0,1],[y.mean(),y.mean()],"--",color="#484F58",linewidth=1.5,label=f"Baseline ({y.mean():.2%})")
        for ii,(nm,model) in enumerate(models.items()):
            X=X1te_s if ds=="D1_test" else X2te_s
            if not _needs_scaling(nm): X=X1te if ds=="D1_test" else X2te
            prob=_get_proba(model,X)
            prec,rec,_=precision_recall_curve(y,prob)
            ap=average_precision_score(y,prob)
            pr_summary.setdefault(nm,{})[ds]={"AP":ap,"prec":prec,"rec":rec}
            ax.plot(rec,prec,color=colors[ii],linewidth=2,label=f"{nm} (AP={ap:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"PR Curve — {tag}")
        ax.legend(fontsize=7.5,loc="lower left"); ax.set_xlim([0,1.02]); ax.set_ylim([0,1.05])
    plt.suptitle("Precision-Recall Analysis",fontsize=14,color="#E6EDF3",y=1.02)
    plt.tight_layout(); _save_fig(fig,plots_dir,"precision_recall_curves.png")

    ap_rows=[]
    for nm,data in pr_summary.items():
        for ds,vals in data.items(): ap_rows.append({"model":nm,"dataset":ds,"average_precision":vals["AP"]})
    ap_df=pd.DataFrame(ap_rows)

    # Threshold analysis
    _log("Generating threshold analysis …")
    families={"DT":("DT_depth20",X2te),"SVM":("SVM_C10",X2te_s),"MLP":("MLP_128_64_32",X2te_s)}
    fig,axes=plt.subplots(1,3,figsize=(21,7)); fig.patch.set_facecolor("#0D1117")
    for ax,(fam,(mname,X)) in zip(axes,families.items()):
        if mname not in models: continue
        model=models[mname]; prob=_get_proba(model,X)
        thresholds=np.linspace(0.1,0.95,50)
        precs,recs,f1s=[],[],[]
        for t in thresholds:
            yp=(prob>=t).astype(int)
            from sklearn.metrics import precision_score as ps_, recall_score as rs_
            precs.append(ps_(y2te,yp,zero_division=0))
            recs.append(rs_(y2te,yp,zero_division=0))
            f1s.append(f1_score(y2te,yp,zero_division=0))
        ax.plot(thresholds,precs,color="#56D364",linewidth=2,label="Precision")
        ax.plot(thresholds,recs, color="#FF7B72",linewidth=2,label="Recall")
        ax.plot(thresholds,f1s,  color="#58A6FF",linewidth=2.5,label="F1",linestyle="--")
        bt=thresholds[np.argmax(f1s)]
        ax.axvline(bt,color="#E3B341",linestyle="--",alpha=0.8,label=f"Best F1 t={bt:.2f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_title(f"{fam} — {mname}")
        ax.legend(fontsize=9); ax.set_xlim([0.1,0.95]); ax.set_ylim([0,1.05])
    plt.suptitle("Threshold Sensitivity Analysis on D2 Test",fontsize=14,color="#E6EDF3",y=1.02)
    plt.tight_layout(); _save_fig(fig,plots_dir,"threshold_analysis.png")

    # Calibration
    _log("Generating calibration curves …")
    fig,axes=plt.subplots(1,2,figsize=(16,7)); fig.patch.set_facecolor("#0D1117")
    for ax,y,ds,tag in[(axes[0],y1te,"D1_test","D1 Test"),(axes[1],y2te,"D2_test","D2 Test")]:
        ax.plot([0,1],[0,1],"--",color="#484F58",linewidth=1.5,label="Perfect calibration")
        for family,mname,c in[("Decision Tree","DT_depth20","#E3B341"),
                                ("SVM","SVM_C10","#56D364"),("MLP","MLP_128_64_32","#58A6FF")]:
            if mname not in models: continue
            X=X1te if ds=="D1_test" else X2te
            if _needs_scaling(mname): X=X1te_s if ds=="D1_test" else X2te_s
            prob=_get_proba(models[mname],X)
            pt,pp=calibration_curve(y,prob,n_bins=12)
            ax.plot(pp,pt,"o-",color=c,linewidth=2,markersize=6,label=f"{family} ({mname})")
        ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration — {tag}"); ax.legend(fontsize=9)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    plt.suptitle("Model Calibration",fontsize=14,color="#E6EDF3",y=1.02)
    plt.tight_layout(); _save_fig(fig,plots_dir,"calibration_curves.png")

    # Classification reports
    _log("Generating classification reports …")
    report_rows=[]
    for mname,model in models.items():
        for ds,X,y in[("D1_test",X1te_s if _needs_scaling(mname) else X1te,y1te),
                       ("D2_test",X2te_s if _needs_scaling(mname) else X2te,y2te)]:
            yp=model.predict(X)
            rep=classification_report(y,yp,target_names=["Non-Chronic","Chronic"],output_dict=True,zero_division=0)
            for cls in["Non-Chronic","Chronic","macro avg","weighted avg"]:
                if cls in rep:
                    r=rep[cls]
                    report_rows.append({"model":mname,"dataset":ds,"class":cls,
                                         "precision":r.get("precision",0),"recall":r.get("recall",0),
                                         "f1-score":r.get("f1-score",0),"support":r.get("support",0)})
    cls_rep=pd.DataFrame(report_rows)

    # Per-class F1
    best_models=["DT_depth20","SVM_C10","MLP_128_64_32"]
    fig,axes=plt.subplots(1,2,figsize=(16,6)); fig.patch.set_facecolor("#0D1117")
    for ax,ds,tag in[(axes[0],"D1_test","D1 Test"),(axes[1],"D2_test","D2 Test")]:
        sub=cls_rep[(cls_rep["dataset"]==ds)&(cls_rep["model"].isin(best_models))&
                    (cls_rep["class"].isin(["Non-Chronic","Chronic"]))]
        for model,c in[("DT_depth20","#E3B341"),("SVM_C10","#56D364"),("MLP_128_64_32","#58A6FF")]:
            d=sub[sub["model"]==model]
            if d.empty: continue
            xp=np.arange(len(d)); offset=best_models.index(model)*0.25-0.25
            ax.bar(xp+offset,d["f1-score"].values,width=0.25,label=model,color=c,alpha=0.85)
        ax.set_xticks([0.25,1.25]); ax.set_xticklabels(["Non-Chronic","Chronic"],fontsize=11)
        ax.set_ylabel("F1 Score"); ax.set_ylim([0.5,1.05]); ax.set_title(f"Per-Class F1 — {tag}"); ax.legend(fontsize=9)
    plt.suptitle("Per-Class Performance",fontsize=14,color="#E6EDF3",y=1.02)
    plt.tight_layout(); _save_fig(fig,plots_dir,"per_class_f1.png")

    # Radar chart
    _log("Generating radar comparison chart …")
    metric_cols=["accuracy","precision","recall","f1_score","roc_auc"]
    labels_r=["Accuracy","Precision","Recall","F1 Score","ROC AUC"]
    N_r=len(metric_cols); angles=[n/float(N_r)*2*np.pi for n in range(N_r)]; angles+=angles[:1]
    fig,axes=plt.subplots(1,2,figsize=(16,8),subplot_kw=dict(polar=True)); fig.patch.set_facecolor("#0D1117")
    colors_r=["#E3B341","#56D364","#58A6FF"]
    for ax,ds,tag in[(axes[0],"D1_test","D1 Test"),(axes[1],"D2_test","D2 Test")]:
        ax.set_facecolor("#161B22")
        for ii,(mname,c) in enumerate(zip(best_models,colors_r)):
            row=metrics_df[(metrics_df["model"]==mname)&(metrics_df["dataset"]==ds)]
            if row.empty: continue
            vals=[row[m].values[0] for m in metric_cols]; vals+=vals[:1]
            ax.plot(angles,vals,"o-",color=c,linewidth=2.5,label=mname,markersize=6)
            ax.fill(angles,vals,color=c,alpha=0.12)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r,fontsize=10,color="#C9D1D9")
        ax.set_ylim([0.5,1.0]); ax.set_yticks([0.6,0.7,0.8,0.9,1.0])
        ax.set_yticklabels(["0.6","0.7","0.8","0.9","1.0"],fontsize=8,color="#8B949E")
        ax.grid(color="#21262D",linewidth=0.7); ax.spines["polar"].set_color("#30363D")
        ax.set_title(f"{tag}",fontsize=13,color="#E6EDF3",pad=18)
        ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.15),fontsize=9,labelcolor="#C9D1D9")
    plt.suptitle("Model Comparison Radar",fontsize=14,color="#E6EDF3",y=1.03)
    plt.tight_layout(); _save_fig(fig,plots_dir,"radar_comparison.png")

    return ap_df, cls_rep
# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
def run_full_pipeline(csv_dict: dict, status_fn=None):
    """
    Run all 5 pipeline stages, return a data dict compatible with the dashboard page functions.
    """
    def _log(msg):
        if status_fn: status_fn(msg)

    tmpdir = tempfile.mkdtemp(prefix="clinicalml_")
    eda_dir = os.path.join(tmpdir, "EDA_plots")
    plots_dir = os.path.join(tmpdir, "plots")
    adv_dir = os.path.join(tmpdir, "advanced_plots")

    _log("▶ Stage 1/5 — Data Preprocessing & EDA")
    d1, d2, splits, scaler, feature_names, eda_stats = run_preprocessing(
        csv_dict, eda_dir, status_fn=status_fn)

    # Robustness: refuse to proceed with degenerate data (prevents downstream crashes).
    if len(d1) == 0 or len(d2) == 0:
        raise ValueError(
            f"Empty dataset after temporal split (D1={len(d1)}, D2={len(d2)}). "
            f"Check that encounter dates span both sides of {TEMPORAL_SPLIT_DATE}.")
    if len(np.unique(splits['y1_train'])) < 2:
        raise ValueError(
            "Dataset 1 training set contains only one class. "
            "Target is likely too rare — cannot train binary classifiers.")

    _log("▶ Stage 2/6 — Model Training & Evaluation")
    metrics_df, roc_data, cm_data, trained_models, complexity_df, feat_imp, model_pkl_paths = run_model_training(
        splits, scaler, feature_names, plots_dir, status_fn=status_fn)

    _log("▶ Stage 3/6 — Hyperparameter Tuning (GridSearchCV + k-fold CV)")
    try:
        hptune = run_hyperparameter_tuning(splits, scaler, feature_names, status_fn=status_fn)
    except Exception as e:
        _log(f"  ✗ Hyperparameter tuning stage failed: {e}")
        hptune = {"summary_df": pd.DataFrame(), "cv_results": {},
                  "tuned_models": {}, "tuned_metrics": pd.DataFrame(),
                  "tuned_roc": {}, "tuned_cm": {}}

    _log("▶ Stage 4/6 — Continual Learning")
    cl_df = run_continual_learning(
        splits, scaler, feature_names, metrics_df, trained_models, plots_dir, status_fn=status_fn)

    _log("▶ Stage 5/6 — Advanced EDA")
    d1e, d2e, psi_df, anomaly_stats, clin_df, pca_ev_df = run_advanced_eda(
        d1.copy(), d2.copy(), feature_names, scaler, adv_dir, status_fn=status_fn)

    _log("▶ Stage 6/6 — PR Curves & Additional Analysis")
    ap_df, cls_rep = run_pr_analysis(
        splits, scaler, trained_models, metrics_df, plots_dir, status_fn=status_fn)

    drift_df = eda_stats.get("drift_df", pd.DataFrame())

    d = {
        "d1": d1, "d2": d2, "d1e": d1e, "d2e": d2e,
        "splits": splits, "feature_names": feature_names, "scaler": scaler,
        "eda_stats": eda_stats,
        "roc_data": roc_data, "cm_data": cm_data,
        "metrics": metrics_df, "complexity": complexity_df,
        "drift": drift_df, "psi": psi_df,
        "feat_imp": feat_imp, "cl_results": cl_df,
        "eda_sum": pd.DataFrame(),
        "anomaly_stats": anomaly_stats,
        "clinical_sum": clin_df, "pca_ev": pca_ev_df,
        "avg_precision": ap_df, "cls_reports": cls_rep,
        "plots": plots_dir, "eda_plots": eda_dir, "adv_plots": adv_dir,
        "model_pkl_paths": model_pkl_paths,
        # Hyperparameter tuning outputs
        "hptune": hptune,
    }
    return d
# ═══════════════════════════════════════════════════════════════
# DASHBOARD HELPERS
# ═══════════════════════════════════════════════════════════════
def kpi(label, value, delta=None, good_direction="up", icon=None):
    sign = "" if delta is None else ("+" if delta >= 0 else "")
    cls = ""
    if delta is not None:
        pos = (good_direction=="up" and delta>=0) or (good_direction=="down" and delta<=0)
        cls = "kpi-delta-good" if pos else "kpi-delta-bad"
    delta_html = f'<div class="{cls}">{sign}{delta:.4f}</div>' if delta is not None else ""
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    return (f'<div class="kpi-card">{icon_html}<div class="kpi-val">{value}</div>'
            f'<div class="kpi-label">{label}</div>{delta_html}</div>')

def img(d, folder_key, fname):
    path = os.path.join(d[folder_key], fname)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.markdown(
            f'<div class="warn-box" style="text-align:center;padding:22px">'
            f'🖼️ <b>Static plot unavailable</b><br>'
            f'<span style="color:#94A3B8;font-size:0.85em">'
            f'Missing: <code>{fname}</code> — re-run the pipeline with '
            f'<code>show_static_plots</code> enabled.</span></div>',
            unsafe_allow_html=True)

def plotly_fig(fig, title=None):
    """Apply shared dark-theme layout to a Plotly figure and render it."""
    layout = dict(PLOTLY_LAYOUT)
    if title is not None:
        layout["title"] = title
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: BIAS-VARIANCE TRADE-OFF (Task 3d)
# ═══════════════════════════════════════════════════════════════
def page_bias_variance(d):
    st.markdown('<div class="sec-head">📐 Bias-Variance Trade-off Analysis</div>', unsafe_allow_html=True)
    metrics  = d.get("metrics",  pd.DataFrame())
    complexity = d.get("complexity", pd.DataFrame())

    st.markdown("""
    <div class="glass" style="margin-bottom:20px">
      <div style="color:#4F8BFF;font-size:1.05em;font-weight:600;margin-bottom:14px">
        🎓 Theoretical Framework
      </div>
      <div style="color:#94A3B8;font-size:0.91em;line-height:1.85">
        The <b style="color:#E2E8F0">bias-variance decomposition</b> states that the expected generalisation error of a
        model can be broken into three components:<br><br>
        <code style="background:rgba(79,139,255,0.1);padding:4px 8px;border-radius:4px;color:#4F8BFF">
          Error = Bias² + Variance + Irreducible Noise
        </code><br><br>
        <b style="color:#E2E8F0">Bias</b> is the systematic error from incorrect assumptions — a high-bias model
        underfits and cannot capture the true relationship in the data.<br>
        <b style="color:#E2E8F0">Variance</b> is sensitivity to fluctuations in training data — a high-variance model
        overfits and memorises noise rather than the underlying signal.<br><br>
        The gap between <span style="color:#2196F3">train accuracy</span> and
        <span style="color:#4CAF50">D1 test accuracy</span> reveals variance (overfitting), while the gap between
        <span style="color:#4CAF50">D1 test accuracy</span> and
        <span style="color:#F44336">D2 test accuracy</span> reveals generalisation failure under distribution shift.
      </div>
    </div>
    """, unsafe_allow_html=True)

    img(d, "plots", "complexity_analysis.png")

    st.markdown('<div class="section-hdr">Decision Tree: Depth vs Bias-Variance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-accent">
      <div style="color:#E2E8F0;font-size:0.92em;line-height:1.85;font-weight:500;margin-bottom:10px">
        🌳 Decision Tree — Complexity Parameter: <code>max_depth ∈ {5, 10, 15, 20}</code>
      </div>
      <div style="color:#94A3B8;font-size:0.88em;line-height:1.85">
        <b style="color:#F59E0B">Depth 5 (High Bias / Low Variance):</b>
        The shallow tree cannot capture the complex, nonlinear interactions between clinical features such as
        age, comorbidities, and lab values. It consistently underestimates the chronic condition rate, yielding
        lower training accuracy. The train-test gap is minimal — the model is too simple to overfit — but
        overall performance is constrained by structural underfitting.<br><br>
        <b style="color:#4F8BFF">Depth 10–15 (Balanced Region):</b>
        These depths occupy the sweet spot: train accuracy improves significantly and test accuracy follows
        closely, indicating the model is learning genuine signal. The bias-variance balance is near-optimal
        for D1, though D2 test accuracy begins to diverge, hinting at temporal overfitting.<br><br>
        <b style="color:#F472B6">Depth 20 (Low Bias / Higher Variance):</b>
        The deepest tree approaches near-perfect training accuracy (close to 1.0), indicating it has memorised
        training patterns including noise. The gap between D1 train and D1 test widens, confirming increased
        variance. Despite this, D2 test accuracy remains competitive because decision trees at high depth can
        still encode clinically meaningful decision rules — the structured nature of EHR features limits
        catastrophic overfitting compared to unstructured domains.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">SVM: Regularisation (C) vs Bias-Variance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-accent">
      <div style="color:#E2E8F0;font-size:0.92em;line-height:1.85;font-weight:500;margin-bottom:10px">
        ⚙️ SVM-RBF — Complexity Parameter: <code>C ∈ {0.1, 1, 10, 100}</code>
      </div>
      <div style="color:#94A3B8;font-size:0.88em;line-height:1.85">
        <b style="color:#F87171">C = 0.1 (Very High Bias / Very Low Variance):</b>
        The strong regularisation forces a wide-margin hyperplane that ignores many support vectors.
        The resulting decision boundary is overly smooth and fails to separate chronic from non-chronic
        patients in the nonlinear RBF feature space. Train accuracy itself is low — this is textbook
        <em>underfitting</em> where the model's capacity is insufficient for the task's complexity.<br><br>
        <b style="color:#F59E0B">C = 1 (Moderate Bias):</b>
        Regularisation eases. The RBF kernel begins to utilise local density patterns in the feature
        space. Training and test accuracy rise together, suggesting the model is approaching an
        appropriate complexity for the data.<br><br>
        <b style="color:#10B981">C = 10–100 (Low Bias / Increasing Variance):</b>
        The margin becomes tight and the model closely fits the training distribution. Train accuracy is
        high but D2 test accuracy drops relative to D1 test, revealing <em>variance-driven temporal
        overfitting</em>: the model has memorised statistical patterns in D1 (pre-2015) that no longer
        hold in D2 (post-2015). This is compounded by distribution shift in features like systolic BP
        and glucose, which the overfitted boundary cannot adapt to.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">MLP: Architecture vs Bias-Variance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-accent">
      <div style="color:#E2E8F0;font-size:0.92em;line-height:1.85;font-weight:500;margin-bottom:10px">
        🧠 MLP — Complexity Parameter: <code>Architecture ∈ {(64,32), (128,64,32)}</code>
      </div>
      <div style="color:#94A3B8;font-size:0.88em;line-height:1.85">
        <b style="color:#F472B6">MLP (64,32) — Smaller Network:</b>
        With ~3,000 parameters, this architecture has limited representational capacity. It exhibits
        moderate bias — the network learns general clinical patterns but may miss subtle multi-way
        feature interactions. Variance is low thanks to early stopping (patience=15) and L2
        regularisation (alpha=0.001). The D1 train–test gap is small.<br><br>
        <b style="color:#4F8BFF">MLP (128,64,32) — Larger Network:</b>
        With ~13,000 parameters and three hidden layers, this network is the most expressive model in
        the suite. It achieves the lowest training error (lowest bias) and, crucially, its test performance
        exceeds the smaller MLP due to its ability to model clinical nonlinearity. Early stopping with
        validation monitoring effectively controls variance: training halts before memorisation sets in.
        The D2 test gap relative to D1 test is moderate — the model generalises better than the
        high-C SVM because its regularisation is distribution-agnostic, unlike the SVM margin which is
        anchored to the training distribution.<br><br>
        <b style="color:#F59E0B">Key takeaway:</b> Among all models, MLP (128,64,32) achieves the best
        bias-variance balance overall. Decision Tree depth 20 achieves the highest raw accuracy, but at the
        cost of higher variance. SVM C=0.1 is the canonical underfitting example across all experiments.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not complexity.empty and not metrics.empty:
        st.markdown('<div class="section-hdr">Bias-Variance Summary Table</div>', unsafe_allow_html=True)
        bv_rows = []
        for _, row in complexity.iterrows():
            train_acc = row.get("train_acc", 0)
            d1_acc    = row.get("test_d1_acc", 0)
            d2_acc    = row.get("test_d2_acc", 0)
            variance_proxy = round(train_acc - d1_acc, 4)
            drift_gap      = round(d1_acc   - d2_acc, 4)
            if train_acc < 0.72:
                bv_label = "High Bias (Underfitting)"
            elif variance_proxy > 0.05:
                bv_label = "High Variance (Overfitting)"
            else:
                bv_label = "Balanced"
            bv_rows.append({
                "Model": row["model"], "Type": row["type"],
                "Train Acc": round(train_acc,4),
                "D1 Test Acc": round(d1_acc,4),
                "D2 Test Acc": round(d2_acc,4),
                "Variance Proxy\n(Train-D1 gap)": variance_proxy,
                "Drift Gap\n(D1-D2 gap)": drift_gap,
                "B-V Assessment": bv_label,
            })
        bv_df = pd.DataFrame(bv_rows)
        st.dataframe(
            bv_df.style.format({c:"{:.4f}" for c in bv_df.columns
                                 if c not in["Model","Type","B-V Assessment"]})
                .applymap(lambda v: "color:#F87171" if "High Bias" in str(v) else
                                    ("color:#F59E0B" if "High Variance" in str(v) else
                                     "color:#10B981"), subset=["B-V Assessment"]),
            use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
def page_home(d):
    # Hero header
    st.markdown('''
    <div style="background:linear-gradient(135deg,rgba(79,139,255,0.08),rgba(244,114,182,0.05));
                border:1px solid rgba(79,139,255,0.15);border-radius:20px;padding:30px 36px;margin-bottom:26px;">
      <div style="font-size:1.95em;font-weight:700;color:#F1F5F9;letter-spacing:-1px;line-height:1.15">
        ⚕️ Chronic Condition Intelligence
      </div>
      <div style="color:#94A3B8;font-size:0.98em;margin-top:8px;line-height:1.6">
        Production-grade AutoML pipeline · Synthea EHR Data · Temporal Shift Detection &amp; Continual Learning
      </div>
    </div>''', unsafe_allow_html=True)

    d1, d2 = d["d1"], d["d2"]

    # ── Pipeline Stepper (replaces wall-of-text) ──────────────
    st.markdown('<div class="section-hdr">📋 Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass" style="padding:18px 18px 14px">
      <div class="stepper">
        <div class="stepper-node">
          <div class="stepper-num">1</div>
          <div class="stepper-title">Data Engineering</div>
          <div class="stepper-desc">Load Synthea CSVs, merge by patient ID, engineer clinical features (vitals, labs, demographics, history).</div>
        </div>
        <div class="stepper-node">
          <div class="stepper-num">2</div>
          <div class="stepper-title">Temporal Split</div>
          <div class="stepper-desc">Pre-2015 → Dataset 1 (historical). 2015+ → Dataset 2 (prospective).</div>
        </div>
        <div class="stepper-node">
          <div class="stepper-num">3</div>
          <div class="stepper-title">Model Training</div>
          <div class="stepper-desc">DT×4 depths, SVM-RBF×4 C-values, MLP×2 architectures. Cross-dataset evaluation exposes temporal gap.</div>
        </div>
        <div class="stepper-node">
          <div class="stepper-num">4</div>
          <div class="stepper-title">Hyperparameter Tuning</div>
          <div class="stepper-desc">GridSearchCV with 3-fold CV on D1 train — tunes DT, SVM-RBF, and MLP families on macro-F1.</div>
        </div>
        <div class="stepper-node">
          <div class="stepper-num">5</div>
          <div class="stepper-title">Continual Learning</div>
          <div class="stepper-desc">Best models fine-tuned on D2 — recovers 3–6% of performance lost to temporal drift.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4:4:4 balanced columns ────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="glass" style="height:100%">
          <div style="color:#F472B6;font-size:1em;font-weight:600;margin-bottom:12px">
            🎯 Clinical Target Definition
          </div>
          <div style="color:#94A3B8;font-size:0.88em;line-height:1.7">
            <span style="color:#E2E8F0;font-weight:600">Task:</span> Binary classification — chronic medical condition at encounter.<br><br>
            <span style="color:#E2E8F0;font-weight:600">Positive class includes:</span><br>
            <span style="color:#CBD5E1">Hypertension · Diabetes · Obesity · CKD · Heart Failure · COPD ·
            Atrial Fibrillation · Hypothyroidism · Osteoarthritis · Cancer · Sleep Apnea + 20 more.</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass" style="height:100%">
          <div style="color:#4F8BFF;font-size:1em;font-weight:600;margin-bottom:12px">
            📅 Temporal Split
          </div>
          <div style="color:#94A3B8;font-size:0.88em;line-height:1.7">
            <span style="color:#E2E8F0;font-weight:600">Split date:</span> 1 January 2015<br><br>
            <span style="color:#4F8BFF;font-weight:600">● Dataset 1 (D1):</span> Historical encounters<br>
            <span style="color:#FB923C;font-weight:600">● Dataset 2 (D2):</span> Current encounters (2015+)<br><br>
            <span style="color:#E2E8F0;font-weight:600">Purpose:</span> Measure how well models trained on historical data generalise to modern clinical patterns.
          </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass" style="height:100%">
          <div style="color:#F59E0B;font-size:1em;font-weight:600;margin-bottom:12px">⚠️ Key Findings</div>
          <div class="insight insight-critical" style="margin:6px 0;font-size:0.85em">
            Most features show significant temporal drift (KS p&lt;0.05).
          </div>
          <div class="insight insight-success" style="margin:6px 0;font-size:0.85em">
            Continual learning recovers up to +6% accuracy.
          </div>
          <div class="insight" style="margin:6px 0;font-size:0.85em">
            Best model: Decision Tree (depth 20) — highest AUC on D1 and D2.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-hdr">📊 Encounter Volume Timeline</div>', unsafe_allow_html=True)
    all_df = pd.concat([d1.assign(Dataset="Historical (D1)"), d2.assign(Dataset="Current (D2)")])
    if "START" in all_df.columns:
        yearly = all_df.groupby([all_df["START"].dt.year,"Dataset"]).size().reset_index()
        yearly.columns = ["Year","Dataset","Encounters"]
        fig = px.bar(yearly, x="Year", y="Encounters", color="Dataset",
                     color_discrete_map={"Historical (D1)": C_D1, "Current (D2)": C_D2},
                     barmode="group")
        fig.add_vline(x=2014.5, line_dash="dash", line_color=C_AMBER,
                      annotation_text="2015 Temporal Split", annotation_position="top right",
                      annotation_font_color=C_AMBER)
        fig.update_layout(**PLOTLY_LAYOUT, height=360,
                          legend=dict(orientation="h", y=1.05, x=0, font_size=12))
        st.plotly_chart(fig, use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════
def page_data_overview(d):
    st.markdown('<div class="sec-head">📊 Data Overview & Statistics</div>', unsafe_allow_html=True)
    d1, d2 = d["d1"], d["d2"]

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("D1 Encounters", f"{len(d1):,}", icon="🗂️"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("D2 Encounters", f"{len(d2):,}", icon="📂"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("D1 Chronic Rate", f"{d1['target'].mean()*100:.1f}%", icon="🩺"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("D2 Chronic Rate", f"{d2['target'].mean()*100:.1f}%",
                              delta=d2["target"].mean()-d1["target"].mean(),
                              good_direction="down", icon="📈"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🍩 Class Distribution", "📋 Feature Statistics", "🏥 Encounter Classes"])

    with tab1:
        st.markdown('<div class="section-hdr">Target Class Split — D1 vs D2</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        for col, df, title, colors_pie, accent in [
            (c1, d1, "Dataset 1 — Historical (Pre-2015)", [C_D1, "#F472B6"], C_D1),
            (c2, d2, "Dataset 2 — Current (2015+)",      [C_D2, "#F472B6"], C_D2)]:
            with col:
                dist = df["target"].value_counts().sort_index()
                fig = go.Figure(go.Pie(
                    labels=["Non-Chronic","Chronic"],
                    values=[dist.get(0,0), dist.get(1,0)],
                    hole=0.52,
                    marker_colors=colors_pie,
                    textfont_size=13,
                    textinfo="label+percent+value",
                    hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>"))
                fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font_size=13), height=340, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div style="text-align:center;color:#64748B;font-size:0.82em;margin-top:-8px">'
                            f'📅 {df["START"].min().date()} → {df["START"].max().date()}</div>',
                            unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-hdr">Statistical Summary with Mean Shift</div>', unsafe_allow_html=True)
        key_feats = [f for f in ["age_at_encounter","systolic_bp","diastolic_bp","heart_rate",
                                   "bmi","glucose","creatinine","body_weight","INCOME","num_prior_encounters"]
                     if f in d1.columns]
        if key_feats:
            s1 = d1[key_feats].describe().T[["mean","std","50%","min","max"]].round(2)
            s2 = d2[key_feats].describe().T[["mean","std","50%","min","max"]].round(2)
            s1.columns = ["D1 Mean","D1 Std","D1 Median","D1 Min","D1 Max"]
            s2.columns = ["D2 Mean","D2 Std","D2 Median","D2 Min","D2 Max"]
            combined = pd.concat([s1, s2], axis=1)
            combined["Mean Shift"] = (combined["D2 Mean"] - combined["D1 Mean"]).round(3)
            st.dataframe(combined.style.format("{:.3f}").background_gradient(
                subset=["Mean Shift"], cmap="RdYlGn_r"), use_container_width=True)

    with tab3:
        st.markdown('<div class="section-hdr">Encounter Class Breakdown</div>', unsafe_allow_html=True)
        if "ENCOUNTERCLASS" in d1.columns:
            ec1 = d1["ENCOUNTERCLASS"].value_counts().reset_index(); ec1.columns=["Class","D1 Count"]
            ec2 = d2["ENCOUNTERCLASS"].value_counts().reset_index(); ec2.columns=["Class","D2 Count"]
            ec = ec1.merge(ec2, on="Class", how="outer").fillna(0)
            ec["D1 %"] = (ec["D1 Count"]/ec["D1 Count"].sum()*100).round(1)
            ec["D2 %"] = (ec["D2 Count"]/ec["D2 Count"].sum()*100).round(1)
            st.dataframe(ec.sort_values("D1 Count", ascending=False), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-hdr">📅 Dataset Timeline Overview</div>', unsafe_allow_html=True)
    img(d, "eda_plots", "dataset_overview.png")
# ═══════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════
def page_eda(d):
    st.markdown('<div class="sec-head">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        analysis = st.selectbox("Select Analysis View", [
            "Feature Distributions (D1 vs D2)", "Correlation Heatmap", "Box Plots",
            "Data Availability", "Class Distribution",
            "Cohort Analysis (Chronic vs Non-Chronic)", "Population Pyramid",
            "Patient Trajectories", "Healthcare Utilisation & Costs"],
            label_visibility="visible")
    with col_info:
        hints = {
            "Feature Distributions (D1 vs D2)": "Overlapping histograms reveal distribution shift between pre- and post-2015 data.",
            "Correlation Heatmap": "Pairwise Pearson correlations among engineered clinical features.",
            "Box Plots": "Quartile distributions of key vitals compared across datasets.",
            "Data Availability": "Missingness patterns per feature and per dataset.",
            "Class Distribution": "Balance of Chronic vs Non-Chronic labels per dataset.",
            "Cohort Analysis (Chronic vs Non-Chronic)": "Side-by-side feature distributions split by target class.",
            "Population Pyramid": "Age-gender breakdown of encounter populations.",
            "Patient Trajectories": "Temporal trends in encounter volume, chronic rates, and vital signs.",
            "Healthcare Utilisation & Costs": "Cost distributions and claim patterns by encounter class.",
        }
        st.markdown(f'<div class="info-box" style="margin-top:24px">ℹ️ {hints.get(analysis,"")}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    if analysis == "Feature Distributions (D1 vs D2)":
        img(d, "eda_plots", "feature_distributions.png")
        st.markdown('<div class="insight">Distribution overlays reveal rightward age shift in D2, '
                    'indicating an aging patient cohort post-2015.</div>', unsafe_allow_html=True)
    elif analysis == "Correlation Heatmap":
        img(d, "eda_plots", "correlation_heatmap.png")
        st.markdown('<div class="insight">High correlations: Systolic↔Diastolic BP (r≈0.56), '
                    'BMI↔Weight (r≈0.94). Stable correlation structure aids transfer learning.</div>',
                    unsafe_allow_html=True)
    elif analysis == "Box Plots":
        img(d, "eda_plots", "box_plots.png")
    elif analysis == "Data Availability":
        img(d, "eda_plots", "data_availability.png")
        st.markdown('<div class="insight insight-critical">⚠️ Clinical vitals missing in ~80% of encounters '
                    '— present only when measured during the visit. Median imputation applied from D1.</div>',
                    unsafe_allow_html=True)
    elif analysis == "Class Distribution":
        img(d, "eda_plots", "class_distribution.png")
    elif analysis == "Cohort Analysis (Chronic vs Non-Chronic)":
        img(d, "adv_plots", "cohort_analysis.png")
        clin = d.get("clinical_sum", pd.DataFrame())
        if not clin.empty:
            st.markdown('<div class="section-hdr">Clinical Feature Means by Group</div>', unsafe_allow_html=True)
            st.dataframe(clin.style.format("{:.2f}", subset=clin.select_dtypes(float).columns),
                         use_container_width=True)
    elif analysis == "Population Pyramid":
        img(d, "adv_plots", "population_pyramid.png")
        st.markdown('<div class="insight">Bimodal age distribution with peaks at 30-50 and 70+. '
                    'The 70+ cohort grows between D1 and D2 — key driver of increasing chronic rates.</div>',
                    unsafe_allow_html=True)
    elif analysis == "Patient Trajectories":
        img(d, "adv_plots", "patient_trajectories.png")
    elif analysis == "Healthcare Utilisation & Costs":
        img(d, "adv_plots", "utilisation_costs.png")
        st.markdown('<div class="insight">Chronic encounters cost 2-3× more than routine visits. '
                    'Early chronic detection has significant economic value.</div>', unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
def page_model_performance(d):
    st.markdown('<div class="sec-head">🤖 Model Performance</div>', unsafe_allow_html=True)
    metrics = d.get("metrics", pd.DataFrame())
    if metrics.empty:
        st.markdown('<div class="warn-box">⚠️ No metrics found — run the pipeline first.</div>',
                    unsafe_allow_html=True); return

    st.markdown('<div class="section-hdr">Cross-Dataset Performance Summary</div>', unsafe_allow_html=True)
    d1m = metrics[metrics["dataset"]=="D1_test"][["model","accuracy","f1_score","roc_auc","precision","recall"]]
    d2m = metrics[metrics["dataset"]=="D2_test"][["model","accuracy","f1_score","roc_auc","precision","recall"]]
    comp = d1m.merge(d2m, on="model", suffixes=("_D1","_D2"))
    comp["Temporal Gap (Acc)"] = (comp["accuracy_D1"]-comp["accuracy_D2"]).round(4)
    comp["Temporal Gap (F1)"]  = (comp["f1_score_D1"] -comp["f1_score_D2"]).round(4)
    st.dataframe(
        comp.rename(columns={"model":"Model"})
        .style.format({c:"{:.4f}" for c in comp.columns if c!="model"})
        .background_gradient(subset=["accuracy_D1","accuracy_D2"], cmap="YlGn")
        .background_gradient(subset=["Temporal Gap (Acc)"], cmap="YlOrRd"),
        use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "📈 ROC Curves","🎯 Precision-Recall","🟦 Confusion Matrices",
        "🎚 Threshold Analysis","🕸 Radar","🌡 Perf. Heatmap","📋 Detailed Metrics"])

    with tab1:
        roc = d.get("roc_data",{})
        if roc:
            colors = MULTI_SERIES_COLORS
            fig = make_subplots(1,2,subplot_titles=("ROC — D1 Test","ROC — D2 Test"))
            for i,(mname,rdata) in enumerate(roc.items()):
                for j,ds in enumerate(["D1_test","D2_test"]):
                    fpr,tpr = rdata[ds]
                    auc_row = metrics[(metrics["model"]==mname)&(metrics["dataset"]==ds)]
                    auc = auc_row["roc_auc"].values[0] if len(auc_row) else 0
                    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                                             name=f"{mname} ({auc:.3f})" if j==0 else None,
                                             showlegend=(j==0),
                                             line=dict(color=colors[i%len(colors)],width=2)),
                                  row=1,col=j+1)
            for c in [1,2]:
                fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                         line=dict(color="rgba(255,255,255,0.2)",dash="dash"),showlegend=False),row=1,col=c)
            fig.update_layout(**PLOTLY_LAYOUT,height=480,legend=dict(font=dict(size=8)))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight">ROC AUC measures separability across all thresholds. '
                    'Best performer: DT depth 20. SVM C=0.1 underperforms due to underfitting.</div>',
                    unsafe_allow_html=True)

    with tab2:
        img(d,"plots","precision_recall_curves.png")
        ap = d.get("avg_precision", pd.DataFrame())
        if not ap.empty:
            st.markdown("**Average Precision (AP) Scores**")
            ap_pivot = ap.pivot(index="model",columns="dataset",values="average_precision").round(4)
            if "D1_test" in ap_pivot.columns and "D2_test" in ap_pivot.columns:
                ap_pivot["AP Gap"] = (ap_pivot["D1_test"]-ap_pivot["D2_test"]).round(4)
            st.dataframe(ap_pivot.style.format("{:.4f}").background_gradient(
                cmap="YlGn",subset=[c for c in ap_pivot.columns if "test" in c.lower()]),
                use_container_width=True)

    with tab3:
        cm_data = d.get("cm_data",{})
        if cm_data:
            sel = st.selectbox("Select Model", list(cm_data.keys()), key="cm_model_sel")
            c1,c2 = st.columns(2)
            for col,ds,accent in [(c1,"D1_test",C_D1),(c2,"D2_test",C_D2)]:
                with col:
                    cm = cm_data[sel][ds]
                    tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
                    fig = go.Figure(go.Heatmap(
                        z=cm, x=["Non-Chronic","Chronic"],
                        y=["Non-Chronic","Chronic"],
                        colorscale=[[0,"#080C14"],[0.5,"rgba(79,139,255,0.4)"],[1,"#4F8BFF"]],
                        text=cm, texttemplate="<b>%{text}</b>", showscale=False,
                        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"))
                    fig.update_layout(**PLOTLY_LAYOUT,
                                      title=dict(text=f"{sel} — {ds}", font_color=accent),
                                      height=360, xaxis_title="Predicted", yaxis_title="Actual")
                    st.plotly_chart(fig, use_container_width=True)
                    spec=tn/(tn+fp+1e-8); sens=tp/(tp+fn+1e-8)
                    st.markdown(
                        f'<div class="glass-accent" style="padding:10px 16px;font-size:0.88em">'
                        f'<span style="color:#10B981">Sensitivity: <b>{sens:.3f}</b></span> &nbsp;|&nbsp; '
                        f'<span style="color:#4F8BFF">Specificity: <b>{spec:.3f}</b></span> &nbsp;|&nbsp; '
                        f'<span style="color:#F87171">FN: <b>{fn}</b></span> &nbsp;|&nbsp; '
                        f'<span style="color:#F59E0B">FP: <b>{fp}</b></span></div>',
                        unsafe_allow_html=True)
        img(d,"plots","per_class_f1.png")

    with tab4:
        img(d,"plots","threshold_analysis.png")
        st.markdown('<div class="insight"><b>Default threshold=0.5 is rarely clinically optimal.</b> '
                    'For screening, lower it (→ higher recall). For confirmatory diagnosis, raise it '
                    '(→ higher precision). Yellow line = F1-optimal threshold.</div>', unsafe_allow_html=True)

    with tab5:
        img(d,"plots","radar_comparison.png")
        st.markdown('<div class="insight">Radar chart compares best model per family on 5 metrics. '
                    'Decision Tree dominates on both datasets due to the structured nature of clinical features.</div>',
                    unsafe_allow_html=True)

    with tab6:
        img(d,"plots","performance_heatmap.png")

    with tab7:
        cls_rep = d.get("cls_reports", pd.DataFrame())
        if not cls_rep.empty:
            sel_ds  = st.selectbox("Dataset",  ["D1_test","D2_test"], key="cls_ds")
            sel_cls = st.selectbox("Class", ["Non-Chronic","Chronic","macro avg","weighted avg"], key="cls_name")
            sub = cls_rep[(cls_rep["dataset"]==sel_ds)&(cls_rep["class"]==sel_cls)]
            if not sub.empty:
                st.dataframe(sub[["model","precision","recall","f1-score","support"]]
                             .style.format({c:"{:.4f}" for c in["precision","recall","f1-score"]})
                             .background_gradient(subset=["f1-score"],cmap="YlGn"),
                             use_container_width=True)
        for ds in ["D1_train","D1_test","D2_test"]:
            subset = metrics[metrics["dataset"]==ds][["model","accuracy","precision","recall","f1_score","roc_auc"]]
            if not subset.empty:
                st.markdown(f"**{ds}**")
                st.dataframe(subset.style.format({c:"{:.4f}" for c in subset.columns if c!="model"}),
                             use_container_width=True)

    # ── Task 3a: Model Download Section ────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-hdr">💾 Download Saved Model Configurations (Task 3a)</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="insight">
    All trained models have been serialised to <code>.pkl</code> files on disk via <code>pickle.dump()</code>.
    Use the buttons below to download any model for offline inference or further analysis.
    Load with: <code>import pickle; model = pickle.load(open("DT_depth20.pkl","rb"))</code>
    </div>""", unsafe_allow_html=True)
    pkl_paths = d.get("model_pkl_paths", {})
    if pkl_paths:
        model_order = ["DT_depth5","DT_depth10","DT_depth15","DT_depth20",
                       "SVM_C0.1","SVM_C1","SVM_C10","SVM_C100",
                       "MLP_64_32","MLP_128_64_32","scaler"]
        cols_dl = st.columns(4)
        for i, key in enumerate([k for k in model_order if k in pkl_paths]):
            path = pkl_paths[key]
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data_bytes = f.read()
                with cols_dl[i % 4]:
                    st.download_button(
                        label=f"⬇ {key}.pkl",
                        data=data_bytes,
                        file_name=f"{key}.pkl",
                        mime="application/octet-stream",
                        key=f"dl_{key}")
    else:
        st.markdown('<div class="info-box">Model .pkl files will appear here after the pipeline runs.</div>',
                    unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: TEMPORAL SHIFT
# ═══════════════════════════════════════════════════════════════
def page_temporal_shift(d):
    st.markdown('<div class="sec-head">📉 Temporal Shift Analysis</div>', unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs([
        "⚖️ Complexity & Overfitting", "📊 Performance Gap",
        "🔬 Data Drift (KS)"])

    with tab1:
        img(d,"plots","complexity_analysis.png")
        st.markdown("""<div class="insight">
        <b>Complexity vs Generalization — see Bias-Variance tab for full analysis:</b><br>
        <span style="color:#F59E0B">●</span> <b>DT depth 5:</b> High bias. Shallow tree generalises well to current distribution.<br>
        <span style="color:#4F8BFF">●</span> <b>DT depth 20:</b> Rich feature capture; best overall D2 test accuracy.<br>
        <span style="color:#F87171">●</span> <b>SVM C=0.1:</b> Severe underfitting — linear margin insufficient for clinical non-linearity.<br>
        <span style="color:#F472B6">●</span> <b>MLP 128,64,32:</b> Best MLP. Early stopping + L2 prevents overfitting.
        </div>""", unsafe_allow_html=True)

    with tab2:
        img(d,"plots","temporal_shift.png")
        metrics = d.get("metrics", pd.DataFrame())
        if not metrics.empty:
            d1t = metrics[metrics["dataset"]=="D1_test"]; d2t = metrics[metrics["dataset"]=="D2_test"]
            comp = d1t.merge(d2t, on="model", suffixes=("_D1","_D2"))
            comp["Gap"] = (comp["accuracy_D1"]-comp["accuracy_D2"]).round(4)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp["model"],y=comp["accuracy_D1"],name="D1 Test",
                                  marker_color=C_D1, marker_line_color="rgba(79,139,255,0.5)",
                                  marker_line_width=1, opacity=0.85))
            fig.add_trace(go.Bar(x=comp["model"],y=comp["accuracy_D2"],name="D2 Test",
                                  marker_color=C_D2, marker_line_color="rgba(248,113,113,0.5)",
                                  marker_line_width=1, opacity=0.85))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                               title="Accuracy: D1 Test vs D2 Test", height=400,
                               xaxis_tickangle=-30,
                               legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        drift = d.get("drift", pd.DataFrame())
        if not drift.empty:
            top = drift.head(20)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top["feature"], y=top["ks_statistic"], name="KS Statistic",
                marker_color=[C_POS if v else C_NEG for v in top["drift_detected"]],
                opacity=0.85,
                hovertemplate="<b>%{x}</b><br>KS: %{y:.4f}<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT, title="KS Statistic by Feature (Top 20)",
                               xaxis_tickangle=-45, height=420,
                               xaxis_title="Feature", yaxis_title="KS Statistic")
            st.plotly_chart(fig, use_container_width=True)
            n_drift = drift["drift_detected"].sum()
            st.markdown(f'<div class="insight"><b>{n_drift}/{len(drift)}</b> features show significant '
                        f'drift (KS test p&lt;0.05). Top drifting: <b>{drift.iloc[0]["feature"]}</b>.</div>',
                        unsafe_allow_html=True)
            img(d, "eda_plots", "data_drift.png")
            st.markdown('<div class="section-hdr">KS Drift Table (Top 20)</div>', unsafe_allow_html=True)
            st.dataframe(
                drift[["feature","ks_statistic","ks_pvalue","mean_shift","drift_detected"]].head(20)
                    .style.format({"ks_statistic":"{:.4f}","ks_pvalue":"{:.4f}","mean_shift":"{:.4f}"}),
                use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════
def page_anomaly(d):
    st.markdown('<div class="sec-head">🔴 Anomaly Detection</div>', unsafe_allow_html=True)
    anom = d.get("anomaly_stats", pd.DataFrame())

    if not anom.empty:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi("D1 Anomaly Rate",  f"{anom.iloc[0]['Anomaly_Rate_pct']:.1f}%"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("D2 Anomaly Rate",  f"{anom.iloc[1]['Anomaly_Rate_pct']:.1f}%"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("D1 Anomaly Chr Rate", f"{anom.iloc[0]['Chronic_Rate_Anomaly_pct']:.1f}%"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("D2 Anomaly Chr Rate", f"{anom.iloc[1]['Chronic_Rate_Anomaly_pct']:.1f}%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_img, col_insight = st.columns([3, 1])
    with col_img:
        st.markdown('<div class="section-hdr">Isolation Forest — Anomaly Score Visualisation</div>',
                    unsafe_allow_html=True)
        img(d, "adv_plots", "anomaly_detection.png")
    with col_insight:
        st.markdown("""
        <div class="glass" style="margin-top:32px">
          <div style="color:#F87171;font-weight:600;margin-bottom:10px">🔴 Method: Isolation Forest</div>
          <div style="color:#94A3B8;font-size:0.88em;line-height:1.7">
            Detects encounters that are statistically anomalous by isolating observations with
            shorter average path lengths in random decision trees.<br><br>
            <span style="color:#10B981;font-weight:600">Key finding:</span>
            Anomalous encounters have significantly higher chronic condition rates —
            patients with unusual clinical profiles are more likely to have complex, chronic conditions.
          </div>
        </div>""", unsafe_allow_html=True)

    if not anom.empty:
        st.markdown('<div class="section-hdr">Anomaly Statistics Table</div>', unsafe_allow_html=True)
        st.dataframe(anom.style.format({c:"{:.2f}" for c in anom.select_dtypes(float).columns}),
                     use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: FEATURE REPRESENTATION ANALYSIS (Task 3f)
# ═══════════════════════════════════════════════════════════════
def page_feature_representation(d):
    st.markdown('<div class="sec-head">🔍 Feature Representation Analysis</div>', unsafe_allow_html=True)
    metrics  = d.get("metrics",  pd.DataFrame())
    splits   = d.get("splits",   {})
    scaler   = d.get("scaler")
    feature_names = d.get("feature_names", [])

    st.markdown("""
    <div class="glass" style="margin-bottom:20px">
      <div style="color:#4F8BFF;font-size:1.05em;font-weight:600;margin-bottom:12px">
        🎯 Task 3f — How Feature Representation Affects Model Performance
      </div>
      <div style="color:#94A3B8;font-size:0.91em;line-height:1.85">
        This analysis compares model performance across <b style="color:#E2E8F0">three feature subsets</b>
        to understand what each feature group contributes to chronic condition classification:<br><br>
        <span style="color:#4F8BFF">①</span> <b style="color:#E2E8F0">Full Feature Set</b> — all engineered features (vitals + labs + demographics + history counts + patient aggregates)<br>
        <span style="color:#F472B6">②</span> <b style="color:#E2E8F0">Clinical Observations Only</b> — vitals and lab values (systolic_bp, diastolic_bp, heart_rate, bmi, glucose, creatinine, etc.)<br>
        <span style="color:#F59E0B">③</span> <b style="color:#E2E8F0">Demographic & History Only</b> — age, gender, race, income, num_conditions, num_medications, num_prior_encounters
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not splits or scaler is None or not feature_names:
        st.markdown('<div class="warn-box">⚠️ Feature representation experiment requires pipeline data. Please run the pipeline first.</div>', unsafe_allow_html=True)
        return

    # ── Define feature subsets ─────────────────────────────────
    obs_keywords   = ["systolic_bp","diastolic_bp","heart_rate","bmi","glucose","creatinine",
                      "body_height","body_weight","respiratory_rate","pain_severity",
                      "calcium","sodium","potassium","chloride","co2","gfr","qaly","daly"]
    demo_keywords  = ["age_at_encounter","gender_M","race_","ethnicity_","marital_","INCOME",
                      "HEALTHCARE_EXPENSES","HEALTHCARE_COVERAGE","num_conditions","num_medications",
                      "num_allergies","num_procedures","num_prior_encounters","encounter_duration_min",
                      "enc_class_"]

    obs_feats   = [f for f in feature_names if any(k in f for k in obs_keywords)]
    demo_feats  = [f for f in feature_names if any(k in f for k in demo_keywords)]
    all_feats   = feature_names

    subsets = {
        "Full Features": all_feats,
        "Clinical Obs Only": obs_feats,
        "Demographics & History Only": demo_feats,
    }

    X1_tr = splits["X1_train"]; X1_te = splits["X1_test"]
    X2_te = splits["X2_test"]
    y1_tr = splits["y1_train"]; y1_te = splits["y1_test"]; y2_te = splits["y2_test"]

    # ── Run lightweight comparison across subsets ──────────────
    from sklearn.tree import DecisionTreeClassifier as DTC
    results_repr = []

    with st.spinner("Running feature representation experiments …"):
        for subset_name, feat_list in subsets.items():
            if not feat_list:
                continue
            # Get indices in the full feature_names list
            idx = [feature_names.index(f) for f in feat_list if f in feature_names]
            if not idx:
                continue

            X1_tr_sub = X1_tr[:, idx]; X1_te_sub = X1_te[:, idx]; X2_te_sub = X2_te[:, idx]

            for depth in [5, 15]:
                clf = DTC(max_depth=depth, random_state=42, class_weight="balanced",
                          min_samples_split=5, min_samples_leaf=2)
                clf.fit(X1_tr_sub, y1_tr)
                acc_d1 = accuracy_score(y1_te, clf.predict(X1_te_sub))
                f1_d1  = f1_score(y1_te, clf.predict(X1_te_sub), average="weighted", zero_division=0)
                acc_d2 = accuracy_score(y2_te, clf.predict(X2_te_sub))
                f1_d2  = f1_score(y2_te, clf.predict(X2_te_sub), average="weighted", zero_division=0)
                results_repr.append({
                    "Feature Set": subset_name,
                    "# Features": len(idx),
                    "DT Depth": depth,
                    "D1 Test Acc": round(acc_d1, 4),
                    "D1 Test F1":  round(f1_d1,  4),
                    "D2 Test Acc": round(acc_d2, 4),
                    "D2 Test F1":  round(f1_d2,  4),
                    "D1-D2 Gap":   round(acc_d1 - acc_d2, 4),
                })

    if not results_repr:
        st.warning("No results generated — feature subsets may be empty for this dataset.")
        return

    repr_df = pd.DataFrame(results_repr)

    # ── Interactive Plotly comparison ──────────────────────────
    st.markdown('<div class="section-hdr">Accuracy Comparison Across Feature Sets</div>', unsafe_allow_html=True)
    fig = go.Figure()
    palette = {
        "Full Features": C_D1,
        "Clinical Obs Only": "#F472B6",
        "Demographics & History Only": C_AMBER,
    }
    for feat_set in repr_df["Feature Set"].unique():
        sub = repr_df[repr_df["Feature Set"] == feat_set]
        labels = [f"Depth {d}" for d in sub["DT Depth"]]
        fig.add_trace(go.Bar(
            name=f"{feat_set} — D1 Test",
            x=labels, y=sub["D1 Test Acc"],
            marker_color=palette.get(feat_set, C_D1),
            opacity=0.9,
            hovertemplate=f"<b>{feat_set}</b><br>D1 Acc: %{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Bar(
            name=f"{feat_set} — D2 Test",
            x=labels, y=sub["D2 Test Acc"],
            marker_color=palette.get(feat_set, C_D1),
            opacity=0.45,
            hovertemplate=f"<b>{feat_set}</b><br>D2 Acc: %{{y:.4f}}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                      title="Feature Set Comparison: D1 Test vs D2 Test Accuracy",
                      height=420, xaxis_title="Model Configuration",
                      yaxis_title="Accuracy",
                      legend=dict(orientation="h", y=-0.25, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # D1-D2 gap chart
    st.markdown('<div class="section-hdr">Temporal Drift Gap by Feature Representation</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    for feat_set in repr_df["Feature Set"].unique():
        sub = repr_df[repr_df["Feature Set"] == feat_set]
        fig2.add_trace(go.Bar(
            name=feat_set,
            x=[f"Depth {d}" for d in sub["DT Depth"]],
            y=sub["D1-D2 Gap"],
            marker_color=palette.get(feat_set, C_D1),
            opacity=0.85,
            hovertemplate=f"<b>{feat_set}</b><br>Gap: %{{y:.4f}}<extra></extra>"))
    fig2.update_layout(**PLOTLY_LAYOUT, barmode="group",
                       title="D1→D2 Accuracy Gap by Feature Set (lower = more robust to drift)",
                       height=360, xaxis_title="Model", yaxis_title="Accuracy Gap (D1−D2)",
                       legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Results table ──────────────────────────────────────────
    st.markdown('<div class="section-hdr">Detailed Results</div>', unsafe_allow_html=True)
    st.dataframe(
        repr_df.style.format({c:"{:.4f}" for c in repr_df.columns
                              if c not in["Feature Set","# Features","DT Depth"]})
            .background_gradient(subset=["D1 Test Acc","D2 Test Acc"], cmap="YlGn")
            .background_gradient(subset=["D1-D2 Gap"], cmap="YlOrRd"),
        use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
def page_feature_importance(d):
    st.markdown('<div class="sec-head">🌟 Feature Importance</div>', unsafe_allow_html=True)
    feat_imp = d.get("feat_imp", pd.DataFrame())

    if not feat_imp.empty:
        col_chart, col_note = st.columns([3, 1])
        with col_chart:
            st.markdown('<div class="section-hdr">Top 20 Features — Gini Importance (Best Decision Tree)</div>',
                        unsafe_allow_html=True)
            top20 = feat_imp.head(20)
            # Gradient color by rank
            n = len(top20)
            bar_colors = [f"rgba({int(0 + (248-0)*i/n)},{int(210 + (113-210)*i/n)},{int(200 + (113-200)*i/n)},0.85)"
                          for i in range(n)]
            fig = go.Figure(go.Bar(
                x=top20["importance"], y=top20["feature"],
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"))
            fig.update_layout(**PLOTLY_LAYOUT,
                               xaxis_title="Gini Importance",
                               height=600)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        with col_note:
            st.markdown("""
            <div class="glass" style="margin-top:32px">
              <div style="color:#4F8BFF;font-weight:600;margin-bottom:10px">📌 Interpretation</div>
              <div style="color:#94A3B8;font-size:0.87em;line-height:1.7">
                <b style="color:#E2E8F0">Gini Importance</b> measures how much each feature reduces
                impurity across all Decision Tree splits.<br><br>
                Patient history counts and age-related features typically dominate in
                chronic condition prediction.<br><br>
                <span style="color:#10B981">High importance</span> → top-of-chart features.<br>
                <span style="color:#F59E0B">Demographic features</span> often provide independent signal.
              </div>
            </div>""", unsafe_allow_html=True)

    img(d, "plots", "feature_importance.png")

    if not feat_imp.empty:
        st.markdown('<div class="section-hdr">Full Importance Table (Top 30)</div>', unsafe_allow_html=True)
        st.dataframe(feat_imp.head(30).style.format({"importance":"{:.4f}"}), use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════
def page_hyperparameter_tuning(d):
    st.markdown('<div class="sec-head">⚙️ Hyperparameter Tuning</div>',
                unsafe_allow_html=True)

    hp = d.get("hptune", {})
    summary_df = hp.get("summary_df", pd.DataFrame())
    cv_results = hp.get("cv_results", {})
    tuned_metrics = hp.get("tuned_metrics", pd.DataFrame())

    if summary_df.empty:
        st.markdown(
            '<div class="warn-box">⚠️ No tuning results found — the tuning '
            'stage did not complete. Re-run the pipeline to generate them.</div>',
            unsafe_allow_html=True)
        return

    st.markdown("""
    <div class="insight">
      Formal hyperparameter tuning using <b>GridSearchCV</b> with
      <b>StratifiedKFold cross-validation</b> on Dataset 1's training set only.
      Test sets are never touched during tuning. Scoring is macro-F1
      (balanced across classes). Best parameters are then used to re-fit
      each model on the full training data and evaluated on both D1_test
      and D2_test.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row: best model per family ─────────────────────────
    st.markdown('<div class="section-hdr">Best Configurations by Model Family</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(summary_df))
    icons = {"DecisionTree": "🌳", "SVM-RBF": "🧮", "MLP": "🧠"}
    for i, (_, row) in enumerate(summary_df.iterrows()):
        family = row["Model Family"]
        with cols[i]:
            st.markdown(kpi(
                f"{family} · Best CV F1",
                f"{row['Best CV F1 (macro)']:.4f}",
                icon=icons.get(family, "⚙️")),
                unsafe_allow_html=True)

    # ── Search statistics strip ────────────────────────────────
    total_configs = int(summary_df["Configs Tested"].sum())
    total_fits    = int(summary_df["Total Fits"].sum())
    st.markdown(f"""
    <div class="glass-accent" style="padding:14px 18px;margin:12px 0">
      <div style="color:#94A3B8;font-size:0.88em;line-height:1.8">
        <span style="color:#4F8BFF">▸</span>
        <b style="color:#E2E8F0">{total_configs}</b> unique parameter combinations tested&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#4F8BFF">▸</span>
        <b style="color:#E2E8F0">{total_fits}</b> total CV fits ({HPTUNE_CV_FOLDS}-fold)&nbsp;&nbsp;·&nbsp;&nbsp;
        <span style="color:#4F8BFF">▸</span>
        Scoring: <b style="color:#E2E8F0">{HPTUNE_SCORING}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs for each family ───────────────────────────────────
    st.markdown('<div class="section-hdr">Search Results by Model Family</div>',
                unsafe_allow_html=True)
    family_tabs = st.tabs([f"{icons.get(f, '⚙️')} {f}"
                            for f in summary_df["Model Family"].tolist()])

    for tab, (_, row) in zip(family_tabs, summary_df.iterrows()):
        family = row["Model Family"]
        with tab:
            # Best params block
            st.markdown(f"""
            <div class="glass" style="padding:18px 22px;margin-bottom:14px">
              <div style="color:#4F8BFF;font-size:0.85em;font-weight:600;
                          text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">
                ✅ Best parameters found
              </div>
              <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:baseline">
                <div>
                  <div style="color:#64748B;font-size:0.72em;text-transform:uppercase;letter-spacing:1px">Best CV F1 (macro)</div>
                  <div style="color:#F472B6;font-size:1.5em;font-weight:700">{row['Best CV F1 (macro)']:.4f}</div>
                  <div style="color:#94A3B8;font-size:0.78em">± {row['CV Std']:.4f} across {HPTUNE_CV_FOLDS} folds</div>
                </div>
                <div style="flex:1;min-width:280px">
                  <div style="color:#64748B;font-size:0.72em;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Parameters</div>
                  <div style="font-family:JetBrains Mono,monospace;font-size:0.92em;color:#E2E8F0;
                              background:rgba(79,139,255,0.08);padding:10px 14px;border-radius:8px;
                              border:1px solid rgba(79,139,255,0.15)">
                    {_format_params(row['Best Params'])}
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Full grid results as a DataFrame
            if family in cv_results:
                cvr = cv_results[family]
                grid_df = _cv_results_to_df(cvr)
                st.markdown(f"**Full grid search results** ({len(grid_df)} configurations, sorted by CV F1)")
                st.dataframe(
                    grid_df.style.format({
                        "mean_test_score": "{:.4f}",
                        "std_test_score":  "{:.4f}",
                        "mean_train_score": "{:.4f}",
                        "mean_fit_time": "{:.2f}s",
                    }).background_gradient(subset=["mean_test_score"], cmap="YlGn"),
                    use_container_width=True, hide_index=True)

                # 2D heatmap if grid is 2-dimensional
                _render_cv_heatmap(cvr, family)

    # ── Comparison: tuned models vs best sweep models ──────────
    st.markdown("---")
    st.markdown('<div class="section-hdr">Tuned Model Performance on Test Sets</div>',
                unsafe_allow_html=True)

    if not tuned_metrics.empty:
        # Join against the original sweep's best model per family for comparison
        sweep_metrics = d.get("metrics", pd.DataFrame())
        rows = []
        for _, r in tuned_metrics.iterrows():
            rows.append({
                "Model":    r["model"],
                "Dataset":  r["dataset"],
                "Accuracy": r["accuracy"],
                "F1":       r["f1_score"],
                "ROC AUC":  r["roc_auc"],
                "Precision":r["precision"],
                "Recall":   r["recall"],
            })
        tuned_display = pd.DataFrame(rows)
        st.markdown("**Tuned model metrics** (re-fitted with best params, evaluated on test sets)")
        st.dataframe(
            tuned_display.style.format({
                c: "{:.4f}" for c in ["Accuracy", "F1", "ROC AUC", "Precision", "Recall"]
            }).background_gradient(
                subset=["F1"], cmap="YlGn"),
            use_container_width=True, hide_index=True)

        # Side-by-side: tuned vs best sweep per family
        if not sweep_metrics.empty:
            st.markdown('<div class="section-hdr">Tuned vs Best-Sweep Comparison</div>',
                        unsafe_allow_html=True)

            compare_rows = []
            family_map = {"DecisionTree": "DecisionTree",
                           "SVM-RBF":      "SVM",
                           "MLP":          "MLP"}
            for fam in summary_df["Model Family"]:
                sweep_type = family_map.get(fam, fam)
                for ds in ["D1_test", "D2_test"]:
                    sweep_sub = sweep_metrics[
                        (sweep_metrics["model_type"] == sweep_type) &
                        (sweep_metrics["dataset"] == ds)]
                    tuned_sub = tuned_metrics[
                        (tuned_metrics["model"] == f"{fam}_tuned") &
                        (tuned_metrics["dataset"] == ds)]
                    if sweep_sub.empty or tuned_sub.empty:
                        continue
                    best_sweep = sweep_sub.loc[sweep_sub["f1_score"].idxmax()]
                    t = tuned_sub.iloc[0]
                    compare_rows.append({
                        "Family":         fam,
                        "Dataset":        ds,
                        "Sweep Best F1":  round(best_sweep["f1_score"], 4),
                        "Tuned F1":       round(t["f1_score"], 4),
                        "ΔF1":            round(t["f1_score"] - best_sweep["f1_score"], 4),
                        "Sweep Best Acc": round(best_sweep["accuracy"], 4),
                        "Tuned Acc":      round(t["accuracy"], 4),
                        "ΔAcc":           round(t["accuracy"] - best_sweep["accuracy"], 4),
                    })
            if compare_rows:
                comp_df = pd.DataFrame(compare_rows)
                st.dataframe(
                    comp_df.style.format({
                        c: "{:+.4f}" if c.startswith("Δ") else "{:.4f}"
                        for c in comp_df.columns
                        if c not in ["Family", "Dataset"]
                    }).background_gradient(subset=["ΔF1", "ΔAcc"], cmap="RdYlGn"),
                    use_container_width=True, hide_index=True)

                # Interpretive note based on results
                pos_deltas = (comp_df["ΔF1"] > 0).sum()
                st.markdown(f"""
                <div class="insight insight-success">
                  <b>Interpretation:</b> Out of {len(comp_df)} family × dataset cells,
                  the tuned model matches or outperforms the best sweep in
                  <b>{pos_deltas}</b>. CV-selected hyperparameters reflect an
                  unbiased estimate of generalization performance — the sweep's
                  test-set "best" may be optimistic because it uses the test set
                  for model selection.
                </div>
                """, unsafe_allow_html=True)

    # ── Methodology footer ─────────────────────────────────────
    with st.expander("📖 Methodology details"):
        st.markdown(f"""
        - **Search strategy:** Exhaustive grid search (`sklearn.model_selection.GridSearchCV`)
        - **Cross-validation:** {HPTUNE_CV_FOLDS}-fold `StratifiedKFold`, shuffle=True, random_state={HPTUNE_RANDOM_STATE}
        - **Scoring metric:** `{HPTUNE_SCORING}` (macro-averaged F1, handles class imbalance)
        - **Data used:** D1 training set only — D1_test and D2_test are held out
        - **SVM-RBF subsample:** {HPTUNE_SVM_MAX_SAMPLES} stratified samples during tuning to keep runtime manageable; best model is re-fit on up to {SVM_MAX_SAMPLES} samples
        - **MLP subsample:** {HPTUNE_MLP_MAX_SAMPLES} samples and `max_iter=100` during tuning; best model is re-fit on full training data with `max_iter=300`
        - **Parallelism:** `n_jobs=-1` — uses all available cores
        - **Parameter grids:**
            - Decision Tree: `max_depth` ∈ {HPTUNE_GRIDS['DecisionTree']['max_depth']}, `min_samples_split` ∈ {HPTUNE_GRIDS['DecisionTree']['min_samples_split']}, `min_samples_leaf` ∈ {HPTUNE_GRIDS['DecisionTree']['min_samples_leaf']}
            - SVM-RBF: `C` ∈ {HPTUNE_GRIDS['SVM-RBF']['C']}, `gamma` ∈ {HPTUNE_GRIDS['SVM-RBF']['gamma']}
            - MLP: `hidden_layer_sizes` ∈ {HPTUNE_GRIDS['MLP']['hidden_layer_sizes']}, `alpha` ∈ {HPTUNE_GRIDS['MLP']['alpha']}
        """)
        
def _format_params(params) -> str:
    """Pretty-print a GridSearchCV best_params_ dict for display."""
    if not isinstance(params, dict):
        return str(params)
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        else:
            parts.append(f"{k}={v}")
    return ",<br>".join(parts)

def _cv_results_to_df(cv_results: dict) -> pd.DataFrame:
    """Turn GridSearchCV.cv_results_ into a tidy DataFrame sorted by score."""
    rows = []
    for i, params in enumerate(cv_results["params"]):
        rows.append({
            **{f"param_{k}": v for k, v in params.items()},
            "mean_test_score":  cv_results["mean_test_score"][i],
            "std_test_score":   cv_results["std_test_score"][i],
            "mean_train_score": cv_results.get("mean_train_score", [np.nan]*len(cv_results["params"]))[i],
            "mean_fit_time":    cv_results["mean_fit_time"][i],
            "rank":             cv_results["rank_test_score"][i],
        })
    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    return df

def _render_cv_heatmap(cv_results: dict, family: str):
    """If the grid is exactly 2-dimensional, render a heatmap of CV scores."""
    param_keys = [k[6:] for k in cv_results.keys() if k.startswith("param_")]
    if len(param_keys) != 2:
        return
    k1, k2 = param_keys
    vals1 = sorted({p[k1] for p in cv_results["params"]}, key=lambda x: str(x))
    vals2 = sorted({p[k2] for p in cv_results["params"]}, key=lambda x: str(x))

    # Build score matrix
    score_mat = np.full((len(vals1), len(vals2)), np.nan)
    for i, params in enumerate(cv_results["params"]):
        r = vals1.index(params[k1])
        c = vals2.index(params[k2])
        score_mat[r, c] = cv_results["mean_test_score"][i]

    fig = go.Figure(go.Heatmap(
        z=score_mat,
        x=[str(v) for v in vals2],
        y=[str(v) for v in vals1],
        colorscale=[[0, "#080C14"], [0.5, "rgba(79,139,255,0.4)"], [1, "#4F8BFF"]],
        text=np.round(score_mat, 3),
        texttemplate="%{text}",
        hovertemplate=f"{k1}: %{{y}}<br>{k2}: %{{x}}<br>CV F1: %{{z:.4f}}<extra></extra>",
        colorbar=dict(title="CV F1"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                       title=f"{family} — CV score heatmap",
                       xaxis_title=k2, yaxis_title=k1,
                       height=320)
    st.plotly_chart(fig, use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: CONTINUAL LEARNING
# ═══════════════════════════════════════════════════════════════
def page_continual_learning(d):
    st.markdown('<div class="sec-head">🔄 Continual Learning</div>', unsafe_allow_html=True)
    cl = d.get("cl_results", pd.DataFrame())
    if cl.empty:
        st.markdown('<div class="warn-box">⚠️ Continual learning results not available.</div>',
                    unsafe_allow_html=True); return

    # Strategy cards
    strategy_cards = [
        ("🌳 Decision Tree", "Weighted Retraining",
         "Retrained on D1+D2 combined with 2× weight on D2. Full retraining required since trees cannot be incrementally updated.",
         "#4F8BFF"),
        ("⚙️ SVM", "Importance-Weighted Training",
         "RBF SVM retrained on balanced sample (up to 7,500 from each dataset), D2 weighted 2×. Better captures current decision boundary.",
         "#F472B6"),
        ("🧠 MLP", "Transfer Learning / Fine-Tuning",
         "Pre-trained D1 weights used as initialisation. Fine-tuned on D2 with 10× lower learning rate (0.0001) to prevent catastrophic forgetting.",
         "#F59E0B"),
    ]
    cols = st.columns(3)
    for col, (icon_title, method, desc, color) in zip(cols, strategy_cards):
        with col:
            st.markdown(f"""
            <div class="glass" style="border-color:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.25);min-height:160px">
              <div style="color:{color};font-weight:600;font-size:0.95em;margin-bottom:6px">{icon_title}</div>
              <div style="color:#E2E8F0;font-size:0.82em;font-weight:500;margin-bottom:8px">{method}</div>
              <div style="color:#64748B;font-size:0.8em;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Before vs After Continual Learning</div>', unsafe_allow_html=True)

    metrics_pairs = [("before_accuracy","after_accuracy","Accuracy"),
                     ("before_f1","after_f1","F1 Score"),
                     ("before_roc_auc","after_roc_auc","ROC AUC")]
    fig = make_subplots(1, 3, subplot_titles=[p[2] for p in metrics_pairs])
    for j,(bcol,acol,title) in enumerate(metrics_pairs):
        if bcol not in cl.columns: continue
        fig.add_trace(go.Bar(name="Before (D1-trained)" if j==0 else None,
                             x=cl["model_type"], y=cl[bcol],
                             marker_color=C_POS, opacity=0.75,
                             showlegend=(j==0), legendgroup="before",
                             hovertemplate="%{x}<br>Before: %{y:.4f}<extra></extra>"),
                      row=1, col=j+1)
        fig.add_trace(go.Bar(name="After Continual Learning" if j==0 else None,
                             x=cl["model_type"], y=cl[acol],
                             marker_color=C_GREEN, opacity=0.88,
                             showlegend=(j==0), legendgroup="after",
                             hovertemplate="%{x}<br>After: %{y:.4f}<extra></extra>"),
                      row=1, col=j+1)
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=400,
                      legend=dict(orientation="h", y=1.12, x=0.3))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-hdr">Accuracy Improvement on D2 Test Set</div>', unsafe_allow_html=True)
    fig2 = go.Figure(go.Bar(
        x=cl["model_type"], y=cl["improvement_pct"],
        marker_color=[C_GREEN if v >= 0 else C_POS for v in cl["improvement_pct"]],
        text=cl["improvement_pct"].round(2).astype(str)+"%",
        textposition="outside",
        hovertemplate="%{x}<br>Improvement: %{y:+.2f}%<extra></extra>"))
    fig2.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig2.update_layout(**PLOTLY_LAYOUT, yaxis_title="Improvement (%)", height=340)
    st.plotly_chart(fig2, use_container_width=True)

    img(d, "plots", "continual_learning.png")

    st.markdown('<div class="section-hdr">Detailed Results Table</div>', unsafe_allow_html=True)
    display = cl[["model_type","before_accuracy","after_accuracy",
                  "before_f1","after_f1","before_roc_auc","after_roc_auc","improvement_pct"]].copy()
    display.columns = ["Model","Before Acc","After Acc","Before F1","After F1",
                        "Before AUC","After AUC","Improvement %"]
    st.dataframe(
        display.style.format({c:"{:.4f}" for c in display.columns if c not in["Model","Improvement %"]}
                              | {"Improvement %":"{:+.2f}%"}),
        use_container_width=True)
# ═══════════════════════════════════════════════════════════════
# PAGE: SUMMARY
# ═══════════════════════════════════════════════════════════════
def _generate_pdf_report(d) -> bytes:
    """
    Build a multi-page PDF summary of the pipeline results using matplotlib's
    PdfPages backend. Returns raw PDF bytes (for st.download_button).
    Purely additive — no dependency beyond matplotlib (already imported).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()

    # Page constants — A4 portrait
    FIG_W, FIG_H = 8.27, 11.69
    BG      = "#0D1321"
    FG      = "#E2E8F0"
    MUTED   = "#94A3B8"
    FAINT   = "#64748B"
    PRIMARY = "#4F8BFF"
    SECOND  = "#F472B6"
    ACCENT  = "#FB923C"
    GOOD    = "#10B981"
    BAD     = "#F87171"

    metrics = d.get("metrics", pd.DataFrame())
    cl      = d.get("cl_results", pd.DataFrame())
    drift   = d.get("drift", pd.DataFrame())
    d1      = d.get("d1", pd.DataFrame())
    d2      = d.get("d2", pd.DataFrame())

    def _page_frame(title_text=None, page_num=None, total_pages=None):
        """Create a blank A4 page with consistent header/footer chrome."""
        fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
        # Top accent bar
        fig.add_artist(plt.Rectangle((0.06, 0.955), 0.88, 0.0035,
                                      color=PRIMARY, alpha=0.9))
        fig.add_artist(plt.Rectangle((0.06, 0.955), 0.30, 0.0035,
                                      color=SECOND, alpha=0.9))
        # Header branding
        fig.text(0.06, 0.97, "ChronicML", fontsize=11, color=PRIMARY,
                 fontweight='bold', family='sans-serif')
        fig.text(0.13, 0.97, " Condition Intelligence Report",
                 fontsize=9, color=MUTED)
        # Date
        fig.text(0.94, 0.97, datetime.now().strftime("%Y-%m-%d %H:%M"),
                 fontsize=8, color=FAINT, ha='right')
        if title_text:
            fig.text(0.06, 0.92, title_text, fontsize=16, color=FG,
                     fontweight='bold')
            fig.add_artist(plt.Rectangle((0.06, 0.907), 0.06, 0.002,
                                          color=PRIMARY))
        # Footer
        if page_num is not None and total_pages is not None:
            fig.text(0.5, 0.03,
                     f"Page {page_num} of {total_pages}  ·  BITS F464 · Assignment 2",
                     fontsize=7, color=FAINT, ha='center')
        return fig

    def _draw_table(fig, headers, rows, y_top, col_widths=None,
                    x_left=0.06, x_right=0.94, row_h=0.028,
                    header_color=PRIMARY, zebra=True, numeric_cols=None):
        """Render a styled table on the figure using matplotlib primitives."""
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [(x_right - x_left) / n_cols] * n_cols
        x_positions = [x_left]
        for w in col_widths[:-1]:
            x_positions.append(x_positions[-1] + w)

        # Header row
        fig.add_artist(plt.Rectangle((x_left, y_top - row_h),
                                      x_right - x_left, row_h,
                                      color=header_color, alpha=0.18))
        for i, h in enumerate(headers):
            fig.text(x_positions[i] + 0.008, y_top - row_h / 2, h,
                     fontsize=8, color=FG, fontweight='bold',
                     verticalalignment='center')

        # Data rows
        y = y_top - row_h
        for r_idx, row in enumerate(rows):
            y -= row_h
            if zebra and r_idx % 2 == 1:
                fig.add_artist(plt.Rectangle((x_left, y),
                                              x_right - x_left, row_h,
                                              color="#FFFFFF", alpha=0.025))
            for i, cell in enumerate(row):
                align_right = numeric_cols and i in numeric_cols
                x_cell = x_positions[i] + col_widths[i] - 0.008 if align_right \
                         else x_positions[i] + 0.008
                ha = 'right' if align_right else 'left'
                fig.text(x_cell, y + row_h / 2, str(cell),
                         fontsize=7.5, color=FG, ha=ha,
                         verticalalignment='center')
        # Bottom border
        fig.add_artist(plt.Line2D([x_left, x_right], [y, y],
                                   color=MUTED, alpha=0.25, lw=0.5))
        return y

    # Precompute total pages
    total_pages = 1  # title + exec summary
    if not metrics.empty: total_pages += 1
    if not cl.empty:      total_pages += 1
    if not drift.empty:   total_pages += 1
    total_pages += 1      # closing page

    with PdfPages(buf) as pdf:
        # ══════════ Page 1: Title + Executive Summary ══════════
        fig = _page_frame(page_num=1, total_pages=total_pages)
        # Hero block
        fig.text(0.5, 0.78, "⚕", ha='center', fontsize=54, color=PRIMARY)
        fig.text(0.5, 0.71, "Chronic Condition Intelligence", ha='center',
                 fontsize=22, color=FG, fontweight='bold')
        fig.text(0.5, 0.66, "Pipeline Report", ha='center',
                 fontsize=14, color=MUTED)
        fig.add_artist(plt.Line2D([0.3, 0.7], [0.625, 0.625],
                                   color=PRIMARY, lw=1))
        fig.text(0.5, 0.595, datetime.now().strftime("Generated  %B %d, %Y · %H:%M"),
                 ha='center', fontsize=10, color=FAINT)
        fig.text(0.5, 0.57, "BITS F464 · Assignment 2  ·  Synthea EHR",
                 ha='center', fontsize=9, color=FAINT)
        # Team details (pulled from TEAM_NUMBER / TEAM_MEMBERS at top of file)
        if TEAM_NUMBER:
            team_line = f"Team {TEAM_NUMBER}"
            if TEAM_MEMBERS:
                team_line += "  ·  " + "  ·  ".join(TEAM_MEMBERS)
            fig.text(0.5, 0.545, team_line,
                     ha='center', fontsize=9, color=PRIMARY, fontweight='bold')

        # Executive summary
        fig.text(0.06, 0.48, "EXECUTIVE SUMMARY", fontsize=10,
                 color=PRIMARY, fontweight='bold')
        fig.add_artist(plt.Rectangle((0.06, 0.473), 0.05, 0.0018,
                                      color=PRIMARY))

        y = 0.44
        lines = []
        if len(d1) or len(d2):
            lines.append(("Dataset",
                f"D1 (historical): {len(d1):,} rows  |  "
                f"D2 (current): {len(d2):,} rows"))
        if not metrics.empty:
            d1t = metrics[metrics["dataset"] == "D1_test"]
            d2t = metrics[metrics["dataset"] == "D2_test"]
            if not d1t.empty and not d2t.empty:
                best_d1 = d1t.loc[d1t["f1_score"].idxmax()]
                best_d2 = d2t.loc[d2t["f1_score"].idxmax()]
                avg_gap = (d1t["accuracy"].values -
                           d2t["accuracy"].values).mean()
                lines.append(("Best D1 model",
                              f"{best_d1['model']}  (F1 = {best_d1['f1_score']:.4f},"
                              f" AUC = {best_d1['roc_auc']:.4f})"))
                lines.append(("Best D2 model",
                              f"{best_d2['model']}  (F1 = {best_d2['f1_score']:.4f},"
                              f" AUC = {best_d2['roc_auc']:.4f})"))
                lines.append(("Avg temporal gap",
                              f"{avg_gap:+.4f} accuracy (D1 − D2)"))
        if not drift.empty:
            n_drift = int(drift["drift_detected"].sum())
            lines.append(("Data drift",
                          f"{n_drift} of {len(drift)} features show significant "
                          f"temporal drift (KS p < 0.05)"))
        if not cl.empty:
            lines.append(("Continual learning",
                          f"Best gain +{cl['improvement_pct'].max():.2f}%,  "
                          f"avg +{cl['improvement_pct'].mean():.2f}%"))

        for label, txt in lines:
            fig.text(0.08, y, "●", fontsize=9, color=PRIMARY)
            fig.text(0.11, y, label.upper(), fontsize=7.5,
                     color=MUTED, fontweight='bold')
            fig.text(0.11, y - 0.018, txt, fontsize=9, color=FG)
            y -= 0.045

        # Key findings callout
        fig.add_artist(plt.Rectangle((0.06, 0.10), 0.88, 0.12,
                                      color=PRIMARY, alpha=0.06))
        fig.add_artist(plt.Rectangle((0.06, 0.10), 0.004, 0.12, color=PRIMARY))
        fig.text(0.08, 0.195, "KEY TAKEAWAYS",
                 fontsize=9, color=PRIMARY, fontweight='bold')
        findings = [
            "• Temporal distribution shift is detectable across most clinical features.",
            "• Models trained on pre-2015 data show measurable degradation on post-2015 data.",
            "• Continual learning meaningfully recovers performance lost to drift.",
            "• Decision Tree (depth 20) is the most robust model family overall.",
        ]
        fy = 0.17
        for f in findings:
            fig.text(0.09, fy, f, fontsize=8.5, color=FG)
            fy -= 0.022

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

        # ══════════ Page 2: Model Performance Table ══════════
        page_idx = 2
        if not metrics.empty:
            fig = _page_frame("Model Performance", page_idx, total_pages)
            page_idx += 1

            d1m = metrics[metrics["dataset"] == "D1_test"][
                ["model", "accuracy", "f1_score", "roc_auc",
                 "precision", "recall"]]
            d2m = metrics[metrics["dataset"] == "D2_test"][
                ["model", "accuracy", "f1_score", "roc_auc"]]
            comp = d1m.merge(d2m, on="model", suffixes=("_D1", "_D2"))
            comp["Gap"] = (comp["accuracy_D1"] - comp["accuracy_D2"]).round(4)

            fig.text(0.06, 0.875, "Cross-Dataset Comparison",
                     fontsize=10, color=PRIMARY, fontweight='bold')
            headers = ["Model", "Acc D1", "Acc D2", "F1 D1", "F1 D2",
                       "AUC D1", "AUC D2", "Gap"]
            rows = []
            for _, r in comp.iterrows():
                rows.append([
                    str(r["model"])[:24],
                    f"{r['accuracy_D1']:.4f}", f"{r['accuracy_D2']:.4f}",
                    f"{r['f1_score_D1']:.4f}", f"{r['f1_score_D2']:.4f}",
                    f"{r['roc_auc_D1']:.4f}", f"{r['roc_auc_D2']:.4f}",
                    f"{r['Gap']:+.4f}",
                ])
            col_widths = [0.23, 0.08, 0.08, 0.08, 0.08, 0.085, 0.085, 0.09]
            _draw_table(fig, headers, rows, y_top=0.85,
                         col_widths=col_widths,
                         numeric_cols=set(range(1, 8)))

            # Mini note
            note = ("A negative Gap means the model performs BETTER on D2 than D1. "
                    "Most models show a positive gap, indicating temporal degradation.")
            fig.text(0.06, 0.27, "NOTE",
                     fontsize=8, color=MUTED, fontweight='bold')
            fig.text(0.06, 0.25, note, fontsize=8.5, color=FG, wrap=True)

            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)

        # ══════════ Page 3: Continual Learning ══════════
        if not cl.empty:
            fig = _page_frame("Continual Learning", page_idx, total_pages)
            page_idx += 1

            headers = ["Model", "Before Acc", "After Acc",
                       "Before F1", "After F1", "Improvement %"]
            rows = []
            for _, r in cl.iterrows():
                rows.append([
                    str(r["model_type"])[:28],
                    f"{r['before_accuracy']:.4f}", f"{r['after_accuracy']:.4f}",
                    f"{r['before_f1']:.4f}", f"{r['after_f1']:.4f}",
                    f"{r['improvement_pct']:+.2f}%",
                ])
            col_widths = [0.28, 0.12, 0.12, 0.12, 0.12, 0.12]
            _draw_table(fig, headers, rows, y_top=0.85,
                         col_widths=col_widths,
                         numeric_cols=set(range(1, 6)))

            # Summary stats block
            fig.text(0.06, 0.60, "SUMMARY STATISTICS",
                     fontsize=9, color=PRIMARY, fontweight='bold')
            fig.add_artist(plt.Rectangle((0.06, 0.595), 0.05, 0.0018,
                                          color=PRIMARY))

            stats = [
                ("Best gain",    f"+{cl['improvement_pct'].max():.2f}%"),
                ("Average gain", f"+{cl['improvement_pct'].mean():.2f}%"),
                ("Models retrained", f"{len(cl)}"),
            ]
            sy = 0.555
            for lbl, val in stats:
                fig.text(0.08, sy, lbl, fontsize=9, color=MUTED)
                fig.text(0.38, sy, val, fontsize=11,
                         color=GOOD if val.startswith('+') else FG,
                         fontweight='bold')
                sy -= 0.032

            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)

        # ══════════ Page 4: Data Drift ══════════
        if not drift.empty:
            fig = _page_frame("Temporal Data Drift",
                               page_idx, total_pages)
            page_idx += 1

            fig.text(0.06, 0.875,
                     f"{int(drift['drift_detected'].sum())} of {len(drift)} "
                     f"features show significant drift (KS test, α = 0.05)",
                     fontsize=9.5, color=FG)

            top15 = drift.head(15)
            headers = ["Rank", "Feature", "KS statistic",
                       "p-value", "Drift?"]
            rows = []
            for i, (_, r) in enumerate(top15.iterrows(), 1):
                rows.append([
                    str(i),
                    str(r["feature"])[:38],
                    f"{r['ks_statistic']:.4f}",
                    f"{r['ks_pvalue']:.4f}",
                    "YES" if r["drift_detected"] else "no",
                ])
            col_widths = [0.06, 0.46, 0.14, 0.14, 0.08]
            _draw_table(fig, headers, rows, y_top=0.83,
                         col_widths=col_widths,
                         numeric_cols={2, 3})

            fig.text(0.06, 0.34, "INTERPRETATION",
                     fontsize=9, color=PRIMARY, fontweight='bold')
            fig.add_artist(plt.Rectangle((0.06, 0.335), 0.055, 0.0018,
                                          color=PRIMARY))
            interp = ("Features at the top of this list have the largest "
                      "distributional shift between pre-2015 and post-2015 "
                      "data. These are the primary drivers of temporal "
                      "performance degradation in the trained classifiers.")
            fig.text(0.06, 0.305, interp, fontsize=9, color=FG, wrap=True)

            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)

        # ══════════ Closing page ══════════
        fig = _page_frame("Conclusions", page_idx, total_pages)
        concl = [
            ("Temporal shift is real and measurable.",
             "The majority of clinical features exhibit statistically "
             "significant shifts between pre-2015 and post-2015 encounter "
             "records, confirming that static models will degrade over time."),
            ("Model robustness varies by family.",
             "Decision Trees (particularly at greater depths) show the "
             "smallest D1 → D2 degradation. SVM-RBF is most sensitive to "
             "the regularisation constant C. MLPs benefit from L2 + early "
             "stopping."),
            ("Continual learning is a viable mitigation.",
             "Fine-tuning the best D1 models on D2 data recovers a "
             "meaningful fraction of the lost performance, supporting "
             "periodic retraining as a deployment strategy."),
            ("Feature-level monitoring is essential.",
             "The drift table above pinpoints which features to watch in "
             "production. A drift dashboard over these features can "
             "trigger retraining before aggregate accuracy degrades."),
        ]
        y = 0.85
        for title, body in concl:
            fig.text(0.06, y, "▸", fontsize=12, color=PRIMARY,
                     fontweight='bold')
            fig.text(0.09, y, title, fontsize=10.5, color=FG,
                     fontweight='bold')
            y -= 0.025
            # Wrap body text manually
            words = body.split()
            line, line_n = "", 0
            for w in words:
                test = line + " " + w if line else w
                if len(test) > 95:
                    fig.text(0.09, y, line, fontsize=9, color=MUTED)
                    y -= 0.022
                    line = w
                    line_n += 1
                else:
                    line = test
            if line:
                fig.text(0.09, y, line, fontsize=9, color=MUTED)
                y -= 0.022
            y -= 0.025

        # Tagline
        fig.add_artist(plt.Rectangle((0.06, 0.08), 0.88, 0.001,
                                      color=PRIMARY, alpha=0.4))
        fig.text(0.5, 0.06, "End of report",
                 ha='center', fontsize=8, color=FAINT, style='italic')

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


def page_summary(d):
    st.markdown('<div class="sec-head">📝 Pipeline Summary</div>', unsafe_allow_html=True)
    metrics  = d.get("metrics",  pd.DataFrame())
    cl       = d.get("cl_results", pd.DataFrame())
    drift    = d.get("drift",    pd.DataFrame())

    # Outcomes hero block
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(79,139,255,0.06),rgba(244,114,182,0.04));
                border:1px solid rgba(79,139,255,0.15);border-radius:16px;padding:24px 28px;margin-bottom:20px">
      <div style="color:#E2E8F0;font-size:1.1em;font-weight:600;margin-bottom:16px">
        🏁 Pipeline Outcomes
      </div>""", unsafe_allow_html=True)

    if not metrics.empty:
        d1t = metrics[metrics["dataset"]=="D1_test"]; d2t = metrics[metrics["dataset"]=="D2_test"]
        best_d1 = d1t.loc[d1t["f1_score"].idxmax()]; best_d2 = d2t.loc[d2t["f1_score"].idxmax()]
        avg_gap = (d1t["accuracy"].values - d2t["accuracy"].values).mean()
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi("Best D1 F1",  f"{best_d1['f1_score']:.4f}", icon="🥇"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Best D2 F1",  f"{best_d2['f1_score']:.4f}", icon="🎯"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Avg Temporal Gap", f"{avg_gap:.4f}", delta=-avg_gap, good_direction="down", icon="📉"), unsafe_allow_html=True)

    if not cl.empty:
        c4,c5,c6 = st.columns(3)
        with c4: st.markdown(kpi("CL Best Gain",f"+{cl['improvement_pct'].max():.2f}%", icon="🚀"), unsafe_allow_html=True)
        with c5: st.markdown(kpi("CL Avg Gain", f"+{cl['improvement_pct'].mean():.2f}%", icon="📈"), unsafe_allow_html=True)
        with c6:
            if not drift.empty:
                n = drift["drift_detected"].sum()
                st.markdown(kpi("Features Drifting",f"{n}/{len(drift)}", icon="🌊"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not metrics.empty:
        st.markdown('<div class="section-hdr">Model Performance Recap</div>', unsafe_allow_html=True)
        d1m = metrics[metrics["dataset"]=="D1_test"][["model","model_type","accuracy","f1_score","roc_auc"]]
        d2m = metrics[metrics["dataset"]=="D2_test"][["model","accuracy","f1_score","roc_auc"]]
        recap = d1m.merge(d2m, on="model", suffixes=("_D1","_D2"))
        recap["Gap"] = (recap["accuracy_D1"]-recap["accuracy_D2"]).round(4)
        st.dataframe(recap.rename(columns={"model":"Model","model_type":"Type"})
                     .style.format({c:"{:.4f}" for c in recap.columns if c not in["model","model_type"]})
                     .background_gradient(subset=["accuracy_D1","accuracy_D2"], cmap="YlGn")
                     .background_gradient(subset=["Gap"], cmap="YlOrRd"),
                     use_container_width=True)

    if not cl.empty:
        st.markdown('<div class="section-hdr">Continual Learning Recap</div>', unsafe_allow_html=True)
        display = cl[["model_type","before_accuracy","after_accuracy","before_f1","after_f1","improvement_pct"]].copy()
        display.columns = ["Model","Before Acc","After Acc","Before F1","After F1","Improvement %"]
        st.dataframe(display.style
                     .format({c:"{:.4f}" for c in display.columns if c not in["Model","Improvement %"]}
                             | {"Improvement %":"{:+.2f}%"})
                     .background_gradient(subset=["After Acc"], cmap="YlGn"),
                     use_container_width=True)

    if not drift.empty:
        st.markdown('<div class="section-hdr">Data Drift Recap</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi("Drifted Features",f"{drift['drift_detected'].sum()}/{len(drift)}", icon="🌊"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Top KS Feature",drift.iloc[0]["feature"] if len(drift) else "N/A", icon="🔝"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Max KS Statistic",f"{drift['ks_statistic'].max():.4f}", icon="📊"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(drift[["feature","ks_statistic","ks_pvalue","drift_detected"]].head(15)
                     .style.format({"ks_statistic":"{:.4f}","ks_pvalue":"{:.4f}"}),
                     use_container_width=True)

    # ── PDF Report Generation ──────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-hdr">📄 Generate PDF Report</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight">
      Compile all the key metrics shown above into a multi-page PDF report with
      executive summary, model comparison tables, continual-learning recap,
      drift analysis, and conclusions. Share it, archive it, or submit it with
      your assignment.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        gen_clicked = st.button("🧾 Generate PDF Report",
                                 use_container_width=True,
                                 type="primary",
                                 key="gen_pdf_btn")
    with col_b:
        st.markdown(
            '<div style="color:#94A3B8;font-size:0.85em;padding-top:10px">'
            'Generated report uses the same results currently shown — '
            'takes ~2 seconds to build.</div>',
            unsafe_allow_html=True)

    if gen_clicked:
        try:
            with st.spinner("Compiling PDF report…"):
                pdf_bytes = _generate_pdf_report(d)
            st.session_state["pdf_report_bytes"] = pdf_bytes
            st.session_state["pdf_report_ts"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.markdown(
                '<div class="success-box">✅ Report generated — use the download button below.</div>',
                unsafe_allow_html=True)
        except Exception as e:
            st.markdown(
                f'<div class="warn-box"><b>Report generation failed:</b> {e}</div>',
                unsafe_allow_html=True)
            import traceback
            with st.expander("🔍 Traceback"):
                st.code(traceback.format_exc())

    if st.session_state.get("pdf_report_bytes"):
        ts = st.session_state.get("pdf_report_ts", "")
        st.download_button(
            label="⬇ Download ChronicML Report (PDF)",
            data=st.session_state["pdf_report_bytes"],
            file_name=f"ChronicML_Report_{ts}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pdf_dl_btn",
        )
# ═══════════════════════════════════════════════════════════════
# MAIN — Team27-STYLE HORIZONTAL TABS + INTERACTIVE SIDEBAR
# ═══════════════════════════════════════════════════════════════
def _render_landing():
    """Slimmed-down landing screen shown before the pipeline is run.
    Focus: hero + clear 3-step call-to-action + required files checklist.
    """
    # Build team sub-line to include in the hero
    team_sub = ""
    if TEAM_NUMBER:
        team_sub = f"Team {TEAM_NUMBER}"
        if TEAM_MEMBERS:
            team_sub += " · " + " · ".join(TEAM_MEMBERS)
        team_sub = (
            f'<div style="margin-top:14px;color:#4F8BFF;font-size:0.82em;'
            f'font-weight:600;letter-spacing:0.4px">'
            f'<span style="color:#F472B6">●</span> {team_sub}'
            f'</div>'
        )

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(79,139,255,0.08),rgba(244,114,182,0.05));
                border:1px solid rgba(79,139,255,0.18);border-radius:20px;
                padding:36px 40px;margin-bottom:22px;text-align:center">
      <div style="font-size:2.6em;margin-bottom:10px">⚕️</div>
      <div style="font-size:1.7em;font-weight:700;color:#F1F5F9;letter-spacing:-0.8px;margin-bottom:6px">
         ChronicML Condition Intelligence
      </div>
      <div style="color:#94A3B8;font-size:0.95em;max-width:600px;margin:0 auto;line-height:1.6">
        Upload your Synthea EHR CSV files and run the full 5-stage ML pipeline in one click.
        Results unlock in the navigation tabs once complete.
      </div>
      {team_sub}
    </div>
    """, unsafe_allow_html=True)

    # Get started stepper — numbered cards
    st.markdown("""
    <div class="glass" style="margin-bottom:18px;padding:20px 22px">
      <div style="color:#4F8BFF;font-size:0.95em;font-weight:600;margin-bottom:14px;
                  text-transform:uppercase;letter-spacing:1px">🚀 Get started in 3 steps</div>
      <div class="stepper">
        <div class="stepper-node">
          <div class="stepper-num">1</div>
          <div class="stepper-title">Upload</div>
          <div class="stepper-desc">Use the sidebar uploader to drop all Synthea CSV files at once.</div>
        </div>
        <div class="stepper-node">
          <div class="stepper-num">2</div>
          <div class="stepper-title">Verify</div>
          <div class="stepper-desc">Confirm all 7 required files show ✅ in the sidebar checklist.</div>
        </div>
        <div class="stepper-node">
          <div class="stepper-num">3</div>
          <div class="stepper-title">Run</div>
          <div class="stepper-desc">Click <b style="color:#E2E8F0">▶ Run ML Pipeline</b> — results in ~5–15 min.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Required files reference — single column, focused
    st.markdown("""
    <div class="glass" style="padding:20px 24px">
      <div style="color:#F472B6;font-weight:600;margin-bottom:12px;
                  text-transform:uppercase;letter-spacing:1px;font-size:0.9em">📋 Required CSV files</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;
                  color:#94A3B8;font-size:0.88em;line-height:1.9">
        <div><span style="color:#4F8BFF">▸</span> <code>patients.csv</code> — Demographics</div>
        <div><span style="color:#4F8BFF">▸</span> <code>encounters.csv</code> — Visit records</div>
        <div><span style="color:#4F8BFF">▸</span> <code>observations.csv</code> — Vitals &amp; labs</div>
        <div><span style="color:#4F8BFF">▸</span> <code>conditions.csv</code> — Diagnoses</div>
        <div><span style="color:#4F8BFF">▸</span> <code>medications.csv</code> — Prescriptions</div>
        <div><span style="color:#4F8BFF">▸</span> <code>allergies.csv</code> — Allergy records</div>
        <div><span style="color:#4F8BFF">▸</span> <code>procedures.csv</code> — Procedures</div>
      </div>
    </div>""", unsafe_allow_html=True)


def _render_sidebar():
    """Decluttered sidebar: upload + checklist + run button stay prominent;
    display options and pipeline config collapse into expanders.
    Returns (csv_dict, view_prefs, run_requested).
    """
    with st.sidebar:
        # ── Brand header ────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:18px 0 10px">
          <div style="font-size:2em;margin-bottom:4px">⚕️</div>
          <div style="font-size:1.15em;font-weight:700;
                      background:linear-gradient(90deg,#4F8BFF,#F472B6);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      color:#4F8BFF">
            ChronicML
          </div>
          <div style="font-size:0.68em;color:#64748B;margin-top:2px;
                      letter-spacing:1.2px;text-transform:uppercase">
            Condition Intelligence
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(79,139,255,0.12);margin:4px 0 14px">',
                    unsafe_allow_html=True)

        # ── Upload section ─────────────────────────────────────
        st.markdown('<div style="color:#E2E8F0;font-size:0.85em;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">'
                    '📁 Synthea CSV Files</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drop CSV files here", type=["csv"], accept_multiple_files=True,
            label_visibility="collapsed")

        csv_dict = {}
        if uploaded_files:
            for f in uploaded_files:
                stem = os.path.splitext(f.name)[0].lower()
                csv_dict[stem] = f

        # ── File detection feedback (vertical checklist) ────────
        if csv_dict:
            st.markdown('<div style="margin-top:10px">', unsafe_allow_html=True)
            # Required files — each on its own line with ✅ / ❌
            for name in sorted(REQUIRED_CSVS):
                if name in csv_dict:
                    st.markdown(
                        f'<div class="file-check ok"><span>{name}.csv</span><b>✓</b></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="file-check missing"><span>{name}.csv</span><b>✗</b></div>',
                        unsafe_allow_html=True)
            # Optional files — only show the ones that are present
            found_opt = [k for k in OPTIONAL_CSVS if k in csv_dict]
            for name in sorted(found_opt):
                st.markdown(
                    f'<div class="file-check opt"><span>{name}.csv</span><b>opt</b></div>',
                    unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:12px 0">',
                    unsafe_allow_html=True)

        # ── Run pipeline button ─────────────────────────────────
        all_required = bool(csv_dict)
        pipeline_done = st.session_state.get("pipeline_done", False)
        run_requested = False

        if all_required:
            btn_label = "🔄 Re-run Pipeline" if pipeline_done else "▶ Run ML Pipeline"
            run_requested = st.button(btn_label, use_container_width=True, type="primary")
        elif not uploaded_files:
            st.markdown('<div class="info-box" style="font-size:0.78em">'
                        '⬆ Upload CSV files to enable the pipeline.</div>',
                        unsafe_allow_html=True)
        else:
            missing = ", ".join(k for k in REQUIRED_CSVS if k not in csv_dict)
            st.markdown(f'<div class="warn-box" style="font-size:0.78em">'
                        f'Missing: {missing}</div>',
                        unsafe_allow_html=True)

        st.markdown('<hr style="border-color:rgba(255,255,255,0.05);margin:12px 0">',
                    unsafe_allow_html=True)

        # ── Display Options (collapsed into expander) ───────────
        with st.expander("⚙️ Display Options", expanded=False):
            top_k = st.slider("Top-K features shown", 5, 30, 20, 5,
                              help="Controls how many features appear in importance / drift plots.")
            show_insights = st.checkbox("Show insight annotations", value=True,
                                         help="Toggle the inline explanatory boxes on each page.")
            show_static_plots = st.checkbox("Show static matplotlib plots", value=True,
                                             help="Additional pre-rendered .png visuals alongside interactive charts.")
            cm_palette = st.selectbox("Confusion matrix palette",
                                       ["Blues", "Viridis", "Plasma", "Cividis"], 0)
            roc_line_width = st.slider("ROC / PR line width", 1, 4, 2, 1)

        # ── Pipeline Config (collapsed into expander) ───────────
        with st.expander("📊 Pipeline Config", expanded=False):
            st.markdown(f"""
            <div style="font-size:0.82em;color:#94A3B8;line-height:2">
              <span style="color:#4F8BFF">▸</span> Split date: <b style="color:#CBD5E1">{TEMPORAL_SPLIT_DATE}</b><br>
              <span style="color:#4F8BFF">▸</span> Test size: <b style="color:#CBD5E1">{TEST_SIZE:.0%}</b><br>
              <span style="color:#4F8BFF">▸</span> Models: <b style="color:#CBD5E1">DT×4, SVM×4, MLP×2</b><br>
              <span style="color:#4F8BFF">▸</span> Saved: <b style="color:#CBD5E1">.pkl (pickle)</b>
            </div>""", unsafe_allow_html=True)

        # ── Footer ──────────────────────────────────────────────
        st.markdown('<div style="flex-grow:1"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.68em;color:#334155;padding:14px 0 6px;text-align:center;
                    border-top:1px solid rgba(255,255,255,0.04);margin-top:18px">
          BITS F464 · Assignment 2<br>
          <span style="color:#1E3A5F">Synthea EHR · Pre/Post 2015</span>
        </div>""", unsafe_allow_html=True)

        view_prefs = {
            "top_k": top_k if all_required else 20,
            "show_insights": show_insights if all_required else True,
            "show_static_plots": show_static_plots if all_required else True,
            "cm_palette": cm_palette if all_required else "Blues",
            "roc_line_width": roc_line_width if all_required else 2,
        }
        return csv_dict, view_prefs, run_requested
# ═══════════════════════════════════════════════════════════════
# PROGRESS CHECKLIST during pipeline run
# ═══════════════════════════════════════════════════════════════
_PIPELINE_STAGES = [
    ("load",    "Data loading &amp; preprocessing"),
    ("train",   "Model training (DT, SVM, MLP)"),
    ("hptune",  "Hyperparameter tuning (GridSearchCV)"),
    ("eval",    "Cross-dataset evaluation"),
    ("cl",      "Continual learning"),
    ("adv",     "Advanced EDA &amp; drift analysis"),
]

_STAGE_KEYWORDS = {
    "load":   ["loading", "preprocess", "pivot", "feature", "scaling", "split"],
    "train":  ["training", "train", "fit", "decision tree", "svm", "mlp"],
    "hptune": ["tuning", "gridsearch", "cv f1", "hyperparameter"],
    "eval":   ["evaluat", "roc", "metric", "confusion", "report"],
    "cl":     ["continual", "fine-tun", "retrain", "weighted"],
    "adv":    ["drift", "anomaly", "cohort", "bias", "representation", "eda"],
}

def _classify_stage(msg: str) -> str:
    """Map a status message to one of the 5 pipeline stages (best-effort)."""
    lm = msg.lower()
    for stage, keywords in _STAGE_KEYWORDS.items():
        if any(kw in lm for kw in keywords):
            return stage
    return "load"  # default — first stage

def _render_checklist(placeholder, active_stage: str, done_stages: set,
                       current_msg: str = "", elapsed_s: float = 0):
    """Render the pipeline progress checklist into a placeholder."""
    rows = []
    for key, label in _PIPELINE_STAGES:
        if key in done_stages:
            rows.append(f'<div class="progress-step done">'
                        f'<span class="progress-icon">✓</span>'
                        f'<span>{label}</span></div>')
        elif key == active_stage:
            rows.append(f'<div class="progress-step active">'
                        f'<span class="progress-icon">●</span>'
                        f'<span>{label}</span></div>')
        else:
            rows.append(f'<div class="progress-step">'
                        f'<span class="progress-icon">○</span>'
                        f'<span>{label}</span></div>')
    mm = int(elapsed_s // 60); ss = int(elapsed_s % 60)
    elapsed_str = f"{mm:02d}:{ss:02d}"

    status_line = (f'<div style="color:#94A3B8;font-size:0.82em;'
                   f'margin:4px 0 6px;font-family:JetBrains Mono,monospace">'
                   f'⏱ {elapsed_str}  ·  {current_msg[:100]}</div>') if current_msg else ""

    placeholder.markdown(
        f'<div class="glass-accent" style="padding:16px 20px">'
        f'<div style="color:#4F8BFF;font-size:0.82em;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px">'
        f'⚡ Pipeline in progress</div>'
        f'{status_line}'
        f'<div class="progress-list">{"".join(rows)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _run_pipeline_with_status(csv_dict):
    """Execute run_full_pipeline with an inline progress checklist.
    Returns True on success."""
    import time
    st.session_state["pipeline_done"] = False
    st.session_state["pipeline_data"] = None
    placeholder = st.empty()

    t0 = time.time()
    state = {"active": "load", "done": set(), "msg": "Initialising…"}

    def _status(msg: str):
        new_stage = _classify_stage(msg)
        if new_stage != state["active"]:
            # When we advance to a new stage, mark all earlier stages as done
            stage_order = [k for k, _ in _PIPELINE_STAGES]
            try:
                new_idx = stage_order.index(new_stage)
                for prev in stage_order[:new_idx]:
                    state["done"].add(prev)
            except ValueError:
                pass
            state["active"] = new_stage
        state["msg"] = msg
        _render_checklist(placeholder, state["active"], state["done"],
                           current_msg=msg, elapsed_s=time.time() - t0)

    # Initial render
    _render_checklist(placeholder, "load", set(),
                       current_msg="Starting pipeline…", elapsed_s=0)

    try:
        with st.spinner("Running pipeline (this may take 5–15 minutes)…"):
            data = run_full_pipeline(csv_dict, status_fn=_status)
        # Mark everything done on success
        st.session_state["pipeline_data"] = data
        st.session_state["pipeline_done"] = True
        total_s = time.time() - t0
        mm, ss = int(total_s // 60), int(total_s % 60)
        placeholder.markdown(
            f'<div class="success-box" style="font-size:0.92em;padding:14px 18px">'
            f'✅ <b>Pipeline complete!</b> '
            f'<span style="color:#94A3B8">Finished in {mm:02d}:{ss:02d}</span>'
            f'</div>',
            unsafe_allow_html=True)
        return True
    except Exception as e:
        placeholder.empty()
        st.markdown(f'<div class="warn-box"><b>Pipeline failed:</b> {e}</div>',
                    unsafe_allow_html=True)
        import traceback
        with st.expander("🔍 Full traceback"):
            st.code(traceback.format_exc())
        return False
# ═══════════════════════════════════════════════════════════════
# TAB STRUCTURE — 11 sections preserved
# ═══════════════════════════════════════════════════════════════
_PAGE_FUNCS = [
    ("🏠 Home",                    page_home),
    ("📊 Data Overview",           page_data_overview),
    ("🔬 EDA",                     page_eda),
    ("🤖 Model Performance",       page_model_performance),
    ("⚙️ Hyperparameter Tuning",   page_hyperparameter_tuning),
    ("📐 Bias-Variance",           page_bias_variance),
    ("🔍 Feature Representation",  page_feature_representation),
    ("📉 Temporal Shift",          page_temporal_shift),
    ("🔴 Anomaly Detection",       page_anomaly),
    ("🌟 Feature Importance",      page_feature_importance),
    ("🔄 Continual Learning",      page_continual_learning),
    ("📝 Summary",                 page_summary),
]
# Grouped structure — 4 super-tabs, nested sub-tabs inside each.
_PAGE_GROUPS = [
    ("🗂️ Data",
        [("🏠 Home",                   page_home),
         ("📊 Data Overview",          page_data_overview),
         ("🔬 EDA",                    page_eda)]),
    ("🧠 Models",
        [("🤖 Performance",            page_model_performance),
         ("⚙️ Hyperparameter Tuning",  page_hyperparameter_tuning),
         ("📐 Bias-Variance",          page_bias_variance),
         ("🔍 Feature Representation", page_feature_representation),
         ("🌟 Feature Importance",     page_feature_importance)]),
    ("📉 Drift & Adaptation",
        [("📉 Temporal Shift",         page_temporal_shift),
         ("🔴 Anomaly Detection",      page_anomaly),
         ("🔄 Continual Learning",     page_continual_learning)]),
    ("📝 Summary",
        [("📝 Summary",                page_summary)]),
]

def _render_status_chip(pipeline_done: bool, running: bool = False) -> str:
    """Build the small status pill shown in the sticky top banner."""
    if running:
        return ('<span class="status-chip chip-run">'
                '<span class="dot" style="background:#4F8BFF"></span>'
                'RUNNING</span>')
    if pipeline_done:
        return ('<span class="status-chip chip-done">'
                '<span class="dot" style="background:#10B981"></span>'
                'COMPLETE</span>')
    return ('<span class="status-chip chip-idle">'
            '<span class="dot" style="background:#F59E0B"></span>'
            'AWAITING UPLOAD</span>')

def main():
    # ── Top sticky banner with status chip + team details ──────
    pipeline_done = st.session_state.get("pipeline_done", False)
    status_html = _render_status_chip(pipeline_done)

    # Build the team line — required by the assignment submission format.
    team_label = f"Team {TEAM_NUMBER}" if TEAM_NUMBER else ""
    if TEAM_MEMBERS:
        members_str = " · ".join(TEAM_MEMBERS)
        team_html = (
            f'<div style="color:#4F8BFF;font-size:0.78em;margin-top:6px;'
            f'font-weight:600;letter-spacing:0.4px">'
            f'  <span style="color:#F472B6">●</span> {team_label}'
            f'  <span style="color:#64748B;font-weight:400;margin-left:10px">'
            f'{members_str}</span>'
            f'</div>'
        )
    elif team_label:
        team_html = (
            f'<div style="color:#4F8BFF;font-size:0.78em;margin-top:6px;'
            f'font-weight:600;letter-spacing:0.4px">'
            f'<span style="color:#F472B6">●</span> {team_label}'
            f'</div>'
        )
    else:
        team_html = ""

    st.markdown(
        f'<div class="top-banner">'
        f'  <div class="top-banner-row">'
        f'    <div>'
        f'      <div style="font-size:1.35em;font-weight:700;color:#F1F5F9;'
        f'letter-spacing:-0.5px;line-height:1.2">'
        f'        ⚕️ ChronicML Condition Intelligence Pipeline'
        f'      </div>'
        f'      <div style="color:#64748B;font-size:0.82em;margin-top:2px">'
        f'        BITS F464 · Assignment 2 · Temporal Shift Detection &amp; Continual Model Adaptation'
        f'      </div>'
        f'      {team_html}'
        f'    </div>'
        f'    <div>{status_html}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True)

    # Sidebar returns config + whether a run was requested this cycle
    # (main-area fallback uploader removed — sidebar uploader is the only entry point)
    csv_dict, view_prefs, run_requested = _render_sidebar()

    # Store view preferences in session_state so page functions could
    # optionally read them (non-breaking — pages that ignore this work fine).
    st.session_state["view_prefs"] = view_prefs

    # Execute pipeline if button was pressed
    if run_requested:
        if _run_pipeline_with_status(csv_dict):
            st.rerun()

    pipeline_done = st.session_state.get("pipeline_done", False)

    # Before pipeline: show landing page
    if not pipeline_done:
        _render_landing()
        return

    data = st.session_state.get("pipeline_data", {})
    if not data:
        st.markdown('<div class="warn-box">Pipeline data missing. Please re-run the pipeline.</div>',
                    unsafe_allow_html=True)
        return

    # ── Grouped two-level tab structure ────────────────────────
    # 11 page functions preserved and invoked — organized as 4 super-tabs.
    # Summary lives in its own top-level tab (group of 1), so for any
    # single-page group we skip the redundant inner tab layer and render
    # the page directly inside the super-tab.
    group_labels = [g[0] for g in _PAGE_GROUPS]
    super_tabs = st.tabs(group_labels)

    for super_tab, (_, sub_pages) in zip(super_tabs, _PAGE_GROUPS):
        with super_tab:
            if len(sub_pages) == 1:
                # Single-page group → render the page flat, no inner tabs
                _, page_fn = sub_pages[0]
                try:
                    page_fn(data)
                except Exception as e:
                    st.markdown(
                        f'<div class="warn-box"><b>Error rendering this section:</b> {e}</div>',
                        unsafe_allow_html=True)
                    import traceback
                    with st.expander("🔍 Traceback"):
                        st.code(traceback.format_exc())
            else:
                # Multi-page group → nested sub-tabs
                sub_labels = [lbl for lbl, _ in sub_pages]
                sub_tabs = st.tabs(sub_labels)
                for sub_tab, (_, page_fn) in zip(sub_tabs, sub_pages):
                    with sub_tab:
                        try:
                            page_fn(data)
                        except Exception as e:
                            st.markdown(
                                f'<div class="warn-box"><b>Error rendering this section:</b> {e}</div>',
                                unsafe_allow_html=True)
                            import traceback
                            with st.expander("🔍 Traceback"):
                                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()