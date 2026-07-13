"""AiVora — multi-user Streamlit dashboard (redesigned).

Runs alongside the single-user ``streamlit_app.py``. This one:

* Requires login (bcrypt password auth).
* Per-user encrypted broker credentials.
* Per-user portfolio + trades in ``data/db/webapp.sqlite``.
* Dark-theme, card-based UI with Inter font.

Launch:

    python -m streamlit run app/multi_user_app.py
"""

from __future__ import annotations

import base64
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402
from aivora.webapp import admin as admin_mod  # noqa: E402
from aivora.webapp import brokers as broker_mod  # noqa: E402
from aivora.webapp import db as db_mod  # noqa: E402
from aivora.webapp import migration as mig_mod  # noqa: E402
from aivora.webapp import portfolios as pf_mod  # noqa: E402
from aivora.webapp import scheduler_manager as sm_mod  # noqa: E402
from aivora.webapp import sessions as sess_mod  # noqa: E402
from aivora.webapp import users as user_mod  # noqa: E402
from aivora.webapp.auth_server import kite_login_url  # noqa: E402

log = get_logger("app.multi_user")


# =============================================================
#  Brand assets — favicon + inline logo data URI
# =============================================================
_ASSETS_DIR = _ROOT / "app" / "assets"
_FAVICON_PATH = _ASSETS_DIR / "favicon" / "favicon-96x96.png"
_LOGO_PATH = _ASSETS_DIR / "favicon" / "web-app-manifest-192x192.png"


def _load_favicon():
    """Return a PIL Image for st.set_page_config(page_icon=…)."""
    try:
        return Image.open(_FAVICON_PATH)
    except Exception:  # pragma: no cover — first-run/dev fallback
        return "📈"


def _logo_data_uri() -> str:
    """Base64 data URI for the AiVora mark, used inline in navbar HTML.

    Cached at module import — a 7 KB PNG becomes a 10 KB inline string, cheap.
    """
    try:
        data = _LOGO_PATH.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
    except Exception:
        return ""


_LOGO_URI = _logo_data_uri()


st.set_page_config(
    page_title="AiVora",
    layout="wide",
    page_icon=_load_favicon(),
    initial_sidebar_state="expanded",
)


# =============================================================
#  Theme — dark, Inter font, tabular-nums for numbers
# =============================================================
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0B1220;
    --sidebar-bg: #111827;
    --card-bg: #172033;
    --card-bg-hover: #1B2540;
    --border: #27344D;
    --border-strong: #334155;
    --primary: #3B82F6;
    --primary-dim: rgba(59, 130, 246, 0.15);
    --success: #16C784;
    --success-dim: rgba(22, 199, 132, 0.15);
    --danger: #EF4444;
    --danger-dim: rgba(239, 68, 68, 0.15);
    --warning: #F59E0B;
    --warning-dim: rgba(245, 158, 11, 0.15);
    --text: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --radius: 14px;
    --radius-sm: 10px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
}

/* Global reset */
html, body, [data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { color: var(--text) !important; }

/* Tabular numbers everywhere they matter */
.av-num, [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
.av-hero-value, .av-metric-value, .av-trade-num {
    font-variant-numeric: tabular-nums;
    font-feature-settings: "tnum";
}

/* Main container padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: var(--text-secondary) !important; }
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 0.75rem 0 !important;
}

/* Text colors */
h1, h2, h3, h4, h5, h6 { color: var(--text) !important; letter-spacing: -0.02em; }
h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; }
p, .stMarkdown, span, label { color: var(--text) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }

/* Buttons — match by data-testid so buttons with help= tooltips
   (wrapped in stTooltipHoverTarget spans) still get the dark theme. */
button[data-testid^="stBaseButton"] {
    min-height: 42px !important;
    padding: 0.55rem 1rem !important;
    font-size: 0.94rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.15s ease !important;
}
/* Exceptions: header buttons (Deploy, hamburger) keep their own look */
button[data-testid="stBaseButton-header"],
button[data-testid="stBaseButton-headerNoPadding"] {
    background: transparent !important;
    border: none !important;
    min-height: auto !important;
}
button[data-testid^="stBaseButton"]:hover:not([data-testid="stBaseButton-header"]):not([data-testid="stBaseButton-headerNoPadding"]) {
    background: var(--card-bg-hover) !important;
    border-color: var(--primary) !important;
    transform: translateY(-1px);
}
/* Primary buttons — red gradient */
button[data-testid^="stBaseButton"][kind="primary"],
button[data-testid^="stBaseButton"][kind*="primaryFormSubmit"] {
    background: linear-gradient(135deg, #EF4444, #DC2626) !important;
    border-color: transparent !important;
    color: white !important;
    font-weight: 600 !important;
}
button[data-testid^="stBaseButton"][kind="primary"]:hover,
button[data-testid^="stBaseButton"][kind*="primaryFormSubmit"]:hover {
    background: linear-gradient(135deg, #DC2626, #B91C1C) !important;
}
/* Inner text (some Streamlit variants wrap in <p>) inherits color */
button[data-testid^="stBaseButton"] p,
button[data-testid^="stBaseButton"] div,
button[data-testid^="stBaseButton"] span { color: inherit !important; background: transparent !important; }

/* Text inputs */
.stTextInput input, .stNumberInput input, .stPasswordInput input,
.stTextArea textarea {
    background: var(--card-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    min-height: 42px !important;
    font-size: 0.94rem !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stPasswordInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-dim) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Radio (mode switcher) */
[role="radiogroup"] { gap: 0.5rem !important; }
[role="radiogroup"] > label {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s ease !important;
}
[role="radiogroup"] > label:has(input:checked) {
    background: var(--primary-dim) !important;
    border-color: var(--primary) !important;
}

/* Metric widget → styled card */
[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.15rem;
    transition: all 0.15s ease;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-strong);
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-weight: 500 !important; }

/* Dividers */
hr, [data-testid="stDivider"] { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* Expander — collapsed AND open state, header + body all dark */
[data-testid="stExpander"] {
    background: var(--card-bg);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] details[open] { background: transparent !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details[open] summary {
    color: var(--text) !important;
    background: transparent !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 1px solid transparent !important;
}
[data-testid="stExpander"] details[open] summary {
    border-bottom: 1px solid var(--border) !important;
    background: var(--sidebar-bg) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--card-bg-hover) !important; }
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div { color: var(--text) !important; }
/* Expander body */
[data-testid="stExpander"] [data-testid="stExpanderDetails"],
[data-testid="stExpander"] > details > div { background: var(--card-bg) !important; padding: 0.5rem 0.75rem !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] * { color: var(--text) !important; }

/* Tooltips (from st.button/st.text_input `help=` param) */
[data-baseweb="tooltip"], [role="tooltip"] {
    background: #0F1A2E !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.55rem 0.75rem !important;
    font-size: 0.82rem !important;
    box-shadow: var(--shadow) !important;
    max-width: 320px !important;
}
[data-baseweb="tooltip"] *, [role="tooltip"] * { color: var(--text) !important; background: transparent !important; }
[data-baseweb="tooltip"] div { color: var(--text) !important; }

/* Tooltip arrow */
[data-baseweb="tooltip"] > div::before,
[data-baseweb="tooltip"] > div::after { border-color: var(--border) transparent !important; }

/* Help icon (?) next to labels */
[data-testid="stTooltipIcon"] svg { color: var(--text-secondary) !important; opacity: 0.7; }
[data-testid="stTooltipIcon"]:hover svg { opacity: 1; color: var(--primary) !important; }

/* Selectbox / dropdown popover */
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow) !important;
}
[data-baseweb="popover"] *, [data-baseweb="menu"] * { color: var(--text) !important; }
[data-baseweb="menu"] li:hover, [role="option"]:hover {
    background: var(--card-bg-hover) !important;
}
[role="listbox"] { background: var(--card-bg) !important; }
[role="option"] { color: var(--text) !important; background: transparent !important; }
[role="option"][aria-selected="true"] { background: var(--primary-dim) !important; color: var(--primary) !important; }

/* Sidebar-specific: expanders inside the sidebar need the same fix */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border-color: var(--border) !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary {
    background: rgba(255,255,255,0.04) !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--card-bg) !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] thead th {
    background: var(--sidebar-bg) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border) !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 2 !important;
}
[data-testid="stDataFrame"] td { color: var(--text) !important; font-size: 0.9rem !important; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    padding: 0.6rem 1rem !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    border: none !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text) !important; background: var(--card-bg) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
    background: transparent !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: var(--card-bg) !important;
    border-radius: var(--radius) !important;
    border-left: 3px solid var(--primary) !important;
    color: var(--text) !important;
}
[data-testid="stAlert"][data-baseweb="notification"] * { color: var(--text) !important; }

/* ========== Custom AiVora components ========== */

/* Top navigation */
.av-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.25rem;
    gap: 1rem;
}
.av-navbar-brand { display: flex; align-items: center; gap: 0.6rem; }
.av-navbar-logo {
    width: 34px; height: 34px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
    background: transparent;
}
.av-navbar-logo img {
    width: 100%; height: 100%;
    object-fit: contain;
    display: block;
}
.av-navbar-title { font-weight: 700; font-size: 1.15rem; letter-spacing: -0.02em; }
.av-navbar-status { display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
.av-status-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255, 255, 255, 0.03);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem; color: var(--text-secondary);
    border: 1px solid var(--border);
}
.av-status-chip strong { color: var(--text); font-weight: 600; }
.av-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.av-dot-green { background: var(--success); box-shadow: 0 0 8px var(--success); }
.av-dot-red   { background: var(--danger); }
.av-dot-blue  { background: var(--primary); box-shadow: 0 0 8px var(--primary); }
.av-dot-amber { background: var(--warning); }

.av-navbar-user {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.25rem 0.6rem 0.25rem 0.25rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 999px;
    border: 1px solid var(--border);
}
.av-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 0.85rem;
}
.av-navbar-user span { color: var(--text); font-size: 0.88rem; font-weight: 500; }

/* Hero card */
.av-hero {
    background: linear-gradient(135deg, var(--card-bg) 0%, #1B2949 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.av-hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.12), transparent 70%);
    pointer-events: none;
}
.av-hero-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 1.5rem;
    position: relative;
    z-index: 1;
}
.av-hero-label {
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.av-hero-value {
    color: var(--text);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.av-hero-value.big { font-size: 2.5rem; }
.av-hero-value.pos { color: var(--success); }
.av-hero-value.neg { color: var(--danger); }
.av-hero-sub {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* Section headings */
.av-h2 {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin: 1.25rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.av-h2-icon {
    width: 22px; height: 22px;
    border-radius: 6px;
    background: var(--primary-dim);
    color: var(--primary);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
}

/* AI Confidence card */
.av-ai-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    position: relative;
}
.av-ai-badge {
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.av-ai-badge.bull { background: var(--success-dim); color: var(--success); }
.av-ai-badge.bear { background: var(--danger-dim); color: var(--danger); }
.av-ai-badge.neutral { background: var(--warning-dim); color: var(--warning); }
.av-ai-badge.waiting { background: rgba(148,163,184,0.15); color: var(--text-secondary); }
.av-confidence-bar {
    height: 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.av-confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--success));
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* Activity timeline */
.av-timeline { display: flex; flex-direction: column; gap: 0.5rem; }
.av-timeline-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.6rem 0.9rem;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    align-items: flex-start;
    transition: all 0.15s ease;
}
.av-timeline-item:hover {
    background: var(--card-bg-hover);
    transform: translateX(2px);
}
.av-timeline-icon {
    width: 28px; height: 28px;
    flex-shrink: 0;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
    background: rgba(255,255,255,0.03);
}
.av-timeline-icon.success { background: var(--success-dim); }
.av-timeline-icon.danger { background: var(--danger-dim); }
.av-timeline-icon.warning { background: var(--warning-dim); }
.av-timeline-icon.info { background: var(--primary-dim); }
.av-timeline-body { flex: 1; min-width: 0; }
.av-timeline-msg {
    color: var(--text);
    font-size: 0.88rem;
    line-height: 1.4;
    word-break: break-word;
}
.av-timeline-time {
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 0.15rem;
}

/* Auth pages — signalled by the .av-auth-mode marker.
   The block-container itself becomes the auth card; no columns needed. */
.stAppViewContainer:has(.av-auth-mode) .block-container {
    max-width: 600px !important;
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    margin-top: 3rem !important;
    margin-bottom: 3rem !important;
}
@media (max-width: 768px) {
    .stAppViewContainer:has(.av-auth-mode) .block-container {
        max-width: 100% !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
}
.av-auth-mode { display: none; }
.av-auth-brand {
    display: flex; align-items: center; justify-content: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}
.av-auth-brand .av-navbar-logo { width: 40px; height: 40px; font-size: 1.3rem; }
.av-auth-brand-name {
    font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
}

/* Banner */
.aivora-banner {
    padding: 0.7rem 1rem;
    border-radius: var(--radius-sm);
    margin-bottom: 1rem;
    background: var(--warning-dim);
    color: var(--warning);
    border-left: 3px solid var(--warning);
}
.aivora-banner.error { background: var(--danger-dim); color: var(--danger); border-color: var(--danger); }
.aivora-banner.ok    { background: var(--success-dim); color: var(--success); border-color: var(--success); }

/* Empty state */
.av-empty {
    text-align: center;
    padding: 2rem 1rem;
    color: var(--text-secondary);
    background: var(--card-bg);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
}
.av-empty-icon { font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.6; }
.av-empty-msg { font-size: 0.94rem; font-weight: 500; margin-bottom: 0.25rem; color: var(--text); }
.av-empty-hint { font-size: 0.82rem; color: var(--text-muted); }

/* Scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-strong); }

/* Responsive */
@media (max-width: 1024px) {
    .av-hero-grid { grid-template-columns: 1fr 1fr; }
    .av-hero-value.big { font-size: 2rem; }
}
@media (max-width: 768px) {
    .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    .av-hero-grid { grid-template-columns: 1fr; gap: 1rem; }
    .av-hero-value.big { font-size: 1.75rem; }
    .av-hero-value { font-size: 1.4rem; }
    .av-navbar { flex-wrap: wrap; padding: 0.6rem 0.85rem; }
    .av-navbar-status { gap: 0.5rem; }
    .av-status-chip { font-size: 0.75rem; padding: 0.3rem 0.55rem; }
    [data-testid="stDataFrame"] table { min-width: 720px !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        min-width: 100% !important;
        flex: 0 0 100% !important;
    }
}
</style>
"""


def inject_css() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# =============================================================
#  Session state helpers
# =============================================================
def _sess_user() -> user_mod.User | None:
    """Return the logged-in user, restoring from the signed cookie if needed."""
    uid = sess_mod.install_from_cookie()
    if uid is None:
        return None
    try:
        u = user_mod.get_by_id(int(uid))
    except Exception:
        sess_mod.revoke()
        return None
    if not admin_mod.is_active(u.id):
        sess_mod.revoke()
        st.session_state["_flash"] = ("error",
            "Your account has been deactivated by an administrator.")
        return None
    return u


def _login(user: user_mod.User) -> None:
    sess_mod.issue(user.id)
    st.session_state.setdefault("mode", "paper")
    st.session_state["_flash"] = ("ok", f"Welcome back, {user.display_name or user.email}.")


def _logout() -> None:
    sess_mod.revoke()


def _flash() -> None:
    msg = st.session_state.pop("_flash", None)
    if not msg:
        return
    cls, text = msg
    st.markdown(f'<div class="aivora-banner {cls}">{text}</div>', unsafe_allow_html=True)


# =============================================================
#  Time / market helpers
# =============================================================
_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(_IST).replace(tzinfo=None)


def _market_state(now: datetime | None = None) -> tuple[bool, str]:
    """Return (is_open, label). NSE hours: Mon-Fri 09:15-15:30 IST."""
    now = now or _now_ist()
    if now.weekday() >= 5:
        return False, "NSE CLOSED"
    t = now.time()
    open_t = datetime.strptime("09:15", "%H:%M").time()
    close_t = datetime.strptime("15:30", "%H:%M").time()
    if open_t <= t <= close_t:
        return True, "NSE OPEN"
    return False, "NSE CLOSED"


def _seconds_to_next_tick(now: datetime | None = None) -> int:
    """Seconds until the next 5-minute :00 boundary + 20s (scheduler offset)."""
    now = now or _now_ist()
    minute = (now.minute // 5 + 1) * 5
    if minute >= 60:
        target = now.replace(minute=0, second=20, microsecond=0) + timedelta(hours=1)
    else:
        target = now.replace(minute=minute, second=20, microsecond=0)
    return max(0, int((target - now).total_seconds()))


def _fmt_countdown(secs: int) -> str:
    m, s = divmod(secs, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"[\s@._-]+", name) if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


# =============================================================
#  AI outlook — parse from most recent event log entries
# =============================================================
def _parse_ai_outlook(events: list[dict]) -> dict:
    """Look through the newest events for probability / gate messages
    and produce a simple {outlook, confidence, reason} snapshot.

    Falls back to WAITING when we can't find anything actionable.
    """
    up_re = re.compile(r"p_up=([\d.]+)")
    dn_re = re.compile(r"p_down=([\d.]+)")
    vr_re = re.compile(r"vr=([\d.]+)")
    conv_re = re.compile(r"(UP|DOWN)\s+conviction\s+([\d.]+)", re.IGNORECASE)

    for e in events[:80]:
        msg = e.get("msg") or ""
        if "opened" in msg.lower() and ("CE" in msg or "PE" in msg):
            side_m = re.search(r"\b(CE|PE)\b", msg)
            if side_m:
                side = side_m.group(1)
                return {
                    "outlook": "BULLISH" if side == "CE" else "BEARISH",
                    "confidence": None,
                    "reason": f"Just entered {side} trade",
                    "kind": "bull" if side == "CE" else "bear",
                }
        m_up = up_re.search(msg)
        m_dn = dn_re.search(msg)
        if m_up and m_dn:
            pu = float(m_up.group(1))
            pd_ = float(m_dn.group(1))
            vr = vr_re.search(msg)
            vr_val = float(vr.group(1)) if vr else None
            if pu > pd_ and pu >= 0.55:
                return {
                    "outlook": "BULLISH", "confidence": pu,
                    "reason": f"UP signal {pu:.0%}"
                              + (f" · vol {vr_val:.0%}" if vr_val is not None else ""),
                    "kind": "bull",
                }
            if pd_ > pu and pd_ >= 0.60:
                return {
                    "outlook": "BEARISH", "confidence": pd_,
                    "reason": f"DOWN signal {pd_:.0%}"
                              + (f" · vol {vr_val:.0%}" if vr_val is not None else ""),
                    "kind": "bear",
                }
            return {
                "outlook": "NEUTRAL",
                "confidence": max(pu, pd_),
                "reason": f"Gates unmet · UP {pu:.0%} / DOWN {pd_:.0%}",
                "kind": "neutral",
            }
        m_conv = conv_re.search(msg)
        if m_conv:
            direction = m_conv.group(1).upper()
            prob = float(m_conv.group(2))
            return {
                "outlook": "BULLISH" if direction == "UP" else "BEARISH",
                "confidence": prob,
                "reason": f"{direction} conviction {prob:.0%} — below threshold",
                "kind": "bull" if direction == "UP" else "bear",
            }

    return {
        "outlook": "AWAITING",
        "confidence": None,
        "reason": "Next inference in <countdown>",
        "kind": "waiting",
    }


# =============================================================
#  Event log → activity timeline
# =============================================================
def _classify_event(msg: str) -> tuple[str, str]:
    """Return (icon, css_class) for a log message."""
    m = msg.lower()
    if "opened" in m and ("ce" in m or "pe" in m):
        return "📈", "info"
    if "closed" in m:
        if re.search(r"[+]", msg):
            return "💰", "success"
        return "🔒", "danger"
    if "trailing" in m or "sl locked" in m:
        return "⏱️", "warning"
    if "cooldown" in m:
        return "⏳", "warning"
    if "tick" in m:
        return "🔄", "info"
    if "error" in m or "fatal" in m or "failed" in m:
        return "❌", "danger"
    if "warn" in m:
        return "⚠️", "warning"
    if "connected" in m or "authenticated" in m:
        return "🔗", "success"
    if "disconnected" in m or "missing" in m:
        return "🔌", "warning"
    return "•", "info"


def _fmt_event_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts[-8:] if ts else ""


# =============================================================
#  Auth pages
# =============================================================
def _auth_brand() -> None:
    st.markdown(
        '<div class="av-auth-brand">'
        f'<div class="av-navbar-logo"><img src="{_LOGO_URI}" alt="AiVora"></div>'
        '<div class="av-auth-brand-name">AiVora</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def login_page() -> None:
    # The .av-auth-mode marker triggers the block-container-as-card CSS above.
    st.markdown('<div class="av-auth-mode"></div>', unsafe_allow_html=True)
    _auth_brand()
    st.markdown("<h2 style='text-align:center;margin-bottom:1.25rem'>Sign in</h2>",
                unsafe_allow_html=True)
    with st.form("login-form", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", width="stretch")
    if submitted:
        try:
            u = user_mod.authenticate(email.strip(), password)
        except Exception as exc:
            st.error(str(exc))
            return
        if u is None:
            st.error("Invalid email or password.")
            return
        _login(u)
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("New here? Create an account", width="stretch"):
        st.session_state["_page"] = "register"
        st.rerun()


def register_page() -> None:
    st.markdown('<div class="av-auth-mode"></div>', unsafe_allow_html=True)
    _auth_brand()
    st.markdown("<h2 style='text-align:center;margin-bottom:0.5rem'>Create account</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:var(--text-secondary);font-size:0.85rem;margin-bottom:1.25rem'>"
                "Password must be at least 8 characters.</p>", unsafe_allow_html=True)
    with st.form("register-form", clear_on_submit=False):
        email = st.text_input("Email")
        display = st.text_input("Display name (optional)")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submit = st.form_submit_button("Create account", width="stretch")
    if submit:
        if password != confirm:
            st.error("Passwords don't match.")
            return
        try:
            u = user_mod.register(email.strip(), password, display_name=display or None)
        except ValueError as exc:
            st.error(str(exc))
            return
        _login(u)
        st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Have an account? Sign in", width="stretch"):
        st.session_state["_page"] = "login"
        st.rerun()


# =============================================================
#  Profile page (broker credentials) — same functionality, dark styling
# =============================================================
def _stored_badge(present: bool) -> str:
    return "✅ Stored" if present else "⚪ Not set"


def _fmt_ist(ts: str | None) -> str:
    if not ts:
        return "never"
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(_IST)
        return ist.strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return str(ts)


def _exchange_and_store_kite_token(user_id: int, request_token: str) -> str:
    creds = broker_mod.get(user_id, "ZERODHA")
    if not creds or not creds.api_key or not creds.api_secret:
        raise RuntimeError("Save API key + secret first.")
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=creds.api_key)
    data = kite.generate_session(request_token.strip(), api_secret=creds.api_secret)
    broker_mod.upsert(
        user_id, "ZERODHA",
        client_id=data.get("user_id") or creds.client_id,
        access_token=data["access_token"],
    )
    return data["access_token"]


def profile_page(user: user_mod.User) -> None:
    st.title("Profile")
    st.caption(f"Signed in as **{user.email}**  ·  user id {user.id}")

    st.info(
        "Secret fields always render blank on purpose — your saved values "
        "stay in the encrypted store; typing nothing keeps them.  Type "
        "something to overwrite; use the Disconnect button to clear."
    )

    st.subheader("Zerodha (Kite Connect)")
    z = broker_mod.get(user.id, "ZERODHA")
    z_has = {
        "api_key":      bool(z and z.api_key),
        "api_secret":   bool(z and z.api_secret),
        "access_token": bool(z and z.access_token),
        "password":     bool(z and z.password),
        "totp_secret":  bool(z and z.totp_secret),
    }
    with st.form("zerodha-form", clear_on_submit=False):
        client_id = st.text_input(
            "Client ID (your Zerodha user id, e.g. BF1234)",
            value=(z.client_id if z else "") or "",
        )
        st.caption(f"API Key: {_stored_badge(z_has['api_key'])}")
        api_key = st.text_input(
            "API Key",
            value="", type="password",
            help="From developers.kite.trade → your app. Leave blank to keep existing.",
        )
        st.caption(f"API Secret: {_stored_badge(z_has['api_secret'])}")
        api_secret = st.text_input(
            "API Secret",
            value="", type="password",
            help="Kept encrypted server-side.  Leave blank to keep existing.",
        )
        st.markdown("**Optional — for one-click TOTP auto-login:**")
        st.caption(f"Zerodha password: {_stored_badge(z_has['password'])}")
        password = st.text_input(
            "Zerodha password", value="", type="password",
            help="Only needed for the TOTP flow.  Leave blank to keep or omit.",
        )
        st.caption(f"TOTP secret: {_stored_badge(z_has['totp_secret'])}")
        totp_secret = st.text_input(
            "TOTP secret (the QR seed, NOT the 6-digit code)",
            value="", type="password",
        )
        save = st.form_submit_button("Save Zerodha credentials", width="stretch")
    if save:
        patch = {"client_id": client_id.strip() or None}
        if api_key: patch["api_key"] = api_key
        if api_secret: patch["api_secret"] = api_secret
        if password: patch["password"] = password
        if totp_secret: patch["totp_secret"] = totp_secret.replace(" ", "")
        try:
            broker_mod.upsert(user.id, "ZERODHA", **patch)
            st.success("Saved.  Any secrets left blank were kept as-is.")
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    if z and z.has_data_creds():
        st.success(f"Access token present (last refreshed {_fmt_ist(z.token_updated_at)}).")

    if z and z.api_key:
        try:
            url = kite_login_url(user.id)
            st.markdown(
                f"[🔑 Connect Zerodha (OAuth)]({url})",
                help=(
                    "Opens Zerodha login in a new tab.  After you log in, "
                    "the callback server (port 8502) captures the token and "
                    "encrypts it against your user only."
                ),
            )
        except Exception as exc:
            st.info(f"OAuth not ready: {exc}")
    else:
        st.caption("Save your API key + secret above to enable one-click OAuth.")

    if z and z.has_data_creds():
        if st.button("Disconnect Zerodha"):
            broker_mod.upsert(user.id, "ZERODHA", access_token="")
            st.success("Access token cleared.")
            st.rerun()

    if z and z.api_key and z.api_secret:
        with st.expander("Paste request_token / redirect URL manually", expanded=False):
            st.markdown(
                "1. Open the login link below in a new tab.\n"
                "2. Log in with your Zerodha password + TOTP.\n"
                "3. Zerodha will redirect you to a URL of the shape "
                "`http://…/?request_token=XYZ&status=success` — copy that "
                "URL (or just the `XYZ` token) and paste it here."
            )
            try:
                from kiteconnect import KiteConnect
                base_url = KiteConnect(api_key=z.api_key).login_url()
                st.markdown(f"[Open Kite login →]({base_url})")
            except Exception as exc:
                st.info(f"Could not build login URL: {exc}")

            pasted = st.text_input(
                "Paste request_token OR full redirect URL",
                key="_kite_manual_paste_multi", value="",
            )
            if st.button("Exchange", key="_kite_manual_btn_multi") and pasted:
                from aivora.live import kite_auth
                rq = kite_auth.extract_request_token(pasted)
                if not rq:
                    st.error("Couldn't find a request_token in that string.")
                else:
                    try:
                        _exchange_and_store_kite_token(user.id, rq)
                        st.session_state["_flash"] = (
                            "ok",
                            "✅ Access token stored and encrypted.",
                        )
                        st.rerun()
                    except Exception as exc:
                        st.session_state["_flash"] = (
                            "error", f"Exchange failed: {exc}",
                        )
                        st.rerun()

    st.divider()
    st.subheader("Change password")
    with st.form("pw-form", clear_on_submit=True):
        old = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        cnew = st.text_input("Confirm new password", type="password")
        pw_save = st.form_submit_button("Update password", width="stretch")
    if pw_save:
        if new != cnew:
            st.error("New passwords don't match.")
        else:
            try:
                user_mod.change_password(user.id, old, new)
                st.success("Password updated.")
            except ValueError as exc:
                st.error(str(exc))


# =============================================================
#  Dashboard components
# =============================================================
def _dashboard_autorefresh() -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=30_000, key="_dash_autorefresh")
    except Exception:
        pass


def _plotly_dark_layout(fig: go.Figure, height: int = 320) -> None:
    """Apply the shared dark theme to a plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#F8FAFC", size=12),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        hoverlabel=dict(bgcolor="#172033", bordercolor="#27344D", font=dict(color="#F8FAFC")),
    )


def _sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0
    return float((daily_returns.mean() / daily_returns.std()) * (252 ** 0.5))


def top_navbar(user: user_mod.User, s: dict) -> None:
    is_open, market_label = _market_state()
    last = s.get("last_data_update")
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            last_str = last_dt.strftime("%H:%M:%S")
        except Exception:
            last_str = str(last)[-8:]
    else:
        last_str = "—"

    ai_active = bool(s.get("master_switch"))
    ai_dot = "av-dot-blue" if ai_active else "av-dot-amber"
    ai_lbl = "ACTIVE" if ai_active else "PAUSED"
    market_dot = "av-dot-green" if is_open else "av-dot-red"
    initials = _initials(user.display_name or user.email)

    st.markdown(
        f"""
        <div class="av-navbar">
            <div class="av-navbar-brand">
                <div class="av-navbar-logo"><img src="{_LOGO_URI}" alt="AiVora"></div>
                <div class="av-navbar-title">AiVora</div>
            </div>
            <div class="av-navbar-status">
                <div class="av-status-chip">
                    <span class="av-dot {market_dot}"></span>
                    <strong>{market_label}</strong>
                </div>
                <div class="av-status-chip">
                    Last Sync <strong>{last_str}</strong>
                </div>
                <div class="av-status-chip">
                    <span class="av-dot {ai_dot}"></span>
                    AI <strong>{ai_lbl}</strong>
                </div>
            </div>
            <div class="av-navbar-user">
                <div class="av-avatar">{initials}</div>
                <span>{user.display_name or user.email.split('@')[0]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_card(s: dict, state: dict) -> None:
    portfolio_val = float(s["current_capital"])
    today_pnl = float(s["today_pnl"])
    pnl_pct = today_pnl / max(float(s["initial_capital"]), 1) * 100
    win_rate = float(s["win_rate_today"]) * 100
    strategy = state.get("settings", {}).get("strategy_variant", "Variant #18")

    pnl_cls = "pos" if today_pnl >= 0 else "neg"
    pnl_sign = "+" if today_pnl >= 0 else ""

    st.markdown(
        f"""
        <div class="av-hero">
            <div class="av-hero-grid">
                <div>
                    <div class="av-hero-label">Portfolio value</div>
                    <div class="av-hero-value big">₹{portfolio_val:,.0f}</div>
                    <div class="av-hero-sub">Initial ₹{float(s['initial_capital']):,.0f} · Unrealised {float(s['unrealized_pnl_total']):+,.0f}</div>
                </div>
                <div>
                    <div class="av-hero-label">Today's P&amp;L</div>
                    <div class="av-hero-value {pnl_cls}">{pnl_sign}₹{today_pnl:,.0f}</div>
                    <div class="av-hero-sub">{pnl_sign}{pnl_pct:.2f}% today</div>
                </div>
                <div>
                    <div class="av-hero-label">Win rate today</div>
                    <div class="av-hero-value">{win_rate:.0f}%</div>
                    <div class="av-hero-sub">{s['trades_today']} trades · {s['n_open_trades']} open</div>
                </div>
                <div>
                    <div class="av-hero-label">Strategy</div>
                    <div class="av-hero-value" style="font-size:1.3rem">{strategy}</div>
                    <div class="av-hero-sub">Binary UP/DOWN · 60-min horizon</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(s: dict, state: dict, outlook: dict) -> None:
    trades = state.get("trades", [])
    closed = [t for t in trades if t.get("realized_pnl") is not None]
    max_dd = float(s.get("drawdown_pct", 0)) * 100

    if closed:
        df = pd.DataFrame(closed)
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df["date"] = df["exit_time"].dt.date
        daily = df.groupby("date")["realized_pnl"].sum().astype(float)
        initial = float(s["initial_capital"])
        daily_returns = daily / initial
        sharpe = _sharpe(daily_returns)
        monthly_pnl = df[df["exit_time"] >= (df["exit_time"].max() - pd.Timedelta(days=30))]["realized_pnl"].sum()
        monthly_return = float(monthly_pnl) / initial * 100
    else:
        sharpe = 0.0
        monthly_return = 0.0

    conf_txt = f"{outlook['confidence']:.0%}" if outlook.get("confidence") else "—"

    cols = st.columns(6)
    metrics = [
        ("Trades today", f"{s['trades_today']}", f"open {s['n_open_trades']}"),
        ("Open positions", f"{s['n_open_trades']}", None),
        ("Max drawdown", f"{max_dd:.2f}%", None),
        ("Sharpe", f"{sharpe:.2f}", None),
        ("AI confidence", conf_txt, outlook.get("outlook", "").title()),
        ("Monthly return", f"{monthly_return:+.2f}%", None),
    ]
    for col, (label, val, sub) in zip(cols, metrics):
        col.metric(label, val, sub)


def ai_confidence_card(outlook: dict) -> None:
    countdown = _fmt_countdown(_seconds_to_next_tick())
    kind = outlook.get("kind", "waiting")
    outlook_txt = outlook.get("outlook", "AWAITING")
    conf = outlook.get("confidence")
    conf_pct = int(conf * 100) if conf else 0
    reason = outlook.get("reason") or ""
    if outlook_txt == "AWAITING":
        reason = f"Next inference in {countdown}"

    dot_icon = {"bull": "🟢", "bear": "🔴", "neutral": "🟡", "waiting": "⚪"}[kind]

    st.markdown(
        f"""
        <div class="av-ai-card">
            <div style="flex:0 0 auto">
                <div class="av-ai-badge {kind}">
                    {dot_icon} {outlook_txt}
                </div>
            </div>
            <div style="flex:1;min-width:0">
                <div style="color:var(--text-secondary);font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem">Model outlook</div>
                <div style="color:var(--text);font-size:1rem;font-weight:500">{reason}</div>
                <div class="av-confidence-bar"><div class="av-confidence-fill" style="width:{conf_pct}%"></div></div>
            </div>
            <div style="flex:0 0 auto;text-align:right">
                <div style="color:var(--text-secondary);font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em">Next tick</div>
                <div style="color:var(--text);font-size:1.2rem;font-weight:700;font-variant-numeric:tabular-nums">{countdown}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _trade_status_html(row) -> str:
    if pd.isna(row.get("exit_time")):
        return '<span style="background:var(--primary-dim);color:var(--primary);padding:0.15rem 0.6rem;border-radius:999px;font-size:0.75rem;font-weight:600">OPEN</span>'
    reason = str(row.get("exit_reason") or "closed").upper()
    color = "success" if str(row.get("_pnl", 0)).lstrip("-").isdigit() else "warning"
    if reason in ("TP", "TRAILING_STOP"):
        color = "success"
    elif reason in ("SL", "STOP"):
        color = "danger"
    elif reason == "HORIZON":
        color = "warning"
    var_map = {"success": ("--success-dim", "--success"),
               "danger": ("--danger-dim", "--danger"),
               "warning": ("--warning-dim", "--warning")}
    bg, fg = var_map[color]
    return f'<span style="background:var({bg});color:var({fg});padding:0.15rem 0.6rem;border-radius:999px;font-size:0.75rem;font-weight:600">{reason}</span>'


def trades_section(state: dict) -> None:
    st.markdown('<div class="av-h2"><span class="av-h2-icon">📋</span>Trades</div>',
                unsafe_allow_html=True)

    trades = state.get("trades", [])
    if not trades:
        st.markdown(
            '<div class="av-empty">'
            '<div class="av-empty-icon">🌱</div>'
            '<div class="av-empty-msg">No trades yet</div>'
            '<div class="av-empty-hint">The scheduler will fire signals during market hours (09:15–15:30 IST).</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(trades)
    df["entry_dt"] = pd.to_datetime(df["entry_time"])
    df["exit_dt"] = pd.to_datetime(df["exit_time"], errors="coerce")

    tab_names = ["Today", "This Week", "All Time"]
    tabs = st.tabs(tab_names)
    now = _now_ist()
    day = now.date()
    week_start = now - timedelta(days=now.weekday())

    for tab, name in zip(tabs, tab_names):
        with tab:
            if name == "Today":
                sub = df[df["entry_dt"].dt.date == day]
            elif name == "This Week":
                sub = df[df["entry_dt"] >= week_start]
            else:
                sub = df
            if sub.empty:
                st.markdown(
                    f'<div class="av-empty">'
                    f'<div class="av-empty-icon">📭</div>'
                    f'<div class="av-empty-msg">No trades in {name.lower()}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
                continue

            def _row(r):
                is_open = pd.isna(r["exit_dt"])
                pnl = r.get("realized_pnl") if not is_open else r.get("unrealized_pnl")
                pnl = float(pnl or 0)
                entry = float(r["entry_premium"])
                current = float(r.get("current_premium") or entry)
                roi = ((current - entry) / entry * 100) if entry else 0
                confidence = r.get("entry_prob")
                dur_ref = r["exit_dt"] if not is_open else pd.Timestamp(now)
                dur_min = int((dur_ref - r["entry_dt"]).total_seconds() // 60)
                reason = str(r.get("exit_reason") or "").upper() if not is_open else "OPEN"
                return pd.Series({
                    "Time": r["entry_dt"].strftime("%H:%M"),
                    "Symbol": r["symbol"],
                    "Side": {"CE": "CALL", "PE": "PUT"}.get(r["side"], r["side"]),
                    "Strike": int(float(r["strike"])),
                    "Lots": int(r["lots"]),
                    "Entry": round(entry, 2),
                    "Current": round(current, 2),
                    "P&L": round(pnl, 2),
                    "ROI %": round(roi, 2),
                    "Confidence": f"{confidence:.0%}" if confidence else "—",
                    "Duration": f"{dur_min}m",
                    "Status": reason,
                })
            show = sub.apply(_row, axis=1).sort_values("Time", ascending=False)
            st.dataframe(show, width="stretch", height=380, hide_index=True)


def charts_section(state: dict, s: dict) -> None:
    st.markdown('<div class="av-h2"><span class="av-h2-icon">📈</span>Performance</div>',
                unsafe_allow_html=True)
    closed = [t for t in state.get("trades", []) if t.get("realized_pnl") is not None]
    if not closed:
        st.markdown(
            '<div class="av-empty">'
            '<div class="av-empty-icon">📊</div>'
            '<div class="av-empty-msg">No realised P&L yet</div>'
            '<div class="av-empty-hint">Equity curve and daily bars appear after the first trade closes.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(closed)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    df["realized_pnl"] = df["realized_pnl"].astype(float)
    df["equity"] = df["realized_pnl"].cumsum() + float(state["initial_capital"])

    tab_eq, tab_bar = st.tabs(["Equity curve", "Daily P&L"])
    with tab_eq:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["exit_time"], y=df["equity"], mode="lines",
            name="Equity",
            line=dict(color="#3B82F6", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.08)",
            hovertemplate="<b>%{y:,.0f}</b><br>%{x|%d %b %H:%M}<extra></extra>",
        ))
        fig.add_hline(y=float(state["initial_capital"]),
                      line_dash="dash", line_color="rgba(148,163,184,0.5)",
                      annotation_text="Initial capital",
                      annotation_position="right",
                      annotation_font_color="#94A3B8")
        _plotly_dark_layout(fig, height=320)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with tab_bar:
        df["date"] = df["exit_time"].dt.date
        daily = df.groupby("date")["realized_pnl"].sum().reset_index()
        colors = ["#16C784" if v >= 0 else "#EF4444" for v in daily["realized_pnl"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["realized_pnl"],
            marker=dict(color=colors),
            hovertemplate="<b>₹%{y:,.0f}</b><br>%{x}<extra></extra>",
        ))
        _plotly_dark_layout(fig, height=320)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def activity_timeline(state: dict) -> None:
    st.markdown('<div class="av-h2"><span class="av-h2-icon">⚡</span>Recent activity</div>',
                unsafe_allow_html=True)
    events = state.get("log", [])
    if not events:
        st.markdown(
            '<div class="av-empty">'
            '<div class="av-empty-icon">💤</div>'
            '<div class="av-empty-msg">No activity yet</div>'
            '</div>', unsafe_allow_html=True,
        )
        return

    top = events[:15]
    items_html = ['<div class="av-timeline">']
    for e in top:
        icon, css = _classify_event(e.get("msg", ""))
        t = _fmt_event_time(e.get("ts", ""))
        # Escape angle brackets to avoid accidental HTML injection from log msgs.
        msg = (e.get("msg") or "").replace("<", "&lt;").replace(">", "&gt;")
        items_html.append(
            f'<div class="av-timeline-item">'
            f'<div class="av-timeline-icon {css}">{icon}</div>'
            f'<div class="av-timeline-body">'
            f'<div class="av-timeline-msg">{msg}</div>'
            f'<div class="av-timeline-time">{t}</div>'
            f'</div></div>'
        )
    items_html.append("</div>")
    st.markdown("".join(items_html), unsafe_allow_html=True)

    if len(events) > 15:
        with st.expander(f"View full log ({len(events)} entries)", expanded=False):
            PAGE = 20
            n_pages = max(1, (len(events) + PAGE - 1) // PAGE)
            st.session_state.setdefault("_evlog_page", 0)
            page = int(st.session_state["_evlog_page"])
            page = max(0, min(page, n_pages - 1))
            slice_ = events[page * PAGE:(page + 1) * PAGE]
            for e in slice_:
                icon, _ = _classify_event(e.get("msg", ""))
                st.text(f"{icon}  {e['ts']}  {e['msg']}")
            c_prev, c_info, c_next = st.columns([1, 2, 1])
            with c_prev:
                if st.button("← Newer", disabled=(page == 0),
                             width="stretch", key="_evlog_prev"):
                    st.session_state["_evlog_page"] = page - 1
                    st.rerun()
            c_info.markdown(
                f"<div style='text-align:center;padding-top:0.6rem;color:var(--text-secondary)'>"
                f"page {page + 1} / {n_pages}</div>",
                unsafe_allow_html=True,
            )
            with c_next:
                if st.button("Older →", disabled=(page >= n_pages - 1),
                             width="stretch", key="_evlog_next"):
                    st.session_state["_evlog_page"] = page + 1
                    st.rerun()


# =============================================================
#  Dashboard page (orchestration)
# =============================================================
def dashboard_page(user: user_mod.User) -> None:
    _dashboard_autorefresh()

    mode = st.session_state.get("mode", "paper")
    portfolio = pf_mod.UserPortfolio(user.id, mode)
    s = portfolio.summary()
    state = portfolio.load()

    try:
        sm_mod.sync_user(user.id, mode, s["master_switch"])
    except Exception as exc:  # noqa: BLE001
        st.info(f"Scheduler: {exc}")

    top_navbar(user, s)
    hero_card(s, state)

    outlook = _parse_ai_outlook(state.get("log", []))
    metric_grid(s, state, outlook)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    ai_confidence_card(outlook)

    # Staleness warning
    last = s.get("last_data_update")
    if last:
        try:
            now_dt = datetime.now()
            last_dt = datetime.fromisoformat(last)
            age_min = (now_dt - last_dt).total_seconds() / 60
            in_session = (
                now_dt.time() >= datetime.strptime("09:15", "%H:%M").time()
                and now_dt.time() <= datetime.strptime("15:30", "%H:%M").time()
            )
            same_day = last_dt.date() == now_dt.date()
            in_first_15 = (
                now_dt.time() >= datetime.strptime("09:15", "%H:%M").time()
                and now_dt.time() <= datetime.strptime("09:30", "%H:%M").time()
            )
            if not same_day and in_first_15:
                st.info(
                    "🌙 Overnight data — first tick of the day fires within "
                    "5 min of market open."
                )
            elif age_min > 10 and in_session and same_day:
                st.warning(
                    f"⚠️ Data is **{age_min:.0f} minutes stale** — click Refresh in sidebar."
                )
        except Exception:
            pass

    # Confirmation modal for live trading (rendered here so it stays near hero)
    if st.session_state.get("_confirm_live"):
        st.error(
            "You're about to enable LIVE trading.  Real money at risk.  "
            "Type CONFIRM below and press Enter."
        )
        typed = st.text_input("Type CONFIRM to enable live trading", key="_confirm_txt")
        if typed.strip() == "CONFIRM":
            portfolio.set_master_switch(True)
            st.session_state["_confirm_live"] = False
            st.session_state["_confirm_txt"] = ""
            st.success("Live trading ARMED.")
            st.rerun()

    left, right = st.columns([3, 2], gap="large")
    with left:
        trades_section(state)
    with right:
        charts_section(state, s)

    activity_timeline(state)


# =============================================================
#  Sidebar
# =============================================================
def sidebar_for(user: user_mod.User) -> str:
    """Return which page to render (dashboard / profile / admin)."""
    initials = _initials(user.display_name or user.email)
    st.sidebar.markdown(
        f'''
        <div style="display:flex;align-items:center;gap:0.6rem;padding:0.5rem 0 1rem">
            <div class="av-avatar" style="width:38px;height:38px">{initials}</div>
            <div>
                <div style="font-weight:600;font-size:0.95rem">{user.display_name or user.email.split("@")[0]}</div>
                <div style="color:var(--text-secondary);font-size:0.75rem">user #{user.id}{" · admin" if user.is_admin else ""}</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    pages = ["Dashboard", "Profile"]
    if user.is_admin:
        pages.append("Admin")
    page = st.sidebar.radio("Navigate", pages, index=0, label_visibility="collapsed")
    which = page.lower()

    if which != "dashboard":
        st.sidebar.divider()
        if st.sidebar.button("Sign out", width="stretch"):
            _logout()
            st.rerun()
        st.sidebar.caption(f"DB: {db_mod.default_db_path().name} · encrypted at rest")
        return which

    # Dashboard-specific sidebar controls
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Trading mode",
        ["paper", "live"],
        index=(0 if st.session_state.get("mode", "paper") == "paper" else 1),
        format_func=lambda m: "📊 Paper" if m == "paper" else "💰 Live",
        horizontal=True,
    )
    if mode != st.session_state.get("mode"):
        st.session_state["mode"] = mode

    portfolio = pf_mod.UserPortfolio(user.id, mode)
    s = portfolio.summary()

    with st.sidebar.expander("Trading control", expanded=True):
        cur = s["master_switch"]
        label = "🟢 Trading ON — Stop" if cur else "🔴 Trading OFF — Start"
        if st.button(label, width="stretch", key="_master_switch"):
            if mode == "live" and not cur:
                st.session_state["_confirm_live"] = True
            else:
                portfolio.set_master_switch(not cur)
                st.rerun()

        if st.button("🔄 Refresh now", width="stretch", key="_refresh_now",
                     help="Fetch latest candles and run one inference tick."):
            with st.spinner("Fetching…"):
                try:
                    from aivora.webapp.trading_engine import MarketDataCache, run_user_tick
                    MarketDataCache._reset()
                    r = run_user_tick(user.id, mode)
                    if r.get("skipped"):
                        st.info(f"Skipped: {r['skipped']}")
                    elif r.get("error"):
                        st.error(r["error"])
                    else:
                        st.success(f"Tick complete — actions={len(r.get('actions', []))}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Tick failed: {exc}")
            st.rerun()

    with st.sidebar.expander("Capital & risk", expanded=False):
        state = portfolio.load()
        settings = state.get("settings", {})
        st.caption(f"Initial capital: ₹{float(state['initial_capital']):,.0f}")
        st.caption(f"Current: ₹{float(s['current_capital']):,.0f}")
        st.caption(f"Max trades/day: {settings.get('max_trades_per_day', 3)}")
        st.caption(f"TP: +{settings.get('take_profit_pct', 0.60) * 100:.0f}% · "
                   f"SL: -{settings.get('stop_loss_pct', 0.30) * 100:.0f}%")

    with st.sidebar.expander("Broker status", expanded=False):
        z = broker_mod.get(user.id, "ZERODHA")
        z_ok = bool(z and z.has_data_creds())
        st.markdown(f"{'🟢' if z_ok else '⚪'} Zerodha (Kite)")
        if not z_ok:
            st.caption("Go to Profile → Zerodha to connect.")

    st.sidebar.divider()
    if st.sidebar.button("🛑 Emergency square off", width="stretch",
                         type="primary", key="_emergency_off"):
        state = portfolio.load()
        now = datetime.now()
        n = 0
        for t in state["trades"]:
            if t.get("exit_time"):
                continue
            current = float(t.get("current_premium") or t["entry_premium"])
            lots = int(t["lots"]); lot_size = int(t["lot_size"])
            gross = (current - float(t["entry_premium"])) * lots * lot_size
            portfolio.close_trade(
                trade_id=t["trade_id"],
                exit_time=now,
                exit_premium=current,
                exit_reason="emergency",
                gross_pnl=gross, costs=0.0,
            )
            n += 1
        st.sidebar.warning(f"Closed {n} position(s).")
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Sign out", width="stretch", key="_signout"):
        _logout()
        st.rerun()
    st.sidebar.caption(f"DB: {db_mod.default_db_path().name} · encrypted at rest")

    return which


# =============================================================
#  Admin page
# =============================================================
def admin_page() -> None:
    st.title("Admin")
    rows = admin_mod.list_with_status()
    df = pd.DataFrame(rows)
    if not df.empty:
        show = df.assign(
            status=df["active"].map({True: "ACTIVE", False: "DEACTIVATED"})
        )[["id", "email", "display_name", "is_admin", "status",
           "created_at", "last_login", "brokers"]]
        st.dataframe(show, width="stretch")

    st.divider()
    st.subheader("Deactivate / reactivate")
    for r in rows:
        cols = st.columns([1, 3, 2, 2])
        cols[0].markdown(f"**{r['id']}**")
        cols[1].markdown(f"{r['email']} {'👑' if r['is_admin'] else ''}")
        cols[2].markdown("🟢 active" if r["active"] else "🔴 deactivated")
        with cols[3]:
            btn_label = "Deactivate" if r["active"] else "Reactivate"
            if st.button(btn_label, key=f"toggle-{r['id']}"):
                admin_mod.set_active(r["id"], not r["active"])
                st.rerun()

    st.caption(
        "Broker secrets remain encrypted — even admins cannot read them "
        "from this UI or by opening the DB file.  Deactivated users cannot log in."
    )


def migration_banner(user: user_mod.User) -> None:
    if st.session_state.get("_mig_shown"):
        return
    preview = mig_mod.preview()
    if preview is None:
        st.session_state["_mig_shown"] = True
        return
    st.warning(
        f"Legacy single-user paper portfolio found at `{preview['path']}` "
        f"({preview['n_trades']} trades, {preview['n_closed']} closed, "
        f"initial ₹{preview.get('initial_capital', 0):,.0f}).  "
        "Import into your paper portfolio?"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Import into my paper portfolio"):
            r = mig_mod.import_into(user.id, "paper")
            mig_mod.deactivate_legacy_file()
            st.session_state["_mig_shown"] = True
            st.success(f"Imported {r['imported']} trades (skipped {r['skipped']}).")
            st.rerun()
    with c2:
        if st.button("Skip for now"):
            st.session_state["_mig_shown"] = True
            st.rerun()


# =============================================================
#  Main dispatcher
# =============================================================
def main() -> None:
    inject_css()
    try:
        db_mod.init_db()
    except Exception as exc:
        st.error(
            "Database not ready.  Run:\n\n"
            "    python -m scripts.init_webapp_db\n\n"
            f"Underlying error: {exc}"
        )
        return

    _flash()
    user = _sess_user()
    if user is None:
        page = st.session_state.get("_page", "login")
        (register_page if page == "register" else login_page)()
        return

    if not st.session_state.get("_sched_registered"):
        sm_mod.set_tick_function(_per_user_tick)
        st.session_state["_sched_registered"] = True

    if st.query_params.get("kite_connected"):
        st.toast("✅ Kite connected.", icon="✅")
        try:
            st.query_params.clear()
        except Exception:
            pass

    which = sidebar_for(user)
    if which == "profile":
        profile_page(user)
    elif which == "admin" and user.is_admin:
        admin_page()
    else:
        migration_banner(user)
        dashboard_page(user)


def _per_user_tick(user_id: int, mode: str) -> None:
    from aivora.webapp.trading_engine import run_user_tick

    try:
        run_user_tick(user_id, mode)
    except Exception as exc:  # noqa: BLE001
        try:
            from aivora.webapp.portfolios import UserPortfolio
            UserPortfolio(user_id, mode).log_event(f"tick fatal: {exc}", "error")
        except Exception:
            log.exception("tick fatal for user_id=%s (event-log write failed)", user_id)


if __name__ == "__main__":
    main()
