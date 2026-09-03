"""
Shared UI theme and components for the GeoPulse Streamlit dashboard.

Every page should call `inject_theme()` right after `st.set_page_config()`
and use the helpers below instead of hand-rolled `st.markdown(html)` blocks.
This keeps all 7 pages visually and behaviorally consistent from one place
instead of each page redefining its own CSS and card markup.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@st.cache_data(ttl=None)
def _load_fips_to_iso3() -> dict[str, str]:
    """
    FIPS 10-4 -> ISO-3166-1 alpha-3 lookup, for the choropleth map (which
    needs ISO-3 codes for Plotly's `locationmode="ISO-3"`) — everything
    else in the app, and the API itself, works in FIPS (GDELT's native
    country coding, e.g. "UP" for Ukraine, "RS" for Russia).

    Loaded by file path from the project's data/iso3_to_fips.py rather
    than `import data.iso3_to_fips`, since `streamlit run` puts this
    file's own directory (streamlit_app/) at sys.path[0], not the project
    root, so the package import isn't reliably available here.
    """
    try:
        path = Path(__file__).resolve().parent.parent / "data" / "iso3_to_fips.py"
        spec = importlib.util.spec_from_file_location("iso3_to_fips", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {fips: iso3 for iso3, fips in module.ISO3_TO_FIPS.items() if fips}
    except Exception:
        return {}


def fips_to_iso3(code: str) -> str:
    """Map a FIPS country code to ISO-3 for map rendering; falls back to the code itself if unmapped."""
    return _load_fips_to_iso3().get(code, code)

# Primary accent — a muted brick-red rather than a pure, saturated red.
# Used for headers, chart lines, and card values across every page.
ACCENT = "#b2504f"

# Softened, desaturated badge/level colors (previously pure #ff4444-style reds).
BADGE_COLORS = {
    "CRITICAL": "#d97a7a",
    "HIGH":     "#d99494",
    "ELEVATED": "#dba9a9",
    "MODERATE": "#d4a988",
    "LOW":      "#8fc08f",
}

# Softened risk-score color ramp, shared by the choropleth, bar charts,
# and network graphs (previously duplicated per page with brighter reds).
RISK_SCALE = [
    (0.80, "#5c3333"),   # CRITICAL — muted dark brick
    (0.65, "#7a4040"),   # HIGH
    (0.50, "#93504f"),   # ELEVATED
    (0.35, "#a8776a"),   # MODERATE
    (0.00, "#4f6b52"),   # LOW — muted green
]

# Contagion color scale for GNN network graphs: cool/muted at low
# contagion, the app's accent red at high contagion. Shared so the
# colorbar legend means the same thing everywhere it's used.
CONTAGION_SCALE = [[0.0, "#2a3540"], [0.5, "#6b5248"], [1.0, "#b2504f"]]


def risk_level(score: float) -> str:
    if score >= 0.80: return "CRITICAL"
    if score >= 0.65: return "HIGH"
    if score >= 0.50: return "ELEVATED"
    if score >= 0.35: return "MODERATE"
    return "LOW"


def risk_color(score: float) -> str:
    """Shared, softened risk-score → color mapping for charts and maps."""
    for threshold, color in RISK_SCALE:
        if score >= threshold:
            return color
    return RISK_SCALE[-1][1]


def trend_arrow(trend: str) -> str:
    return {"increasing": "↑", "stable": "→", "decreasing": "↓"}.get(trend, "")


# ---------------------------------------------------------------------------
# Theme — one stylesheet shared by every page, including restyled native
# Streamlit widgets (st.metric / st.alert / sidebar) so they don't clash
# with the custom dark theme.
# ---------------------------------------------------------------------------

_THEME_CSS = """
<style>
:root {
    --bg: #0d0d0d;
    --surface: #141414;
    --border: #2a2a2a;
    --text: #e0e0e0;
    --muted: #888;
    --accent: #b2504f;
}

.stApp { background-color: var(--bg); color: var(--text); }
h1, h2, h3 { font-family: 'Courier New', monospace; color: var(--accent); }

/* Native st.metric */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px 6px;
}
div[data-testid="stMetricLabel"] > div { color: var(--muted) !important; font-family: 'Courier New', monospace; }
div[data-testid="stMetricValue"] { color: var(--text) !important; }

/* Native st.error / st.warning / st.info / st.success. Streamlit paints
   the color on an inner .stAlertContainer (semi-transparent yellow/red/
   blue/green), not the outer stAlert div, so both must be overridden. */
div[data-testid="stAlert"], .stAlertContainer {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

/* Toggles / checkboxes / sliders — Streamlit's default primary color is
   a bright saturated red that doesn't follow --accent. The toggle track
   is the first <div> inside its <label data-baseweb="checkbox">; only
   recolor it when checked, so the "off" state stays neutral gray. */
div[data-testid="stSlider"] [role="slider"] { background-color: var(--accent) !important; }
label[data-baseweb="checkbox"]:has(input:checked) > div:first-child {
    background: var(--accent) !important;
}

/* Tabs */
button[data-baseweb="tab"] { font-family: 'Courier New', monospace; color: var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }

/* Shared card component — replaces per-page .metric-card / .cluster-card / etc. */
.gp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 0;
}
.gp-card-accent { border-left: 3px solid var(--accent); }
.gp-label { font-family: 'Courier New', monospace; font-size: 11px; letter-spacing: 2px; color: var(--muted); }
.gp-sublabel { color: #666; font-size: 12px; margin-top: 4px; }
.gp-value { font-size: 30px; font-weight: bold; color: var(--accent); margin: 4px 0; }
.gp-badge {
    display: inline-block; padding: 3px 10px; border-radius: 3px;
    font-size: 12px; font-weight: bold; letter-spacing: 1px;
    background: #000; border: 1px solid currentColor;
}

/* Responsive metric grid — replaces st.columns(N) + st.metric for summary
   rows. st.columns() are fixed-ratio flex items that just squish into
   slivers on a narrow viewport; this reflows into fewer columns instead. */
.gp-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin: 10px 0;
}
.gp-metric-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    min-width: 0;
}
.gp-metric-label {
    font-family: 'Courier New', monospace;
    font-size: 11px; letter-spacing: 1px; color: var(--muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.gp-metric-value {
    font-size: 22px; font-weight: bold; color: var(--text);
    margin-top: 2px; overflow-wrap: break-word;
}
.gp-metric-sub { font-size: 11px; color: #666; margin-top: 2px; }
</style>
"""


def inject_theme() -> None:
    """Apply the shared GeoPulse dark theme. Call once per page, right after st.set_page_config()."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Country selection — persisted across pages via st.session_state, so
# picking a country on one page keeps it selected when you navigate to
# another (previously every page had its own disconnected selectbox).
# ---------------------------------------------------------------------------

SELECTED_COUNTRY_KEY = "gp_selected_country"


@st.cache_data(ttl=300)
def fetch_countries() -> tuple[list[str], dict[str, str]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/countries", timeout=10)
        resp.raise_for_status()
        entries = resp.json().get("countries", [])
        codes = [r["country"] for r in entries]
        names = {r["country"]: r.get("name", r["country"]) for r in entries}
        return codes, names
    except Exception:
        return [], {}


def country_picker(
    label: str = "Country",
    key: str = SELECTED_COUNTRY_KEY,
    container=None,
) -> tuple[Optional[str], dict[str, str]]:
    """
    A country selector backed by st.session_state[key]. Using the same key
    (the default) on every page means the selection carries over across
    page navigation instead of resetting each time.
    """
    codes, names = fetch_countries()
    target = container if container is not None else st

    if not codes:
        target.error("Backend unavailable or no countries loaded yet.")
        return None, {}

    if key not in st.session_state or st.session_state[key] not in codes:
        st.session_state[key] = codes[0]

    selected = target.selectbox(
        label, codes,
        format_func=lambda x: names.get(x, x),
        key=key,
    )
    return selected, names


# ---------------------------------------------------------------------------
# Reusable HTML components — replace hand-rolled f-string HTML per page.
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sublabel: str = "", badge: Optional[str] = None) -> None:
    """A themed metric card with an optional risk-level badge."""
    badge_html = ""
    if badge:
        color = BADGE_COLORS.get(badge, "#888")
        badge_html = f'<span class="gp-badge" style="color:{color};">{badge}</span>'
    sub_html = f'<div class="gp-sublabel">{sublabel}</div>' if sublabel else ""
    # NOTE: built as one unbroken line, not a multi-line indented f-string.
    # st.markdown(unsafe_allow_html=True) still runs the text through a
    # CommonMark parser first; any line indented 4+ spaces is treated as a
    # literal code block, which silently prints raw tags instead of
    # rendering them. This only surfaces once real data reaches the call
    # site, so it's easy to miss — every HTML-building helper here must
    # avoid embedded newlines/indentation for the same reason.
    html = (
        f'<div class="gp-card gp-card-accent">'
        f'<div class="gp-label">{label}</div>'
        f'<div class="gp-value">{value}</div>'
        f'{sub_html}{badge_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def metric_grid(items: list[tuple[str, str, str]]) -> None:
    """
    A responsive grid of metric cells, replacing st.columns(N) + st.metric.
    Reflows to fewer columns on narrow viewports instead of squishing.

    items: list of (label, value, sublabel) — sublabel may be "".
    """
    def _cell(label: str, value: str, sub: str) -> str:
        sub_html = f'<div class="gp-metric-sub">{sub}</div>' if sub else ""
        return (
            f'<div class="gp-metric-cell">'
            f'<div class="gp-metric-label">{label}</div>'
            f'<div class="gp-metric-value">{value}</div>'
            f'{sub_html}'
            f'</div>'
        )

    cells = "".join(_cell(label, value, sub) for label, value, sub in items)
    st.markdown(f'<div class="gp-metric-grid">{cells}</div>', unsafe_allow_html=True)


def risk_header(pred: dict, display_name: str) -> None:
    """Standard risk summary row, used at the top of most pages."""
    # `.get(key, 0)` only substitutes when the key is *missing* — the API
    # can return an explicit null for a scored field, so `or 0` guards
    # against that too (a bare `.get` crashed here in production).
    risk  = pred.get("risk_score") or 0
    level = pred.get("level", "UNKNOWN")
    conf  = pred.get("confidence") or 0
    trend = pred.get("trend", "stable")
    war   = pred.get("war_probability") or 0

    st.markdown(
        f"<h3 style='color:#888; font-family:Courier New;'>{display_name}</h3>",
        unsafe_allow_html=True,
    )
    metric_grid([
        ("Risk Score", f"{risk:.3f}", ""),
        ("Level", level, ""),
        ("Trend", f"{trend_arrow(trend)} {trend}", ""),
        ("Confidence", f"{conf:.0%}", ""),
        ("War Prob.", f"{war:.3f}", ""),
    ])

    advisory = pred.get("advisory", "")
    drivers = pred.get("major_drivers") or []
    if drivers:
        # The advisory sentence alone doesn't say *why* — fold the top
        # drivers in so this one line is self-contained instead of sending
        # the reader to the Escalation Drivers tab just to find out.
        driver_text = ", ".join(d.replace("_", " ") for d in drivers[:3])
        advisory = f"{advisory} Primary drivers: {driver_text}." if advisory else f"Primary drivers: {driver_text}."
    if advisory:
        st.info(advisory)


@contextmanager
def loading(message: str):
    """Shorthand for st.spinner so slow API calls show a visible loading state."""
    with st.spinner(message):
        yield
