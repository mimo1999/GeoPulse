"""
Phase 2 — Country Drilldown Page.

Shows:
  A. Risk Timeline (30d / 90d / 1yr)
  B. Key Event Clusters (protest, military, terrorism, sanctions, diplomatic)
  C. Feature Attributions (Integrated Gradients)
  D. Spillover Network (top related countries)
  E. Proxy Label Inspection
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Country Drilldown — GeoPulse",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'Courier New', monospace; color: #cc2222; }
    .cluster-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .cluster-title { font-size: 13px; letter-spacing: 1px; color: #888; }
    .cluster-count { font-size: 28px; font-weight: bold; color: #cc2222; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_countries() -> tuple[list[str], dict[str, str]]:
    """Return (list_of_codes, code→name mapping)."""
    try:
        resp = requests.get(f"{BACKEND_URL}/countries", timeout=10)
        entries = resp.json().get("countries", [])
        codes   = [r["country"] for r in entries]
        names   = {r["country"]: r.get("name", r["country"]) for r in entries}
        return codes, names
    except Exception:
        return [], {}


@st.cache_data(ttl=120)
def get_timeline(country: str, days: int) -> pd.DataFrame:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/timeline",
            params={"days": days}, timeout=10,
        )
        return pd.DataFrame(resp.json().get("timeline", []))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def get_events(country: str, days: int) -> list[dict]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/events",
            params={"days": days}, timeout=10,
        )
        return resp.json().get("events", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_heatmap() -> pd.DataFrame:
    try:
        resp = requests.get(f"{BACKEND_URL}/global/heatmap", timeout=10)
        return pd.DataFrame(resp.json().get("countries", []))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_spillover(country: str) -> list[dict]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/spillover",
            params={"top_n": 6}, timeout=10,
        )
        return resp.json().get("neighbors", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_attributions(country: str) -> dict:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/attributions",
            timeout=15,
        )
        return resp.json()
    except Exception:
        return {}


@st.cache_data(ttl=120)
def get_risk_score(country: str) -> dict:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/riskscore",
            json={"country": country}, timeout=15,
        )
        return resp.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "protest":    "#b8860b",
    "military":   "#8b0000",
    "terrorism":  "#4b0000",
    "sanctions":  "#4a4a8a",
    "diplomatic": "#2a5a2a",
}

CATEGORY_ICONS = {
    "protest":    "✊",
    "military":   "⚔️",
    "terrorism":  "💣",
    "sanctions":  "🚫",
    "diplomatic": "🤝",
}


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.markdown(
        "<h1>🔍 Country Intelligence Drilldown</h1>",
        unsafe_allow_html=True,
    )

    # --- Country selector (same source as the home page map: latest_country_risk) ---
    df_heat = get_heatmap()
    if df_heat.empty:
        st.error("Backend unavailable or no data loaded yet.")
        return
    df_heat = df_heat.sort_values("risk_score", ascending=False)
    countries = df_heat["country"].tolist()
    name_map = dict(zip(df_heat["country"], df_heat["name"]))

    col_sel, col_window = st.columns([2, 1])
    with col_sel:
        country = st.selectbox(
            "Select Country", countries,
            format_func=lambda x: name_map.get(x, x),
            key="drilldown_country",
        )
    with col_window:
        window = st.select_slider("Timeline Window", [30, 60, 90, 180, 365], value=90)

    if not country:
        return

    display_name = name_map.get(country, country)

    # --- Header: same values as the map. (The live /riskscore endpoint scores a
    # 90-day window ending today, which is empty when ingestion lags, and then
    # contradicts the map.) ---
    row = df_heat[df_heat["country"] == country].iloc[0]
    trend = row.get("trend") or "stable"
    trend_arrow = {"increasing": "↑", "stable": "→", "decreasing": "↓"}.get(trend, "")
    st.markdown(f"<h3 style='color:#888; font-family:Courier New;'>{display_name}</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Activity Index", f"{row['risk_score']:.3f}")
    c2.metric("Level", row.get("level", "UNKNOWN"))
    c3.metric("Trend", f"{trend_arrow} {trend}")
    c4.metric("Confidence", f"{(row.get('confidence') or 0):.0%}")
    c5.metric("Data as of", str(row.get("feature_date", "")))
    st.caption("Heuristic index of recent GDELT event mix; describes current activity, not a prediction.")

    st.divider()

    # ========== A. Risk Timeline ==========
    st.markdown("### Activity Timeline")
    df_timeline = get_timeline(country, window)
    if not df_timeline.empty:
        df_timeline["feature_date"] = pd.to_datetime(df_timeline["feature_date"])
        fig = _build_multitrace_timeline(df_timeline, display_name)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("No timeline data available.")

    st.divider()



    # ========== D. Spillover Network ==========
    st.markdown("### Related Countries (correlated activity)")
    neighbors = get_spillover(country)

    if neighbors:
        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown("**Top Related Countries**")
            for n in neighbors:
                neighbor    = n.get("neighbor", "")
                nb_name     = name_map.get(neighbor, neighbor)
                weight      = n.get("spillover_weight", 0)
                corr        = n.get("risk_correlation", 0)
                is_adj      = n.get("is_adjacent", False)
                border_note = " 🗺️ border" if is_adj else ""
                st.markdown(
                    f"**{nb_name}**{border_note}  \n"
                    f"Spillover: `{weight:.3f}` · Corr: `{corr:.3f}`"
                )

        with col_r:
            fig_spill = _build_spillover_chart(display_name, neighbors, name_map)
            st.plotly_chart(fig_spill, use_container_width=True,
                            config={"displayModeBar": False})
    else:
        st.info("No spillover data. Run POST /analyze/spillover to compute.")




# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_multitrace_timeline(df: pd.DataFrame, country: str) -> go.Figure:
    fig = go.Figure()
    if "risk_score" in df:
        fig.add_trace(go.Scatter(
            x=df["feature_date"], y=df["risk_score"],
            name="Risk Score", line=dict(color="#cc2222", width=2.5),
            fill="tozeroy", fillcolor="rgba(204,34,34,0.08)",
        ))
    for col, name, color in [
        ("violence_score",    "Violence",       "#8b0000"),
        ("protest_score",     "Protests",       "#b8860b"),
        ("diplomatic_stress", "Diplo. Stress",  "#4a4a8a"),
        ("terrorism_score",   "Terrorism",      "#6b1111"),
        ("economic_stress",   "Econ. Stress",   "#3a5a3a"),
    ]:
        if col in df:
            fig.add_trace(go.Scatter(
                x=df["feature_date"], y=df[col],
                name=name, line=dict(color=color, width=1, dash="dot"),
                visible="legendonly",
            ))
    fig.update_layout(
        title=f"{country} — Activity Timeline",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a1a",
                   tickfont=dict(color="#888")),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a",
                    font=dict(color="#888", size=11)),
        font=dict(color="#888"), height=300,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _build_cluster_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for cat, color in CATEGORY_COLORS.items():
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        icon = CATEGORY_ICONS.get(cat, "")
        fig.add_trace(go.Bar(
            x=sub["cluster_date"],
            y=sub["total_mentions"],
            name=f"{icon} {cat.title()}",
            marker_color=color,
            opacity=0.85,
        ))
    fig.update_layout(
        barmode="stack",
        title="Event Mentions by Category Over Time",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=False, tickfont=dict(color="#888")),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888"),
                   title="Total Mentions"),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a",
                    font=dict(color="#888", size=10)),
        font=dict(color="#888"), height=250,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _show_actor_pairs(events: list[dict]) -> None:
    import json
    all_pairs: dict[tuple, int] = {}
    for ev in events:
        pairs = ev.get("top_actor_pairs")
        if pairs:
            if isinstance(pairs, str):
                try:
                    pairs = json.loads(pairs)
                except Exception:
                    continue
            for p in pairs:
                key = (p.get("actor1", ""), p.get("actor2", ""))
                all_pairs[key] = all_pairs.get(key, 0) + p.get("count", 1)

    if not all_pairs:
        return

    sorted_pairs = sorted(all_pairs.items(), key=lambda x: -x[1])[:10]
    rows = [
        {"Actor 1": p[0], "Actor 2": p[1], "Events": c}
        for (p, c) in sorted_pairs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_spillover_chart(
    country: str, neighbors: list[dict], name_map: dict[str, str] | None = None
) -> go.Figure:
    # Simple bar chart of spillover weights
    nm      = name_map or {}
    labels  = [nm.get(n["neighbor"], n["neighbor"]) for n in neighbors]
    weights = [n["spillover_weight"] for n in neighbors]
    corrs   = [n.get("risk_correlation", 0) for n in neighbors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=weights,
        name="Spillover Weight",
        marker_color="#8b0000",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=[abs(c) for c in corrs],
        name="|Risk Corr.|",
        mode="markers",
        marker=dict(color="#cc6622", size=10, symbol="diamond"),
        yaxis="y",
    ))
    fig.update_layout(
        title=f"Spillover Network — {country}",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(tickfont=dict(color="#ccc")),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888"),
                   title="Weight / |Correlation|", range=[0, 1]),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a",
                    font=dict(color="#888")),
        font=dict(color="#888"), height=260,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _build_label_chart(df: pd.DataFrame, country: str) -> go.Figure:
    fig = go.Figure()
    label_styles = [
        ("instability_label", "Instability", "#cc2222"),
        ("war_label",         "War",         "#8b0000"),
        ("terrorism_label",   "Terrorism",   "#4b0000"),
        ("financial_label",   "Financial",   "#3a5a7a"),
    ]
    for col, name, color in label_styles:
        if col in df:
            fig.add_trace(go.Scatter(
                x=df["label_date"], y=df[col],
                name=name, line=dict(color=color, width=1.5),
            ))
    fig.update_layout(
        title=f"{country} — Proxy Labels (Training Ground Truth)",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a1a",
                   tickfont=dict(color="#888")),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a",
                    font=dict(color="#888", size=10)),
        font=dict(color="#888"), height=260,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


main()
