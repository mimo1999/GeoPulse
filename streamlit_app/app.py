"""
GeoPulse Streamlit Dashboard — Global Risk Intelligence Map.

Layout:
  ┌────────────────────────────────────┐
  │        GLOBAL RISK MAP             │
  │   Black → Maroon → Crimson scale   │
  └────────────────────────────────────┘
  ┌──────────────┬─────────────────────┐
  │ Country Info │ Activity Timeline   │
  └──────────────┴─────────────────────┘
  ┌────────────────────────────────────┐
  │ Highest Activity Countries         │
  └────────────────────────────────────┘
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.iso3_to_fips import ISO3_TO_FIPS  # noqa: E402

FIPS_TO_ISO3 = {v: k for k, v in ISO3_TO_FIPS.items() if v is not None}

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GeoPulse Activity Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — intelligence-dashboard aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    :root {
        --bg: #0d0d0d;
        --surface: #141414;
        --border: #2a2a2a;
        --text: #e0e0e0;
        --muted: #888;
        --critical: #000000;
        --high: #3b0000;
        --elevated: #6b0000;
        --moderate: #8b1a1a;
        --low: #b22222;
        --accent: #cc2222;
    }

    .stApp { background-color: var(--bg); color: var(--text); }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .risk-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .badge-CRITICAL { background: #000; color: #ff4444; border: 1px solid #ff4444; }
    .badge-HIGH     { background: #1a0000; color: #ff6b6b; }
    .badge-ELEVATED { background: #2d0000; color: #ff9999; }
    .badge-MODERATE { background: #3d1000; color: #ffaa77; }
    .badge-LOW      { background: #0d1a0d; color: #66cc66; }
    h1, h2, h3 { color: var(--text); font-family: 'Courier New', monospace; }
    .sidebar .sidebar-content { background-color: var(--surface); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_heatmap() -> pd.DataFrame:
    try:
        resp = requests.get(f"{BACKEND_URL}/global/heatmap", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data["countries"])
    except Exception as e:
        st.warning(f"Backend unavailable: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=120)
def fetch_timeline(country: str, days: int = 90) -> pd.DataFrame:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/country/{country}/timeline",
            params={"days": days},
            timeout=10,
        )
        resp.raise_for_status()
        return pd.DataFrame(resp.json()["timeline"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_risk_score(country: str) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/riskscore",
            json={"country": country},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Risk color mapping
# ---------------------------------------------------------------------------

RISK_COLORS = {
    "CRITICAL": "#000000",
    "HIGH":     "#1a0000",
    "ELEVATED": "#6b0000",
    "MODERATE": "#8b1a1a",
    "LOW":      "#2d4a2d",
}

def risk_color(score: float) -> str:
    if score >= 0.80: return RISK_COLORS["CRITICAL"]
    if score >= 0.65: return RISK_COLORS["HIGH"]
    if score >= 0.50: return RISK_COLORS["ELEVATED"]
    if score >= 0.35: return RISK_COLORS["MODERATE"]
    return RISK_COLORS["LOW"]


def risk_level(score: float) -> str:
    if score >= 0.80: return "CRITICAL"
    if score >= 0.65: return "HIGH"
    if score >= 0.50: return "ELEVATED"
    if score >= 0.35: return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# Highest-risk bar chart
# ---------------------------------------------------------------------------

TREND_COLORS = {
    "increasing": "#d94040",  # red
    "decreasing": "#4a7fb5",  # blue
}
STABLE_HIGH_COLOR = "#d9b23c"  # yellow, stable with risk > 0.5
STABLE_LOW_COLOR = "#4caf50"   # green, stable with risk <= 0.5
NONE_TREND_COLOR = "#000000"   # black, trend unknown


def _bar_color(trend: str | None, risk_score: float) -> str:
    if trend is None or (isinstance(trend, float) and math.isnan(trend)):
        return NONE_TREND_COLOR
    if trend == "stable":
        return STABLE_HIGH_COLOR if risk_score > 0.5 else STABLE_LOW_COLOR
    return TREND_COLORS.get(trend, NONE_TREND_COLOR)


def build_risk_bar_chart(df: pd.DataFrame, visible_bars: int = 20) -> go.Figure:
    """Bar chart of every tracked country's latest risk score, sorted
    descending. Color encodes trend direction (and, for a stable trend,
    whether risk is currently above or below the 0.5 midpoint) rather than
    risk magnitude, since risk_score already drives bar height."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available", paper_bgcolor="#0d0d0d")
        return fig

    df = df.sort_values("risk_score", ascending=False).copy()
    colors = [_bar_color(t, r) for t, r in zip(df["trend"], df["risk_score"])]
    confidence_text = df["confidence"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    updated_text = df["feature_date"].astype(str) if "feature_date" in df.columns else ""
    trend_text = df["trend"].fillna("unknown")

    # Two raw country codes can resolve to the same display name (e.g. a
    # legacy FIPS "US" row and an ISO3 "USA" row both showing "United
    # States") when upstream data has both conventions for one country.
    # A single Bar trace with a repeated x-category renders the bars
    # stacked on top of each other rather than side by side, which looks
    # like one impossible bar exceeding 1.0 -- disambiguate the label with
    # the raw code so every bar gets its own x position instead.
    dup_name = df["name"].duplicated(keep=False)
    label = df["name"].where(~dup_name, df["name"] + " (" + df["country"].astype(str) + ")")

    fig = go.Figure(go.Bar(
        x=label, y=df["risk_score"], marker_color=colors,
        customdata=list(zip(confidence_text, updated_text, trend_text)),
        hovertemplate=(
            "<b>%{x}</b><br>Index: %{y:.4f}<br>Trend: %{customdata[2]}<br>"
            "Confidence: %{customdata[0]}<br>Last update: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d", font=dict(color="#ccc"),
        margin=dict(l=10, r=10, t=10, b=10), height=420,
        yaxis=dict(title="Activity index", gridcolor="#333"),
        xaxis=dict(
            title="Country", range=[-0.5, visible_bars - 0.5],
            rangeslider=dict(visible=True, thickness=0.08, bgcolor="#1a1a1a"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def build_choropleth(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available", paper_bgcolor="#0d0d0d")
        return fig

    df = df.copy()
    df["risk_score"] = df["risk_score"].fillna(0.0)
    if "name" not in df.columns:
        df["name"] = df["country"]
    df["text"] = df.apply(
        lambda r: (
            f"<b>{r['name']}</b><br>"
            f"Index: {r['risk_score']:.2f}<br>"
            f"Level: {risk_level(r['risk_score'])}<br>"
            f"Confidence: {r.get('confidence', 0):.2f}<br>"
            f"Trend: {r.get('trend', 'N/A')}"
        ),
        axis=1,
    )

    # `country` is mostly FIPS 10-4 (GDELT's coding, e.g. "UP" = Ukraine), which
    # Plotly's ISO-3 location mode can't place; a few countries also have a
    # duplicate ISO-3-coded row. Convert FIPS -> ISO-3 and keep one row per
    # country, preferring the FIPS-coded one (the dominant convention).
    df["iso3"] = df["country"].map(lambda c: c if len(c) == 3 else FIPS_TO_ISO3.get(c))
    df = df.dropna(subset=["iso3"])
    df = df.assign(_is_iso3=df["country"].str.len() == 3).sort_values("_is_iso3")
    df = df.drop_duplicates("iso3", keep="first")

    fig = go.Figure(go.Choropleth(
        locations=df["iso3"],
        locationmode="ISO-3",
        z=df["risk_score"],
        text=df["text"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0,  "#1a3a1a"],    # Low    — dark green
            [0.35, "#4a1a00"],    # Moderate
            [0.50, "#6b0000"],    # Elevated
            [0.65, "#3b0000"],    # High
            [0.80, "#1a0000"],    # Critical
            [1.0,  "#000000"],    # Maximum
        ],
        zmin=0.0,
        zmax=1.0,
        showscale=True,
        colorbar=dict(
            title=dict(text="Activity Index", font=dict(color="#888")),
            tickfont=dict(color="#888"),
            bgcolor="#141414",
            bordercolor="#2a2a2a",
            len=0.8,
        ),
        marker=dict(line=dict(color="#1a1a1a", width=0.5)),
    ))

    fig.update_layout(
        title=dict(
            text="GLOBAL ACTIVITY MONITOR",
            font=dict(size=18, color="#cc2222", family="Courier New"),
            x=0.5,
        ),
        geo=dict(
            bgcolor="#0d0d0d",
            landcolor="#1a1a1a",
            oceancolor="#0a0a0f",
            showocean=True,
            showland=True,
            showcoastlines=True,
            coastlinecolor="#2a2a2a",
            showframe=False,
            projection_type="natural earth",
            lakecolor="#0a0a0f",
            countrycolor="#1f1f1f",
        ),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )
    return fig


# ---------------------------------------------------------------------------
# Timeline figure
# ---------------------------------------------------------------------------

def build_timeline(df: pd.DataFrame, country: str) -> go.Figure:
    if df.empty:
        return go.Figure()

    df = df.copy()
    df["feature_date"] = pd.to_datetime(df["feature_date"])

    fig = go.Figure()

    # Main risk score line
    if "risk_score" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["feature_date"],
            y=df["risk_score"],
            name="Activity Index",
            line=dict(color="#cc2222", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(204,34,34,0.10)",
        ))

    # Component traces
    component_colors = {
        "violence_score":    ("Violence", "#8b0000"),
        "protest_score":     ("Protests", "#b8860b"),
        "diplomatic_stress": ("Diplo. Stress", "#4a4a8a"),
        "terrorism_score":   ("Terrorism", "#6b2222"),
    }
    for col, (name, color) in component_colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["feature_date"],
                y=df[col],
                name=name,
                line=dict(color=color, width=1, dash="dot"),
                opacity=0.7,
                visible="legendonly",
            ))

    fig.update_layout(
        title=f"{country} — Activity Timeline",
        xaxis=dict(
            showgrid=True, gridcolor="#1a1a1a",
            tickfont=dict(color="#888"),
            title="",
        ),
        yaxis=dict(
            range=[0, 1],
            showgrid=True, gridcolor="#1a1a1a",
            tickfont=dict(color="#888"),
            title="Score (0–1)",
            title_font=dict(color="#666"),
        ),
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        legend=dict(
            bgcolor="#141414",
            bordercolor="#2a2a2a",
            font=dict(color="#888", size=11),
        ),
        font=dict(color="#888"),
        height=280,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    df_heatmap = fetch_heatmap()
    as_of = str(df_heatmap["feature_date"].max()) if not df_heatmap.empty and "feature_date" in df_heatmap else "unknown"

    # Header
    st.markdown(
        "<h1 style='text-align:center; font-family:Courier New; "
        "color:#cc2222; letter-spacing:3px; margin-bottom:4px;'>"
        "⬛ GLOBAL ACTIVITY MONITOR</h1>"
        "<p style='text-align:center; color:#555; font-size:13px; "
        "font-family:Courier New; margin-top:0;'>"
        f"Activity Monitor · data as of {as_of}"
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- World Map ----
    fig_map = build_choropleth(df_heatmap)
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ---- Stats row ----
    if not df_heatmap.empty:
        col1, col2, col3, col4 = st.columns(4)
        critical_count = int((df_heatmap["risk_score"] >= 0.80).sum())
        high_count     = int(((df_heatmap["risk_score"] >= 0.65) & (df_heatmap["risk_score"] < 0.80)).sum())
        avg_risk       = float(df_heatmap["risk_score"].mean())

        with col1:
            st.metric("Countries Tracked", len(df_heatmap))
        with col2:
            st.metric("CRITICAL", critical_count, delta=None)
        with col3:
            st.metric("HIGH", high_count)
        with col4:
            st.metric("Global Avg Index", f"{avg_risk:.3f}")

    st.caption(
        "Scores are a heuristic index of each country's recent GDELT event mix (protest, violence, "
        "diplomatic and economic stress, terrorism). They describe current activity; they are not "
        "predictions and have not been validated as a measure of risk."
    )

    st.divider()

    # ---- Country drilldown ----
    st.markdown("### Country Drilldown")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        if not df_heatmap.empty:
            sorted_df = df_heatmap.sort_values("risk_score", ascending=False)
            countries_list = sorted_df["country"].tolist()
            # Build name lookup: code → display name
            name_map = {}
            if "name" in sorted_df.columns:
                name_map = dict(zip(sorted_df["country"], sorted_df["name"]))
        else:
            countries_list = []
            name_map = {}

        selected = st.selectbox(
            "Select Country",
            countries_list,
            format_func=lambda x: name_map.get(x, x),
            key="home_country",
        )

        timeline_days = st.select_slider(
            "Timeline Window",
            options=[30, 60, 90, 180, 365],
            value=90,
        )

        if selected:
            # Use the same source as the map/table (latest_country_risk). The live
            # /riskscore endpoint scores a 90-day window ending today, which is empty
            # whenever ingestion lags, and then contradicts the map (e.g. CRITICAL on
            # the map, LOW at 0% confidence here).
            row = df_heatmap[df_heatmap["country"] == selected].iloc[0].to_dict()
            pred = {
                "risk_score": row["risk_score"],
                "level": row.get("level", "UNKNOWN"),
                "confidence": row.get("confidence") or 0,
                "trend": row.get("trend") or "stable",
                "name": row.get("name"),
            }
            if pred:
                score       = pred.get("risk_score", 0)
                level       = pred.get("level", "UNKNOWN")
                conf        = pred.get("confidence", 0)
                trend       = pred.get("trend", "stable")
                display_name = pred.get("name") or name_map.get(selected, selected)
                trend_arrow = {"increasing": "↑", "stable": "→", "decreasing": "↓"}.get(trend, "")

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-family:Courier New; font-size:11px; color:#555; letter-spacing:2px;">ACTIVITY INDEX</div>
                        <div style="font-family:Courier New; font-size:14px; color:#888; margin-bottom:4px;">{display_name}</div>
                        <div style="font-size:36px; font-weight:bold; color:#cc2222; margin:8px 0;">{score:.3f}</div>
                        <span class="risk-badge badge-{level}">{level}</span>
                        <span style="margin-left:8px; color:#888; font-size:13px;">{trend_arrow} {trend}</span>
                        <div style="margin-top:12px; color:#666; font-size:12px;">
                            Confidence: {conf:.0%}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    with col_right:
        if selected:
            df_timeline = fetch_timeline(selected, days=timeline_days)
            display_label = name_map.get(selected, selected)
            fig_timeline = build_timeline(df_timeline, display_label)
            st.plotly_chart(fig_timeline, use_container_width=True,
                            config={"displayModeBar": False})

            # Feature breakdown
            if not df_timeline.empty:
                latest = df_timeline.iloc[-1]
                st.markdown("**Latest Feature Snapshot**")
                feat_cols = [
                    ("protest_score", "Protests"),
                    ("violence_score", "Violence"),
                    ("diplomatic_stress", "Diplo. Stress"),
                    ("economic_stress", "Econ. Stress"),
                    ("terrorism_score", "Terrorism"),
                    ("avg_sentiment", "Avg Sentiment"),
                ]
                fc1, fc2, fc3 = st.columns(3)
                for i, (col, label) in enumerate(feat_cols):
                    val = latest.get(col)
                    if val is not None:
                        [fc1, fc2, fc3][i % 3].metric(label, f"{float(val):.3f}")

    # ---- Top risk table ----
    st.divider()
    st.markdown("### Highest Activity Countries")
    if not df_heatmap.empty:
        top_df = df_heatmap.sort_values("risk_score", ascending=False).copy()
        if "name" not in top_df.columns:
            top_df["name"] = top_df["country"]
        st.plotly_chart(build_risk_bar_chart(top_df), use_container_width=True,
                        config={"displayModeBar": False})

    # ---- Footer ----
    st.markdown(
        "<hr style='border-color:#1a1a1a;'/>"
        "<p style='text-align:center; color:#333; font-size:11px; font-family:Courier New;'>"
        "GeoPulse Activity Monitor · Powered by GDELT · "
        "Exploratory analytics only · Not for operational, safety or targeting decisions"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
