"""
Global Intelligence — country
summaries, a highlighted-events feed, and an activity heatmap, all served
by backend/routers/intelligence.py (graph.* / Postgres, independent of the
parked PIT/risk-scoring pages).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from streamlit_app import data_source  # noqa: E402
from ui import inject_theme, loading, metric_grid  # noqa: E402

st.set_page_config(
    page_title="Global Intelligence — GeoPulse",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

TAG_COLORS = {
    "Where media is looking": "#4a7fb5",
    "Where people are looking": "#b2504f",
    "What no one saw coming": "#c98a2c",
}


# ---------------------------------------------------------------------------
# Data helpers (live backend when reachable, otherwise the exported assets)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def fetch_default_since() -> date:
    """Anchored to the data's own latest resolvable date (the backend's /meta),
    never wall-clock date.today(): the newest events may have no resolvable
    location and would show an empty window."""
    try:
        return data_source.gi_default_since()
    except Exception as e:
        st.warning(f"Could not fetch data-anchored default window ({e}); using a wall-clock fallback "
                   "that may show no data.")
        return date.today() - timedelta(days=90)


@st.cache_data(ttl=600)
def fetch_countries() -> list[str]:
    try:
        return data_source.gi_countries()
    except Exception as e:
        st.warning(f"Data unavailable: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_heatmap(since: str) -> pd.DataFrame:
    try:
        return data_source.gi_heatmap(since)
    except Exception as e:
        st.warning(f"Data unavailable: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_highlighted(since: str, iso3: str | None) -> list[dict]:
    try:
        return data_source.gi_highlighted(since, iso3)
    except Exception as e:
        st.warning(f"Data unavailable: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_country_summary(iso3: str, since: str, include_by_type: bool = False) -> dict | None:
    try:
        return data_source.gi_country_summary(iso3, since, include_by_type)
    except Exception as e:
        st.warning(f"Data unavailable: {e}")
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def build_activity_choropleth(df: pd.DataFrame, metric: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available", paper_bgcolor="#0d0d0d")
        return fig

    df = df.copy()
    df["text"] = df.apply(
        lambda r: (
            f"<b>{r['iso3']}</b><br>"
            f"Events: {r['total_events']}<br>"
            f"Conflict share: {(r['conflict_share'] or 0):.1%}<br>"
            f"Avg intensity: {(r['avg_intensity'] or 0):.2f}<br>"
            f"Mentions: {r['total_mentions']}"
        ),
        axis=1,
    )
    fig = go.Figure(go.Choropleth(
        locations=df["iso3"], locationmode="ISO-3", z=df[metric], text=df["text"],
        colorscale=[[0, "#1a1a1a"], [0.5, "#b2504f"], [1, "#e8b4b0"]],
        hoverinfo="text", marker_line_color="#333", marker_line_width=0.5,
        colorbar=dict(title=metric.replace("_", " ").title(), tickfont=dict(color="#aaa")),
    ))
    fig.update_layout(
        geo=dict(bgcolor="#0d0d0d", showframe=False, showcoastlines=False, projection_type="natural earth"),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d", margin=dict(l=0, r=0, t=10, b=0), height=500,
    )
    return fig


def render_highlighted_events(events: list[dict]):
    if not events:
        st.info("No highlighted events for this window.")
        return
    for e in events[:30]:
        badges = " ".join(
            f"<span style='background:{TAG_COLORS.get(t,'#555')};color:#fff;padding:2px 8px;"
            f"border-radius:10px;font-size:11px;margin-right:4px;'>{t}</span>"
            for t in e["tags"]
        )
        z = e.get("country_day_z_score")
        z_str = f" · spike z={z:.1f}" if z is not None else ""
        st.markdown(
            f"<div style='padding:8px 0;border-bottom:1px solid #333;'>"
            f"{badges}<br>"
            f"<b>{e.get('country_iso3') or '—'}</b> · {e['event_date']} · "
            f"{e.get('interaction_type') or 'untyped'} · "
            f"mentions={e.get('num_mentions') or 0} · intensity={e.get('intensity') or 0:.1f}{z_str}<br>"
            f"<a href='{e.get('source_url','')}' target='_blank' style='color:#7a9bc4;font-size:12px;'>"
            f"{(e.get('source_url') or '')[:90]}</a>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_country_summary(summary: dict, iso3: str, since: str):
    metric_grid([
        ("Total Events", str(summary["total_events"]), ""),
        ("Cooperation", str(summary["interaction_mix"].get("cooperation", 0)), ""),
        ("Consultation", str(summary["interaction_mix"].get("consultation", 0)), ""),
        ("Conflict", str(summary["interaction_mix"].get("conflict", 0)), ""),
    ])

    trend = pd.DataFrame(summary["monthly_volume"], columns=["month", "n"])
    if not trend.empty:
        fig = go.Figure(go.Scatter(x=trend["month"], y=trend["n"], mode="lines+markers",
                                    line=dict(color="#b2504f")))
        fig.update_layout(
            title="Monthly event volume", paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            font=dict(color="#ccc"), height=300, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    counterparts = summary["top_counterparts"]
    if counterparts:
        st.markdown("**Top counterpart countries** (all interaction types)")
        st.dataframe(pd.DataFrame(counterparts, columns=["Country", "Events"]), hide_index=True)
    else:
        st.caption("No resolved cross-country counterparts for this window.")

    with st.expander("By interaction type (slow, often sparse — see note)"):
        st.caption(
            "GDELT rarely gives an explicit, non-inferred country for the 'other side' of an "
            "interaction. Splitting counterparts by type frequently returns little or nothing, "
            "even when the all-types view above has solid data — this is a real data-coverage "
            "limit, not a loading error. Also "
            "slow (~50s, 3 expensive queries) -- loaded only on request, not by default."
        )
        if not data_source.backend_available():
            st.caption("Not available in the static snapshot.")
        elif st.button("Load breakdown by type", key=f"load_by_type_{iso3}"):
            with loading(f"Loading by-type breakdown for {iso3} (up to ~90s)..."):
                detailed = fetch_country_summary(iso3, since, include_by_type=True)
            if detailed:
                for itype, rows in detailed["top_counterparts_by_type"].items():
                    st.markdown(f"*{itype}*")
                    if rows:
                        st.dataframe(pd.DataFrame(rows, columns=["Country", "Events"]), hide_index=True)
                    else:
                        st.caption("— no resolved matches —")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='text-align:center; font-family:Courier New; color:#b2504f; "
    "letter-spacing:3px; margin-bottom:4px;'>🌐 GLOBAL INTELLIGENCE</h1>"
    "<p style='text-align:center; color:#555; font-size:13px; font-family:Courier New;'>"
    "Country summaries · highlighted events · activity heatmap</p>",
    unsafe_allow_html=True,
)
st.divider()

if data_source.backend_available():
    since = st.sidebar.date_input("Window start", value=fetch_default_since())
else:
    windows = data_source.gi_windows()
    days = st.sidebar.selectbox("Window", windows, index=windows.index(90) if 90 in windows else 0,
                                format_func=lambda d: f"Last {d} days")
    since = data_source.gi_since_for_window(days)
    st.sidebar.caption(
        f"Static snapshot: events through {data_source.meta().get('gi_latest_event_date')}, "
        f"exported {data_source.meta().get('updated')}."
    )
since_str = since.isoformat()

tab_heatmap, tab_events, tab_country = st.tabs(["🗺️ Activity Heatmap", "⚡ Highlighted Events", "🏳️ Country Summary"])

with tab_heatmap:
    # Fixed to the last 3 months, independent of the sidebar window (the backend's default window).
    heat_since = fetch_default_since()
    st.caption(f"Last 3 months, since {heat_since.isoformat()}")
    metric = st.selectbox("Color by", ["total_events", "conflict_share", "avg_intensity", "total_mentions"], index=1)
    with loading("Loading activity heatmap..."):
        df_heat = fetch_heatmap(heat_since.isoformat())
    st.plotly_chart(build_activity_choropleth(df_heat, metric), use_container_width=True,
                     config={"displayModeBar": False})
    if not df_heat.empty:
        metric_grid([
            ("Countries with activity", str(len(df_heat)), "last 3 months"),
            ("Avg conflict share", f"{df_heat['conflict_share'].mean():.1%}", "last 3 months"),
        ])

with tab_events:
    countries = fetch_countries()
    iso3_filter = st.selectbox("Filter by country (optional)", ["(all)"] + countries)
    iso3_filter = None if iso3_filter == "(all)" else iso3_filter
    with loading("Loading highlighted events..."):
        events = fetch_highlighted(since_str, iso3_filter)
    st.caption(
        "Every event keeps all three significance signals — media attention, Goldstein "
        "magnitude, and country-level activity spikes — tagged rather than blended into one score."
    )
    render_highlighted_events(events)

with tab_country:
    countries = fetch_countries()
    if data_source.backend_available():
        st.caption(
            "Can take up to ~60 seconds to load: every figure here is a live aggregate over "
            "the full event table. The interaction-type breakdown below is even slower and "
            "loads only on request."
        )
    if countries:
        # UKR, not USA, as the default -- USA is among the most expensive
        # queries in this dataset and a bad first impression on page load.
        default_iso3 = "UKR" if "UKR" in countries else countries[0]
        iso3 = st.selectbox("Country", countries, index=countries.index(default_iso3))
        with loading(f"Loading summary for {iso3}..."):
            summary = fetch_country_summary(iso3, since_str)
        if summary:
            render_country_summary(summary, iso3, since_str)
        else:
            st.info(f"No events found for {iso3} in this window.")
    else:
        st.warning("No countries available — is the backend running?")
