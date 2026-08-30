"""
Country Intelligence Drilldown — the single per-country deep-dive page.

Consolidates every country-level view that previously lived on its own
sub-dashboard (Event Explorer, Spillover Network, Escalation Forecast,
GNN Network, RAG Advisory) into one page as tabs, all driven by the one
country picker at the top. Only truly global-only views (the full GNN
graph, global escalation alerts) remain on the Global Risk Map page.

Tabs:
  Timeline          — risk score trend only (component lines removed —
                       hidden behind "legendonly" where nobody found them,
                       and Escalation Drivers already shows what's driving
                       the score better than five overlapping dotted lines)
  Forecast          — 4-step bi-weekly escalation forecast; per-task
                       breakdown collapsed (rarely diverges from the
                       composite line)
  Event Clusters    — GDELT event clusters, filterable by category; actor
                       interactions and the raw event log are collapsed
  Escalation Drivers — feature attribution (Integrated Gradients)
  Network           — spillover + GNN contagion ego network + inspector
                       (Spillover and GNN previously visualized the same
                       edges twice with different node coloring — merged).
                       "Top Influencers" is derived from the same filtered
                       edge list as the graph, not a separately-thresholded
                       endpoint, so the two can't disagree on the count.
  RAG Advisory      — historical analogues lead the tab (the one thing
                       here a plain risk score can't give you); the
                       narrative text is collapsed below since it mostly
                       restates the header's advisory sentence

The rule-based advisory is shown once, at the top of the page (via
risk_header, which also folds in the top risk drivers so that one line
is self-contained) — nothing below repeats it.

Proxy Labels is not a tab — it's a collapsed "Model Internals" section
below the tabs, since it answers a model-validation question, not a
risk-intelligence one.
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from ui import (
    BACKEND_URL, SELECTED_COUNTRY_KEY, CONTAGION_SCALE, country_picker,
    inject_theme, loading, metric_grid, risk_header,
)

# Default GNN edge-weight threshold. Previously a slider on every visit;
# 0.20 is a sensible fixed default and the graph is still explorable via
# hover/click, so the extra widget wasn't earning its space.
DEFAULT_MIN_EDGE_WEIGHT = 0.20

st.set_page_config(
    page_title="Country Drilldown — GeoPulse",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()
st.markdown("""
<style>
    .cluster-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .cluster-title { font-size: 13px; letter-spacing: 1px; color: #888; }
    .cluster-count { font-size: 28px; font-weight: bold; color: #b2504f; }
    .node-card {
        background: #141414; border: 1px solid #2a2a2a;
        border-radius: 6px; padding: 14px 18px; margin: 6px 0;
    }
    .node-card-desc { color: #777; font-size: 11px; margin-top: 2px; line-height: 1.5; }
    .advisory-box {
        background: #0f0f0f; border: 1px solid #2a2a2a; border-left: 3px solid #b2504f;
        border-radius: 6px; padding: 18px 22px; font-family: 'Courier New', monospace;
        font-size: 13px; line-height: 1.8; color: #ddd; white-space: pre-wrap; margin: 8px 0;
    }
    .context-card {
        background: #141414; border: 1px solid #222; border-radius: 5px;
        padding: 12px 16px; margin: 6px 0; font-size: 12px;
    }
    .similarity-badge {
        display: inline-block; padding: 2px 8px; background: #1a1a3a;
        border-radius: 3px; color: #6666cc; font-size: 11px; font-family: Courier New;
    }
</style>
""", unsafe_allow_html=True)

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
CATEGORIES = ["All", "Military", "Terrorism", "Protest", "Sanctions", "Diplomatic"]

PENDING_COUNTRY_KEY = "gp_pending_country"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def get_timeline(country: str, days: int) -> pd.DataFrame:
    try:
        resp = requests.get(f"{BACKEND_URL}/country/{country}/timeline", params={"days": days}, timeout=10)
        return pd.DataFrame(resp.json().get("timeline", []))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def get_events(country: str, days: int) -> list[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/country/{country}/events", params={"days": days}, timeout=10)
        return resp.json().get("events", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_attributions(country: str) -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/country/{country}/attributions", timeout=15)
        return resp.json()
    except Exception:
        return {}


@st.cache_data(ttl=120)
def get_risk_score(country: str) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}/riskscore", json={"country": country}, timeout=15)
        return resp.json()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_forecast(country: str) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/country/{country}/forecast", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=600)
def get_gnn_network(min_weight: float = 0.20) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/global/gnn_network", params={"min_weight": min_weight}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_gnn_influence(country: str) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/country/{country}/gnn_influence", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def get_rag_advisory(country: str) -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/country/{country}/rag_advisory", params={"include_retrieved": True}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=600)
def get_corpus_stats() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/advisory/corpus/stats", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Plain-language interpretation (GNN)
# ---------------------------------------------------------------------------

def _describe_contagion(score: float) -> str:
    if score >= 0.65:
        return "High — a large share of this country's risk position is shaped by its neighbors."
    if score >= 0.45:
        return "Moderate — this country's risk is partly shaped by its neighbors."
    return "Low — this country's risk is mostly driven by its own local signals."


def _describe_amplification(delta: float) -> str:
    if delta > 0.05:
        return f"Network effects push this country's risk UP by {delta:.3f} beyond its own model signal."
    if delta < -0.05:
        return f"Network effects pull this country's risk DOWN by {abs(delta):.3f} relative to its own model signal."
    return "Network effects have little net impact on this country's risk here."


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_multitrace_timeline(df: pd.DataFrame, country: str) -> go.Figure:
    # Previously also plotted 5 component traces (violence, protests, etc.)
    # hidden behind "legendonly" — invisible unless a viewer knew to click
    # the legend, and the Escalation Drivers tab already shows what's
    # driving the score better than five overlapping dotted lines would.
    # This chart now does one job: show the trend.
    fig = go.Figure()
    if "risk_score" in df:
        fig.add_trace(go.Scatter(
            x=df["feature_date"], y=df["risk_score"],
            name="Risk Score", line=dict(color="#b2504f", width=2.5),
            fill="tozeroy", fillcolor="rgba(204,34,34,0.08)",
        ))
    fig.update_layout(
        title=f"{country} — Risk Timeline",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888", size=11)),
        font=dict(color="#888"), height=300, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def _build_category_chart(df: pd.DataFrame, metric: str = "Mentions") -> go.Figure:
    """
    Category-over-time chart. Mentions and Intensity were previously two
    separate always-rendered charts showing the same category breakdown
    with a different y-metric; merged into one with a toggle.
    """
    fig = go.Figure()
    if metric == "Mentions":
        for cat, color in CATEGORY_COLORS.items():
            sub = df[df["category"] == cat]
            if sub.empty:
                continue
            icon = CATEGORY_ICONS.get(cat, "")
            fig.add_trace(go.Bar(
                x=sub["cluster_date"], y=sub["total_mentions"],
                name=f"{icon} {cat.title()}", marker_color=color, opacity=0.85,
            ))
        title, ytitle, barmode = "Event Mentions by Category Over Time", "Total Mentions", "stack"
    else:
        for cat, color in CATEGORY_COLORS.items():
            sub = df[df["category"] == cat]
            if sub.empty:
                continue
            y = sub["max_intensity"].abs() if "max_intensity" in sub else sub["avg_goldstein"].abs()
            fig.add_trace(go.Scatter(
                x=sub["cluster_date"], y=y, name=cat.title(),
                mode="markers+lines", line=dict(color=color, width=1), marker=dict(color=color, size=6),
            ))
        title, ytitle, barmode = "Event Intensity by Category Over Time", "|Intensity|", None

    fig.update_layout(
        barmode=barmode, title=title,
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=False, tickfont=dict(color="#888")),
        yaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888"), title=ytitle),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888", size=10)),
        font=dict(color="#888"), height=260, margin=dict(l=60, r=20, t=40, b=40),
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
        st.caption("No actor-pair data for the selected filters.")
        return
    sorted_pairs = sorted(all_pairs.items(), key=lambda x: -x[1])[:10]
    rows = [{"Actor 1": p[0], "Actor 2": p[1], "Events": c} for (p, c) in sorted_pairs]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def build_forecast_chart(country: str, forecast: dict, current_risk: dict) -> go.Figure:
    steps = forecast.get("forecasts", [])
    if not steps:
        return go.Figure()

    dates     = [s["target_date"] for s in steps]
    risk_mean = [s["risk_score"] for s in steps]
    lower     = [s["lower_bound"] for s in steps]
    upper     = [s["upper_bound"] for s in steps]

    current_date  = forecast.get("forecast_date", str(datetime.today().date()))
    current_score = current_risk.get("risk_score", None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(204, 34, 34, 0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence Interval", showlegend=True, hoverinfo="skip",
    ))
    all_x = ([current_date] + dates) if current_score is not None else dates
    all_y = ([current_score] + risk_mean) if current_score is not None else risk_mean
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y, mode="lines+markers", name="Forecast Risk",
        line=dict(color="#b2504f", width=2.5),
        marker=dict(size=[10] + [8] * len(dates),
                    color=["#c88a4a"] + ["#b2504f"] * len(dates),
                    symbol=["diamond"] + ["circle"] * len(dates)),
    ))
    for level, y, color in [("CRITICAL", 0.80, "#550000"), ("HIGH", 0.65, "#3b0000"), ("ELEVATED", 0.50, "#2a2a00")]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1,
                      annotation_text=level, annotation_position="right", annotation_font_color=color)
    fig.update_layout(
        title=f"{country} — Escalation Forecast (4-step, bi-weekly)",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888"), title="Target Date"),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888"), title="Predicted Risk Score"),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888")),
        font=dict(color="#888"), height=380, margin=dict(l=60, r=80, t=50, b=50),
    )
    return fig


def build_task_ribbon_chart(country: str, steps: list[dict]) -> go.Figure:
    dates = [s["target_date"] for s in steps]
    tasks = [
        ("instability", "Instability", "#cc4422"), ("war_probability", "War Risk", "#8b0000"),
        ("terrorism_risk", "Terrorism", "#6b2200"), ("financial_stress", "Financial", "#3a5a7a"),
    ]
    fig = go.Figure()
    for key, name, color in tasks:
        y = [s[key] for s in steps]
        fig.add_trace(go.Scatter(x=dates, y=y, name=name, mode="lines+markers",
                                  line=dict(color=color, width=1.5, dash="dot"), marker=dict(size=6)))
    fig.update_layout(
        title=f"{country} — Per-Task Forecast Breakdown",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888", size=10)),
        font=dict(color="#888"), height=260, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def build_ego_gnn_graph(center: str, network_data: dict, name_map: dict[str, str] | None = None,
                         max_neighbors: int = 14) -> go.Figure:
    nm = name_map or {}
    nodes = {n["country"]: n for n in network_data.get("nodes", [])}
    edges = network_data.get("edges", [])

    if center not in nodes:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No GNN data yet for {nm.get(center, center)}.\nRun POST /analyze/gnn or click Recompute.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color="#666", size=14),
        )
        fig.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                           xaxis=dict(visible=False), yaxis=dict(visible=False), height=520)
        return fig

    neighbor_edges = sorted(
        (e for e in edges if e.get("source") == center or e.get("target") == center),
        key=lambda e: -float(e.get("weight") or 0),
    )[:max_neighbors]

    n = len(neighbor_edges)
    positions = {center: (0.0, 0.0)}
    neighbor_codes = []
    for i, e in enumerate(neighbor_edges):
        other = e["target"] if e.get("source") == center else e["source"]
        neighbor_codes.append(other)
        angle = 2 * math.pi * i / max(n, 1)
        w = float(e.get("weight") or 0)
        radius = 1.6 - min(w, 0.9)
        positions[other] = (radius * math.cos(angle), radius * math.sin(angle))

    fig = go.Figure()
    for e in neighbor_edges:
        other = e["target"] if e.get("source") == center else e["source"]
        if other not in positions:
            continue
        w = float(e.get("weight") or 0)
        x0, y0 = positions[center]
        x1, y1 = positions[other]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=f"rgba(178,80,79,{min(0.25 + w * 0.6, 0.9):.2f})", width=max(1.5, w * 6)),
            hoverinfo="none", showlegend=False,
        ))

    all_codes = [center] + [c for c in neighbor_codes if c in positions]
    xs    = [positions[c][0] for c in all_codes]
    ys    = [positions[c][1] for c in all_codes]
    names = [nm.get(c, c) for c in all_codes]
    # .get(key, default) only substitutes when the key is *missing* — the
    # API can return an explicit null for a scored field, so `or 0` guards
    # against that too (this crashed in production for a real country).
    risk_vals      = [float(nodes.get(c, {}).get("network_adjusted_risk") or nodes.get(c, {}).get("risk_score") or 0) for c in all_codes]
    contagion_vals = [float(nodes.get(c, {}).get("contagion_score") or 0) for c in all_codes]
    sizes = [(max(16, 16 + 34 * r) if c == center else max(10, 10 + 26 * r)) for c, r in zip(all_codes, risk_vals)]
    border_colors = ["#e8c15a" if c == center else "#333" for c in all_codes]
    border_widths = [3 if c == center else 0.5 for c in all_codes]

    hover = [
        f"<b>{name}</b>{' — selected' if c == center else ''}<br>"
        f"Network-adjusted risk: {r:.3f}<br>Contagion: {ct:.3f}"
        + ("" if c == center else "<br><i>Click to inspect</i>")
        for c, name, r, ct in zip(all_codes, names, risk_vals, contagion_vals)
    ]
    customdata = [[c, h] for c, h in zip(all_codes, hover)]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(
            size=sizes, color=contagion_vals, colorscale=CONTAGION_SCALE, cmin=0, cmax=1,
            colorbar=dict(title=dict(text="Contagion", font=dict(color="#888")),
                           tickfont=dict(color="#888"), bgcolor="#141414", bordercolor="#2a2a2a", len=0.6),
            line=dict(color=border_colors, width=border_widths),
        ),
        text=names, textposition="top center", textfont=dict(color="#aaa", size=10),
        customdata=customdata, hovertemplate="%{customdata[1]}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        title=f"Ego Network — {nm.get(center, center)} and its {n} strongest connections",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(visible=False, range=[-1.9, 1.9]), yaxis=dict(visible=False, range=[-1.9, 1.9]),
        font=dict(color="#888"), height=520, margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _build_label_chart(df: pd.DataFrame, country: str) -> go.Figure:
    fig = go.Figure()
    label_styles = [
        ("instability_label", "Instability", "#b2504f"), ("war_label", "War", "#8b0000"),
        ("terrorism_label", "Terrorism", "#4b0000"), ("financial_label", "Financial", "#3a5a7a"),
    ]
    for col, name, color in label_styles:
        if col in df:
            fig.add_trace(go.Scatter(x=df["label_date"], y=df[col], name=name, line=dict(color=color, width=1.5)))
    fig.update_layout(
        title=f"{country} — Proxy Labels (Training Ground Truth)",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
        legend=dict(bgcolor="#141414", bordercolor="#2a2a2a", font=dict(color="#888", size=10)),
        font=dict(color="#888"), height=260, margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    # Apply any pending country change from a click inside a tab (e.g. the
    # GNN ego network) *before* the country_picker widget below renders —
    # Streamlit forbids writing to a widget's session_state key after that
    # widget has already been instantiated in the same run.
    pending = st.session_state.pop(PENDING_COUNTRY_KEY, None)
    if pending:
        st.session_state[SELECTED_COUNTRY_KEY] = pending

    st.markdown("<h1>🔍 Country Intelligence Drilldown</h1>", unsafe_allow_html=True)

    col_sel, col_window = st.columns([2, 1])
    with col_sel:
        country, name_map = country_picker("Select Country")
    with col_window:
        window = st.select_slider("Timeline Window", [30, 60, 90, 180, 365], value=90)

    if not country:
        return

    display_name = name_map.get(country, country)

    with loading("Scoring country..."):
        pred = get_risk_score(country)
    if pred:
        risk_header(pred, display_name)

    st.divider()

    # Proxy Labels isn't a peer tab here — it answers "what did the model
    # learn from," a model-validation question for whoever built the
    # scorer, not a risk-intelligence question for an analyst deciding
    # what to flag. It's a collapsed section below the tabs instead.
    (tab_timeline, tab_forecast, tab_events, tab_attr,
     tab_network, tab_rag) = st.tabs([
        "Timeline", "Forecast", "Event Clusters", "Escalation Drivers",
        "Network", "RAG Advisory",
    ])

    # ========== Timeline ==========
    with tab_timeline:
        with loading("Loading timeline..."):
            df_timeline = get_timeline(country, window)
        if not df_timeline.empty:
            df_timeline["feature_date"] = pd.to_datetime(df_timeline["feature_date"])
            fig = _build_multitrace_timeline(df_timeline, display_name)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning("No timeline data available.")

    # ========== Forecast ==========
    with tab_forecast:
        with loading("Computing forecast..."):
            forecast = get_forecast(country)

        if "error" in forecast or "forecasts" not in forecast:
            st.warning(
                "No forecast available for this country. Train and deploy the forecaster "
                "model first, or the country may have insufficient feature data."
            )
            if pred:
                current_rs = pred.get("risk_score")
                rs_text = f"{current_rs:.3f}" if isinstance(current_rs, (int, float)) else "N/A"
                st.info(f"Current risk score: **{rs_text}**  |  Trend: {pred.get('trend', 'N/A')}")
        else:
            steps = forecast.get("forecasts", [])
            if steps and pred:
                # Trimmed to 3 essentials — the chart right below already
                # plots every step, so a per-step metric row (previously 5
                # metrics) and a step-card grid restating the same numbers
                # as text were pure duplication. Confidence interval and
                # per-step confidence are visible on the chart itself.
                current_score = pred.get("risk_score") or 0
                final_score   = steps[-1].get("risk_score") or 0
                metric_grid([
                    ("Current Risk", f"{current_score:.3f}", ""),
                    ("Next Forecast (14d)", f"{(steps[0].get('risk_score') or 0):.3f}",
                     f"{(steps[0].get('risk_score') or 0) - current_score:+.3f}"),
                    ("Final Forecast (56d)", f"{final_score:.3f}", f"{final_score - current_score:+.3f}"),
                ])

            st.plotly_chart(build_forecast_chart(display_name, forecast, pred or {}),
                             use_container_width=True, config={"displayModeBar": False})

            # Collapsed: the 4 task lines are usually near-parallel (every
            # task elevated together), so this rarely adds information over
            # the composite chart above — only worth opening when a task
            # diverges from the others.
            with st.expander("Per-Task Breakdown (instability / war / terrorism / financial)"):
                st.plotly_chart(build_task_ribbon_chart(display_name, steps),
                                 use_container_width=True, config={"displayModeBar": False})

    # ========== Event Clusters ==========
    with tab_events:
        col_cat, col_metric = st.columns([2, 1])
        with col_cat:
            category = st.selectbox("Category", CATEGORIES, key="drilldown_category")
        with col_metric:
            chart_metric = st.radio("Chart metric", ["Mentions", "Intensity"], horizontal=True, key="drilldown_ev_metric")
        with loading("Loading event clusters..."):
            events = get_events(country, days=window)
        if category != "All":
            events = [e for e in events if e.get("category") == category.lower()]

        if events:
            df_ev = pd.DataFrame(events)
            df_ev["cluster_date"] = pd.to_datetime(df_ev["cluster_date"])

            cat_summary = (
                df_ev.groupby("category")
                .agg(total_events=("event_count", "sum"), total_mentions=("total_mentions", "sum"),
                     avg_goldstein=("avg_goldstein", "mean"))
                .reset_index()
            )
            cols = st.columns(min(len(cat_summary), 5))
            for i, row in cat_summary.iterrows():
                cat, color = row["category"], CATEGORY_COLORS.get(row["category"], "#555")
                icon = CATEGORY_ICONS.get(cat, "📌")
                goldstein = row["avg_goldstein"]
                goldstein_text = f"{goldstein:.2f}" if pd.notna(goldstein) else "N/A"
                with cols[i % 5]:
                    st.markdown(
                        f'<div class="cluster-card" style="border-left: 3px solid {color};">'
                        f'<div class="cluster-title">{icon} {cat.upper()}</div>'
                        f'<div class="cluster-count">{int(row["total_events"])}</div>'
                        f'<div style="color:#666; font-size:12px;">'
                        f'{int(row["total_mentions"]):,} mentions<br/>'
                        f'Goldstein: {goldstein_text}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            st.plotly_chart(_build_category_chart(df_ev, chart_metric), use_container_width=True, config={"displayModeBar": False})

            # Collapsed: a niche question ("which two actors interact most")
            # that's frequently empty and rarely changes what an analyst
            # does next — kept available, not forced on every visit.
            with st.expander("Top Actor Interactions"):
                _show_actor_pairs(events)

            with st.expander("Event Log"):
                display_cols = [c for c in ["cluster_date", "category", "event_count", "total_mentions",
                                             "avg_goldstein", "avg_tone", "max_intensity"] if c in df_ev.columns]
                display_df = df_ev[display_cols].copy()
                display_df["cluster_date"] = display_df["cluster_date"].dt.date
                for col in ["avg_goldstein", "avg_tone", "max_intensity"]:
                    if col in display_df:
                        display_df[col] = display_df[col].round(3)
                display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No event cluster data for the selected filters.")

    # ========== Escalation Drivers ==========
    with tab_attr:
        with loading("Computing feature attributions..."):
            attr_data = get_attributions(country)

        if attr_data.get("attributions"):
            attr = attr_data["attributions"]
            sorted_pairs = sorted(attr.items(), key=lambda x: abs(x[1]), reverse=True)
            names_s, values_s = zip(*sorted_pairs) if sorted_pairs else ([], [])
            colors = ["#b2504f" if v > 0 else "#2255cc" for v in values_s]

            fig_attr = go.Figure(go.Bar(
                x=list(values_s), y=list(names_s), orientation="h", marker_color=colors,
                text=[f"{v:+.4f}" for v in values_s], textposition="outside",
            ))
            fig_attr.update_layout(
                title="Feature Attribution (Integrated Gradients) — Red: increases risk",
                paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                xaxis=dict(showgrid=True, gridcolor="#1a1a1a", tickfont=dict(color="#888")),
                yaxis=dict(tickfont=dict(color="#ccc")),
                font=dict(color="#888"), height=300, margin=dict(l=150, r=80, t=40, b=20),
            )
            st.plotly_chart(fig_attr, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info(f"ℹ️ {attr_data.get('message', 'Attributions not available.')}")

    # ========== Network ==========
    # Merged Spillover + GNN Network: both previously drew an ego graph
    # over the *same* country_spillover edges, just colored by a different
    # score (raw risk vs. GNN contagion) — one graph now, using the richer
    # GNN-enriched data (a superset: it carries risk, network-adjusted
    # risk, and contagion together per node).
    with tab_network:
        with st.expander("ℹ️ How to read this graph", expanded=False):
            st.markdown(
                "- **Dot size** = risk *after* accounting for neighboring countries.\n"
                "- **Dot color** = *contagion* — how much of that risk is imported from "
                "neighbors rather than generated locally (darker red = more imported).\n"
                "- **Lines** = spillover connections; thicker/brighter = stronger.\n"
                "- **Gold ring** marks the selected country. **Click any other dot** to "
                "jump the whole page to that country."
            )

        with loading("Loading network..."):
            network_data = get_gnn_network(DEFAULT_MIN_EDGE_WEIGHT)

        if "error" in network_data:
            st.info("No network data available. Trigger POST /analyze/gnn to compute it.")
            with st.expander("Admin"):
                if st.button("🔄 Recompute GNN"):
                    try:
                        r = requests.post(f"{BACKEND_URL}/analyze/gnn", json={}, timeout=10)
                        st.success("GNN computation triggered (background)") if r.ok else st.error(f"Error: {r.status_code}")
                    except Exception as e:
                        st.error(str(e))
                    st.cache_data.clear()
        else:
            fig = build_ego_gnn_graph(country, network_data, name_map)
            event = st.plotly_chart(
                fig, use_container_width=True, config={"displayModeBar": True},
                on_select="rerun", selection_mode="points", key="gnn_ego_chart",
            )
            points = []
            if event:
                sel = event.get("selection") if hasattr(event, "get") else getattr(event, "selection", None)
                if sel:
                    points = sel.get("points") if hasattr(sel, "get") else getattr(sel, "points", [])
            if points:
                cd = points[0].get("customdata") if hasattr(points[0], "get") else getattr(points[0], "customdata", None)
                clicked_code = cd[0] if isinstance(cd, (list, tuple)) and cd else None
                if clicked_code and clicked_code != country:
                    st.session_state[PENDING_COUNTRY_KEY] = clicked_code
                    st.rerun()

            with st.expander("Admin"):
                if st.button("🔄 Recompute GNN"):
                    try:
                        r = requests.post(f"{BACKEND_URL}/analyze/gnn", json={}, timeout=10)
                        st.success("GNN computation triggered (background)") if r.ok else st.error(f"Error: {r.status_code}")
                    except Exception as e:
                        st.error(str(e))
                    st.cache_data.clear()

        with loading("Loading network enrichment..."):
            gnn = get_gnn_influence(country)
        if "error" in gnn:
            st.info("No network enrichment data available for this country yet.")
        else:
            contagion = gnn.get("contagion_score") or 0
            amplif    = gnn.get("risk_amplification") or 0
            adj_risk  = gnn.get("network_adjusted_risk") or 0
            st.markdown(
                f'<div class="node-card">'
                f'<div style="font-family:Courier New; font-size:11px; color:#555;">GNN ENRICHMENT — {display_name}</div>'
                f'<br/>'
                f'<b style="color:#8888ff">Contagion Score</b><br/>'
                f'<div style="font-size:22px; color:#8888ff;">{contagion:.3f}</div>'
                f'<div class="node-card-desc">{_describe_contagion(contagion)}</div>'
                f'<br/>'
                f'<b style="color:#cc6622">Risk Amplification</b><br/>'
                f'<div style="font-size:22px; color:#cc6622;">{amplif:+.3f}</div>'
                f'<div class="node-card-desc">{_describe_amplification(amplif)}</div>'
                f'<br/>'
                f'<b style="color:#b2504f">Network-Adjusted Risk</b><br/>'
                f'<div style="font-size:22px; color:#b2504f;">{adj_risk:.3f}</div>'
                f'<div class="node-card-desc">This country\'s risk score once its network neighborhood is taken into account.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Built from the *same* filtered/sorted edges as the graph above
            # (network_data), not the separate gnn_influence endpoint's own
            # top_influencers list — those two used independent thresholds,
            # so the graph could say "2 strongest connections" while this
            # list showed 3 different countries. Same source now, so the
            # count and names here always match what's drawn above.
            if "error" not in network_data:
                neighbor_edges = sorted(
                    (e for e in network_data.get("edges", [])
                     if e.get("source") == country or e.get("target") == country),
                    key=lambda e: -float(e.get("weight") or 0),
                )
                if neighbor_edges:
                    st.markdown("**Top Influencers**")
                    for e in neighbor_edges[:5]:
                        other = e["target"] if e.get("source") == country else e["source"]
                        other_name = name_map.get(other, other)
                        st.markdown(
                            f"<div style='font-family:Courier New; font-size:12px; color:#888; margin:3px 0;'>"
                            f"→ <b style='color:#ccc'>{other_name}</b>  "
                            f"<span style='color:#555'>w={(e.get('weight') or 0):.3f}</span></div>",
                            unsafe_allow_html=True,
                        )

    # ========== RAG Advisory ==========
    # The rule-based advisory is already shown once at the top of the page
    # (risk_header renders pred["advisory"]), so this tab no longer repeats
    # it in a side-by-side comparison. Retrieved analogues lead the tab —
    # "this looks like situation X" is the one thing here a plain risk
    # score can't give you; the RAG narrative text mostly restates the
    # header sentence, so it's collapsed below instead of leading.
    with tab_rag:
        with loading("Generating advisory..."):
            rag_data = get_rag_advisory(country)

        retrieved = rag_data.get("retrieved_contexts", []) if "error" not in rag_data else []
        if retrieved:
            st.markdown("#### Similar Risk Profiles")
            st.caption(f"Top-{len(retrieved)} profiles retrieved by TF-IDF cosine similarity — synthetic templates unless marked historical")
            for ctx in retrieved:
                rl = ctx.get("risk_level", "")
                badge_color = {"CRITICAL": "#cc0000", "HIGH": "#8b0000", "ELEVATED": "#6b3300",
                               "MODERATE": "#4a4a00", "LOW": "#1a3a1a"}.get(rl, "#333")
                source = ctx.get("source", "seed_template")
                if source == "ucdp_ged":
                    source_badge = '<span style="color:#4a9; font-size:10px; border:1px solid #4a9; padding:1px 4px; border-radius:3px;">&#x2022; historical</span>'
                elif source == "event_cluster":
                    source_badge = '<span style="color:#69a; font-size:10px; border:1px solid #69a; padding:1px 4px; border-radius:3px;">cluster</span>'
                else:
                    source_badge = '<span style="color:#555; font-size:10px; border:1px solid #444; padding:1px 4px; border-radius:3px;">template</span>'
                st.markdown(
                    f'<div class="context-card">'
                    f'<span class="similarity-badge">sim={(ctx.get("similarity") or 0):.3f}</span>'
                    f'&nbsp;<span style="color:{badge_color}; font-weight:bold; font-size:11px;">{rl}</span>'
                    f'&nbsp;<span style="color:#666; font-size:11px;">{ctx.get("situation_type", "").replace("_", " ").title()}</span>'
                    f'&nbsp;{source_badge}'
                    f'<br/><br/><span style="color:#bbb;">{ctx.get("text", "")}</span><br/>'
                    f'<span style="color:#444; font-size:10px;">{" · ".join(ctx.get("tags", [])[:6])}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif "error" not in rag_data:
            st.info("No analogues retrieved. The TF-IDF corpus may need rebuilding.")

        with st.expander("RAG-Enhanced Advisory (narrative text)"):
            if "error" in rag_data:
                st.warning(f"RAG unavailable: {rag_data['error']}")
            else:
                st.markdown(f'<div class="advisory-box">{rag_data.get("advisory", "No advisory generated.")}</div>', unsafe_allow_html=True)
                if rag_data.get("rag_confidence"):
                    st.caption(f"RAG confidence: {rag_data['rag_confidence']:.2f}")

        with st.expander("Corpus Management"):
            stats = get_corpus_stats()
            if stats:
                st.caption(f"Seed entries: {stats.get('seed_entries', 'N/A')}  ·  DB entries: {stats.get('db_entries', 'N/A')}")
            if st.button("🔄 Rebuild Corpus"):
                try:
                    r = requests.post(f"{BACKEND_URL}/advisory/corpus/rebuild", json={}, timeout=15)
                    st.success(f"Corpus rebuilt: {r.json().get('corpus_size', '?')} entries") if r.ok else st.error(f"Error {r.status_code}")
                except Exception as e:
                    st.error(str(e))
                st.cache_data.clear()

    # "Key Risk Drivers" removed here — it duplicated pred["major_drivers"],
    # which risk_header() now folds directly into the advisory sentence at
    # the top of the page instead of requiring a click to find.

    # ========== Model Internals (Proxy Labels) ==========
    # Demoted from a peer tab: this shows the training ground-truth labels
    # the model learned from — a model-validation question, not something
    # that changes what an analyst flags. Available, not front-and-center.
    st.divider()
    with st.expander("🔧 Model Internals — Proxy Labels (training ground truth)"):
        try:
            resp = requests.get(f"{BACKEND_URL}/country/{country}/labels", params={"days": window}, timeout=10)
            labels = resp.json().get("labels", [])
            if labels:
                df_labels = pd.DataFrame(labels)
                df_labels["label_date"] = pd.to_datetime(df_labels["label_date"])
                df_labels = df_labels.sort_values("label_date")
                st.plotly_chart(_build_label_chart(df_labels, display_name), use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No labels generated yet. Run POST /analyze/labels.")
        except Exception:
            st.info("Labels unavailable.")


main()
