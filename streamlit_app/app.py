"""
GeoPulse Streamlit Dashboard — Global Risk Intelligence Map.

Layout:
  ┌────────────────────────────────────┐
  │        GLOBAL RISK MAP             │
  │   Black → Maroon → Crimson scale   │
  └────────────────────────────────────┘
  ┌──────────────┬─────────────────────┐
  │ Country Info │ Risk Timeline       │
  └──────────────┴─────────────────────┘
  ┌────────────────────────────────────┐
  │ Latest Escalation Events           │
  └────────────────────────────────────┘
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from ui import BACKEND_URL, CONTAGION_SCALE, inject_theme, loading, metric_grid, risk_level

# NOTE: no st.set_page_config() / inject_theme() call here. Under
# st.navigation, whichever page runs must call st.set_page_config() itself
# as its own first Streamlit command (see main() below and each pages/*.py
# file) so the browser tab title updates per page instead of staying fixed.


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


@st.cache_data(ttl=300)
def fetch_escalation_alerts(min_risk: float = 0.60) -> list[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/global/escalation_alerts", params={"min_risk": min_risk}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("alerts", [])
    except Exception:
        return []


@st.cache_data(ttl=600)
def fetch_gnn_network(min_weight: float = 0.20) -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}/global/gnn_network", params={"min_weight": min_weight}, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


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
            f"Risk: {r['risk_score']:.2f}<br>"
            f"Level: {risk_level(r['risk_score'])}<br>"
            f"Confidence: {r.get('confidence', 0):.2f}<br>"
            f"Trend: {r.get('trend', 'N/A')}"
        ),
        axis=1,
    )

    fig = go.Figure(go.Choropleth(
        locations=df["country"],
        locationmode="ISO-3",
        z=df["risk_score"],
        text=df["text"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=[
            [0.0,  "#2a3f2a"],    # Low    — muted green
            [0.35, "#4a3a30"],    # Moderate — muted brown
            [0.50, "#5c3a3a"],    # Elevated
            [0.65, "#4a2e2e"],    # High
            [0.80, "#332222"],    # Critical
            [1.0,  "#1c1414"],    # Maximum — near-black, not pure black
        ],
        zmin=0.0,
        zmax=1.0,
        showscale=True,
        colorbar=dict(
            title=dict(text="Risk Score", font=dict(color="#888")),
            tickfont=dict(color="#888"),
            bgcolor="#141414",
            bordercolor="#2a2a2a",
            len=0.8,
        ),
        marker=dict(line=dict(color="#1a1a1a", width=0.5)),
    ))

    fig.update_layout(
        title=dict(
            text="GLOBAL GEOPOLITICAL RISK MONITOR",
            font=dict(size=18, color="#b2504f", family="Courier New"),
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
# Global GNN network figure (full graph — the country-specific ego view
# lives on the Country Drilldown page instead)
# ---------------------------------------------------------------------------

def build_gnn_graph_full(network_data: dict, name_map: dict[str, str] | None = None) -> go.Figure:
    nodes = network_data.get("nodes", [])
    edges = network_data.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.add_annotation(
            text="No GNN network data. Run POST /analyze/gnn to compute.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#666", size=14),
        )
        fig.update_layout(
            paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=500,
        )
        return fig

    nm = name_map or {}
    # Order the circle by network-adjusted risk (descending) so the layout
    # itself carries meaning — sweeping clockwise from riskiest to safest.
    # `.get(key, default)` only substitutes when the key is *missing* — the
    # API can return an explicit null for a scored field, so `or 0` guards
    # against that too (this crashed in production for a real country).
    nodes_sorted = sorted(nodes, key=lambda n: -float(n.get("network_adjusted_risk") or n.get("risk_score") or 0))
    n = len(nodes_sorted)
    angles = [2 * math.pi * i / n for i in range(n)]
    country_pos = {node["country"]: (math.cos(a), math.sin(a)) for node, a in zip(nodes_sorted, angles)}

    fig = go.Figure()
    for edge in edges:
        src, dst, w = edge.get("source", ""), edge.get("target", ""), float(edge.get("weight") or 0)
        if src not in country_pos or dst not in country_pos:
            continue
        x0, y0 = country_pos[src]
        x1, y1 = country_pos[dst]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=f"rgba(178,80,79,{min(0.15 + w * 0.5, 0.85):.2f})", width=max(1, w * 4)),
            hoverinfo="none", showlegend=False,
        ))

    country_codes  = [node["country"] for node in nodes_sorted]
    country_names  = [nm.get(c, c) for c in country_codes]
    x_nodes        = [country_pos[c][0] for c in country_codes]
    y_nodes        = [country_pos[c][1] for c in country_codes]
    risk_vals      = [float(node.get("network_adjusted_risk") or node.get("risk_score") or 0) for node in nodes_sorted]
    contagion_vals = [float(node.get("contagion_score") or 0) for node in nodes_sorted]
    node_sizes     = [max(7, 7 + 20 * r) for r in risk_vals]

    hover_texts = [
        f"<b>{name}</b><br>Network-adjusted risk: {r:.3f}<br>Contagion: {ct:.3f}"
        for name, r, ct in zip(country_names, risk_vals, contagion_vals)
    ]
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes, mode="markers",
        marker=dict(
            size=node_sizes, color=contagion_vals, colorscale=CONTAGION_SCALE, cmin=0, cmax=1,
            colorbar=dict(title=dict(text="Contagion", font=dict(color="#888")),
                           tickfont=dict(color="#888"), bgcolor="#141414", bordercolor="#2a2a2a", len=0.6),
            line=dict(color="#333", width=0.5),
        ),
        customdata=hover_texts, hovertemplate="%{customdata}<extra></extra>", showlegend=False,
    ))
    fig.update_layout(
        title="Full Network — arranged clockwise by risk (riskiest first)",
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.4, 1.4]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-1.4, 1.4]),
        font=dict(color="#888"), height=600, margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Global Risk Map — GeoPulse",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()

    # Header
    st.markdown(
        "<h1 style='text-align:center; font-family:Courier New; "
        "color:#b2504f; letter-spacing:3px; margin-bottom:4px;'>"
        "⬛ GLOBAL RISK INTELLIGENCE</h1>"
        "<p style='text-align:center; color:#555; font-size:13px; "
        "font-family:Courier New; margin-top:0;'>"
        f"Geopolitical Escalation Monitor · {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- World Map ----
    with loading("Loading global risk map..."):
        df_heatmap = fetch_heatmap()
    fig_map = build_choropleth(df_heatmap)
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # ---- Stats row ----
    if not df_heatmap.empty:
        critical_count = int((df_heatmap["risk_score"] >= 0.80).sum())
        high_count     = int(((df_heatmap["risk_score"] >= 0.65) & (df_heatmap["risk_score"] < 0.80)).sum())
        avg_risk       = float(df_heatmap["risk_score"].mean())

        metric_grid([
            ("Countries Tracked", str(len(df_heatmap)), ""),
            ("CRITICAL", str(critical_count), ""),
            ("HIGH", str(high_count), ""),
            ("Global Avg Risk", f"{avg_risk:.3f}", ""),
        ])

    st.divider()

    # ---- Top risk table ----
    st.markdown("### Highest Risk Countries")
    if not df_heatmap.empty:
        top_df = df_heatmap.sort_values("risk_score", ascending=False).head(20).copy()
        if "name" not in top_df.columns:
            top_df["name"] = top_df["country"]
        display_df = top_df[["name", "country", "risk_score", "confidence", "trend", "feature_date"]].copy()
        display_df["risk_score"] = display_df["risk_score"].map(lambda x: f"{x:.4f}")
        display_df["confidence"] = display_df["confidence"].map(
            lambda x: f"{x:.2f}" if x is not None else "N/A"
        )
        display_df.columns = ["Country", "Code", "Risk Score", "Confidence", "Trend", "Last Updated"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption("For a single country's full breakdown — timeline, forecast, drivers, network, advisory — use **Country Drilldown** in the sidebar.")

    st.divider()

    # ---- Global Escalation Alerts (global-only; per-country forecasts live on Country Drilldown) ----
    with st.expander("🚨 Global Escalation Alerts", expanded=False):
        alert_thresh = st.slider("Alert Threshold", 0.40, 0.90, 0.60, 0.05, key="home_alert_thresh")
        st.caption(f"Countries predicted to exceed risk {alert_thresh:.2f} in the next forecast period")
        with loading("Loading escalation alerts..."):
            alerts = fetch_escalation_alerts(alert_thresh)
        if alerts:
            df_alerts = pd.DataFrame(alerts)
            if "country" in df_alerts.columns and not df_heatmap.empty and "name" in df_heatmap.columns:
                name_lookup = dict(zip(df_heatmap["country"], df_heatmap["name"]))
                df_alerts.insert(0, "Country", df_alerts["country"].map(lambda c: name_lookup.get(c, c)))
            cols_show = [c for c in ["Country", "predicted_risk", "current_risk", "delta", "confidence", "target_date"]
                         if c in df_alerts.columns]
            if cols_show:
                st.dataframe(
                    df_alerts[cols_show].sort_values(
                        "predicted_risk" if "predicted_risk" in df_alerts.columns else cols_show[0],
                        ascending=False,
                    ).head(20),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("No escalation alerts. The forecaster may not be trained yet, or no countries exceed the threshold.")

    # ---- Global Contagion Network (global-only; per-country ego view lives on Country Drilldown) ----
    with st.expander("🕸️ Global Contagion Network", expanded=False):
        min_weight = st.slider("Min Edge Weight", 0.10, 0.60, 0.20, 0.05, key="home_gnn_min_weight")
        with loading("Loading GNN network graph..."):
            network_data = fetch_gnn_network(min_weight)

        if "error" in network_data:
            st.info("No GNN network data available. Trigger POST /analyze/gnn to compute the graph.")
        else:
            name_lookup = dict(zip(df_heatmap["country"], df_heatmap["name"])) if not df_heatmap.empty and "name" in df_heatmap.columns else {}
            fig_gnn = build_gnn_graph_full(network_data, name_lookup)
            st.plotly_chart(fig_gnn, use_container_width=True, config={"displayModeBar": True})

            nodes = network_data.get("nodes", [])
            edges = network_data.get("edges", [])
            if nodes:
                avg_contagion = sum(n.get("contagion_score") or 0 for n in nodes) / max(len(nodes), 1)
                metric_grid([
                    ("Countries (Nodes)", str(len(nodes)), ""),
                    ("Connections (Edges)", str(len(edges)), ""),
                    ("Avg Contagion Score", f"{avg_contagion:.3f}", ""),
                ])
                st.markdown("**Highest Contagion Scores** (most risk imported)")
                df_nodes = pd.DataFrame(nodes)
                if "contagion_score" in df_nodes.columns:
                    top_c = df_nodes.sort_values("contagion_score", ascending=False).head(10).copy()
                    top_c.insert(0, "Country", top_c["country"].map(lambda c: name_lookup.get(c, c)))
                    display_cols = [c for c in ["Country", "contagion_score", "risk_amplification",
                                                "network_adjusted_risk", "risk_score"] if c in top_c.columns]
                    st.dataframe(top_c[display_cols], use_container_width=True, hide_index=True)

    # ---- Footer ----
    st.markdown(
        "<hr style='border-color:#1a1a1a;'/>"
        "<p style='text-align:center; color:#333; font-size:11px; font-family:Courier New;'>"
        "GeoPulse Risk Intelligence · Powered by GDELT · "
        "Geopolitical analytics and escalation monitoring only · Not for operational targeting"
        "</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation — just two pages. Every country-level view (events,
# forecast, spillover, GNN, RAG advisory) now lives as a tab on Country
# Drilldown; only genuinely global-only views stay on the Global Risk Map
# (as expanders: Escalation Alerts, Contagion Network).
# ---------------------------------------------------------------------------

pg_home      = st.Page(main, title="Global Risk Map", icon="🌍", default=True)
pg_drilldown = st.Page("pages/01_country_drilldown.py", title="Country Drilldown", icon="🔍")

nav = st.navigation({
    "Overview": [pg_home],
    "Country Intelligence": [pg_drilldown],
})
nav.run()
