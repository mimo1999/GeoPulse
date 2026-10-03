"""Export the dashboard's data as files under assets/ for the static (GitHub Pages) build.

Home / Drilldown assets come from the real FastAPI routes (run in-process), so they match
what the backend serves. Global Intelligence assets are SQL rollups of graph.* that reproduce
the backend's per-window results without a database (see streamlit_app/data_source.py).
Needs the local Postgres.

    python scripts/export_static_assets.py                    # everything
    python scripts/export_static_assets.py --only home,gi     # subset: home, gi, counterparts
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.country_summary import latest_resolvable_event_date  # noqa: E402
from preprocessing.highlighted_events import (  # noqa: E402
    BASELINE_WINDOW_DAYS,
    MIN_BASELINE_DAYS,
    merge_dimensions,
)

OUT = ROOT / "assets"
WINDOWS = (30, 90, 180, 365)
ROLLUP_DAYS = 370          # daily rollup shipped to the browser (the longest window is 365 days)
ZSCORE_LOOKBACK_DAYS = 450  # history used to compute spike baselines (30 rows before the earliest window day)
FEED_LIMIT = 20             # per dimension, matching the API default
FEED_SHOWN = 30             # the page renders at most this many
COUNTERPARTS = 10


def _dsn() -> str:
    from backend.routers.intelligence import _load_dsn
    return _load_dsn()


# ---------------------------------------------------------------------------
# Home / Drilldown
# ---------------------------------------------------------------------------

def _snapshot_dates() -> set[str]:
    """The activity index is a series of bi-weekly snapshots (one parquet per date). Daily rows
    written by live ingestion use a different scale, so they are left out of the site."""
    return {f"{f.name[:4]}-{f.name[4:6]}-{f.name[6:8]}"
            for f in (ROOT / "data" / "real_cache").glob("*_features.parquet")}


def export_home() -> dict:
    from fastapi.testclient import TestClient
    from advisory.rule_engine import classify_risk
    from backend.main import app

    snaps = _snapshot_dates()
    with TestClient(app) as c:
        api_heat = pd.DataFrame(c.get("/global/heatmap").json()["countries"]).set_index("country")

        timelines, spill = [], []
        for country in api_heat.index:
            r = c.get(f"/country/{country}/timeline", params={"days": 365})
            if r.status_code == 200:
                df = pd.DataFrame(r.json()["timeline"])
                df = df[df["feature_date"].astype(str).isin(snaps)]
                if not df.empty:
                    timelines.append(df.assign(country=country))
            r = c.get(f"/country/{country}/spillover", params={"top_n": 6})
            if r.status_code == 200 and r.json()["neighbors"]:
                spill.append(pd.DataFrame(r.json()["neighbors"]).assign(country=country))

        timeline = pd.concat(timelines, ignore_index=True)
        timeline.to_csv(OUT / "timelines.csv", index=False)
        print(f"timelines.csv: {len(timeline)} rows, {len(timelines)} countries")

        # Latest snapshot per country; trend and name come from the API, level is re-derived.
        latest = timeline.sort_values("feature_date").groupby("country").tail(1).set_index("country")
        heat = api_heat.loc[latest.index].copy()
        heat["risk_score"] = latest["risk_score"]
        heat["confidence"] = latest["confidence"]
        heat["feature_date"] = latest["feature_date"]
        heat["level"] = heat["risk_score"].map(classify_risk)
        heat.reset_index().to_csv(OUT / "heatmap.csv", index=False)
        print(f"heatmap.csv: {len(heat)} countries, as of {heat['feature_date'].max()}")

        pd.concat(spill, ignore_index=True).to_csv(OUT / "spillover.csv", index=False)
        print(f"spillover.csv: {sum(map(len, spill))} rows, {len(spill)} countries")
    return {"home_data_as_of": str(heat["feature_date"].max())}


# ---------------------------------------------------------------------------
# Global Intelligence
# ---------------------------------------------------------------------------

_DAILY_SQL = """
    SELECT l.country_iso3, e.event_date,
           COUNT(*) AS total_events,
           COUNT(*) FILTER (WHERE e.interaction_type = 'cooperation') AS cooperation,
           COUNT(*) FILTER (WHERE e.interaction_type = 'consultation') AS consultation,
           COUNT(*) FILTER (WHERE e.interaction_type = 'conflict') AS conflict,
           COUNT(e.intensity) AS intensity_n,
           COALESCE(SUM(e.intensity), 0) AS intensity_sum,
           COALESCE(SUM(e.num_mentions), 0) AS mentions
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE l.country_iso3 IS NOT NULL AND e.source = 'gdelt' AND e.event_date >= %s
    GROUP BY 1, 2
"""

_POOL_SQL = """
    SELECT event_id, event_date, country_iso3, interaction_type, num_mentions, intensity, source_url
    FROM (
        SELECT e.event_id, e.event_date, l.country_iso3, e.interaction_type,
               e.num_mentions, e.intensity, e.source_url,
               ROW_NUMBER() OVER (PARTITION BY l.country_iso3, e.event_date ORDER BY {order}) AS rn
        FROM graph.event e
        JOIN graph.location l ON l.location_id = e.location_id
        WHERE e.source = 'gdelt' AND {notnull} AND e.event_date >= %s
    ) t WHERE rn <= %s
"""

_COLS = ["event_id", "event_date", "country_iso3", "interaction_type",
         "num_mentions", "intensity", "source_url"]


def _query(conn, sql: str, params: tuple, columns: list[str] | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = columns or [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _zscores(daily: pd.DataFrame) -> pd.DataFrame:
    """Same as highlighted_events.country_day_zscores: trailing-30-row baseline per country."""
    out = []
    for iso3, g in daily.sort_values("event_date").groupby("country_iso3"):
        n = g["total_events"].astype(float)
        prev = n.shift(1)
        mean = prev.rolling(BASELINE_WINDOW_DAYS, min_periods=MIN_BASELINE_DAYS).mean()
        std = prev.rolling(BASELINE_WINDOW_DAYS, min_periods=MIN_BASELINE_DAYS).std(ddof=1)
        z = ((n - mean) / std).where(std > 0)
        out.append(pd.DataFrame({"country_iso3": iso3, "event_date": g["event_date"], "z": z}))
    return pd.concat(out, ignore_index=True)


def _rows(df: pd.DataFrame) -> list[tuple]:
    return [(int(r.event_id), r.event_date, r.country_iso3, r.interaction_type,
             None if pd.isna(r.num_mentions) else int(r.num_mentions),
             None if pd.isna(r.intensity) else float(r.intensity), r.source_url)
            for r in df.itertuples(index=False)]


def export_gi(conn) -> dict:
    latest = latest_resolvable_event_date(conn)
    print(f"latest located event date: {latest}")
    conn.cursor().execute("SET work_mem = '512MB'")

    # --- daily rollup (heatmap + country summary) and spike baselines -------------------------
    daily_all = _query(conn, _DAILY_SQL, (latest - timedelta(days=ZSCORE_LOOKBACK_DAYS),))
    daily = daily_all[daily_all["event_date"] >= latest - timedelta(days=ROLLUP_DAYS)]
    daily.sort_values(["country_iso3", "event_date"]).to_csv(OUT / "gi_daily.csv", index=False)
    print(f"gi_daily.csv: {len(daily)} rows")
    z = _zscores(daily_all)

    countries = _query(conn, """
        SELECT DISTINCT l.country_iso3 AS iso3 FROM graph.location l
        WHERE l.country_iso3 IS NOT NULL
          AND EXISTS (SELECT 1 FROM graph.event e WHERE e.location_id = l.location_id)
        ORDER BY 1""", ())
    countries.to_csv(OUT / "gi_countries.csv", index=False)
    print(f"gi_countries.csv: {len(countries)}")

    # --- highlighted-events pools: per (country, day) top-N, then exact per-window top-N ------
    since_max = latest - timedelta(days=max(WINDOWS))
    media = _query(conn, _POOL_SQL.format(order="e.num_mentions DESC", notnull="e.num_mentions IS NOT NULL"),
                   (since_max, FEED_LIMIT), _COLS)
    gold = _query(conn, _POOL_SQL.format(order="abs(e.intensity) DESC", notnull="e.intensity IS NOT NULL"),
                  (since_max, FEED_LIMIT), _COLS)
    print(f"pools: media {len(media)}, goldstein {len(gold)}")
    gold["_abs"] = gold["intensity"].abs()
    rep = media.sort_values("num_mentions", ascending=False).drop_duplicates(["country_iso3", "event_date"])
    rep = rep.set_index(["country_iso3", "event_date"])

    out_rows = []
    scopes = [None] + countries["iso3"].tolist()
    for window in WINDOWS:
        since = latest - timedelta(days=window)
        m_w, g_w = media[media["event_date"] >= since], gold[gold["event_date"] >= since]
        z_w = z[(z["event_date"] >= since) & z["z"].notna()]
        for scope in scopes:
            m_s = m_w if scope is None else m_w[m_w["country_iso3"] == scope]
            g_s = g_w if scope is None else g_w[g_w["country_iso3"] == scope]
            z_s = z_w if scope is None else z_w[z_w["country_iso3"] == scope]
            spikes = []
            for r in z_s.nlargest(FEED_LIMIT, "z").itertuples(index=False):
                key = (r.country_iso3, r.event_date)
                if key in rep.index:
                    e = rep.loc[key]
                    spikes.append(((int(e["event_id"]), key[1], key[0], e["interaction_type"],
                                    int(e["num_mentions"]), None if pd.isna(e["intensity"]) else float(e["intensity"]),
                                    e["source_url"]), float(r.z)))
            feed = merge_dimensions(
                _rows(m_s.nlargest(FEED_LIMIT, "num_mentions")),
                _rows(g_s.nlargest(FEED_LIMIT, "_abs")),
                spikes,
            )[:FEED_SHOWN]
            for h in feed:
                out_rows.append({
                    "window": window, "scope": scope or "", "event_id": h.event_id,
                    "event_date": h.event_date, "country_iso3": h.country_iso3,
                    "interaction_type": h.interaction_type, "num_mentions": h.num_mentions,
                    "intensity": h.intensity, "source_url": h.source_url,
                    "country_day_z_score": h.country_day_z_score, "tags": "|".join(h.tags),
                })
        print(f"  window {window}d done")
    events = pd.DataFrame(out_rows)
    events.to_csv(OUT / "gi_events.csv", index=False)
    print(f"gi_events.csv: {len(events)} rows")

    return {
        "gi_latest_event_date": str(latest),
        "gi_default_since": str(latest - timedelta(days=90)),
        "windows": list(WINDOWS),
    }


# ---------------------------------------------------------------------------
# Counterparts (all-time, as in country_summary.top_counterparts)
# ---------------------------------------------------------------------------

_COUNTERPARTS_SQL = """
    WITH ev AS (
        SELECT ea.event_id,
               array_agg(DISTINCT a.country_iso3) FILTER (WHERE a.country_iso3 IS NOT NULL) AS any_c,
               array_agg(DISTINCT a.country_iso3)
                   FILTER (WHERE a.country_iso3 IS NOT NULL AND a.country_inferred = FALSE) AS explicit_c
        FROM graph.event_actor ea
        JOIN graph.actor a ON a.actor_id = ea.actor_id
        GROUP BY ea.event_id
    )
    SELECT x.c AS iso3, y.c AS counterpart, COUNT(*) AS n
    FROM ev, unnest(ev.any_c) AS x(c), unnest(ev.explicit_c) AS y(c)
    WHERE x.c <> y.c
    GROUP BY 1, 2
"""


def export_counterparts(conn) -> None:
    conn.cursor().execute("SET work_mem = '1GB'")
    df = _query(conn, _COUNTERPARTS_SQL, ())
    df = df.sort_values(["iso3", "n"], ascending=[True, False])
    df = df.groupby("iso3").head(COUNTERPARTS).copy()
    df["rank"] = df.groupby("iso3").cumcount() + 1
    df.to_csv(OUT / "gi_counterparts.csv", index=False)
    print(f"gi_counterparts.csv: {len(df)} rows, {df['iso3'].nunique()} countries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="home,gi,counterparts")
    args = parser.parse_args()
    parts = set(args.only.split(","))

    OUT.mkdir(exist_ok=True)
    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if "home" in parts:
        meta.update(export_home())
    if parts & {"gi", "counterparts"}:
        conn = psycopg2.connect(_dsn())
        conn.autocommit = True
        try:
            if "gi" in parts:
                meta.update(export_gi(conn))
            if "counterparts" in parts:
                export_counterparts(conn)
        finally:
            conn.close()

    meta["updated"] = date.today().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"meta.json: {meta}")


if __name__ == "__main__":
    main()
