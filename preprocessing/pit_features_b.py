"""
Phase B features: B1 (fine-grained CAMEO), B2 (actor-structural), B4 (media
amplification), B5 (geographic dispersion, partial), B7 (tone distribution) --
built from the raw gdelt_events backfill (2023-01 -> 2025-12, Step 5), not
the pre-aggregated parquet cache Phase A used.

Same PIT design as pit_labels.py and pit_features.py: for run_date t, the
"current" window is (t - 37d, t - 7d] -- a 30-day lookback ending at the
same 7-day publication-lag buffer used everywhere else in this plan. B6
("surge" temporal derivatives) is implemented narrowly here: only the B1
log-count family gets a current-vs-prior delta (prior window (t - 67d,
t - 37d]), because that is specifically the family the plan calls out as
carrying escalation visibility the percentile-ranked parquet features
cannot ("the latter makes absolute escalation visible, which parquet cannot
do"). A full B6 pass over every B-family scalar was cut to control feature
volume -- Phase A's result (Step 4) already showed that piling on features
without a clear reason each one earns its place makes a model worse, not
better; this is not a decision to revisit lightly.

Not implemented, out of scope for this pass:
  - B3 (graph scalars) -- the existing `graph` schema was built over a
    single month (June 2026) as a knowledge-graph prototype, not the
    2023-2025 backfill window this evaluation needs; rebuilding it at that
    scale is separate work, not a quick addition here.
  - B5's centroid-to-capital distance and land-border proximity -- both
    need a capital-city / border-geometry reference table this repo doesn't
    have; distinct-adm1 count and adm1 Shannon entropy (the two B5 features
    that only need action_geo_adm1, which Step 5 just added) are included.

Single set-based query (CROSS JOIN run_dates x countries, aggregated with
FILTER clauses and one CTE for the adm1 entropy), matching the anti-N+1
discipline established in pit_labels.py -- not a query per country per date.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date

import numpy as np
import pandas as pd
import psycopg2

logger = logging.getLogger("preprocessing.pit_features_b")

DSN = os.environ.get(
    "GEOPULSE_DSN",
    "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432",
)

PIT_BUFFER_DAYS = 7
WINDOW_DAYS = 30

# B1: named singleton CAMEO base codes from the plan -- riot, arrest,
# repression, assault subtypes, fight subtypes, mass violence.
B1_CODES = [145, 173, 175, 180, 181, 182, 183, 184, 185, 186,
            190, 191, 192, 193, 194, 195, 196, 200, 201, 202, 203, 204]

# B2: CAMEO actor-type sector codes relevant to conflict, per the plan.
B2_SECTORS = ["GOV", "MIL", "REB", "INS", "OPP", "COP", "CVL"]


def _b1_select(alias: str) -> str:
    parts = []
    for c in B1_CODES:
        parts.append(
            f"SUM(mentions) FILTER (WHERE base_code = {c})::float "
            f"/ NULLIF(SUM(mentions), 0) AS b1_{c}_share"
        )
        parts.append(
            f"COUNT(*) FILTER (WHERE base_code = {c}) AS b1_{c}_count"
        )
    return ",\n        ".join(parts)


def _b2_select() -> str:
    parts = []
    for s in B2_SECTORS:
        parts.append(
            f"COUNT(*) FILTER (WHERE actor1_type1 = '{s}')::float "
            f"/ NULLIF(COUNT(*) FILTER (WHERE actor1_type1 IS NOT NULL), 0) AS b2_actor1_{s.lower()}_share"
        )
        parts.append(
            f"COUNT(*) FILTER (WHERE actor2_type1 = '{s}')::float "
            f"/ NULLIF(COUNT(*) FILTER (WHERE actor2_type1 IS NOT NULL), 0) AS b2_actor2_{s.lower()}_share"
        )
    return ",\n        ".join(parts)


_PANEL_SQL = f"""
WITH run_dates AS (
    SELECT generate_series(%(start)s::date, %(end)s::date, interval '1 month')::date AS run_date
),
countries AS (
    SELECT DISTINCT fips2 FROM ucdp.gw_country_map WHERE fips2 IS NOT NULL
),
panel AS (
    SELECT r.run_date, c.fips2 FROM run_dates r CROSS JOIN countries c
),
cur_events AS (
    SELECT p.run_date, p.fips2,
           e.event_base_code AS base_code, e.num_mentions AS mentions,
           e.num_sources, e.actor1_type1, e.actor2_type1,
           e.actor1_country, e.action_geo_country, e.action_geo_adm1,
           e.avg_tone
    FROM panel p
    JOIN gdelt_events e
      ON e.action_geo_country = p.fips2
     AND e.event_date >  p.run_date - INTERVAL '{WINDOW_DAYS + PIT_BUFFER_DAYS} days'
     AND e.event_date <= p.run_date - INTERVAL '{PIT_BUFFER_DAYS} days'
),
prior_events AS (
    SELECT p.run_date, p.fips2, e.event_base_code AS base_code
    FROM panel p
    JOIN gdelt_events e
      ON e.action_geo_country = p.fips2
     AND e.event_date >  p.run_date - INTERVAL '{2 * WINDOW_DAYS + PIT_BUFFER_DAYS} days'
     AND e.event_date <= p.run_date - INTERVAL '{WINDOW_DAYS + PIT_BUFFER_DAYS} days'
),
adm1_counts AS (
    SELECT run_date, fips2, action_geo_adm1, COUNT(*) AS n
    FROM cur_events
    WHERE action_geo_adm1 IS NOT NULL
    GROUP BY 1, 2, 3
),
adm1_agg AS (
    SELECT run_date, fips2,
           COUNT(*) AS b5_n_adm1,
           SUM(n) AS b5_adm1_total
    FROM adm1_counts
    GROUP BY 1, 2
),
adm1_entropy_terms AS (
    SELECT a.run_date, a.fips2,
           -SUM((a.n::float / t.b5_adm1_total) * ln(a.n::float / t.b5_adm1_total)) AS b5_adm1_entropy
    FROM adm1_counts a
    JOIN adm1_agg t USING (run_date, fips2)
    GROUP BY a.run_date, a.fips2
),
cur_agg AS (
    SELECT
        run_date, fips2,
        COUNT(*) AS n_events_b,
        {_b1_select("cur_events")},
        {_b2_select()},
        COUNT(*) FILTER (
            WHERE (actor1_type1 = 'REB' AND actor2_type1 = 'GOV')
               OR (actor1_type1 = 'GOV' AND actor2_type1 = 'REB')
        )::float / NULLIF(COUNT(*), 0) AS b2_reb_gov_dyad_share,
        COUNT(*) FILTER (
            WHERE actor1_country IS NOT NULL AND actor1_country != action_geo_country
        )::float / NULLIF(COUNT(*) FILTER (WHERE actor1_country IS NOT NULL), 0) AS b2_crossborder_share,
        AVG(mentions) AS b4_avg_mentions,
        AVG(num_sources) AS b4_avg_sources,
        MAX(mentions) AS b4_max_mentions,
        MAX(mentions)::float / NULLIF(SUM(mentions), 0) AS b4_top1_mention_share,
        COUNT(*) FILTER (WHERE num_sources = 1)::float / NULLIF(COUNT(*), 0) AS b4_single_source_share,
        STDDEV(avg_tone) AS b7_tone_std,
        PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY avg_tone) AS b7_tone_p10,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY avg_tone) AS b7_tone_p90,
        COUNT(*) FILTER (WHERE avg_tone < -5)::float
            / NULLIF(COUNT(*) FILTER (WHERE avg_tone IS NOT NULL), 0) AS b7_tone_below_neg5_share
    FROM cur_events
    GROUP BY run_date, fips2
),
prior_agg AS (
    SELECT run_date, fips2,
        {", ".join(f"COUNT(*) FILTER (WHERE base_code = {c}) AS prior_b1_{c}_count" for c in B1_CODES)}
    FROM prior_events
    GROUP BY run_date, fips2
)
SELECT c.*, COALESCE(a.b5_n_adm1, 0) AS b5_n_adm1, e.b5_adm1_entropy,
       p.*
FROM cur_agg c
LEFT JOIN adm1_agg a USING (run_date, fips2)
LEFT JOIN adm1_entropy_terms e USING (run_date, fips2)
LEFT JOIN prior_agg p USING (run_date, fips2)
"""


def build_phase_b_panel(start: date, end: date, dsn: str = DSN) -> pd.DataFrame:
    conn = psycopg2.connect(dsn)
    try:
        df = pd.read_sql(_PANEL_SQL, conn, params={"start": start, "end": end})
    finally:
        conn.close()

    # pandas read_sql with USING-joined columns can duplicate run_date/fips2
    # column labels; drop the duplicates, keep the first (they're identical).
    df = df.loc[:, ~df.columns.duplicated()]

    # B6: log-count delta for the B1 family (the one family the plan flags
    # as carrying absolute-escalation visibility -- current vs prior window).
    for c in B1_CODES:
        cur_col, prior_col = f"b1_{c}_count", f"prior_b1_{c}_count"
        df[f"b1_{c}_log_count"] = np.log1p(df[cur_col].fillna(0))
        df[f"b1_{c}_log_count_delta"] = df[f"b1_{c}_log_count"] - np.log1p(df[prior_col].fillna(0))
        df = df.drop(columns=[cur_col, prior_col])

    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date
    return df


B_FEATURE_PREFIXES = ("b1_", "b2_", "b4_", "b5_", "b7_")


def b_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(B_FEATURE_PREFIXES)]


def attach_phase_b_to_labels(labels_df: pd.DataFrame, b_panel: pd.DataFrame) -> pd.DataFrame:
    """Exact (fips2, run_date) merge -- both are on the same monthly grid,
    unlike the parquet's bi-weekly snapshots, so no as-of join is needed."""
    return labels_df.merge(b_panel, on=["fips2", "run_date"], how="left")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    panel = build_phase_b_panel(date(2023, 1, 1), date(2025, 11, 1))
    cols = b_feature_columns(panel)
    logger.info("Phase B panel: %d rows, %d feature columns", len(panel), len(cols))
    coverage = panel[cols[0]].notna().mean() if cols else 0
    logger.info("Coverage of first feature column: %.1f%%", coverage * 100)
    out_path = "data/pit_phase_b_panel.parquet"
    panel.to_parquet(out_path)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
