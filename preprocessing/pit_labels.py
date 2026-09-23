"""
Point-in-time UCDP label generator -> ucdp.country_pit_labels, ucdp.dyad_pit_labels.

For each (country, run_date) on a monthly grid:

    current window  = (run_date, run_date + 30d]   -- the label
    prior window     = (run_date - 30d, run_date]   -- needed for escalation_delta
    trailing 12m      = (run_date - 365d, run_date]  -- dyad candidate set

PIT discipline lives entirely in the SQL windows above: every aggregate that
becomes part of a *label* is drawn strictly from date_start > run_date. The
"prior"/trailing aggregates describe history up to and including the run
date and are never fed forward as if they were future information -- they
exist to make escalation_delta and the dyad candidate set computable, and
later double as A3 features (preprocessing/pit_features.py), which is exactly
why country_pit_labels stores the raw aggregates alongside the derived
targets: a feature-side consumer and a label-side consumer read the same
numbers under the same PIT boundary.

Base filter: date_prec <= date_prec_max (default 4, excludes date_prec=5
"within the year" precision) and event_clarity IN clarity_filter (default
just '1' = Clear; pass clarity_filter=(1,2) for the sensitivity run named in
the plan).

Writes a row for every (fips2, run_date) in scope -- including all zeros --
so a missing row is never mistaken for a true negative. Single set-based
queries (CROSS JOIN run_dates x countries, aggregated with one JOIN each),
not one query per country per date -- that N+1 pattern (fixed in feature_extractor.py) is not worth repeating here.

Usage:
    python -m preprocessing.pit_labels --start 2015-01-01 --end 2025-11-01
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from datetime import date, timedelta
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger("preprocessing.pit_labels")

DSN = os.environ.get(
    "GEOPULSE_DSN",
    "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432",
)

LABEL_VERSION = "ged-v26.1"
ESCALATION_THRESHOLD = 0.5   # log1p-delta magnitude for up/down vs flat
MINOR_THRESHOLD = 25         # UCDP's own "minor armed conflict" line
WAR_THRESHOLD = 100          # ~1,000/yr war line, annualized to a month
ONSET_QUIET_MONTHS = 3       # months with no minor_conflict required before an "onset"


# ---------------------------------------------------------------------------
# Country-level panel
# ---------------------------------------------------------------------------

_COUNTRY_PANEL_SQL = """
WITH run_dates AS (
    SELECT generate_series(%(start)s::date, %(end)s::date, interval '1 month')::date AS run_date
),
countries AS (
    SELECT DISTINCT gw_id, fips2 FROM ucdp.gw_country_map WHERE fips2 IS NOT NULL
),
panel AS (
    SELECT r.run_date, c.gw_id, c.fips2 FROM run_dates r CROSS JOIN countries c
),
cur_agg AS (
    SELECT p.run_date, p.gw_id,
           COUNT(*)                                              AS n_events,
           COALESCE(SUM(e.best), 0)                               AS deaths_best,
           COALESCE(SUM(e.low), 0)                                AS deaths_low,
           COALESCE(SUM(e.high), 0)                                AS deaths_high,
           COALESCE(SUM(e.deaths_civilians), 0)                    AS deaths_civilians,
           COALESCE(SUM(e.best) FILTER (WHERE e.type_of_violence = 1), 0) AS deaths_sb,
           COALESCE(SUM(e.best) FILTER (WHERE e.type_of_violence = 2), 0) AS deaths_ns,
           COALESCE(SUM(e.best) FILTER (WHERE e.type_of_violence = 3), 0) AS deaths_os,
           COUNT(*) FILTER (WHERE e.type_of_violence = 1)          AS n_events_sb,
           COUNT(*) FILTER (WHERE e.type_of_violence = 2)          AS n_events_ns,
           COUNT(*) FILTER (WHERE e.type_of_violence = 3)          AS n_events_os,
           COUNT(DISTINCT e.dyad_new_id)                           AS n_dyads,
           COUNT(DISTINCT e.adm_1)                                 AS n_adm1
    FROM panel p
    JOIN ucdp.ged_events e
      ON e.country_id = p.gw_id
     AND e.date_start >  p.run_date
     AND e.date_start <= p.run_date + INTERVAL '30 days'
     AND e.date_prec <= %(date_prec_max)s
     AND e.event_clarity = ANY(%(clarity)s)
    GROUP BY p.run_date, p.gw_id
),
prev_agg AS (
    SELECT p.run_date, p.gw_id,
           COUNT(*)                     AS prev_n_events,
           COALESCE(SUM(e.best), 0)      AS prev_deaths_best
    FROM panel p
    JOIN ucdp.ged_events e
      ON e.country_id = p.gw_id
     AND e.date_start >  p.run_date - INTERVAL '30 days'
     AND e.date_start <= p.run_date
     AND e.date_prec <= %(date_prec_max)s
     AND e.event_clarity = ANY(%(clarity)s)
    GROUP BY p.run_date, p.gw_id
)
SELECT
    p.run_date, p.gw_id, p.fips2,
    COALESCE(cur.n_events, 0), COALESCE(cur.deaths_best, 0), COALESCE(cur.deaths_low, 0),
    COALESCE(cur.deaths_high, 0), COALESCE(cur.deaths_civilians, 0),
    COALESCE(cur.deaths_sb, 0), COALESCE(cur.deaths_ns, 0), COALESCE(cur.deaths_os, 0),
    COALESCE(cur.n_events_sb, 0), COALESCE(cur.n_events_ns, 0), COALESCE(cur.n_events_os, 0),
    COALESCE(cur.n_dyads, 0), COALESCE(cur.n_adm1, 0),
    COALESCE(prev.prev_deaths_best, 0), COALESCE(prev.prev_n_events, 0)
FROM panel p
LEFT JOIN cur_agg  cur  ON cur.run_date = p.run_date AND cur.gw_id = p.gw_id
LEFT JOIN prev_agg prev ON prev.run_date = p.run_date AND prev.gw_id = p.gw_id
ORDER BY p.gw_id, p.run_date
"""

_COUNTRY_COLUMNS = [
    "fips2", "gw_id", "run_date", "label_version",
    "n_events", "deaths_best", "deaths_low", "deaths_high", "deaths_civilians",
    "deaths_sb", "deaths_ns", "deaths_os", "n_events_sb", "n_events_ns", "n_events_os",
    "n_dyads", "n_adm1", "prev_deaths_best", "prev_n_events",
    "escalation_delta", "escalation_dir", "civ_share",
    "any_violence", "minor_conflict", "war_intensity", "sb_minor", "ns_minor", "os_minor",
    "onset_3m", "months_since_last_minor",
    "date_prec_max", "clarity_filter",
]

_UPSERT_COUNTRY_SQL = f"""
    INSERT INTO ucdp.country_pit_labels ({", ".join(_COUNTRY_COLUMNS)})
    VALUES %s
    ON CONFLICT (fips2, run_date, label_version) DO UPDATE SET
        n_events = EXCLUDED.n_events, deaths_best = EXCLUDED.deaths_best,
        deaths_low = EXCLUDED.deaths_low, deaths_high = EXCLUDED.deaths_high,
        deaths_civilians = EXCLUDED.deaths_civilians,
        deaths_sb = EXCLUDED.deaths_sb, deaths_ns = EXCLUDED.deaths_ns, deaths_os = EXCLUDED.deaths_os,
        n_events_sb = EXCLUDED.n_events_sb, n_events_ns = EXCLUDED.n_events_ns, n_events_os = EXCLUDED.n_events_os,
        n_dyads = EXCLUDED.n_dyads, n_adm1 = EXCLUDED.n_adm1,
        prev_deaths_best = EXCLUDED.prev_deaths_best, prev_n_events = EXCLUDED.prev_n_events,
        escalation_delta = EXCLUDED.escalation_delta, escalation_dir = EXCLUDED.escalation_dir,
        civ_share = EXCLUDED.civ_share,
        any_violence = EXCLUDED.any_violence, minor_conflict = EXCLUDED.minor_conflict,
        war_intensity = EXCLUDED.war_intensity,
        sb_minor = EXCLUDED.sb_minor, ns_minor = EXCLUDED.ns_minor, os_minor = EXCLUDED.os_minor,
        onset_3m = EXCLUDED.onset_3m, months_since_last_minor = EXCLUDED.months_since_last_minor,
        date_prec_max = EXCLUDED.date_prec_max, clarity_filter = EXCLUDED.clarity_filter,
        computed_at = NOW()
"""


def _escalation(cur_deaths: int, prev_deaths: int) -> tuple[float, int]:
    delta = math.log1p(cur_deaths) - math.log1p(prev_deaths)
    if delta > ESCALATION_THRESHOLD:
        d = 1
    elif delta < -ESCALATION_THRESHOLD:
        d = -1
    else:
        d = 0
    return round(delta, 6), d


def build_country_rows(
    conn, start: date, end: date, date_prec_max: int = 4, clarity_filter: tuple[int, ...] = (1,),
) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            _COUNTRY_PANEL_SQL,
            {
                "start": start, "end": end,
                "date_prec_max": date_prec_max, "clarity": list(clarity_filter),
            },
        )
        raw = cur.fetchall()

    clarity_str = ",".join(str(c) for c in clarity_filter)

    # Sequential state per country for onset_3m / months_since_last_minor --
    # these describe trajectory over time and can't be computed in the
    # set-based query above. raw is already ORDER BY gw_id, run_date.
    rows: list[tuple] = []
    last_minor_run_date: dict[int, date] = {}
    minor_history: dict[int, list[bool]] = {}  # trailing minor_conflict flags per country

    for r in raw:
        (run_date, gw_id, fips2, n_events, deaths_best, deaths_low, deaths_high,
         deaths_civilians, deaths_sb, deaths_ns, deaths_os, n_events_sb, n_events_ns,
         n_events_os, n_dyads, n_adm1, prev_deaths_best, prev_n_events) = r

        escalation_delta, escalation_dir = _escalation(deaths_best, prev_deaths_best)
        civ_share = round(deaths_civilians / deaths_best, 6) if deaths_best >= MINOR_THRESHOLD else None

        any_violence = n_events > 0
        minor_conflict = deaths_best >= MINOR_THRESHOLD
        war_intensity = deaths_best >= WAR_THRESHOLD
        sb_minor = deaths_sb >= MINOR_THRESHOLD
        ns_minor = deaths_ns >= MINOR_THRESHOLD
        os_minor = deaths_os >= MINOR_THRESHOLD

        hist = minor_history.setdefault(gw_id, [])
        quiet_before = len(hist) >= ONSET_QUIET_MONTHS and not any(hist[-ONSET_QUIET_MONTHS:])
        onset_3m = bool(minor_conflict and quiet_before) if len(hist) >= ONSET_QUIET_MONTHS else None
        hist.append(minor_conflict)

        if minor_conflict:
            months_since_last_minor = 0
            last_minor_run_date[gw_id] = run_date
        elif gw_id in last_minor_run_date:
            months_since_last_minor = round((run_date - last_minor_run_date[gw_id]).days / 30.44)
        else:
            months_since_last_minor = None

        rows.append((
            fips2, gw_id, run_date, LABEL_VERSION,
            n_events, deaths_best, deaths_low, deaths_high, deaths_civilians,
            deaths_sb, deaths_ns, deaths_os, n_events_sb, n_events_ns, n_events_os,
            n_dyads, n_adm1, prev_deaths_best, prev_n_events,
            escalation_delta, escalation_dir, civ_share,
            any_violence, minor_conflict, war_intensity, sb_minor, ns_minor, os_minor,
            onset_3m, months_since_last_minor,
            date_prec_max, clarity_str,
        ))
    return rows


# ---------------------------------------------------------------------------
# Dyad-level panel (P2)
# ---------------------------------------------------------------------------

_DYAD_CANDIDATE_SQL = """
WITH run_dates AS (
    SELECT generate_series(%(start)s::date, %(end)s::date, interval '1 month')::date AS run_date
),
countries AS (
    SELECT DISTINCT gw_id, fips2 FROM ucdp.gw_country_map WHERE fips2 IS NOT NULL
),
panel AS (
    SELECT r.run_date, c.gw_id, c.fips2 FROM run_dates r CROSS JOIN countries c
),
trailing_dyads AS (
    SELECT p.run_date, p.gw_id, p.fips2, e.dyad_new_id,
           MAX(e.dyad_name) AS dyad_name, MAX(e.side_a) AS side_a, MAX(e.side_b) AS side_b,
           MAX(e.date_start) AS last_event_date
    FROM panel p
    JOIN ucdp.ged_events e
      ON e.country_id = p.gw_id
     AND e.date_start >  p.run_date - INTERVAL '365 days'
     AND e.date_start <= p.run_date
     AND e.date_prec <= %(date_prec_max)s
     AND e.event_clarity = ANY(%(clarity)s)
    GROUP BY p.run_date, p.gw_id, p.fips2, e.dyad_new_id
),
current_active AS (
    SELECT p.run_date, p.gw_id, e.dyad_new_id,
           MAX(e.dyad_name) AS dyad_name, MAX(e.side_a) AS side_a, MAX(e.side_b) AS side_b,
           COALESCE(SUM(e.best), 0) AS deaths_best
    FROM panel p
    JOIN ucdp.ged_events e
      ON e.country_id = p.gw_id
     AND e.date_start >  p.run_date
     AND e.date_start <= p.run_date + INTERVAL '30 days'
     AND e.date_prec <= %(date_prec_max)s
     AND e.event_clarity = ANY(%(clarity)s)
    GROUP BY p.run_date, p.gw_id, e.dyad_new_id
)
SELECT
    COALESCE(t.fips2, (SELECT fips2 FROM countries WHERE gw_id = c.gw_id)) AS fips2,
    COALESCE(t.run_date, c.run_date)     AS run_date,
    COALESCE(t.gw_id, c.gw_id)           AS gw_id,
    COALESCE(t.dyad_new_id, c.dyad_new_id) AS dyad_new_id,
    COALESCE(t.dyad_name, c.dyad_name), COALESCE(t.side_a, c.side_a), COALESCE(t.side_b, c.side_b),
    (t.dyad_new_id IS NOT NULL)          AS in_candidate_set,
    (c.dyad_new_id IS NOT NULL)          AS active,
    COALESCE(c.deaths_best, 0)           AS deaths_best,
    t.last_event_date
FROM trailing_dyads t
FULL OUTER JOIN current_active c
  ON c.run_date = t.run_date AND c.gw_id = t.gw_id AND c.dyad_new_id = t.dyad_new_id
ORDER BY 3, 4, 2
"""

_DYAD_COLUMNS = [
    "fips2", "run_date", "dyad_new_id", "label_version",
    "dyad_name", "side_a", "side_b",
    "in_candidate_set", "active", "deaths_best", "is_new_dyad",
    "months_since_last_active", "active_months_trailing12",
]

_UPSERT_DYAD_SQL = f"""
    INSERT INTO ucdp.dyad_pit_labels ({", ".join(_DYAD_COLUMNS)})
    VALUES %s
    ON CONFLICT (fips2, run_date, dyad_new_id, label_version) DO UPDATE SET
        dyad_name = EXCLUDED.dyad_name, side_a = EXCLUDED.side_a, side_b = EXCLUDED.side_b,
        in_candidate_set = EXCLUDED.in_candidate_set, active = EXCLUDED.active,
        deaths_best = EXCLUDED.deaths_best, is_new_dyad = EXCLUDED.is_new_dyad,
        months_since_last_active = EXCLUDED.months_since_last_active,
        active_months_trailing12 = EXCLUDED.active_months_trailing12,
        computed_at = NOW()
"""


def build_dyad_rows(
    conn, start: date, end: date, date_prec_max: int = 4, clarity_filter: tuple[int, ...] = (1,),
) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(
            _DYAD_CANDIDATE_SQL,
            {
                "start": start, "end": end,
                "date_prec_max": date_prec_max, "clarity": list(clarity_filter),
            },
        )
        raw = cur.fetchall()

    # Sequential state per (gw_id, dyad_new_id): active-month history to
    # compute months_since_last_active / active_months_trailing12. raw is
    # ORDER BY gw_id, dyad_new_id, run_date.
    rows: list[tuple] = []
    active_history: dict[tuple[int, int], list[tuple[date, bool]]] = {}

    for r in raw:
        (fips2, run_date, gw_id, dyad_new_id, dyad_name, side_a, side_b,
         in_candidate_set, active, deaths_best, last_event_date) = r

        key = (gw_id, dyad_new_id)
        hist = active_history.setdefault(key, [])

        active_months_trailing12 = sum(
            1 for d, a in hist if a and d > run_date - timedelta(days=365)
        )
        prior_active_dates = [d for d, a in hist if a]
        months_since_last_active = (
            round((run_date - prior_active_dates[-1]).days / 30.44) if prior_active_dates else None
        )
        # "New dyad" per the schema: active this window, unseen in this
        # month's own trailing-12m candidate set. Deliberately a per-window
        # definition, not lifetime-since-2015 -- a dyad quiet for 13+ months
        # counts as new again if it reactivates, which is the right read for
        # a rolling monitoring feature (months_since_last_active already
        # carries the "this isn't really new" information for that case).
        is_new_dyad = bool(active and not in_candidate_set)

        rows.append((
            fips2, run_date, dyad_new_id, LABEL_VERSION,
            dyad_name, side_a, side_b,
            bool(in_candidate_set), bool(active), int(deaths_best), is_new_dyad,
            months_since_last_active, active_months_trailing12,
        ))

        hist.append((run_date, bool(active)))

    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(start: date, end: date, dsn: str = DSN,
        date_prec_max: int = 4, clarity_filter: tuple[int, ...] = (1,)) -> dict[str, int]:
    conn = psycopg2.connect(dsn)
    try:
        logger.info("Building country panel %s -> %s ...", start, end)
        country_rows = build_country_rows(conn, start, end, date_prec_max, clarity_filter)
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _UPSERT_COUNTRY_SQL, country_rows, page_size=2000)
        conn.commit()
        logger.info("country_pit_labels: %d rows", len(country_rows))

        logger.info("Building dyad panel %s -> %s ...", start, end)
        dyad_rows = build_dyad_rows(conn, start, end, date_prec_max, clarity_filter)
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, _UPSERT_DYAD_SQL, dyad_rows, page_size=2000)
        conn.commit()
        logger.info("dyad_pit_labels: %d rows", len(dyad_rows))

        return {"country_pit_labels": len(country_rows), "dyad_pit_labels": len(dyad_rows)}
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build UCDP point-in-time labels")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-11-01",
                    help="last run_date -- must leave a full 30d label window inside GED's coverage")
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument("--date-prec-max", type=int, default=4)
    ap.add_argument("--clarity", default="1", help="comma-separated event_clarity values, e.g. '1,2'")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    clarity = tuple(int(c) for c in args.clarity.split(","))

    results = run(start, end, dsn=args.dsn, date_prec_max=args.date_prec_max, clarity_filter=clarity)
    for table, n in results.items():
        print(f"{table}: {n:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
