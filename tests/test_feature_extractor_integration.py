"""
Integration test for the N+1 fix in preprocessing/feature_extractor.py
(TODO.md P1). Skipped if Postgres isn't reachable -- see
tests/test_pit_labels.py for the same pattern and rationale.
"""

from __future__ import annotations

from datetime import date

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from preprocessing.feature_extractor import FeatureExtractor

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc}); skipping feature_extractor integration tests")
    yield c
    c.close()


def _a_date_with_events(conn) -> date | None:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT event_date FROM gdelt_events
               GROUP BY event_date HAVING COUNT(*) > 100 ORDER BY event_date LIMIT 1"""
        )
        row = cur.fetchone()
        return row[0] if row else None


def test_grouped_query_matches_raw_event_counts(conn):
    """The core correctness check for the N+1 fix: total_events per country,
    computed by the new single-query-grouped-in-Python path, must equal a
    direct COUNT(*) over gdelt_events for that (country, date) -- no country
    silently dropped or double-counted by the grouping.

    Does NOT assert country_daily_features' total row count for the date --
    that table is also written by ingestion/polecat_pipeline.py and the
    parquet-seeded path (scripts/seed_db_from_cache.py), on possibly
    different, overlapping country-code conventions (see TODO.md's
    country-code key-space note), so its row count for a date is not solely
    a function of this extractor's input."""
    d = _a_date_with_events(conn)
    if d is None:
        pytest.skip("no dates with events in gdelt_events")

    fe = FeatureExtractor(DSN)
    n_written = fe.compute_daily_features(d)
    assert n_written > 0

    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(DISTINCT action_geo_country) FROM gdelt_events
               WHERE event_date = %s AND action_geo_country IS NOT NULL AND action_geo_country != ''""",
            (d,),
        )
        expected_countries = cur.fetchone()[0]
    assert n_written == expected_countries

    with conn.cursor() as cur:
        cur.execute(
            """SELECT g.action_geo_country
               FROM gdelt_events g
               WHERE g.event_date = %s AND g.action_geo_country IS NOT NULL AND g.action_geo_country != ''
               EXCEPT
               SELECT country FROM country_daily_features WHERE feature_date = %s""",
            (d, d),
        )
        dropped = [r[0] for r in cur.fetchall()]
    assert not dropped, f"countries with events but no feature row: {dropped}"

    with conn.cursor() as cur:
        cur.execute(
            """SELECT cdf.country, cdf.total_events, raw.n
               FROM country_daily_features cdf
               JOIN (
                   SELECT action_geo_country AS country, COUNT(*) AS n
                   FROM gdelt_events WHERE event_date = %s GROUP BY 1
               ) raw ON raw.country = cdf.country
               WHERE cdf.feature_date = %s""",
            (d, d),
        )
        mismatches = [(c, fe_n, raw_n) for c, fe_n, raw_n in cur.fetchall() if fe_n != raw_n]
    assert not mismatches, f"total_events drifted from raw counts: {mismatches[:5]}"


def test_country_filter_still_works(conn):
    """The optional `countries` argument must still restrict output, even
    though the fetch itself is no longer per-country."""
    d = _a_date_with_events(conn)
    if d is None:
        pytest.skip("no dates with events in gdelt_events")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT action_geo_country FROM gdelt_events WHERE event_date = %s LIMIT 2",
            (d,),
        )
        two_countries = [r[0] for r in cur.fetchall()]
    if len(two_countries) < 2:
        pytest.skip("fewer than 2 countries on this date")

    fe = FeatureExtractor(DSN)
    n_written = fe.compute_daily_features(d, countries=two_countries)
    assert n_written == len(two_countries)
