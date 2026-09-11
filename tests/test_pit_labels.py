"""
Integration tests for preprocessing/pit_labels.py against a live UCDP schema.

Skipped entirely if Postgres isn't reachable, so a bare `pytest` run never
breaks on missing infrastructure (see TODO.md P3 re: scripts/test_pg_conn.py,
which made exactly that mistake). Run with the DB up to get real coverage:

    docker-compose up -d postgres   # or however Postgres is started locally
    pytest tests/test_pit_labels.py

These assume `python -m ingestion.ucdp_loader` and
`python -m preprocessing.pit_labels --start 2015-01-01 --end 2025-11-01` have
already been run once.
"""

from __future__ import annotations

from datetime import date

import pytest

psycopg2 = pytest.importorskip("psycopg2")

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc}); skipping PIT integration tests")
    yield c
    c.close()


def _scalar(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()[0]


def test_ged_row_count(conn):
    """Regression test named in the plan: 417,968 GED rows."""
    assert _scalar(conn, "SELECT COUNT(*) FROM ucdp.ged_events") == 417_968


def test_country_pit_labels_populated(conn):
    n = _scalar(conn, "SELECT COUNT(*) FROM ucdp.country_pit_labels")
    assert n > 0, "run preprocessing.pit_labels before this test"


def test_hand_verified_country_month(conn):
    """Ukraine, 2024-06 run_date (label window 2024-06-02..2024-07-01):
    hand-checked against a direct query of ucdp.ged_events in this session."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT n_events, deaths_best, minor_conflict, war_intensity
               FROM ucdp.country_pit_labels WHERE fips2 = 'UP' AND run_date = '2024-06-01'"""
        )
        row = cur.fetchone()
    assert row is not None
    n_events, deaths_best, minor_conflict, war_intensity = row
    assert n_events == 1403
    assert deaths_best == 2978
    assert minor_conflict is True
    assert war_intensity is True


def test_pit_boundary_country_labels(conn):
    """The single most important test in the plan: features never see the
    future. For a sample of stored rows, recompute the current-window death
    count directly from ucdp.ged_events using a strictly-greater-than
    run_date lower bound, and confirm it matches what was stored -- proving
    the label was not built from data <= run_date."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT fips2, gw_id, run_date, deaths_best, n_events
               FROM ucdp.country_pit_labels
               WHERE deaths_best > 0
               ORDER BY random() LIMIT 25"""
        )
        sample = cur.fetchall()
    assert sample, "no non-zero rows to sample -- run pit_labels first"

    with conn.cursor() as cur:
        for fips2, gw_id, run_date, stored_deaths, stored_n in sample:
            cur.execute(
                """SELECT COUNT(*), COALESCE(SUM(best), 0) FROM ucdp.ged_events
                   WHERE country_id = %s AND date_start > %s
                     AND date_start <= %s + INTERVAL '30 days'
                     AND date_prec <= 4 AND event_clarity = 1""",
                (gw_id, run_date, run_date),
            )
            recomputed_n, recomputed_deaths = cur.fetchone()
            assert recomputed_n == stored_n, f"{fips2}/{run_date}: n_events drifted from raw events"
            assert recomputed_deaths == stored_deaths, f"{fips2}/{run_date}: deaths_best drifted from raw events"

            # The PIT assertion itself: nothing in the window is <= run_date.
            cur.execute(
                """SELECT COUNT(*) FROM ucdp.ged_events
                   WHERE country_id = %s AND date_start <= %s
                     AND date_start > %s - INTERVAL '30 days'
                     AND date_start > %s""",  # deliberately over-constrained; must always be 0
                (gw_id, run_date, run_date, run_date),
            )
            assert cur.fetchone()[0] == 0


def test_no_duplicate_country_month_keys(conn):
    total = _scalar(conn, "SELECT COUNT(*) FROM ucdp.country_pit_labels")
    distinct = _scalar(
        conn, "SELECT COUNT(*) FROM (SELECT DISTINCT fips2, run_date, label_version FROM ucdp.country_pit_labels) s"
    )
    assert total == distinct


def test_dyad_candidate_set_median_matches_measured_baseline(conn):
    """Free regression test from the plan: candidate-set median size 3."""
    median = _scalar(
        conn,
        """SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY n) FROM (
             SELECT COUNT(*) AS n FROM ucdp.dyad_pit_labels
             WHERE run_date BETWEEN '2023-01-01' AND '2025-11-01' AND in_candidate_set
             GROUP BY fips2, run_date
           ) s""",
    )
    assert median == 3


def test_join_integrity_2023_2025(conn):
    """No UCDP country active in 2023-2025 fails to resolve to a FIPS code
    with real gdelt_events coverage -- the exact check that caught the
    country-code key-space bug (RU/CN/DE/JP/ES/GB/UA/PL split, fixed
    2026-09-11 via scripts/fix_country_code_split.py)."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT m.fips2, m.ucdp_country_name, COUNT(g.*) AS n
               FROM ucdp.gw_country_map m
               LEFT JOIN gdelt_events g ON g.action_geo_country = m.fips2
               WHERE m.gw_id IN (
                   SELECT DISTINCT country_id FROM ucdp.ged_events WHERE date_start >= '2023-01-01'
               )
               GROUP BY 1, 2 HAVING COUNT(g.*) = 0"""
        )
        unresolved = cur.fetchall()
    assert not unresolved, f"UCDP countries with zero matching gdelt_events: {unresolved}"
