"""Integration tests for preprocessing/activity_heatmap.py."""

from __future__ import annotations

from datetime import date

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from preprocessing.activity_heatmap import country_activity, country_activity_timeseries

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"
SINCE = date(2025, 12, 1)
UNTIL = date(2026, 1, 1)


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc})")
    yield c
    c.close()


def test_country_activity_matches_raw_total(conn):
    rows = country_activity(conn, since=SINCE, until=UNTIL)
    assert rows
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM graph.event e JOIN graph.location l ON l.location_id = e.location_id
               WHERE l.country_iso3 IS NOT NULL AND e.source = 'gdelt'
                 AND e.event_date >= %s AND e.event_date < %s""",
            (SINCE, UNTIL),
        )
        expected = cur.fetchone()[0]
    assert sum(r.total_events for r in rows) == expected


def test_conflict_share_bounded_zero_to_one(conn):
    rows = country_activity(conn, since=SINCE, until=UNTIL)
    for r in rows:
        assert r.conflict_share is None or 0.0 <= r.conflict_share <= 1.0
        assert r.conflict_events <= r.total_events


def test_country_activity_timeseries_weeks_are_chronological_per_country(conn):
    rows = country_activity_timeseries(conn, since=SINCE)
    by_country: dict[str, list] = {}
    for iso3, week, n in rows:
        by_country.setdefault(iso3, []).append(week)
    for iso3, weeks in by_country.items():
        assert weeks == sorted(weeks), f"{iso3} weeks not chronological"
