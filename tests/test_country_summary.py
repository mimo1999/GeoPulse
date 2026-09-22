"""
Integration tests for preprocessing/country_summary.py. Skipped if
Postgres isn't reachable. Uses real graph.* data (small, quiet countries
where possible, to keep runtime reasonable at 48.9M-event scale).
"""

from __future__ import annotations

from datetime import date

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from preprocessing.country_summary import (
    INTERACTION_TYPES,
    build_country_summary,
    event_volume_trend,
    interaction_mix,
    list_countries_with_activity,
    top_counterparts,
    top_counterparts_by_type,
    untyped_event_count,
)

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc})")
    yield c
    c.close()


@pytest.fixture(scope="module")
def a_quiet_country(conn) -> str:
    """Pick a country with real but low activity, so tests run fast."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT l.country_iso3, COUNT(*) AS n
               FROM graph.event e JOIN graph.location l ON l.location_id = e.location_id
               WHERE l.country_iso3 IS NOT NULL AND e.interaction_type IS NOT NULL
               GROUP BY 1 HAVING COUNT(*) BETWEEN 50 AND 500
               ORDER BY 2 LIMIT 1"""
        )
        row = cur.fetchone()
    if row is None:
        pytest.skip("no low-activity country found in this window")
    return row[0]


def test_interaction_mix_sums_to_typed_total(conn, a_quiet_country):
    mix = interaction_mix(conn, a_quiet_country)
    assert set(mix.keys()) == set(INTERACTION_TYPES)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM graph.event e
               JOIN graph.location l ON l.location_id = e.location_id
               WHERE l.country_iso3 = %s AND e.interaction_type IS NOT NULL""",
            (a_quiet_country,),
        )
        expected_total = cur.fetchone()[0]
    assert sum(mix.values()) == expected_total


def test_untyped_plus_typed_equals_raw_total(conn, a_quiet_country):
    mix = interaction_mix(conn, a_quiet_country)
    untyped = untyped_event_count(conn, a_quiet_country)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM graph.event e JOIN graph.location l ON l.location_id = e.location_id "
            "WHERE l.country_iso3 = %s",
            (a_quiet_country,),
        )
        raw_total = cur.fetchone()[0]
    assert sum(mix.values()) + untyped == raw_total


def test_event_volume_trend_matches_raw_count(conn, a_quiet_country):
    trend = event_volume_trend(conn, a_quiet_country)
    assert trend == sorted(trend)  # chronological
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM graph.event e JOIN graph.location l ON l.location_id = e.location_id "
            "WHERE l.country_iso3 = %s AND e.source = 'gdelt'",
            (a_quiet_country,),
        )
        expected = cur.fetchone()[0]
    assert sum(n for _, n in trend) == expected


def test_since_filter_narrows_results(conn, a_quiet_country):
    full = event_volume_trend(conn, a_quiet_country)
    if not full:
        pytest.skip("no events for this country")
    cutoff = full[-1][0]  # last month only
    narrowed = event_volume_trend(conn, a_quiet_country, since=cutoff)
    assert sum(n for _, n in narrowed) <= sum(n for _, n in full)
    assert all(m >= cutoff for m, _ in narrowed)


def test_top_counterparts_excludes_self(conn, a_quiet_country):
    counterparts = top_counterparts(conn, a_quiet_country, limit=20)
    countries = [c for c, _ in counterparts]
    assert a_quiet_country not in countries


def test_top_counterparts_by_type_rejects_invalid_type(conn, a_quiet_country):
    with pytest.raises(ValueError):
        top_counterparts_by_type(conn, a_quiet_country, "not_a_real_type")


def test_build_country_summary_is_internally_consistent(conn, a_quiet_country):
    s = build_country_summary(conn, a_quiet_country, n_counterparts=5)
    assert s.iso3 == a_quiet_country
    assert s.total_events == sum(s.interaction_mix.values())
    assert set(s.top_counterparts_by_type.keys()) == set(INTERACTION_TYPES)


def test_list_countries_with_activity_nonempty_and_valid(conn):
    countries = list_countries_with_activity(conn)
    assert len(countries) > 50  # sanity: real data has well over 50 active locations
    assert all(len(c) == 3 for c in countries)  # ISO3
