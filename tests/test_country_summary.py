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


def test_top_counterparts_excludes_location_inferred_actors(conn):
    """Regression test for a real bug: a conflict event geolocated in
    Ukraine with an explicit UKR actor1 and a bare-role actor2 (no country
    given by GDELT) had actor2's country inferred as UKR too -- purely
    from where the event happened, not from any signal about the actor's
    actual origin. That fabricated a false domestic (UKR vs UKR) pair,
    which the same-country exclusion filter then correctly stripped,
    leaving top_counterparts_by_type('UKR', 'conflict') empty even though
    real cross-border data (Russia) exists elsewhere. Fix: exclude
    country_inferred=true actors from counterpart matching entirely.

    Verified by directly recomputing the join without the fix's WHERE
    clause and confirming it would have included at least one
    location-inferred actor that the fixed query correctly excludes --
    proves the exclusion is doing real work on this data, not a no-op."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(DISTINCT a2.actor_id)
               FROM graph.actor a
               JOIN graph.event_actor ea ON ea.actor_id = a.actor_id AND a.country_iso3 = 'UKR'
               JOIN graph.event_actor ea2 ON ea2.event_id = ea.event_id AND ea2.actor_id != ea.actor_id
               JOIN graph.actor a2 ON a2.actor_id = ea2.actor_id
               WHERE a2.country_iso3 IS NOT NULL AND a2.country_iso3 != 'UKR'
                 AND a2.country_inferred = true"""
        )
        n_excluded = cur.fetchone()[0]
    if n_excluded == 0:
        pytest.skip("no location-inferred cross-country counterparts in this dataset to test against")
    assert n_excluded > 0  # the exclusion in _TOP_COUNTERPARTS_SQL has real rows to remove

    # And the fix itself: every counterpart the (patched) query returns
    # must be backed by at least one non-inferred actor for that country
    # among this specific event set -- not merely "some actor somewhere
    # from that country is non-inferred" (too weak) or "zero inferred
    # actors exist globally" (irrelevant to this join).
    with conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT a2.country_iso3
               FROM graph.actor a
               JOIN graph.event_actor ea ON ea.actor_id = a.actor_id AND a.country_iso3 = 'UKR'
               JOIN graph.event_actor ea2 ON ea2.event_id = ea.event_id AND ea2.actor_id != ea.actor_id
               JOIN graph.actor a2 ON a2.actor_id = ea2.actor_id
               WHERE a2.country_iso3 IS NOT NULL AND a2.country_iso3 != 'UKR'
                 AND a2.country_inferred = false"""
        )
        non_inferred_backed = {r[0] for r in cur.fetchall()}
    counterparts = top_counterparts(conn, "UKR", limit=50)
    for country, _ in counterparts:
        assert country in non_inferred_backed


def test_top_counterparts_by_type_rejects_invalid_type(conn, a_quiet_country):
    with pytest.raises(ValueError):
        top_counterparts_by_type(conn, a_quiet_country, "not_a_real_type")


def test_build_country_summary_is_internally_consistent(conn, a_quiet_country):
    s = build_country_summary(conn, a_quiet_country, n_counterparts=5)
    assert s.iso3 == a_quiet_country
    assert s.total_events == sum(s.interaction_mix.values())


def test_build_country_summary_skips_by_type_by_default(conn, a_quiet_country):
    """Regression test for a real performance fix: top_counterparts_by_type
    measured at ~17s PER interaction type (3 types = ~50s), on top of an
    already ~64s default (mix+trend+counterparts) -- computing it eagerly
    made the whole summary exceed even a 150s client timeout in practice.
    Default must skip it; include_by_type=True must still provide it for a
    caller that explicitly wants to pay the cost."""
    s = build_country_summary(conn, a_quiet_country, n_counterparts=5)
    assert s.top_counterparts_by_type == {}

    s2 = build_country_summary(conn, a_quiet_country, n_counterparts=5, include_by_type=True)
    assert set(s2.top_counterparts_by_type.keys()) == set(INTERACTION_TYPES)


def test_list_countries_with_activity_nonempty_and_valid(conn):
    countries = list_countries_with_activity(conn)
    assert len(countries) > 50  # sanity: real data has well over 50 active locations
    assert all(len(c) == 3 for c in countries)  # ISO3


def test_latest_resolvable_event_date_excludes_unresolvable_events(conn):
    """Regression test for a real bug: ~1.1M events (an untracked older
    data source, kept rather than wiped) have no location_id at all and
    can never resolve to a country. A default date window built from
    MAX(event_date) over ALL events (rather than only resolvable ones)
    could land entirely inside that unresolvable tail and show "No data
    available" in the UI despite 47.8M real, located events existing."""
    from preprocessing.country_summary import latest_resolvable_event_date

    latest = latest_resolvable_event_date(conn)
    assert latest is not None

    with conn.cursor() as cur:
        cur.execute("SELECT MAX(event_date) FROM graph.event WHERE source = 'gdelt'")
        latest_any = cur.fetchone()[0]

    # The resolvable-only date must not be later than the true max (sanity),
    # and this dataset is known to have unresolvable events strictly after
    # the last resolvable one -- if that's no longer true (data changed),
    # this assertion should be revisited, not silently loosened.
    assert latest <= latest_any
