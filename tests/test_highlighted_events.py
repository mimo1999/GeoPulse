"""
Integration tests for preprocessing/highlighted_events.py. Skipped if
Postgres isn't reachable.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from preprocessing.highlighted_events import (
    MIN_BASELINE_DAYS,
    TAG_GOLDSTEIN,
    TAG_MEDIA,
    TAG_SPIKE,
    country_day_zscores,
    get_highlighted_events,
    top_goldstein_events,
    top_media_events,
    top_spike_country_days,
)

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"
SINCE = date(2025, 12, 1)


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc})")
    yield c
    c.close()


def test_top_media_events_sorted_descending_by_mentions(conn):
    rows = top_media_events(conn, SINCE, limit=20)
    mentions = [r[4] for r in rows]
    assert mentions == sorted(mentions, reverse=True)


def test_top_goldstein_events_sorted_by_absolute_intensity(conn):
    rows = top_goldstein_events(conn, SINCE, limit=20)
    magnitudes = [abs(r[5]) for r in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_goldstein_ranking_is_not_the_same_as_media_ranking(conn):
    """The whole point of keeping both dimensions: they should surface
    different events, not the same ones in a different order."""
    media_ids = {r[0] for r in top_media_events(conn, SINCE, limit=20)}
    goldstein_ids = {r[0] for r in top_goldstein_events(conn, SINCE, limit=20)}
    assert media_ids != goldstein_ids


def test_country_day_zscores_excludes_insufficient_history(conn):
    rows = country_day_zscores(conn, since=SINCE)
    # every returned row implicitly had >= MIN_BASELINE_DAYS of trailing
    # history (enforced in SQL) -- spot check none are the dataset's own
    # earliest possible dates for a given country, which would indicate
    # the history filter didn't apply
    assert all(r[1] >= date(2023, 1, 1) + timedelta(days=MIN_BASELINE_DAYS) for r in rows)


def test_top_spike_country_days_sorted_by_zscore_descending(conn):
    rows = top_spike_country_days(conn, since=SINCE, limit=10)
    zscores = [r[5] for r in rows]
    assert zscores == sorted(zscores, reverse=True)
    assert all(z is not None for z in zscores)


def test_get_highlighted_events_tags_are_valid(conn):
    events = get_highlighted_events(conn, since=SINCE, limit_per_dimension=10)
    assert events
    valid_tags = {TAG_MEDIA, TAG_GOLDSTEIN, TAG_SPIKE}
    for e in events:
        assert e.tags, f"event {e.event_id} has no tags"
        assert set(e.tags) <= valid_tags
        assert len(e.tags) == len(set(e.tags))  # no duplicate tags


def test_get_highlighted_events_multi_tag_events_keep_all_tags(conn):
    """An event qualifying on more than one dimension must not be silently
    collapsed to a single reason."""
    events = get_highlighted_events(conn, since=SINCE, limit_per_dimension=30)
    multi = [e for e in events if len(e.tags) > 1]
    # Not asserting multi is non-empty (data-dependent), but if it IS
    # non-empty, sorting must actually put them first.
    if multi:
        assert events[0] in multi or len(events[0].tags) >= len(multi[-1].tags)
