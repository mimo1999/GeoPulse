"""
Highlighted-events feed -- the second of the three broadened use-case
deliverables (country summaries, highlighted events, activity heatmap).

Every event keeps all three significance dimensions, never collapsed into
one score, per instruction:

  - **media**      -- num_mentions. "Where media is looking."
  - **goldstein**   -- abs(intensity). "Where people are looking." Magnitude,
                       not sign: an extremely cooperative event is exactly as
                       significant to someone watching geopolitics closely as
                       an extremely conflictual one -- direction is a
                       separate question (interaction_type already answers
                       it), this dimension is about *how much*.
  - **spike**       -- z-score of the event's own country's event volume on
                       its own date vs a trailing 30-day baseline for that
                       country. "What no one saw coming": a sudden burst of
                       activity for a country, not a property of the event's
                       own fields at all -- two identical, unremarkable
                       events are equally "unexpected" if they land on a day
                       their country suddenly went from typically-quiet to
                       very active.

An event can rank on more than one dimension -- that isn't deduplicated
away, since an event that's simultaneously heavily covered, intensely
positive/negative, AND part of an unexpected burst is genuinely the most
significant kind of event, not a data quality issue to resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

MIN_BASELINE_DAYS = 14   # minimum trailing history before a country's
                          # z-score is trusted -- otherwise every country's
                          # first ~2 weeks would spuriously look like spikes
BASELINE_WINDOW_DAYS = 30


@dataclass
class HighlightedEvent:
    event_id: int
    event_date: date
    country_iso3: Optional[str]
    interaction_type: Optional[str]
    num_mentions: Optional[int]
    intensity: Optional[float]
    source_url: Optional[str]
    country_day_z_score: Optional[float]
    tags: list[str]


# ---------------------------------------------------------------------------
# Dimension 1: media attention
# ---------------------------------------------------------------------------

_TOP_MEDIA_SQL = """
    SELECT e.event_id, e.event_date, l.country_iso3, e.interaction_type,
           e.num_mentions, e.intensity, e.source_url
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE e.source = 'gdelt' AND e.num_mentions IS NOT NULL
      AND e.event_date >= %(since)s
      AND (%(iso3)s::text IS NULL OR l.country_iso3 = %(iso3)s)
    ORDER BY e.num_mentions DESC LIMIT %(limit)s
"""


def top_media_events(conn, since: date, iso3: Optional[str] = None, limit: int = 20) -> list[tuple]:
    """Highest num_mentions. Bounded by `since` -- an unbounded global sort
    over 48.9M events would be a full-table sort; scoping to a recent
    window keeps this to the window's row count, which the existing
    (source, event_date) index makes cheap to select before sorting."""
    with conn.cursor() as cur:
        cur.execute(_TOP_MEDIA_SQL, {"since": since, "iso3": iso3, "limit": limit})
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Dimension 2: Goldstein magnitude
# ---------------------------------------------------------------------------

_TOP_GOLDSTEIN_SQL = """
    SELECT e.event_id, e.event_date, l.country_iso3, e.interaction_type,
           e.num_mentions, e.intensity, e.source_url
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE e.source = 'gdelt' AND e.intensity IS NOT NULL
      AND e.event_date >= %(since)s
      AND (%(iso3)s::text IS NULL OR l.country_iso3 = %(iso3)s)
    ORDER BY abs(e.intensity) DESC LIMIT %(limit)s
"""


def top_goldstein_events(conn, since: date, iso3: Optional[str] = None, limit: int = 20) -> list[tuple]:
    """Highest |Goldstein intensity| -- magnitude of cooperation or
    conflict, whichever is more extreme, not signed."""
    with conn.cursor() as cur:
        cur.execute(_TOP_GOLDSTEIN_SQL, {"since": since, "iso3": iso3, "limit": limit})
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Dimension 3: unexpected country-level activity spikes
# ---------------------------------------------------------------------------

_COUNTRY_DAY_ZSCORES_SQL = """
    WITH daily_counts AS (
        SELECT l.country_iso3, e.event_date, COUNT(*) AS n
        FROM graph.event e
        JOIN graph.location l ON l.location_id = e.location_id
        WHERE l.country_iso3 IS NOT NULL AND e.source = 'gdelt'
        GROUP BY 1, 2
    ),
    with_baseline AS (
        SELECT country_iso3, event_date, n,
               AVG(n) OVER w AS baseline_mean,
               STDDEV(n) OVER w AS baseline_std,
               COUNT(*) OVER w AS baseline_days
        FROM daily_counts
        WINDOW w AS (
            PARTITION BY country_iso3 ORDER BY event_date
            ROWS BETWEEN %(window)s PRECEDING AND 1 PRECEDING
        )
    )
    SELECT country_iso3, event_date, n, baseline_mean, baseline_std,
           CASE WHEN baseline_std > 0 THEN (n - baseline_mean) / baseline_std END AS z_score
    FROM with_baseline
    WHERE baseline_days >= %(min_days)s
      AND (%(iso3)s::text IS NULL OR country_iso3 = %(iso3)s)
      AND (%(since)s::date IS NULL OR event_date >= %(since)s)
"""


def country_day_zscores(conn, iso3: Optional[str] = None, since: Optional[date] = None,
                         window: int = BASELINE_WINDOW_DAYS,
                         min_days: int = MIN_BASELINE_DAYS) -> list[tuple]:
    """(country_iso3, event_date, n, baseline_mean, baseline_std, z_score)
    for every country-day with enough trailing history to trust the
    baseline. A country's first `min_days` days never appear here -- there
    is no way to know if they were unusual without history to compare
    against, and a NULL/assumed baseline would silently manufacture a
    fake spike out of a cold start."""
    with conn.cursor() as cur:
        cur.execute(_COUNTRY_DAY_ZSCORES_SQL,
                    {"iso3": iso3, "since": since, "window": window, "min_days": min_days})
        return cur.fetchall()


def top_spike_country_days(conn, iso3: Optional[str] = None, since: Optional[date] = None,
                            limit: int = 20) -> list[tuple]:
    rows = country_day_zscores(conn, iso3=iso3, since=since)
    rows = [r for r in rows if r[5] is not None]
    rows.sort(key=lambda r: r[5], reverse=True)
    return rows[:limit]


_REPRESENTATIVE_EVENT_SQL = """
    SELECT e.event_id, e.event_date, l.country_iso3, e.interaction_type,
           e.num_mentions, e.intensity, e.source_url
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE l.country_iso3 = %(iso3)s AND e.event_date = %(event_date)s AND e.source = 'gdelt'
    ORDER BY e.num_mentions DESC NULLS LAST LIMIT 1
"""


def top_spike_events(conn, iso3: Optional[str] = None, since: Optional[date] = None,
                      limit: int = 20) -> list[tuple[tuple, float]]:
    """For each of the top-N spiking country-days, the single most-mentioned
    event that day as the spike's headline -- not every event from that
    day, which for a genuinely large spike could be thousands of near-
    duplicate wire-service events."""
    spike_days = top_spike_country_days(conn, iso3=iso3, since=since, limit=limit)
    results = []
    with conn.cursor() as cur:
        for country_iso3, event_date, n, mean, std, z in spike_days:
            cur.execute(_REPRESENTATIVE_EVENT_SQL, {"iso3": country_iso3, "event_date": event_date})
            row = cur.fetchone()
            if row:
                results.append((row, z))
    return results


# ---------------------------------------------------------------------------
# Combined feed
# ---------------------------------------------------------------------------

TAG_MEDIA = "Where media is looking"
TAG_GOLDSTEIN = "Where people are looking"
TAG_SPIKE = "What no one saw coming"


def get_highlighted_events(conn, since: date, iso3: Optional[str] = None,
                            limit_per_dimension: int = 20) -> list[HighlightedEvent]:
    """All three dimensions, each event tagged with which one(s) it earned.
    An event appearing in more than one top-N list keeps every tag it
    earned -- not deduplicated to a single "reason", by design (see module
    docstring)."""
    by_id: dict[int, HighlightedEvent] = {}

    def _add(row: tuple, z: Optional[float], tag: str):
        event_id, event_date, country_iso3, interaction_type, num_mentions, intensity, source_url = row
        if event_id not in by_id:
            by_id[event_id] = HighlightedEvent(
                event_id=event_id, event_date=event_date, country_iso3=country_iso3,
                interaction_type=interaction_type, num_mentions=num_mentions,
                intensity=intensity, source_url=source_url,
                country_day_z_score=z, tags=[],
            )
        if tag not in by_id[event_id].tags:
            by_id[event_id].tags.append(tag)
        if z is not None:
            by_id[event_id].country_day_z_score = z

    for row in top_media_events(conn, since, iso3, limit_per_dimension):
        _add(row, None, TAG_MEDIA)
    for row in top_goldstein_events(conn, since, iso3, limit_per_dimension):
        _add(row, None, TAG_GOLDSTEIN)
    for row, z in top_spike_events(conn, iso3, since, limit_per_dimension):
        _add(row, z, TAG_SPIKE)

    # Most tags earned first; most recent as the tiebreak within that.
    return sorted(by_id.values(), key=lambda h: (-len(h.tags), -h.event_date.toordinal()))
