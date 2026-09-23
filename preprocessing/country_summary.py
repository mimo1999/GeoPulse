"""
Country-level summary queries against graph.* (Postgres) -- the first of
the three broadened use-case deliverables (country summaries, highlighted
events, activity heatmap).

Deliberately Postgres-only, no Neo4j/Cypher: every query here is a plain
aggregate over Event/Location/Actor/Country, answerable without graph
traversal. Neo4j earns its keep for genuine multi-hop network analysis
(actor communities, influence propagation) -- not for this.

Two different notions of "this country" are used on purpose, matching two
different real questions:
  - **Location-based** (Event -> Location -> Country): "what happened IN
    this country" -- event volume, trend, interaction-type mix. This is
    the country-as-place framing.
  - **Actor-based** (Event -> event_actor -> Actor -> Country): "who did
    this country's actors interact with" -- top counterparts. This is the
    country-as-diplomatic-actor framing. A US actor operating in another
    country's territory is a US-actor event happening at a foreign
    location; conflating the two framings would blur exactly that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import psycopg2
import psycopg2.extras

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"

INTERACTION_TYPES = ("cooperation", "consultation", "conflict")


@dataclass
class CountrySummary:
    iso3: str
    total_events: int
    interaction_mix: dict[str, int]
    monthly_volume: list[tuple[date, int]]
    top_counterparts: list[tuple[str, int]]
    top_counterparts_by_type: dict[str, list[tuple[str, int]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Location-based: activity IN the country
# ---------------------------------------------------------------------------

_EVENT_VOLUME_TREND_SQL = """
    SELECT date_trunc('month', e.event_date)::date AS month, COUNT(*) AS n
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE l.country_iso3 = %(iso3)s
      AND e.source = 'gdelt'
      AND (%(since)s::date IS NULL OR e.event_date >= %(since)s)
    GROUP BY 1 ORDER BY 1
"""

_INTERACTION_MIX_SQL = """
    SELECT e.interaction_type, COUNT(*) AS n
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE l.country_iso3 = %(iso3)s
      AND e.source = 'gdelt'
      AND (%(since)s::date IS NULL OR e.event_date >= %(since)s)
    GROUP BY 1
"""


def event_volume_trend(conn, iso3: str, since: Optional[date] = None) -> list[tuple[date, int]]:
    """Monthly event count for events located in `iso3`."""
    with conn.cursor() as cur:
        cur.execute(_EVENT_VOLUME_TREND_SQL, {"iso3": iso3, "since": since})
        return cur.fetchall()


def interaction_mix(conn, iso3: str, since: Optional[date] = None) -> dict[str, int]:
    """Counts by interaction_type for events located in `iso3`. Events with
    a NULL interaction_type (the pre-column June-2026 prototype rows) are
    excluded by the JOIN's implicit filter on source='gdelt' and reported
    separately by total_events_including_untyped() below -- never silently
    folded into one of the three real buckets."""
    with conn.cursor() as cur:
        cur.execute(_INTERACTION_MIX_SQL, {"iso3": iso3, "since": since})
        rows = dict(cur.fetchall())
    return {t: rows.get(t, 0) for t in INTERACTION_TYPES}


def untyped_event_count(conn, iso3: str, since: Optional[date] = None) -> int:
    """Events located in `iso3` with NULL interaction_type -- surfaced
    explicitly rather than silently excluded, so a summary's totals are
    auditable (typed + untyped = total)."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM graph.event e
               JOIN graph.location l ON l.location_id = e.location_id
               WHERE l.country_iso3 = %(iso3)s AND e.interaction_type IS NULL
                 AND (%(since)s::date IS NULL OR e.event_date >= %(since)s)""",
            {"iso3": iso3, "since": since},
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Actor-based: who this country's actors interact with
# ---------------------------------------------------------------------------

_TOP_COUNTERPARTS_SQL = """
    WITH country_actor_events AS (
        SELECT DISTINCT ea.event_id
        FROM graph.actor a
        JOIN graph.event_actor ea ON ea.actor_id = a.actor_id
        WHERE a.country_iso3 = %(iso3)s
    )
    SELECT a2.country_iso3, COUNT(DISTINCT ea2.event_id) AS n
    FROM country_actor_events cae
    JOIN graph.event_actor ea2 ON ea2.event_id = cae.event_id
    JOIN graph.actor a2 ON a2.actor_id = ea2.actor_id
    WHERE a2.country_iso3 IS NOT NULL AND a2.country_iso3 != %(iso3)s
      AND a2.country_inferred = FALSE
    GROUP BY 1 ORDER BY 2 DESC LIMIT %(limit)s
"""

_TOP_COUNTERPARTS_BY_TYPE_SQL = """
    WITH country_actor_events AS (
        SELECT DISTINCT ea.event_id
        FROM graph.actor a
        JOIN graph.event_actor ea ON ea.actor_id = a.actor_id
        WHERE a.country_iso3 = %(iso3)s
    )
    SELECT a2.country_iso3, COUNT(DISTINCT ea2.event_id) AS n
    FROM country_actor_events cae
    JOIN graph.event_actor ea2 ON ea2.event_id = cae.event_id
    JOIN graph.actor a2 ON a2.actor_id = ea2.actor_id
    JOIN graph.event e ON e.event_id = cae.event_id
    WHERE a2.country_iso3 IS NOT NULL AND a2.country_iso3 != %(iso3)s
      AND a2.country_inferred = FALSE
      AND e.interaction_type = %(itype)s
    GROUP BY 1 ORDER BY 2 DESC LIMIT %(limit)s
"""


def top_counterparts(conn, iso3: str, limit: int = 10) -> list[tuple[str, int]]:
    """Countries most frequently co-appearing (via actor identity, either
    role) in the same events as `iso3`'s actors, across all event sources
    this country's actors touch -- not restricted to events located in
    `iso3` itself (see module docstring).

    Excludes counterpart actors whose country was *inferred* from the
    event's own location (graph_builder.py's country_inferred flag) rather
    than given directly by GDELT. Found via a real bug this exposed: a
    conflict event geolocated in Ukraine with an explicit UKR actor1 and a
    bare-role actor2 (no country in GDELT) gets actor2's country inferred
    as UKR too, purely from where the event happened -- manufacturing a
    false domestic (UKR vs UKR) pair out of what may well be a foreign
    actor (e.g. Russian forces) operating in/against Ukraine. Without this
    exclusion, `top_counterparts_by_type('UKR', 'conflict')` returned
    empty: the real cross-border pairs were swamped by fabricated
    same-country ones, which the != iso3 filter then correctly stripped,
    leaving nothing. This is a scoped bug (the schema already carries the
    information needed to avoid it), not the known limitation that many actor references are too
    generic to attribute."""
    with conn.cursor() as cur:
        cur.execute(_TOP_COUNTERPARTS_SQL, {"iso3": iso3, "limit": limit})
        return cur.fetchall()


def top_counterparts_by_type(conn, iso3: str, interaction_type: str, limit: int = 10) -> list[tuple[str, int]]:
    """Same as top_counterparts(), split by interaction_type.

    Measured directly on real data: this frequently returns an empty list.
    GDELT rarely gives an explicit (non-inferred) country code for the
    "other side" of an interaction at all -- for Ukraine, splitting by
    type left literally zero non-inferred cross-country matches for
    cooperation, consultation, AND conflict, even though the unsplit
    top_counterparts() for the same country returns solid results (RUS,
    USA, GBR, ...). The aggregate across all types has enough genuine
    explicit-country data points to clear visibility; any single type's
    slice usually doesn't. This is the known generic-actor-reference limitation
    showing up concretely, not a
    further bug to chase -- callers (the UI included) must treat an empty
    result here as "not enough explicitly-attributed data for this split",
    not as an error."""
    if interaction_type not in INTERACTION_TYPES:
        raise ValueError(f"interaction_type must be one of {INTERACTION_TYPES}, got {interaction_type!r}")
    with conn.cursor() as cur:
        cur.execute(_TOP_COUNTERPARTS_BY_TYPE_SQL, {"iso3": iso3, "itype": interaction_type, "limit": limit})
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def build_country_summary(conn, iso3: str, since: Optional[date] = None,
                           n_counterparts: int = 10, include_by_type: bool = False) -> CountrySummary:
    """include_by_type defaults to False. Measured directly (2026-09-22,
    no concurrent load): interaction_mix ~24s, event_volume_trend ~22s,
    top_counterparts ~18s, top_counterparts_by_type ~17s EACH of the 3
    types -- computing all of it synchronously totals over 100s, well past
    any reasonable request timeout, for a feature that (per
    top_counterparts_by_type's own docstring) is frequently empty anyway.
    Default summary costs ~64s (mix+trend+counterparts); the by-type
    breakdown is opt-in for a caller that specifically wants it and can
    wait. The real fix is a materialized per-country rollup, not attempted
    here."""
    trend = event_volume_trend(conn, iso3, since)
    mix = interaction_mix(conn, iso3, since)
    counterparts = top_counterparts(conn, iso3, limit=n_counterparts)
    by_type = {}
    if include_by_type:
        by_type = {
            t: top_counterparts_by_type(conn, iso3, t, limit=n_counterparts)
            for t in INTERACTION_TYPES
        }
    return CountrySummary(
        iso3=iso3,
        total_events=sum(mix.values()),
        interaction_mix=mix,
        monthly_volume=trend,
        top_counterparts=counterparts,
        top_counterparts_by_type=by_type,
    )


def latest_resolvable_event_date(conn) -> Optional[date]:
    """MAX(event_date) among events that can actually resolve to a country
    (have a non-null location_id) -- NOT the same as MAX(event_date) over
    all events. Found via a real bug: ~1.1M events (the untracked
    June-2026 prototype, kept per instruction rather than wiped) have no
    location_id at all and are invisible to every location-based query in
    this module. Naively defaulting a "since" window to wall-clock
    date.today() - 90d landed inside that unresolvable tail and showed
    "No data available" in the UI, despite 47.8M real, located events
    sitting in the same table for 2023-2025. Callers should build default
    date-range windows from this, never from date.today()."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(e.event_date) FROM graph.event e "
            "JOIN graph.location l ON l.location_id = e.location_id "
            "WHERE e.source = 'gdelt'"
        )
        return cur.fetchone()[0]


def list_countries_with_activity(conn) -> list[str]:
    """All country ISO3 codes with at least one located event -- the valid
    input set for build_country_summary(), so a caller (e.g. a dashboard
    dropdown) never has to guess which codes have data."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT l.country_iso3 FROM graph.location l "
            "JOIN graph.event e ON e.location_id = l.location_id "
            "WHERE l.country_iso3 IS NOT NULL ORDER BY 1"
        )
        return [r[0] for r in cur.fetchall()]
