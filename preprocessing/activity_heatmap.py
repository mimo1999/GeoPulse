"""
Activity heatmap -- third of the three broadened use-case deliverables
(country summaries, highlighted events, activity heatmap).

Country-level choropleth data by default: reuses the same Event->Location
->Country join as country_summary.py, aggregated across all countries at
once rather than one country at a time. Pragmatic default -- fast, and the
UI can already render a choropleth (streamlit_app/ui.py has the theme and
color-scale utilities from the parked PIT dashboard). Point-density
(lat/lon-level) heatmap is a documented extension below, not built yet --
same query shape, just grouped by (lat, lon) instead of country_iso3.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

INTERACTION_TYPES = ("cooperation", "consultation", "conflict")


@dataclass
class CountryActivity:
    iso3: str
    total_events: int
    conflict_events: int
    conflict_share: Optional[float]
    avg_intensity: Optional[float]
    total_mentions: int


_COUNTRY_ACTIVITY_SQL = """
    SELECT l.country_iso3,
           COUNT(*) AS total_events,
           COUNT(*) FILTER (WHERE e.interaction_type = 'conflict') AS conflict_events,
           AVG(e.intensity) AS avg_intensity,
           COALESCE(SUM(e.num_mentions), 0) AS total_mentions
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE l.country_iso3 IS NOT NULL AND e.source = 'gdelt'
      AND (%(since)s::date IS NULL OR e.event_date >= %(since)s)
      AND (%(until)s::date IS NULL OR e.event_date < %(until)s)
    GROUP BY 1
"""


def country_activity(conn, since: Optional[date] = None,
                      until: Optional[date] = None) -> list[CountryActivity]:
    """One row per country with any located activity in the window --
    the choropleth's data source. A country with zero events in the window
    simply doesn't appear (an empty map cell), which is correct: it is not
    the same claim as "zero conflict share" (that needs a denominator)."""
    with conn.cursor() as cur:
        cur.execute(_COUNTRY_ACTIVITY_SQL, {"since": since, "until": until})
        rows = cur.fetchall()
    out = []
    for iso3, total, conflict, avg_intensity, mentions in rows:
        share = conflict / total if total else None
        out.append(CountryActivity(
            iso3=iso3, total_events=total, conflict_events=conflict,
            conflict_share=share, avg_intensity=avg_intensity, total_mentions=mentions,
        ))
    return out


_COUNTRY_ACTIVITY_TIMESERIES_SQL = """
    SELECT l.country_iso3, date_trunc('week', e.event_date)::date AS week, COUNT(*) AS n
    FROM graph.event e
    JOIN graph.location l ON l.location_id = e.location_id
    WHERE l.country_iso3 IS NOT NULL AND e.source = 'gdelt'
      AND (%(since)s::date IS NULL OR e.event_date >= %(since)s)
    GROUP BY 1, 2 ORDER BY 1, 2
"""


def country_activity_timeseries(conn, since: Optional[date] = None) -> list[tuple]:
    """(country_iso3, week, event_count) -- feeds an animated/time-slider
    version of the choropleth, the "how it evolves over time" requirement
    applied to the heatmap specifically rather than just country summaries."""
    with conn.cursor() as cur:
        cur.execute(_COUNTRY_ACTIVITY_TIMESERIES_SQL, {"since": since})
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Point-density extension (not wired into a UI yet -- documented shape only)
# ---------------------------------------------------------------------------
#
# SELECT l.lat, l.lon, COUNT(*) AS n
# FROM graph.event e JOIN graph.location l ON l.location_id = e.location_id
# WHERE e.source = 'gdelt' AND e.event_date >= %(since)s
# GROUP BY 1, 2
#
# Same shape as country_activity() with country_iso3 replaced by (lat, lon);
# 245,075 distinct points total, a bounded-since window will be far fewer.
# Left undeveloped because usecase.md's own choropleth precedent
# (streamlit_app/ui.py) and country_summary.py's existing country-shaped
# aggregates make the country-level view reusable immediately, while a
# point map needs its own rendering choice (Plotly density_mapbox / kepler)
# not yet made.
