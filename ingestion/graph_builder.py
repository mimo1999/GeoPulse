"""
Builds the actor-interaction graph (graph.country/actor/location/event/
event_actor) from gdelt_events, to transform GDELT event data
into a graph model representing actors and their interactions over time.

Entity resolution (actor -> country):
    1. GDELT's own actor{1,2}_country field, when present -- this is GDELT's
       own parsed-out country code, not a guess.
    2. When absent: if the actor's raw code is composed entirely of known
       CAMEO Actor Type ("role") codes (data/cameo_actor_types.py) -- e.g. a
       bare "GOV" or concatenated "GOVMIL" -- infer the country from the
       event's own location (action_geo_country). This is CAMEO's own
       coding convention (a role-only actor implicitly belongs to wherever
       the event happened), not an invented heuristic.
    3. Otherwise: leave country_iso3 NULL, country_inferred = False. Never
       guess past what (1) or (2) can support -- this is exactly the
       known limitation that many actor references are too generic to
       attribute to a country, and it should show up as NULLs a reader can query, not be
       silently papered over.

Actor identity is (source, raw_code, country_iso3) -- the same raw role code
in two different countries (GOV/USA vs GOV/CAN) is two different actors,
never merged. This was a real, previously-found bug class in the original
(uncommitted) prototype build.

Interaction typing: graph.event.interaction_type is computed once per event
from data/cameo_codes.interaction_type(), the single source of truth for the
CAMEO -> {cooperation, consultation, conflict} mapping.

Performance: actor resolution needs per-row Python logic (bare-role-code
detection can't be pushed into SQL without duplicating the CAMEO type list
there), but the number of *distinct* (raw_code, resolved country) pairs is
tiny relative to event volume -- they're cached in memory for the life of a
build run, so the DB only ever sees a bounded number of actor upserts
regardless of how many events reference the same actor.

Usage:
    python -m ingestion.graph_builder --start 2023-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from typing import Optional

import psycopg2
import psycopg2.extras

from data.cameo_actor_types import is_bare_role_code
from data.cameo_codes import interaction_type
from data.iso3_to_fips import ISO3_TO_FIPS

logger = logging.getLogger("ingestion.graph_builder")

DSN = os.environ.get(
    "GEOPULSE_DSN",
    "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432",
)

FIPS_TO_ISO3 = {fips: iso3 for iso3, fips in ISO3_TO_FIPS.items() if fips}

BATCH_DAYS = 1  # one gdelt_events partition-friendly day per fetch
SOURCE = "gdelt"

_EVENTS_SQL = """
    SELECT global_event_id, event_date,
           actor1_code, actor1_name, actor1_country,
           actor2_code, actor2_name, actor2_country,
           event_root_code, quad_class, goldstein, avg_tone,
           num_mentions, action_geo_country, latitude, longitude, source_url
    FROM gdelt_events
    WHERE event_date = %s
"""


class GraphBuilder:
    def __init__(self, dsn: str = DSN):
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False
        # In-memory caches, keyed for the life of one process run.
        self._country_cache: set[str] = set()
        self._actor_cache: dict[tuple[str, Optional[str], Optional[str]], int] = {}
        self._location_cache: dict[tuple[Optional[str], Optional[float], Optional[float]], int] = {}
        self._preload_caches()

    def close(self):
        self._conn.close()

    def _preload_caches(self):
        with self._conn.cursor() as cur:
            cur.execute("SELECT iso3 FROM graph.country")
            self._country_cache = {r[0] for r in cur.fetchall()}
            cur.execute("SELECT source, raw_code, country_iso3, actor_id FROM graph.actor")
            for source, raw_code, country_iso3, actor_id in cur.fetchall():
                self._actor_cache[(source, raw_code, country_iso3)] = actor_id
            cur.execute("SELECT country_iso3, lat, lon, location_id FROM graph.location")
            for country_iso3, lat, lon, location_id in cur.fetchall():
                self._location_cache[(country_iso3, lat, lon)] = location_id
        logger.info("Preloaded caches: %d countries, %d actors, %d locations",
                    len(self._country_cache), len(self._actor_cache), len(self._location_cache))

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _ensure_country(self, iso3: Optional[str], cur) -> Optional[str]:
        if not iso3:
            return None
        if iso3 not in self._country_cache:
            cur.execute(
                "INSERT INTO graph.country (iso3) VALUES (%s) ON CONFLICT (iso3) DO NOTHING",
                (iso3,),
            )
            self._country_cache.add(iso3)
        return iso3

    def _resolve_actor_country(self, gdelt_country_fips: Optional[str],
                                raw_code: Optional[str],
                                event_geo_iso3: Optional[str]) -> tuple[Optional[str], bool]:
        if gdelt_country_fips:
            iso3 = FIPS_TO_ISO3.get(gdelt_country_fips)
            if iso3:
                return iso3, False
        if is_bare_role_code(raw_code) and event_geo_iso3:
            return event_geo_iso3, True
        return None, False

    def _ensure_actor(self, raw_code: Optional[str], raw_name: Optional[str],
                       gdelt_country_fips: Optional[str], event_geo_iso3: Optional[str],
                       cur) -> Optional[int]:
        if not raw_code:
            return None
        country_iso3, inferred = self._resolve_actor_country(gdelt_country_fips, raw_code, event_geo_iso3)
        if country_iso3:
            self._ensure_country(country_iso3, cur)

        key = (SOURCE, raw_code, country_iso3)
        if key in self._actor_cache:
            return self._actor_cache[key]

        cur.execute(
            """INSERT INTO graph.actor (source, raw_code, raw_name, country_iso3, country_inferred)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (source, raw_code, country_iso3) DO UPDATE SET raw_code = EXCLUDED.raw_code
               RETURNING actor_id""",
            (SOURCE, raw_code, raw_name, country_iso3, inferred),
        )
        actor_id = cur.fetchone()[0]
        self._actor_cache[key] = actor_id
        return actor_id

    def _ensure_location(self, country_iso3: Optional[str], lat: Optional[float],
                          lon: Optional[float], cur) -> Optional[int]:
        if country_iso3 is None and lat is None and lon is None:
            return None
        # Round coordinates -- GDELT's lat/lon carries more precision than
        # meaningfully distinguishes locations for this graph's purposes,
        # and unrounded floats would create a near-unique location per event.
        lat_r = round(lat, 2) if lat is not None else None
        lon_r = round(lon, 2) if lon is not None else None
        key = (country_iso3, lat_r, lon_r)
        if key in self._location_cache:
            return self._location_cache[key]

        if country_iso3:
            self._ensure_country(country_iso3, cur)
        cur.execute(
            """INSERT INTO graph.location (country_iso3, lat, lon) VALUES (%s, %s, %s)
               ON CONFLICT (country_iso3, lat, lon) DO UPDATE SET country_iso3 = EXCLUDED.country_iso3
               RETURNING location_id""",
            (country_iso3, lat_r, lon_r),
        )
        location_id = cur.fetchone()[0]
        self._location_cache[key] = location_id
        return location_id

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_date(self, d: date) -> dict[str, int]:
        with self._conn.cursor() as cur:
            cur.execute(_EVENTS_SQL, (d,))
            rows = cur.fetchall()

        if not rows:
            return {"events": 0, "event_actor": 0}

        event_rows = []
        event_actor_rows = []

        with self._conn.cursor() as cur:
            for (gid, edate, a1_code, a1_name, a1_country,
                 a2_code, a2_name, a2_country,
                 root_code, quad_class, goldstein, avg_tone,
                 mentions, geo_fips, lat, lon, source_url) in rows:

                geo_iso3 = FIPS_TO_ISO3.get(geo_fips) if geo_fips else None
                location_id = self._ensure_location(geo_iso3, lat, lon, cur)

                itype = interaction_type(root_code=root_code, quad_class=quad_class)

                event_rows.append((
                    SOURCE, str(gid), edate, str(root_code) if root_code is not None else None,
                    goldstein, avg_tone, None, None, None, mentions, location_id, source_url, itype,
                ))

                actor1_id = self._ensure_actor(a1_code, a1_name, a1_country, geo_iso3, cur)
                actor2_id = self._ensure_actor(a2_code, a2_name, a2_country, geo_iso3, cur)
                if actor1_id:
                    event_actor_rows.append((str(gid), actor1_id, "actor1"))
                if actor2_id:
                    event_actor_rows.append((str(gid), actor2_id, "actor2"))

            if event_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO graph.event (
                           source, source_event_id, event_date, event_type,
                           intensity, avg_tone, fatalities_best, fatalities_low, fatalities_high,
                           num_mentions, location_id, source_url, interaction_type
                       ) VALUES %s
                       ON CONFLICT (source, source_event_id) DO NOTHING""",
                    event_rows,
                    page_size=5000,
                )

            # event_id lookup for the event_actor batch (needed since we
            # inserted by source_event_id, not the generated event_id).
            cur.execute(
                "SELECT source_event_id, event_id FROM graph.event WHERE source = %s AND event_date = %s",
                (SOURCE, d),
            )
            id_map = dict(cur.fetchall())

            ea_rows = [
                (id_map[gid], actor_id, role)
                for gid, actor_id, role in event_actor_rows
                if gid in id_map
            ]
            if ea_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO graph.event_actor (event_id, actor_id, role) VALUES %s
                       ON CONFLICT (event_id, actor_id, role) DO NOTHING""",
                    ea_rows,
                    page_size=5000,
                )

        self._conn.commit()
        return {"events": len(event_rows), "event_actor": len(ea_rows)}


def daterange(start: date, end: date):
    d = start
    from datetime import timedelta
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the GDELT actor-interaction graph")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dsn", default=DSN)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    builder = GraphBuilder(args.dsn)
    try:
        total_events = total_ea = 0
        n_days = (end - start).days + 1
        for i, d in enumerate(daterange(start, end), 1):
            stats = builder.build_date(d)
            total_events += stats["events"]
            total_ea += stats["event_actor"]
            if i % 30 == 0 or i == n_days:
                logger.info("Progress: %d/%d days -- %d events, %d event_actor edges so far",
                            i, n_days, total_events, total_ea)
        logger.info("Done: %d events, %d event_actor edges", total_events, total_ea)
    finally:
        builder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
