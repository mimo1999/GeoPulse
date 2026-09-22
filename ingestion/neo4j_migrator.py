"""
Postgres graph.* -> Neo4j migration loader, per docs/neo4j_schema.md.

Scale: 244 countries, 51,849 actors, 245,075 locations, 48.9M events,
79.6M event_actor edges. This is the reason for every design choice below:

- **Keyset pagination, never OFFSET.** graph.event/actor/location have a
  single bigserial PK; graph.event_actor's PK is composite
  (event_id, actor_id, role). OFFSET-based paging degrades to effectively
  O(n^2) as the offset grows -- at 79.6M rows in ~16k batches, that would
  turn a multi-hour job into one that never finishes. `WHERE (pk) > (last)`
  is index-backed and O(1) per batch regardless of how far in we are.

- **MERGE, not CREATE, for every node and relationship.** The exact same
  reason every other long-running job in this project uses ON CONFLICT:
  this session has already had one background job silently killed and
  needing a clean restart (the first graph_builder.py backfill attempt).
  MERGE against the NODE KEY constraints from scripts/init_neo4j_schema.py
  makes every batch idempotent -- a restart from the last logged checkpoint
  re-processes a little overlap harmlessly instead of risking duplicates.

- **Python date objects passed as query parameters, not Cypher date()
  calls.** The neo4j driver (bolt v2+) converts datetime.date parameters to
  Neo4j's native temporal type automatically -- one less thing to get wrong
  by string-formatting dates by hand.

Usage:
    python -m ingestion.neo4j_migrator                  # full migration
    python -m ingestion.neo4j_migrator --only country,actor,location
    python -m ingestion.neo4j_migrator --only event --resume-from 12345678
    python -m ingestion.neo4j_migrator --only event_actor --resume-from "1,0,''"
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Iterator

import psycopg2

from ingestion.neo4j_client import Neo4jClient, Neo4jConfig

logger = logging.getLogger("ingestion.neo4j_migrator")

PG_DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"
BATCH_SIZE = 5_000
LOG_EVERY_N_BATCHES = 20


# ---------------------------------------------------------------------------
# Postgres keyset-paginated fetchers
# ---------------------------------------------------------------------------

def fetch_batches_single_pk(
    pg_conn, table: str, columns: list[str], pk_col: str,
    batch_size: int = BATCH_SIZE, start_after: Any = 0,
) -> Iterator[list[tuple]]:
    """Keyset pagination on a single-column PK (event_id/actor_id/location_id)."""
    last = start_after
    cols_sql = ", ".join(columns)
    while True:
        with pg_conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols_sql} FROM {table} WHERE {pk_col} > %s ORDER BY {pk_col} LIMIT %s",
                (last, batch_size),
            )
            rows = cur.fetchall()
        if not rows:
            return
        yield rows
        last = rows[-1][columns.index(pk_col)]


def fetch_event_actor_batches(
    pg_conn, batch_size: int = BATCH_SIZE, start_after: tuple = (0, 0, ""),
) -> Iterator[list[tuple]]:
    """Keyset pagination on graph.event_actor's composite PK
    (event_id, actor_id, role), using Postgres row-wise comparison -- index-
    backed against the same btree that already enforces the PK."""
    last = start_after
    while True:
        with pg_conn.cursor() as cur:
            cur.execute(
                """SELECT event_id, actor_id, role FROM graph.event_actor
                   WHERE (event_id, actor_id, role) > (%s, %s, %s)
                   ORDER BY event_id, actor_id, role LIMIT %s""",
                (*last, batch_size),
            )
            rows = cur.fetchall()
        if not rows:
            return
        yield rows
        last = rows[-1]


# ---------------------------------------------------------------------------
# Cypher, one MERGE statement per entity type
# ---------------------------------------------------------------------------

_MERGE_COUNTRY = """
UNWIND $rows AS row
MERGE (c:Country {iso3: row.iso3})
SET c.name = row.name
"""

_MERGE_ACTOR = """
UNWIND $rows AS row
MERGE (a:Actor {actor_id: row.actor_id})
SET a.source = row.source, a.raw_code = row.raw_code, a.raw_name = row.raw_name,
    a.country_iso3 = row.country_iso3, a.country_inferred = row.country_inferred
WITH a, row WHERE row.country_iso3 IS NOT NULL
MERGE (c:Country {iso3: row.country_iso3})
MERGE (a)-[:BELONGS_TO]->(c)
"""

_MERGE_LOCATION = """
UNWIND $rows AS row
MERGE (l:Location {location_id: row.location_id})
SET l.lat = row.lat, l.lon = row.lon
WITH l, row WHERE row.country_iso3 IS NOT NULL
MERGE (c:Country {iso3: row.country_iso3})
MERGE (l)-[:IN_COUNTRY]->(c)
"""

_MERGE_EVENT = """
UNWIND $rows AS row
MERGE (e:Event {event_id: row.event_id})
SET e.source = row.source, e.source_event_id = row.source_event_id,
    e.event_date = row.event_date, e.event_type = row.event_type,
    e.interaction_type = row.interaction_type, e.intensity = row.intensity,
    e.avg_tone = row.avg_tone, e.num_mentions = row.num_mentions,
    e.fatalities_best = row.fatalities_best, e.fatalities_low = row.fatalities_low,
    e.fatalities_high = row.fatalities_high, e.source_url = row.source_url
WITH e, row WHERE row.location_id IS NOT NULL
MERGE (l:Location {location_id: row.location_id})
MERGE (e)-[:OCCURRED_AT]->(l)
"""

_MERGE_INITIATED = """
UNWIND $rows AS row
MATCH (a:Actor {actor_id: row.actor_id})
MATCH (e:Event {event_id: row.event_id})
MERGE (a)-[:INITIATED]->(e)
"""

_MERGE_TARGETED = """
UNWIND $rows AS row
MATCH (a:Actor {actor_id: row.actor_id})
MATCH (e:Event {event_id: row.event_id})
MERGE (e)-[:TARGETED]->(a)
"""

_MERGE_CANONICAL = """
UNWIND $rows AS row
MATCH (a:Actor {actor_id: row.actor_id})
MATCH (c:Actor {actor_id: row.canonical_actor_id})
MERGE (a)-[:CANONICAL_OF]->(c)
"""


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------

def migrate_country(pg_conn, client: Neo4jClient) -> int:
    with pg_conn.cursor() as cur:
        cur.execute("SELECT iso3, name FROM graph.country")
        rows = [{"iso3": r[0], "name": r[1]} for r in cur.fetchall()]
    if rows:
        client.run(_MERGE_COUNTRY, {"rows": rows})
    logger.info("country: %d rows", len(rows))
    return len(rows)


def migrate_actor(pg_conn, client: Neo4jClient, start_after: int = 0) -> int:
    cols = ["actor_id", "source", "raw_code", "raw_name", "country_iso3", "country_inferred"]
    total = 0
    for i, batch in enumerate(fetch_batches_single_pk(pg_conn, "graph.actor", cols, "actor_id",
                                                       start_after=start_after)):
        rows = [dict(zip(cols, r)) for r in batch]
        client.run(_MERGE_ACTOR, {"rows": rows})
        total += len(rows)
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("actor: %d rows so far (last actor_id=%s)", total, rows[-1]["actor_id"])
    logger.info("actor: %d rows total", total)
    return total


def migrate_location(pg_conn, client: Neo4jClient, start_after: int = 0) -> int:
    cols = ["location_id", "country_iso3", "lat", "lon"]
    total = 0
    for i, batch in enumerate(fetch_batches_single_pk(pg_conn, "graph.location", cols, "location_id",
                                                       start_after=start_after)):
        rows = [dict(zip(cols, r)) for r in batch]
        client.run(_MERGE_LOCATION, {"rows": rows})
        total += len(rows)
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("location: %d rows so far (last location_id=%s)", total, rows[-1]["location_id"])
    logger.info("location: %d rows total", total)
    return total


def migrate_event(pg_conn, client: Neo4jClient, start_after: int = 0) -> int:
    cols = ["event_id", "source", "source_event_id", "event_date", "event_type",
            "interaction_type", "intensity", "avg_tone", "num_mentions",
            "fatalities_best", "fatalities_low", "fatalities_high", "location_id", "source_url"]
    total = 0
    for i, batch in enumerate(fetch_batches_single_pk(pg_conn, "graph.event", cols, "event_id",
                                                       start_after=start_after)):
        rows = [dict(zip(cols, r)) for r in batch]
        client.run(_MERGE_EVENT, {"rows": rows})
        total += len(rows)
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("event: %d rows so far (last event_id=%s)", total, rows[-1]["event_id"])
    logger.info("event: %d rows total", total)
    return total


def migrate_event_actor(pg_conn, client: Neo4jClient, start_after: tuple = (0, 0, "")) -> int:
    total = 0
    for i, batch in enumerate(fetch_event_actor_batches(pg_conn, start_after=start_after)):
        actor1_rows = [{"event_id": e, "actor_id": a} for e, a, role in batch if role == "actor1"]
        actor2_rows = [{"event_id": e, "actor_id": a} for e, a, role in batch if role == "actor2"]
        if actor1_rows:
            client.run(_MERGE_INITIATED, {"rows": actor1_rows})
        if actor2_rows:
            client.run(_MERGE_TARGETED, {"rows": actor2_rows})
        total += len(batch)
        if i % LOG_EVERY_N_BATCHES == 0:
            logger.info("event_actor: %d rows so far (last key=%s)", total, batch[-1])
    logger.info("event_actor: %d rows total", total)
    return total


def migrate_canonical(pg_conn, client: Neo4jClient) -> int:
    with pg_conn.cursor() as cur:
        cur.execute("SELECT actor_id, canonical_actor_id FROM graph.actor WHERE canonical_actor_id IS NOT NULL")
        rows = [{"actor_id": r[0], "canonical_actor_id": r[1]} for r in cur.fetchall()]
    if rows:
        client.run(_MERGE_CANONICAL, {"rows": rows})
    logger.info("canonical_of: %d rows", len(rows))
    return len(rows)


def verify(pg_conn, client: Neo4jClient) -> dict[str, tuple[int, int]]:
    """Reconciles Postgres row counts against Neo4j node/relationship counts.

    This exists because of a real, silent-failure mode found while testing
    this migrator: Cypher's MATCH simply produces zero rows for an UNWIND
    row whose referenced node doesn't exist yet -- MERGE on that path then
    creates nothing, with no error and no warning. If event_actor is ever
    migrated before actor/event finish (wrong --only order, an interrupted
    run resumed incorrectly), edges go missing silently. This is the same
    "a missing row must never be mistaken for a true negative" principle
    already applied to the UCDP PIT labels and the country-code migration
    checks elsewhere in this project -- just checked here instead of
    assumed from correct step ordering.
    """
    checks = [
        ("country", "SELECT COUNT(*) FROM graph.country", "MATCH (n:Country) RETURN count(n) AS n"),
        ("actor", "SELECT COUNT(*) FROM graph.actor", "MATCH (n:Actor) RETURN count(n) AS n"),
        ("location", "SELECT COUNT(*) FROM graph.location", "MATCH (n:Location) RETURN count(n) AS n"),
        ("event", "SELECT COUNT(*) FROM graph.event", "MATCH (n:Event) RETURN count(n) AS n"),
        ("initiated (actor1)",
         "SELECT COUNT(*) FROM graph.event_actor WHERE role = 'actor1'",
         "MATCH ()-[r:INITIATED]->() RETURN count(r) AS n"),
        ("targeted (actor2)",
         "SELECT COUNT(*) FROM graph.event_actor WHERE role = 'actor2'",
         "MATCH ()-[r:TARGETED]->() RETURN count(r) AS n"),
        ("canonical_of",
         "SELECT COUNT(*) FROM graph.actor WHERE canonical_actor_id IS NOT NULL",
         "MATCH ()-[r:CANONICAL_OF]->() RETURN count(r) AS n"),
    ]
    results = {}
    all_match = True
    for name, pg_sql, cy in checks:
        with pg_conn.cursor() as cur:
            cur.execute(pg_sql)
            pg_count = cur.fetchone()[0]
        neo_count = client.run(cy)[0]["n"]
        match = pg_count == neo_count
        all_match &= match
        results[name] = (pg_count, neo_count)
        logger.info("%-22s postgres=%-10d neo4j=%-10d %s", name, pg_count, neo_count,
                    "OK" if match else "MISMATCH")
    if not all_match:
        logger.warning("Verification found mismatches -- migration is incomplete "
                        "or ran out of order. Re-run the missing --only step(s).")
    return results


STEPS = {
    "country": lambda pg, cl, resume: migrate_country(pg, cl),
    "actor": lambda pg, cl, resume: migrate_actor(pg, cl, start_after=resume or 0),
    "location": lambda pg, cl, resume: migrate_location(pg, cl, start_after=resume or 0),
    "event": lambda pg, cl, resume: migrate_event(pg, cl, start_after=resume or 0),
    "event_actor": lambda pg, cl, resume: migrate_event_actor(pg, cl, start_after=resume or (0, 0, "")),
    "canonical": lambda pg, cl, resume: migrate_canonical(pg, cl),
}
STEP_ORDER = ["country", "actor", "location", "event", "event_actor", "canonical"]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Migrate graph.* from Postgres into Neo4j")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of steps to run: " + ",".join(STEP_ORDER))
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint value to resume a single --only step from "
                         "(an integer PK for actor/location/event, or "
                         "'event_id,actor_id,role' for event_actor)")
    ap.add_argument("--pg-dsn", default=PG_DSN)
    ap.add_argument("--verify", action="store_true",
                    help="skip migration, just reconcile Postgres vs Neo4j counts")
    args = ap.parse_args()

    if args.verify:
        pg_conn = psycopg2.connect(args.pg_dsn)
        try:
            with Neo4jClient(Neo4jConfig.from_yaml()) as client:
                verify(pg_conn, client)
        finally:
            pg_conn.close()
        return 0

    steps = args.only.split(",") if args.only else STEP_ORDER
    unknown = set(steps) - set(STEP_ORDER)
    if unknown:
        raise SystemExit(f"Unknown step(s): {unknown}. Valid: {STEP_ORDER}")

    resume = None
    if args.resume_from:
        if len(steps) != 1:
            raise SystemExit("--resume-from requires exactly one --only step")
        if steps[0] == "event_actor":
            eid, aid, role = args.resume_from.split(",")
            resume = (int(eid), int(aid), role.strip("'\""))
        else:
            resume = int(args.resume_from)

    pg_conn = psycopg2.connect(args.pg_dsn)
    try:
        with Neo4jClient(Neo4jConfig.from_yaml()) as client:
            for step in steps:
                logger.info("=== Migrating: %s ===", step)
                STEPS[step](pg_conn, client, resume)
            if steps == STEP_ORDER:
                logger.info("=== Verifying ===")
                verify(pg_conn, client)
    finally:
        pg_conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
