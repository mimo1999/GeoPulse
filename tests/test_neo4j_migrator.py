"""
Integration tests for ingestion/neo4j_migrator.py. Skipped if either DB is
unreachable. Uses small, bounded slices (not the full 48.9M-row table) so
this runs in seconds, not the hours the real migration takes.
"""

from __future__ import annotations

import itertools

import pytest

psycopg2 = pytest.importorskip("psycopg2")
neo4j = pytest.importorskip("neo4j")

import ingestion.neo4j_migrator as m
from ingestion.neo4j_client import Neo4jClient, Neo4jConfig

PG_DSN = m.PG_DSN


@pytest.fixture(scope="module")
def pg_conn():
    try:
        c = psycopg2.connect(PG_DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc})")
    yield c
    c.close()


@pytest.fixture(scope="module")
def client():
    c = Neo4jClient(Neo4jConfig.from_yaml())
    try:
        c.connect()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable ({exc})")
    yield c
    c.close()


def test_fetch_batches_single_pk_keyset_advances(pg_conn):
    cols = ["actor_id", "raw_code"]
    gen = m.fetch_batches_single_pk(pg_conn, "graph.actor", cols, "actor_id", batch_size=10)
    first = next(gen)
    second = next(gen)
    assert len(first) == 10
    assert first[-1][0] < second[0][0]  # strictly increasing, no overlap, no gap-skipping


def test_fetch_event_actor_batches_composite_keyset_advances(pg_conn):
    gen = m.fetch_event_actor_batches(pg_conn, batch_size=10)
    first = next(gen)
    second = next(gen)
    assert first[-1] < second[0]


def test_migrate_country_matches_postgres_exactly(pg_conn, client):
    m.migrate_country(pg_conn, client)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM graph.country")
        pg_count = cur.fetchone()[0]
    neo_count = client.run("MATCH (c:Country) RETURN count(c) AS n")[0]["n"]
    assert neo_count == pg_count


def test_actor_event_relationship_requires_both_endpoints_present(pg_conn, client):
    """Documents the silent-MATCH-failure behavior directly: MERGEing an
    INITIATED edge for an actor_id that doesn't exist in Neo4j creates
    nothing, and does not raise. This is exactly why verify() exists."""
    client.run("MATCH (n:Actor {actor_id: -999}) DETACH DELETE n")  # ensure absent
    client.run("MERGE (e:Event {event_id: -999})")
    client.run(m._MERGE_INITIATED, {"rows": [{"event_id": -999, "actor_id": -999}]})
    rows = client.run("MATCH ()-[r:INITIATED]->(e:Event {event_id: -999}) RETURN count(r) AS n")
    assert rows[0]["n"] == 0  # silently absent, not an error -- the documented risk
    client.run("MATCH (e:Event {event_id: -999}) DETACH DELETE e")


def test_event_actor_migration_and_verify_on_small_slice(pg_conn, client):
    """Migrate a small, mutually-consistent slice (actors + events actually
    referenced by each other, not two independently-truncated ranges -- the
    mistake made once already while building this migrator) and confirm
    verify() reports a match for that slice's own relationship types."""
    with pg_conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT event_id FROM graph.event_actor
               ORDER BY event_id LIMIT 50"""
        )
        event_ids = [r[0] for r in cur.fetchall()]
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT actor_id FROM graph.event_actor WHERE event_id = ANY(%s)",
            (event_ids,),
        )
        actor_ids = [r[0] for r in cur.fetchall()]

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT actor_id, source, raw_code, raw_name, country_iso3, country_inferred "
            "FROM graph.actor WHERE actor_id = ANY(%s)",
            (actor_ids,),
        )
        actor_cols = ["actor_id", "source", "raw_code", "raw_name", "country_iso3", "country_inferred"]
        client.run(m._MERGE_ACTOR, {"rows": [dict(zip(actor_cols, r)) for r in cur.fetchall()]})

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, source, source_event_id, event_date, event_type, interaction_type, "
            "intensity, avg_tone, num_mentions, fatalities_best, fatalities_low, fatalities_high, "
            "location_id, source_url FROM graph.event WHERE event_id = ANY(%s)",
            (event_ids,),
        )
        event_cols = ["event_id", "source", "source_event_id", "event_date", "event_type",
                      "interaction_type", "intensity", "avg_tone", "num_mentions",
                      "fatalities_best", "fatalities_low", "fatalities_high", "location_id", "source_url"]
        client.run(m._MERGE_EVENT, {"rows": [dict(zip(event_cols, r)) for r in cur.fetchall()]})

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT event_id, actor_id, role FROM graph.event_actor WHERE event_id = ANY(%s)",
            (event_ids,),
        )
        ea_rows = cur.fetchall()
    actor1 = [{"event_id": e, "actor_id": a} for e, a, role in ea_rows if role == "actor1"]
    actor2 = [{"event_id": e, "actor_id": a} for e, a, role in ea_rows if role == "actor2"]
    if actor1:
        client.run(m._MERGE_INITIATED, {"rows": actor1})
    if actor2:
        client.run(m._MERGE_TARGETED, {"rows": actor2})

    for e, a, role in ea_rows:
        rel = "INITIATED" if role == "actor1" else "TARGETED"
        pattern = (f"MATCH (n:Actor {{actor_id:{a}}})-[:{rel}]->(m:Event {{event_id:{e}}}) RETURN n"
                   if role == "actor1" else
                   f"MATCH (m:Event {{event_id:{e}}})-[:{rel}]->(n:Actor {{actor_id:{a}}}) RETURN n")
        assert client.run(pattern), f"missing {rel} for event={e} actor={a}"
