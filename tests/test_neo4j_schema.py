"""Verifies the Neo4j schema from docs/neo4j_schema.md is applied. Skipped
if Neo4j isn't reachable. Run scripts/init_neo4j_schema.py first."""

from __future__ import annotations

import pytest

neo4j = pytest.importorskip("neo4j")

from ingestion.neo4j_client import Neo4jClient, Neo4jConfig

EXPECTED_CONSTRAINTS = {
    ("Country", "iso3"),
    ("Actor", "actor_id"),
    ("Location", "location_id"),
    ("Event", "event_id"),
}


@pytest.fixture(scope="module")
def client():
    c = Neo4jClient(Neo4jConfig.from_yaml())
    try:
        c.connect()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable ({exc}); skipping schema tests")
    yield c
    c.close()


def test_node_key_constraints_present(client):
    rows = client.run("SHOW CONSTRAINTS YIELD labelsOrTypes, properties, type WHERE type = 'NODE_KEY'")
    present = {(r["labelsOrTypes"][0], r["properties"][0]) for r in rows}
    assert EXPECTED_CONSTRAINTS <= present


def test_event_date_and_interaction_type_indexed(client):
    rows = client.run(
        "SHOW INDEXES YIELD labelsOrTypes, properties WHERE 'Event' IN labelsOrTypes"
    )
    indexed_props = {p for r in rows for p in r["properties"]}
    assert "event_date" in indexed_props
    assert "interaction_type" in indexed_props


def test_actor_country_indexed(client):
    rows = client.run(
        "SHOW INDEXES YIELD labelsOrTypes, properties WHERE 'Actor' IN labelsOrTypes"
    )
    indexed_props = {p for r in rows for p in r["properties"]}
    assert "country_iso3" in indexed_props
