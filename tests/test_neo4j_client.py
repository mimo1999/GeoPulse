"""Integration test for ingestion/neo4j_client.py. Skipped if the local
Neo4j instance isn't reachable, same pattern as the Postgres DB-integration
tests elsewhere in this suite."""

from __future__ import annotations

import pytest

neo4j = pytest.importorskip("neo4j")

from ingestion.neo4j_client import Neo4jClient, Neo4jConfig


@pytest.fixture(scope="module")
def client():
    c = Neo4jClient(Neo4jConfig.from_yaml())
    try:
        c.connect()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable ({exc}); skipping Neo4j integration tests")
    yield c
    c.close()


def test_config_loads_from_yaml():
    cfg = Neo4jConfig.from_yaml()
    assert cfg.uri.startswith("neo4j://") or cfg.uri.startswith("bolt://")
    assert cfg.user


def test_simple_query(client):
    rows = client.run("RETURN 1 AS ok")
    assert rows == [{"ok": 1}]


def test_write_and_read_roundtrip(client):
    client.run("MERGE (n:GeoPulseSmokeTest {id: 'test'}) SET n.value = 42")
    rows = client.run("MATCH (n:GeoPulseSmokeTest {id: 'test'}) RETURN n.value AS value")
    assert rows == [{"value": 42}]
    client.run("MATCH (n:GeoPulseSmokeTest {id: 'test'}) DELETE n")
