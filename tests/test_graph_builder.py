"""
Integration tests for ingestion/graph_builder.py. Skipped if Postgres isn't
reachable, same pattern as tests/test_pit_labels.py.

Uses a date outside 2023-2025 (2026-02-21, sparse pre-Phase-5 ingestion
data) deliberately, so these tests can run concurrently with a 2023-2025
build without racing on the same event_date partitions.
"""

from __future__ import annotations

from datetime import date

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from data.cameo_codes import interaction_type
from ingestion.graph_builder import GraphBuilder

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"
TEST_DATE = date(2026, 2, 21)


@pytest.fixture(scope="module")
def raw_conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc}); skipping graph_builder tests")
    yield c
    c.close()


@pytest.fixture(scope="module")
def builder(raw_conn):
    with raw_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gdelt_events WHERE event_date = %s", (TEST_DATE,))
        if cur.fetchone()[0] == 0:
            pytest.skip(f"no gdelt_events for {TEST_DATE} to build against")
    b = GraphBuilder(DSN)
    b.build_date(TEST_DATE)  # idempotent: ON CONFLICT DO NOTHING/UPDATE throughout
    yield b
    b.close()


def test_event_count_matches_raw(raw_conn, builder):
    with raw_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM gdelt_events WHERE event_date = %s", (TEST_DATE,))
        raw_count = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM graph.event WHERE source = 'gdelt' AND event_date = %s",
            (TEST_DATE,),
        )
        graph_count = cur.fetchone()[0]
    assert graph_count == raw_count


def test_interaction_type_matches_cameo_mapping(raw_conn, builder):
    """For a sample of graph.event rows on TEST_DATE, interaction_type must
    equal what data.cameo_codes.interaction_type() computes independently
    from the raw event's own root code -- the single source of truth is not
    silently drifting from what got stored."""
    with raw_conn.cursor() as cur:
        cur.execute(
            """SELECT g.event_type, g.interaction_type, e.event_root_code, e.quad_class
               FROM graph.event g
               JOIN gdelt_events e ON e.global_event_id::text = g.source_event_id
               WHERE g.source = 'gdelt' AND g.event_date = %s
               LIMIT 200""",
            (TEST_DATE,),
        )
        rows = cur.fetchall()
    assert rows
    for event_type, stored_itype, root_code, quad_class in rows:
        expected = interaction_type(root_code=root_code, quad_class=quad_class)
        assert stored_itype == expected, (event_type, stored_itype, expected, root_code)


def test_no_actor_country_collapse(raw_conn, builder):
    """The bug class this schema exists to prevent: the same raw role code
    in two different countries must be two different actor rows, never
    merged into one."""
    with raw_conn.cursor() as cur:
        cur.execute(
            """SELECT raw_code, COUNT(DISTINCT country_iso3)
               FROM graph.actor
               WHERE source = 'gdelt' AND raw_code = 'GOV' AND country_iso3 IS NOT NULL
               GROUP BY raw_code"""
        )
        row = cur.fetchone()
    if row is None:
        pytest.skip("no resolved GOV actors yet to check")
    _, n_distinct_countries = row
    assert n_distinct_countries > 1, "expected GOV to resolve to multiple distinct countries"


def test_rerun_is_idempotent(raw_conn, builder):
    """Running build_date twice must not create duplicate events or edges."""
    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM graph.event WHERE source = 'gdelt' AND event_date = %s",
            (TEST_DATE,),
        )
        before = cur.fetchone()[0]

    builder.build_date(TEST_DATE)

    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM graph.event WHERE source = 'gdelt' AND event_date = %s",
            (TEST_DATE,),
        )
        after = cur.fetchone()[0]
    assert after == before


def test_bare_role_actor_gets_inferred_country(raw_conn, builder):
    with raw_conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM graph.actor
               WHERE source = 'gdelt' AND raw_code = 'GOV'
                 AND country_inferred = true AND country_iso3 IS NOT NULL"""
        )
        n = cur.fetchone()[0]
    assert n > 0, "expected at least one bare-role GOV actor with an inferred country"
