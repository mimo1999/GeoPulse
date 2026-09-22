"""
Applies the Neo4j schema (constraints + indexes) from docs/neo4j_schema.md.

Schema-only -- no data is loaded here. Idempotent (all statements use
IF NOT EXISTS), safe to re-run.

Usage:
    python -m scripts.init_neo4j_schema
"""

from __future__ import annotations

import logging

from ingestion.neo4j_client import Neo4jClient, Neo4jConfig

logger = logging.getLogger("scripts.init_neo4j_schema")

STATEMENTS = [
    "CREATE CONSTRAINT country_key IF NOT EXISTS FOR (c:Country) REQUIRE c.iso3 IS NODE KEY",
    "CREATE CONSTRAINT actor_key IF NOT EXISTS FOR (a:Actor) REQUIRE a.actor_id IS NODE KEY",
    "CREATE CONSTRAINT location_key IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS NODE KEY",
    "CREATE CONSTRAINT event_key IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS NODE KEY",
    "CREATE INDEX event_date_idx IF NOT EXISTS FOR (e:Event) ON (e.event_date)",
    "CREATE INDEX event_interaction_idx IF NOT EXISTS FOR (e:Event) ON (e.interaction_type)",
    "CREATE INDEX actor_country_idx IF NOT EXISTS FOR (a:Actor) ON (a.country_iso3)",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with Neo4jClient(Neo4jConfig.from_yaml()) as client:
        for stmt in STATEMENTS:
            logger.info("Applying: %s", stmt)
            client.run(stmt)

        constraints = client.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties")
        indexes = client.run(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE type <> 'LOOKUP'"
        )
        logger.info("%d constraints, %d indexes now present", len(constraints), len(indexes))
        for c in constraints:
            print(f"  constraint {c['name']}: {c['labelsOrTypes']} {c['properties']} ({c['type']})")
        for i in indexes:
            print(f"  index {i['name']}: {i['labelsOrTypes']} {i['properties']} ({i['type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
