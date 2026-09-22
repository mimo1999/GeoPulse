-- ============================================================
-- Indexes for country-level rollup queries (country summaries,
-- highlighted events, activity heatmap) against graph.*.
--
-- graph.event had no index on location_id -- every "events in country X"
-- query (Event -> Location -> Country) would force a sequential scan of
-- the full 48.9M-row event table. graph.actor.country_iso3 was only
-- indexed as the non-leading tail of a composite unique constraint, not
-- usable for a plain "actors from country X" filter (though actor is only
-- 51,849 rows, so this one is a smaller win -- included for consistency).
--
-- CONCURRENTLY: the Neo4j migration (ingestion/neo4j_migrator.py) is
-- actively read-scanning graph.event/graph.event_actor while this runs.
-- A plain CREATE INDEX takes a SHARE lock that would block the
-- migration's writes to graph itself if it had any (it doesn't -- it's
-- read-only against Postgres) but CONCURRENTLY avoids any lock
-- contention risk and is the correct choice on a live table regardless.
-- Cannot run inside a transaction block -- run this file's statements
-- individually if using a client that wraps scripts in one transaction.
--
-- Run: psql -U gldt -d gdelt_risk -f scripts/add_graph_country_indexes.sql
-- ============================================================

CREATE INDEX CONCURRENTLY IF NOT EXISTS event_location_id_idx
    ON graph.event (location_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS actor_country_iso3_idx
    ON graph.actor (country_iso3);
