-- ============================================================
-- GeoPulse -- actor-interaction graph schema (usecase.md).
--
-- This formalizes a schema that already existed live in this database from
-- an earlier exploratory session, built by ad hoc scripts that were never
-- saved to the repo -- there was no producing script for `graph.*` before
-- this file. All CREATE TABLE statements below match the live schema
-- exactly (verified via \d against the running DB) and use IF NOT EXISTS,
-- so running this against that same database is a safe no-op on the
-- existing tables; running it against a fresh database creates them.
--
-- Source-agnostic by design (`source` column on actor/event): GDELT is the
-- only source populated today, but graph.polecat_stage/ucdp_stage exist as
-- staging tables for wiring in POLECAT and UCDP as additional sources later,
-- entity-resolved against the same actor registry.
--
-- New in this revision: `event.interaction_type`, backing usecase.md's
-- "map CAMEO event codes to interpretable interaction types (e.g.,
-- cooperation, consultation, conflict)". Computed in Python
-- (data/cameo_codes.py) rather than duplicated as SQL CASE logic, so there
-- is exactly one place the CAMEO taxonomy is encoded.
--
-- Run: psql -U gldt -d gdelt_risk -f scripts/init_graph_schema.sql
-- ============================================================

CREATE SCHEMA IF NOT EXISTS graph;

CREATE TABLE IF NOT EXISTS graph.country (
    iso3    TEXT PRIMARY KEY,
    name    TEXT
);

CREATE TABLE IF NOT EXISTS graph.actor (
    actor_id            BIGSERIAL PRIMARY KEY,
    source              TEXT NOT NULL,
    raw_code            TEXT,
    raw_name            TEXT,
    country_iso3        TEXT REFERENCES graph.country(iso3),
    canonical_actor_id  BIGINT REFERENCES graph.actor(actor_id),
    country_inferred    BOOLEAN DEFAULT FALSE,
    UNIQUE (source, raw_code, country_iso3)
);

CREATE TABLE IF NOT EXISTS graph.location (
    location_id     BIGSERIAL PRIMARY KEY,
    country_iso3    TEXT REFERENCES graph.country(iso3),
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    UNIQUE (country_iso3, lat, lon)
);

CREATE TABLE IF NOT EXISTS graph.event (
    event_id            BIGSERIAL PRIMARY KEY,
    source              TEXT NOT NULL,
    source_event_id     TEXT NOT NULL,
    event_date          DATE NOT NULL,
    event_type          TEXT,               -- CAMEO root code, as text
    intensity           DOUBLE PRECISION,   -- Goldstein scale
    avg_tone            DOUBLE PRECISION,
    fatalities_best      DOUBLE PRECISION,
    fatalities_low       DOUBLE PRECISION,
    fatalities_high      DOUBLE PRECISION,
    num_mentions        INT,
    location_id         BIGINT REFERENCES graph.location(location_id),
    source_url          TEXT,
    UNIQUE (source, source_event_id)
);

-- Additive: derived from event_type via data/cameo_codes.interaction_type().
-- CHECK, not FK -- this is a fixed 3-value enum, not a growing reference table.
ALTER TABLE graph.event
    ADD COLUMN IF NOT EXISTS interaction_type TEXT
        CHECK (interaction_type IN ('cooperation', 'consultation', 'conflict'));

CREATE INDEX IF NOT EXISTS event_event_date_idx ON graph.event (event_date);
CREATE INDEX IF NOT EXISTS event_source_event_date_idx ON graph.event (source, event_date);
CREATE INDEX IF NOT EXISTS event_interaction_type_date_idx ON graph.event (interaction_type, event_date);

CREATE TABLE IF NOT EXISTS graph.event_actor (
    event_id    BIGINT NOT NULL REFERENCES graph.event(event_id),
    actor_id    BIGINT NOT NULL REFERENCES graph.actor(actor_id),
    role        TEXT NOT NULL CHECK (role IN ('actor1', 'actor2')),
    PRIMARY KEY (event_id, actor_id, role)
);

CREATE INDEX IF NOT EXISTS event_actor_actor_id_role_idx ON graph.event_actor (actor_id, role);
CREATE INDEX IF NOT EXISTS event_actor_event_id_role_idx ON graph.event_actor (event_id, role);

-- Staging tables for future sources, pre-entity-resolution -- flat POLECAT/
-- UCDP fields, not yet mapped onto graph.actor/event. Not wired into the
-- GDELT builder this revision adds; left exactly as they exist live.
CREATE TABLE IF NOT EXISTS graph.polecat_stage (
    event_id            TEXT,
    event_date          DATE,
    event_type          TEXT,
    intensity           DOUBLE PRECISION,
    quad_code           TEXT,
    actor_name          TEXT,
    actor_country       TEXT,
    actor_sector        TEXT,
    recipient_name      TEXT,
    recipient_country   TEXT,
    recipient_sector    TEXT,
    geo_country         TEXT,
    lat                 DOUBLE PRECISION,
    lon                 DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS graph.ucdp_stage (
    event_date      DATE,
    country         TEXT,
    side_a          TEXT,
    side_b          TEXT,
    violence_type   INT,
    best            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION
);
