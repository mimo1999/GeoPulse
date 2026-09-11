-- ============================================================
-- GeoPulse — UCDP ground-truth schema
--
-- Replaces the GDELT-derived proxy labels (preprocessing/label_generator.py,
-- train_real_data.py::compute_proxy_labels) with independent, human-coded
-- ground truth from the UCDP Georeferenced Event Dataset.
--
-- Point-in-time discipline: for run_date t, features come only from data
-- dated <= t - 7d, and labels only from (t, t + 30d]. The label tables here
-- store BOTH the raw aggregates and the derived targets, so any threshold is
-- re-derivable without re-ingesting, and a reviewer can verify a definition.
--
-- Run:
--   psql -U gldt -d gdelt_risk -f scripts/init_ucdp.sql
-- ============================================================

-- gldt is not superuser and PG15+ revokes CREATE on schema public,
-- so UCDP objects live in their own schema (same pattern as `graph`).
CREATE SCHEMA IF NOT EXISTS ucdp;

-- ------------------------------------------------------------
-- Raw GED events — 1:1 with GEDEvent_v26_1.csv
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ucdp.ged_events (
    id                  BIGINT PRIMARY KEY,
    relid               TEXT,
    year                INT,
    active_year         BOOLEAN,
    code_status         TEXT,
    type_of_violence    SMALLINT,        -- 1=state-based 2=non-state 3=one-sided
    conflict_dset_id    TEXT,
    conflict_new_id     INT,
    conflict_name       TEXT,
    dyad_dset_id        TEXT,
    dyad_new_id         INT,
    dyad_name           TEXT,
    side_a_dset_id      TEXT,
    side_a_new_id       INT,
    side_a              TEXT,
    side_b_dset_id      TEXT,
    side_b_new_id       INT,
    side_b              TEXT,
    number_of_sources   INT,
    source_article      TEXT,
    source_office       TEXT,
    source_date         TEXT,
    source_headline     TEXT,
    source_original     TEXT,
    where_prec          SMALLINT,
    where_coordinates   TEXT,
    where_description   TEXT,
    adm_1               TEXT,
    adm_2               TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    geom_wkt            TEXT,
    priogrid_gid        INT,
    country             TEXT,
    country_id          INT,             -- Gleditsch-Ward number
    region              TEXT,
    event_clarity       SMALLINT,
    date_prec           SMALLINT,        -- 1=exact day ... 5=within the year
    date_start          DATE,
    date_end            DATE,
    deaths_a            INT,
    deaths_b            INT,
    deaths_civilians    INT,
    deaths_unknown      INT,
    best                INT,             -- best fatality estimate
    high                INT,
    low                 INT,
    gwnoa               TEXT,
    gwnob               TEXT,
    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ged_country_date ON ucdp.ged_events (country_id, date_start);
CREATE INDEX IF NOT EXISTS idx_ged_date         ON ucdp.ged_events (date_start);
CREATE INDEX IF NOT EXISTS idx_ged_dyad         ON ucdp.ged_events (dyad_new_id);
CREATE INDEX IF NOT EXISTS idx_ged_tov          ON ucdp.ged_events (type_of_violence);

-- ------------------------------------------------------------
-- Canonical actor registry — from Actor_v26_1.csv
-- Includes rename/split lineage, which is why we resolve GDELT actors
-- *against* this rather than building alias matching from scratch.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ucdp.actor (
    actor_id                INT PRIMARY KEY,
    name_data               TEXT,
    name_orig               TEXT,
    name_orig_full_eng      TEXT,
    org                     TEXT,
    conflict_ids            TEXT,
    dyad_ids                TEXT,
    primary_party           TEXT,
    name_change             TEXT,
    new_name                TEXT,
    name_prev               TEXT,
    actor_id_prev           TEXT,
    location                TEXT,
    gwno_loc                TEXT,
    region                  TEXT,
    version                 TEXT,
    loaded_at               TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Gleditsch-Ward -> ISO3 -> FIPS2 bridge.
-- This repo has already shipped one FIPS/ISO-3 bug that rendered zero
-- countries on the map; unmapped rows must be explicit, never silent.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ucdp.gw_country_map (
    gw_id               INT PRIMARY KEY,
    ucdp_country_name   TEXT NOT NULL,
    iso3                CHAR(3),
    fips2               CHAR(2),
    note                TEXT,            -- reason when fips2 IS NULL
    loaded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gwmap_fips ON ucdp.gw_country_map (fips2);

-- ------------------------------------------------------------
-- Country-level point-in-time label panel.
-- One row per (country, run_date) INCLUDING ZEROS — absence of a row must
-- never be mistaken for absence of violence.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ucdp.country_pit_labels (
    fips2               CHAR(2) NOT NULL,
    gw_id               INT,
    run_date            DATE NOT NULL,       -- first day of the label month
    label_version       TEXT NOT NULL DEFAULT 'ged-v26.1',

    -- raw aggregates over the label window (t, t+30d]
    n_events            INT NOT NULL DEFAULT 0,
    deaths_best         INT NOT NULL DEFAULT 0,
    deaths_low          INT NOT NULL DEFAULT 0,
    deaths_high         INT NOT NULL DEFAULT 0,
    deaths_civilians    INT NOT NULL DEFAULT 0,
    deaths_sb           INT NOT NULL DEFAULT 0,
    deaths_ns           INT NOT NULL DEFAULT 0,
    deaths_os           INT NOT NULL DEFAULT 0,
    n_events_sb         INT NOT NULL DEFAULT 0,
    n_events_ns         INT NOT NULL DEFAULT 0,
    n_events_os         INT NOT NULL DEFAULT 0,
    n_dyads             INT NOT NULL DEFAULT 0,
    n_adm1              INT NOT NULL DEFAULT 0,

    -- prior window (t-30d, t] — needed for the escalation delta
    prev_deaths_best    INT NOT NULL DEFAULT 0,
    prev_n_events       INT NOT NULL DEFAULT 0,

    -- Tier 1 targets (prediction)
    escalation_delta    DOUBLE PRECISION,    -- log1p(deaths) - log1p(prev_deaths)
    escalation_dir      SMALLINT,            -- -1 down / 0 flat / +1 up
    civ_share           DOUBLE PRECISION,    -- NULL unless deaths_best >= 25

    -- Tier 2 targets (monitoring; persistence-dominated, report BSS only)
    any_violence        BOOLEAN NOT NULL DEFAULT FALSE,
    minor_conflict      BOOLEAN NOT NULL DEFAULT FALSE,  -- >=25, UCDP's own threshold
    war_intensity       BOOLEAN NOT NULL DEFAULT FALSE,  -- >=100
    sb_minor            BOOLEAN NOT NULL DEFAULT FALSE,
    ns_minor            BOOLEAN NOT NULL DEFAULT FALSE,
    os_minor            BOOLEAN NOT NULL DEFAULT FALSE,

    -- descriptive only — too rare to claim at the 2023-2025 scope
    onset_3m                BOOLEAN,
    months_since_last_minor INT,

    -- provenance of the filter used to build this row
    date_prec_max       SMALLINT NOT NULL DEFAULT 4,
    clarity_filter      TEXT NOT NULL DEFAULT '1',
    computed_at         TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (fips2, run_date, label_version)
);

CREATE INDEX IF NOT EXISTS idx_cpl_run   ON ucdp.country_pit_labels (run_date);
CREATE INDEX IF NOT EXISTS idx_cpl_fips  ON ucdp.country_pit_labels (fips2, run_date);

-- ------------------------------------------------------------
-- Dyad-level point-in-time panel — target P2 (who will be fighting).
-- Candidate set = dyads seen in the trailing 12 months for that country;
-- 93.7% of next-month-active dyads fall inside it.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ucdp.dyad_pit_labels (
    fips2                   CHAR(2) NOT NULL,
    run_date                DATE NOT NULL,
    dyad_new_id             INT NOT NULL,
    label_version           TEXT NOT NULL DEFAULT 'ged-v26.1',

    dyad_name               TEXT,
    side_a                  TEXT,
    side_b                  TEXT,

    in_candidate_set        BOOLEAN NOT NULL DEFAULT TRUE,
    active                  BOOLEAN NOT NULL DEFAULT FALSE,  -- active in (t, t+30d]
    deaths_best             INT NOT NULL DEFAULT 0,
    is_new_dyad             BOOLEAN NOT NULL DEFAULT FALSE,  -- unseen in trailing 12m
    months_since_last_active INT,
    active_months_trailing12 INT NOT NULL DEFAULT 0,

    computed_at             TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (fips2, run_date, dyad_new_id, label_version)
);

CREATE INDEX IF NOT EXISTS idx_dpl_run  ON ucdp.dyad_pit_labels (run_date);
CREATE INDEX IF NOT EXISTS idx_dpl_fips ON ucdp.dyad_pit_labels (fips2, run_date);
CREATE INDEX IF NOT EXISTS idx_dpl_dyad ON ucdp.dyad_pit_labels (dyad_new_id);

-- ------------------------------------------------------------
-- GRANTS
-- ------------------------------------------------------------
GRANT ALL PRIVILEGES ON ALL TABLES    IN SCHEMA ucdp TO gldt;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ucdp TO gldt;
