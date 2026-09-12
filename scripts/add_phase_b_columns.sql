-- ============================================================
-- Phase B parse-time additions to gdelt_events (plan Step 5 / B2, B5).
--
-- Purely additive: nullable columns, no default requiring a table rewrite,
-- so this is metadata-only on a partitioned table and safe to run against a
-- live table with existing rows (which keep NULL for these columns until
-- re-parsed or newly ingested). All 8 columns are already present in the
-- raw GDELT v1 export -- ingestion/gdelt_parser.py was simply discarding
-- them; this migration just gives them somewhere to land.
--
-- Run: psql -U gldt -d gdelt_risk -f scripts/add_phase_b_columns.sql
-- ============================================================

ALTER TABLE gdelt_events
    ADD COLUMN IF NOT EXISTS action_geo_adm1            TEXT,
    ADD COLUMN IF NOT EXISTS actor1_known_group_code    TEXT,
    ADD COLUMN IF NOT EXISTS actor2_known_group_code    TEXT,
    ADD COLUMN IF NOT EXISTS actor1_type2               TEXT,
    ADD COLUMN IF NOT EXISTS actor1_type3               TEXT,
    ADD COLUMN IF NOT EXISTS actor2_type2               TEXT,
    ADD COLUMN IF NOT EXISTS actor2_type3               TEXT,
    ADD COLUMN IF NOT EXISTS is_root_event               BOOLEAN;

-- B5 geographic-dispersion features group by adm1 within a country/date;
-- this index is what makes that GROUP BY cheap instead of a seq scan.
CREATE INDEX IF NOT EXISTS idx_gdelt_geo_adm1
    ON gdelt_events (action_geo_country, action_geo_adm1, event_date DESC);
