-- ============================================================
-- ONE-TIME ADMIN SCRIPT -- run as postgres (or another superuser), not gldt.
--
-- gdelt_events and all 84 of its monthly partitions were created by
-- postgres; ALTER TABLE gdelt_events OWNER TO gldt does NOT cascade to
-- partitions (each is a separate object with its own owner), so
-- ADD COLUMN on the partitioned parent fails atomically the moment it
-- reaches the first partition gldt doesn't own -- nothing gets applied,
-- not even to the parent, since it's one DDL statement across all of them.
--
-- Also grants CREATE on schema public, needed for scripts/add_phase_b_columns.sql's
-- new index (PG15+ revokes this by default; gldt already has it on the
-- dedicated `ucdp` and `graph` schemas via their own init scripts, just not
-- on public, where gdelt_events happens to already live).
--
-- Run: psql -U postgres -d gdelt_risk -f scripts/admin_grant_gldt_ownership.sql
-- ============================================================

ALTER TABLE gdelt_events OWNER TO gldt;

DO $$
DECLARE
    part RECORD;
BEGIN
    FOR part IN
        SELECT inhrelid::regclass AS partition_name
        FROM pg_inherits
        WHERE inhparent = 'gdelt_events'::regclass
    LOOP
        EXECUTE format('ALTER TABLE %s OWNER TO gldt', part.partition_name);
    END LOOP;
END
$$;

GRANT CREATE ON SCHEMA public TO gldt;
