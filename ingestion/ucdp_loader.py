"""
UCDP GED + Actor loader -> `ucdp` Postgres schema.

Ground-truth ingestion for the point-in-time label pipeline. Unlike the GDELT
path, this is a one-shot bulk load from a versioned release archive, not a
streaming pipeline -- UCDP publishes GED annually (with a monthly candidate
series), so re-running this on a new release is the whole update story.

Sources (both shipped in data/UCDP/):
    ged261-csv.zip          -> GEDEvent_v26_1.csv   (417,968 events, 1989-2025)
    ucdp-actor-261-csv.zip  -> Actor_v26_1.csv      (canonical actor registry)

The GED CSV is 1:1 with ucdp.ged_events -- every column is preserved, nothing
is derived here. Derivation happens in preprocessing/pit_labels.py, which reads
this table. Keeping load and derivation separate means a label-definition
change never requires re-reading a 39MB archive, and a reviewer can check any
aggregate against the raw rows.

Usage:
    python -m ingestion.ucdp_loader              # load both, skip if populated
    python -m ingestion.ucdp_loader --truncate   # force full reload
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator

import psycopg2
import psycopg2.extras

logger = logging.getLogger("ingestion.ucdp_loader")

REPO_ROOT = Path(__file__).resolve().parent.parent
UCDP_DIR = REPO_ROOT / "data" / "UCDP"

GED_ZIP = UCDP_DIR / "ged261-csv.zip"
GED_MEMBER = "GEDEvent_v26_1.csv"
ACTOR_ZIP = UCDP_DIR / "ucdp-actor-261-csv.zip"
ACTOR_MEMBER = "Actor_v26_1.csv"

DSN = os.environ.get(
    "GEOPULSE_DSN",
    "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432",
)

# CSV field order == ucdp.ged_events column order (verified against the header).
GED_COLUMNS = [
    "id", "relid", "year", "active_year", "code_status", "type_of_violence",
    "conflict_dset_id", "conflict_new_id", "conflict_name",
    "dyad_dset_id", "dyad_new_id", "dyad_name",
    "side_a_dset_id", "side_a_new_id", "side_a",
    "side_b_dset_id", "side_b_new_id", "side_b",
    "number_of_sources", "source_article", "source_office", "source_date",
    "source_headline", "source_original",
    "where_prec", "where_coordinates", "where_description", "adm_1", "adm_2",
    "latitude", "longitude", "geom_wkt", "priogrid_gid",
    "country", "country_id", "region",
    "event_clarity", "date_prec", "date_start", "date_end",
    "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown",
    "best", "high", "low", "gwnoa", "gwnob",
]

GED_INT_COLS = {
    "id", "year", "type_of_violence", "conflict_new_id", "dyad_new_id",
    "side_a_new_id", "side_b_new_id", "number_of_sources", "where_prec",
    "priogrid_gid", "country_id", "event_clarity", "date_prec",
    "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown",
    "best", "high", "low",
}
GED_FLOAT_COLS = {"latitude", "longitude"}
GED_DATE_COLS = {"date_start", "date_end"}
GED_BOOL_COLS = {"active_year"}

# Actor CSV header -> ucdp.actor column. The CSV carries 35 columns of
# coalition/alliance/splinter bookkeeping; we keep the identity and lineage
# fields, which are what entity resolution needs.
ACTOR_MAP = {
    "ActorId": "actor_id",
    "NameData": "name_data",
    "NameOrig": "name_orig",
    "NameOrigFullEng": "name_orig_full_eng",
    "Org": "org",
    "ConflictId": "conflict_ids",
    "DyadId": "dyad_ids",
    "PrimaryParty": "primary_party",
    "NameChange": "name_change",
    "NewName": "new_name",
    "NamePrev": "name_prev",
    "ActorIdPrev": "actor_id_prev",
    "Location": "location",
    "GWNOLoc": "gwno_loc",
    "Region": "region",
    "Version": "version",
}
ACTOR_COLUMNS = list(ACTOR_MAP.values())

BATCH = 5000


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

def _coerce_ged(field: str, raw: str) -> Any:
    """Empty string -> NULL; UCDP uses '' for every missing value, and psycopg2
    would otherwise write '' into an INT column and fail the whole batch."""
    s = raw.strip()
    if s == "":
        return None
    if field in GED_INT_COLS:
        try:
            return int(float(s))  # some int fields arrive as '615.0'
        except ValueError:
            return None
    if field in GED_FLOAT_COLS:
        try:
            return float(s)
        except ValueError:
            return None
    if field in GED_BOOL_COLS:
        return s.lower() in ("true", "1", "t", "yes")
    if field in GED_DATE_COLS:
        return s[:10]  # '1992-03-17 00:00:00.000' -> '1992-03-17'
    return s


def _iter_ged(path: Path, member: str) -> Iterator[tuple]:
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            missing = set(GED_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{member} is missing expected GED columns: {sorted(missing)}. "
                    "This is probably the wrong UCDP file -- the Violent Political "
                    "Protest variant has a different shape and has been mistaken "
                    "for the full GED in this repo before."
                )
            for row in reader:
                yield tuple(_coerce_ged(c, row.get(c, "")) for c in GED_COLUMNS)


def _iter_actor(path: Path, member: str) -> Iterator[tuple]:
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as fh:
            # The Actor release is cp1252, not UTF-8 -- actor names carry
            # accented Latin characters (e.g. 'Frente Sandinista de Liberación').
            # GED is UTF-8; the two files disagree, so don't unify the encoding.
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="cp1252"))
            for row in reader:
                if not (row.get("ActorId") or "").strip():
                    continue  # the release ends with blank padding rows
                vals = []
                for src, dst in ACTOR_MAP.items():
                    v = (row.get(src) or "").strip()
                    if dst == "actor_id":
                        vals.append(int(float(v)) if v else None)
                    else:
                        vals.append(v or None)
                yield tuple(vals)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _bulk_load(conn, table: str, columns: list[str], rows: Iterator[tuple],
               conflict_key: str) -> int:
    cols = ", ".join(columns)
    sql = (
        f"INSERT INTO {table} ({cols}) VALUES %s "
        f"ON CONFLICT ({conflict_key}) DO NOTHING"
    )
    total = 0
    batch: list[tuple] = []
    with conn.cursor() as cur:
        for row in rows:
            batch.append(row)
            if len(batch) >= BATCH:
                psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH)
                total += len(batch)
                batch.clear()
                if total % (BATCH * 10) == 0:
                    logger.info("  %s: %d rows", table, total)
        if batch:
            psycopg2.extras.execute_values(cur, sql, batch, page_size=BATCH)
            total += len(batch)
    conn.commit()
    return total


def _count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]


def load(truncate: bool = False, dsn: str = DSN) -> dict[str, int]:
    for p in (GED_ZIP, ACTOR_ZIP):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download from https://ucdp.uu.se/downloads/ "
                "(GED Global and the UCDP Actor Dataset) into data/UCDP/."
            )

    conn = psycopg2.connect(dsn)
    try:
        if truncate:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE ucdp.ged_events, ucdp.actor")
            conn.commit()
            logger.info("Truncated ucdp.ged_events, ucdp.actor")

        results = {}
        for table, cols, it, key in (
            ("ucdp.ged_events", GED_COLUMNS, _iter_ged(GED_ZIP, GED_MEMBER), "id"),
            ("ucdp.actor", ACTOR_COLUMNS, _iter_actor(ACTOR_ZIP, ACTOR_MEMBER), "actor_id"),
        ):
            existing = _count(conn, table)
            if existing and not truncate:
                logger.info("%s already has %d rows; skipping (use --truncate to reload)",
                            table, existing)
                results[table] = existing
                continue
            logger.info("Loading %s ...", table)
            _bulk_load(conn, table, cols, it, key)
            results[table] = _count(conn, table)
            logger.info("%s: %d rows", table, results[table])

        return results
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Load UCDP GED + Actor into Postgres")
    ap.add_argument("--truncate", action="store_true",
                    help="wipe and reload (needed after a new UCDP release)")
    ap.add_argument("--dsn", default=DSN)
    args = ap.parse_args()

    results = load(truncate=args.truncate, dsn=args.dsn)
    for table, n in results.items():
        print(f"{table}: {n:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
