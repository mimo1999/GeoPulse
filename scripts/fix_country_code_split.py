"""
One-time fix: undo the FIPS/ISO country-code split in gdelt_events.

Root cause: ingestion/event_cleaner.py::EventCleaner.FIPS_TO_ISO used to
rewrite 8 of GDELT's native FIPS 10-4 codes to ISO 3166-1 alpha-2 on ingest
(RS->RU, CH->CN, GM->DE, JA->JP, SP->ES, UK->GB, UP->UA, PO->PL), while every
other country stayed FIPS and every other consumer in the repo
(data/iso3_to_fips.py, the parquet feature cache, streamlit_app/ui.py) assumes
gdelt_events.action_geo_country is FIPS-only. Found while joining UCDP GED
(keyed by Gleditsch-Ward -> ISO3 -> FIPS2) against gdelt_events: Russia,
Ukraine, China, Germany and Spain resolved to zero rows because their events
were silently split across two codes.

This UPDATEs the historical rows back to FIPS. Safe: gdelt_events' primary
key is (global_event_id, event_date), which doesn't include the country
column, so there is no collision risk. The bug in event_cleaner.py has
separately been disabled (FIPS_TO_ISO = {}) so this does not need to be re-run
after future ingestion.

country_daily_features is NOT touched here -- it has 83 pre-existing FIPS-coded
rows per country from the parquet-seeded path (seed_db_from_cache.py) that
would collide on the (country, feature_date) PK with the renamed rows, and
that table is being superseded by the UCDP PIT panel, not
repaired in place.

Usage:
    python -m scripts.fix_country_code_split           # dry run, prints counts
    python -m scripts.fix_country_code_split --apply    # actually UPDATE
"""

from __future__ import annotations

import argparse
import os

import psycopg2

DSN = os.environ.get(
    "GEOPULSE_DSN",
    "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432",
)

# ISO code (as wrongly written) -> correct FIPS code.
ISO_TO_FIPS_FIX = {
    "RU": "RS",  # Russia
    "CN": "CH",  # China
    "DE": "GM",  # Germany
    "JP": "JA",  # Japan
    "ES": "SP",  # Spain
    "GB": "UK",  # United Kingdom
    "UA": "UP",  # Ukraine
    "PL": "PO",  # Poland
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="run the UPDATE (default: dry run)")
    ap.add_argument("--dsn", default=DSN)
    args = ap.parse_args()

    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cur:
            total = 0
            for iso, fips in ISO_TO_FIPS_FIX.items():
                cur.execute(
                    "SELECT COUNT(*) FROM gdelt_events WHERE action_geo_country = %s",
                    (iso,),
                )
                n = cur.fetchone()[0]
                total += n
                print(f"  {iso} -> {fips}: {n:,} rows")
                if args.apply and n:
                    for field in ("action_geo_country", "actor1_country", "actor2_country"):
                        cur.execute(
                            f"UPDATE gdelt_events SET {field} = %s WHERE {field} = %s",
                            (fips, iso),
                        )
            if args.apply:
                conn.commit()
                print(f"Applied. {total:,} action_geo_country rows corrected.")
            else:
                print(f"Dry run. {total:,} rows would be corrected. Re-run with --apply.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
