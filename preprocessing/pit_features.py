"""
Phase A features: data/real_cache/*.parquet -> A1 temporal derivatives,
joined onto the UCDP PIT label panel. No new ingestion -- this is the
"works on data already on disk" phase of the plan.

A1 temporal derivatives (per plan): for each country's bi-weekly snapshot
series (83 files, 2023-01-01 -> 2026-03-22, 7 raw features f0-f6 each),
compute, as of the snapshot nearest but strictly before (run_date - 7d):
    current value, delta vs 1 snapshot back (~14d) and 2 back (~28d),
    rolling mean over trailing 3/6/13 snapshots, rolling std over trailing 13,
    z-score vs trailing-13 mean/std.

The 7-day buffer before run_date matches the PIT design used everywhere else
in this plan (preprocessing/pit_labels.py): it is not merely "before the
label window", it is "before the point where GDELT/whatever-produced-this-
snapshot could plausibly still be catching up on data".

Country key normalization (join hazard #2 from the plan): the parquet mixes
FIPS-2 codes (~90%) with CAMEO-3 codes (~10%, `actor1_country_code` fallback
in scripts/train_real_data.py -- GDELT's CAMEO-3 actor country codes are the
same alphabet as ISO 3166-1 alpha-3). Normalized via data/iso3_to_fips.py,
which already had 5 entries fixed this session while building the UCDP join.
Three codes (AFR, EUR, SEA) are regional/supranational CAMEO actors, not
countries -- dropped and counted, not silently absorbed.

Coverage caveat that must travel with any result built on this: the parquet
only goes back to 2023-01-01, so the rolling windows for the earliest 2023
test folds are built on very little trailing history (min_periods=1, so they
degrade to fewer-snapshot averages rather than NaN, but they are not the
full 6-month picture a later fold gets). HistGradientBoostingClassifier
handles the resulting NaNs (e.g. delta vs. a snapshot that doesn't exist yet)
natively -- no imputation needed, same pattern as the A3 features already in
evaluation/pit_backtest.py.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from data.iso3_to_fips import ISO3_TO_FIPS

logger = logging.getLogger("preprocessing.pit_features")

RAW_FEATURE_COLS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
PIT_BUFFER_DAYS = 7

A1_STATS = ["cur", "delta1", "delta2", "roll_mean3", "roll_mean6", "roll_mean13", "roll_std13", "zscore13"]
A1_FEATURES = [f"{c}_{s}" for c in RAW_FEATURE_COLS for s in A1_STATS]  # 56 columns


# ---------------------------------------------------------------------------
# Load + normalize the parquet cache
# ---------------------------------------------------------------------------

def _fips_for_code(code: str) -> str | None:
    if len(code) == 2:
        return code
    if len(code) == 3:
        return ISO3_TO_FIPS.get(code)
    return None


def load_parquet_panel(glob_pattern: str = "data/real_cache/*.parquet") -> pd.DataFrame:
    """Returns a long (fips2, snapshot_date, f0..f6) frame, one row per
    country per snapshot, FIPS-2 and CAMEO-3 codes both normalized to FIPS-2."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched {glob_pattern}")

    frames = []
    dropped_regional = 0
    for path in files:
        m = re.search(r"(\d{8})_features\.parquet$", os.path.basename(path))
        if not m:
            logger.warning("Skipping %s -- doesn't match YYYYMMDD_features.parquet", path)
            continue
        snapshot_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        df = pd.read_parquet(path, columns=["country"] + RAW_FEATURE_COLS)
        df["fips2"] = df["country"].map(_fips_for_code)
        unresolved = df[df["fips2"].isna()]
        dropped_regional += len(unresolved)
        df = df.dropna(subset=["fips2"])
        df["snapshot_date"] = snapshot_date
        frames.append(df[["fips2", "snapshot_date"] + RAW_FEATURE_COLS])

    panel = pd.concat(frames, ignore_index=True)
    # A country can appear under both its native FIPS-2 row and a normalized
    # CAMEO-3 row in the same snapshot (e.g. "US" and "USA" both present) --
    # average them rather than silently keeping whichever pandas saw last.
    dupes = panel.duplicated(subset=["fips2", "snapshot_date"], keep=False).sum()
    if dupes:
        logger.info("%d rows share a (fips2, snapshot_date) after normalization "
                    "(native + CAMEO-3 duplicate) -- averaging", dupes)
    panel = panel.groupby(["fips2", "snapshot_date"], as_index=False)[RAW_FEATURE_COLS].mean()

    logger.info("Loaded %d files, %d (country, snapshot) rows, %d countries, "
                "dropped %d regional/unresolved rows",
                len(files), len(panel), panel["fips2"].nunique(), dropped_regional)
    return panel


# ---------------------------------------------------------------------------
# A1 temporal derivatives
# ---------------------------------------------------------------------------

def _build_a1_per_country(g: pd.DataFrame, fips2: str) -> pd.DataFrame:
    """g: one country's snapshots (fips2 column already excluded by the
    caller), sorted by snapshot_date. Returns the same rows with
    A1_FEATURES columns added, using only *prior* snapshots for every
    derived stat (row i's delta/rolling stats never include row i's own
    current value in the window -- they describe the trajectory *up to* the
    previous snapshot, then 'cur' itself is that previous snapshot's raw
    value; see attach_a1_to_labels for how this gets picked per run_date)."""
    g = g.sort_values("snapshot_date").reset_index(drop=True)
    out = {}
    for c in RAW_FEATURE_COLS:
        s = g[c]
        out[f"{c}_cur"] = s
        out[f"{c}_delta1"] = s - s.shift(1)
        out[f"{c}_delta2"] = s - s.shift(2)
        out[f"{c}_roll_mean3"] = s.rolling(3, min_periods=1).mean()
        out[f"{c}_roll_mean6"] = s.rolling(6, min_periods=1).mean()
        out[f"{c}_roll_mean13"] = s.rolling(13, min_periods=1).mean()
        roll_std13 = s.rolling(13, min_periods=2).std()
        out[f"{c}_roll_std13"] = roll_std13
        roll_mean13 = out[f"{c}_roll_mean13"]
        out[f"{c}_zscore13"] = (s - roll_mean13) / roll_std13.replace(0, np.nan)
    a1 = pd.DataFrame(out, index=g.index)
    a1.insert(0, "fips2", fips2)
    return pd.concat([g[["snapshot_date"]], a1], axis=1)


def build_a1_snapshot_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-(fips2, snapshot_date) A1 features, computed independently per
    country so no cross-country leakage is possible."""
    parts = [
        _build_a1_per_country(g.drop(columns=["fips2"]), fips2)
        for fips2, g in panel.groupby("fips2")
    ]
    return pd.concat(parts, ignore_index=True)


def attach_a1_to_labels(labels_df: pd.DataFrame, a1_snapshots: pd.DataFrame,
                         buffer_days: int = PIT_BUFFER_DAYS) -> pd.DataFrame:
    """For each (fips2, run_date) in labels_df, attach the A1 features from
    the most recent snapshot with snapshot_date <= run_date - buffer_days.
    A country/run_date with no eligible snapshot (before parquet coverage
    starts, or after a country drops out) gets all-NaN A1 columns --
    HistGradientBoostingClassifier handles this natively."""
    labels_df = labels_df.copy()
    labels_df["_cutoff"] = pd.to_datetime(labels_df["run_date"]) - pd.Timedelta(days=buffer_days)

    a1 = a1_snapshots.sort_values("snapshot_date").copy()
    a1["snapshot_date"] = pd.to_datetime(a1["snapshot_date"])

    merged_parts = []
    for fips2, lg in labels_df.groupby("fips2"):
        ag = a1[a1["fips2"] == fips2].sort_values("snapshot_date")
        if ag.empty:
            part = lg.copy()
            for col in A1_FEATURES:
                part[col] = np.nan
            merged_parts.append(part)
            continue
        lg_sorted = lg.sort_values("_cutoff")
        merged = pd.merge_asof(
            lg_sorted, ag.drop(columns=["fips2"]),
            left_on="_cutoff", right_on="snapshot_date", direction="backward",
        )
        merged_parts.append(merged)

    result = pd.concat(merged_parts, ignore_index=True)
    return result.drop(columns=["_cutoff", "snapshot_date"], errors="ignore")


def build_phase_a_panel(labels_df: pd.DataFrame,
                         glob_pattern: str = "data/real_cache/*.parquet") -> pd.DataFrame:
    """Full Phase A pipeline: load parquet -> A1 snapshot features -> attach
    to the label panel by PIT-correct as-of join. Returns labels_df with
    A1_FEATURES columns added."""
    panel = load_parquet_panel(glob_pattern)
    a1_snapshots = build_a1_snapshot_features(panel)
    return attach_a1_to_labels(labels_df, a1_snapshots)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from evaluation.pit_backtest import load_country_panel

    labels = load_country_panel()
    merged = build_phase_a_panel(labels)
    coverage = merged[A1_FEATURES[0]].notna().mean()
    logger.info("Phase A panel: %d rows, %.1f%% have at least one A1 feature populated",
                len(merged), coverage * 100)
    out_path = "data/pit_phase_a_panel.parquet"
    merged.to_parquet(out_path)
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
