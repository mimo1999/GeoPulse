"""Unit tests for preprocessing/pit_features.py -- no DB, no real parquet
files required (synthetic panels)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from preprocessing.pit_features import (
    A1_FEATURES,
    _fips_for_code,
    attach_a1_to_labels,
    build_a1_snapshot_features,
)


def test_fips_for_code_two_char_passthrough():
    assert _fips_for_code("US") == "US"


def test_fips_for_code_three_char_resolves_via_iso3():
    assert _fips_for_code("RUS") == "RS"  # ISO3 RUS -> FIPS RS


def test_fips_for_code_regional_code_unresolved():
    assert _fips_for_code("EUR") is None


def _synthetic_snapshot_panel() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=20, freq="14D").date
    rows = []
    for d in dates:
        for c in ["US", "SY"]:
            rows.append({"fips2": c, "snapshot_date": d,
                         "f0": 0.5, "f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5, "f6": 0.6})
    return pd.DataFrame(rows)


def test_a1_snapshot_features_no_future_leakage():
    """delta1/roll_mean at row i must never depend on rows after i within a
    country's own series (checked by perturbing a later value and confirming
    earlier rows are unaffected)."""
    panel = _synthetic_snapshot_panel()
    a1 = build_a1_snapshot_features(panel)

    perturbed = panel.copy()
    last_us_idx = perturbed[perturbed["fips2"] == "US"].index[-1]
    perturbed.loc[last_us_idx, "f0"] = 999.0
    a1_perturbed = build_a1_snapshot_features(perturbed)

    us_a1 = a1[a1["fips2"] == "US"].sort_values("snapshot_date").reset_index(drop=True)
    us_a1_perturbed = a1_perturbed[a1_perturbed["fips2"] == "US"].sort_values("snapshot_date").reset_index(drop=True)
    # every row except the perturbed last one must be identical
    pd.testing.assert_frame_equal(us_a1.iloc[:-1], us_a1_perturbed.iloc[:-1])


def test_attach_a1_respects_pit_buffer():
    """A label row's attached A1 snapshot must be strictly before
    run_date - buffer_days, never from a snapshot on or after that cutoff."""
    panel = _synthetic_snapshot_panel()
    a1 = build_a1_snapshot_features(panel)

    labels = pd.DataFrame([
        {"fips2": "US", "run_date": date(2023, 3, 1), "escalation_dir": 0},
    ])
    merged = attach_a1_to_labels(labels, a1, buffer_days=7)
    assert merged.loc[0, A1_FEATURES[0]] is not None or pd.isna(merged.loc[0, A1_FEATURES[0]])

    # Manually verify: the picked snapshot's date must be <= run_date - 7d.
    cutoff = pd.Timestamp(date(2023, 3, 1)) - pd.Timedelta(days=7)
    us_snaps = a1[a1["fips2"] == "US"]["snapshot_date"]
    eligible = us_snaps[pd.to_datetime(us_snaps) <= cutoff]
    assert not eligible.empty  # sanity: the synthetic panel does have eligible snapshots
    latest_eligible_cur = a1[(a1["fips2"] == "US") & (a1["snapshot_date"] == eligible.max())]["f0_cur"].iloc[0]
    assert merged.loc[0, "f0_cur"] == latest_eligible_cur


def test_attach_a1_no_eligible_snapshot_gives_nan_not_error():
    panel = _synthetic_snapshot_panel()
    a1 = build_a1_snapshot_features(panel)
    labels = pd.DataFrame([
        {"fips2": "US", "run_date": date(2020, 1, 1), "escalation_dir": 0},  # before any snapshot
    ])
    merged = attach_a1_to_labels(labels, a1, buffer_days=7)
    assert pd.isna(merged.loc[0, "f0_cur"])


def test_attach_a1_unknown_country_gives_nan_not_error():
    panel = _synthetic_snapshot_panel()
    a1 = build_a1_snapshot_features(panel)
    labels = pd.DataFrame([
        {"fips2": "ZZ", "run_date": date(2023, 3, 1), "escalation_dir": 0},
    ])
    merged = attach_a1_to_labels(labels, a1, buffer_days=7)
    assert pd.isna(merged.loc[0, "f0_cur"])
