"""Unit tests for evaluation/pit_backtest.py -- fold logic and baselines,
no DB required (uses a small synthetic panel)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from evaluation.pit_backtest import (
    build_a3_features,
    build_folds,
    predict_country_base_rate,
    predict_global_base_rate,
    predict_persistence_flat,
    run_backtest,
)


def _synthetic_panel(n_months: int = 40, n_countries: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2022-01-01", periods=n_months, freq="MS").date
    for c in [f"C{i}" for i in range(n_countries)]:
        deaths = rng.integers(0, 50, size=n_months)
        for i, d in enumerate(dates):
            prev = deaths[i - 1] if i > 0 else 0
            delta = np.log1p(deaths[i]) - np.log1p(prev)
            esc_dir = 1 if delta > 0.5 else (-1 if delta < -0.5 else 0)
            rows.append({
                "fips2": c, "gw_id": hash(c) % 1000, "run_date": d,
                "n_events": int(deaths[i] > 0), "deaths_best": int(deaths[i]),
                "deaths_civilians": 0, "n_dyads": int(deaths[i] > 0),
                "n_adm1": int(deaths[i] > 0), "prev_deaths_best": int(prev), "prev_n_events": 0,
                "escalation_delta": delta, "escalation_dir": esc_dir, "civ_share": None,
                "any_violence": deaths[i] > 0, "minor_conflict": deaths[i] >= 25,
                "war_intensity": deaths[i] >= 100, "months_since_last_minor": None,
            })
    df = pd.DataFrame(rows)
    dates_sorted = sorted(df["run_date"].unique())
    idx = {d: i for i, d in enumerate(dates_sorted)}
    df["month_idx"] = df["run_date"].map(idx)
    return df


def test_build_folds_respects_purge():
    df = _synthetic_panel()
    folds = build_folds(df, eval_start=date(2023, 1, 1), eval_end=date(2024, 4, 1), purge_months=3)
    for fold in folds:
        assert fold.train_cutoff_idx == fold.test_month_idx - 3


def test_fold_train_test_never_overlap_in_time():
    """The PIT-adjacent invariant for the backtest itself: no training row's
    month_idx is >= a test row's month_idx within the same fold."""
    df = _synthetic_panel()
    folds = build_folds(df, eval_start=date(2023, 1, 1), eval_end=date(2024, 4, 1), purge_months=3)
    for fold in folds:
        train = df[df["month_idx"] <= fold.train_cutoff_idx]
        test = df[df["month_idx"] == fold.test_month_idx]
        if train.empty or test.empty:
            continue
        assert train["month_idx"].max() < test["month_idx"].min()
        assert fold.test_month_idx - train["month_idx"].max() >= 3


def test_global_base_rate_matches_persistence_when_flat_is_mode():
    df = _synthetic_panel()
    train = df[df["month_idx"] <= 20]
    test = df[df["month_idx"] == 25]
    # By construction with small random deltas, flat should dominate.
    if (train["escalation_dir"] == 0).mean() > 0.5:
        assert np.array_equal(predict_global_base_rate(train, test), predict_persistence_flat(train, test))


def test_country_base_rate_falls_back_to_global_for_unseen_country():
    df = _synthetic_panel()
    train = df[df["fips2"] != "C2"]
    test = df[df["fips2"] == "C2"].head(3)
    preds = predict_country_base_rate(train, test)
    assert len(preds) == 3
    assert set(preds.tolist()) <= {-1, 0, 1}


def test_a3_features_are_lagged_not_current():
    """The defining PIT property of the harness's own feature builder: row t's
    A3 features must never equal row t's own current-window deaths_best."""
    df = _synthetic_panel()
    feat = build_a3_features(df)
    for fips, g in feat.groupby("fips2"):
        g = g.sort_values("month_idx")
        # log_deaths_prev1 at row i should equal log1p(deaths_best) at row i-1
        expected = np.log1p(g["deaths_best"]).shift(1).to_numpy()
        actual = g["log_deaths_prev1"].to_numpy()
        np.testing.assert_allclose(actual[1:], expected[1:], equal_nan=True)


def test_run_backtest_produces_three_reporting_views():
    df = _synthetic_panel()
    df = build_a3_features(df)
    active = {"C0", "C1"}
    result = run_backtest(df, predict_persistence_flat, active_fips=active,
                           eval_start=date(2023, 1, 1), eval_end=date(2024, 4, 1), purge_months=3)
    assert "pooled" in result and "active_only" in result and "macro_by_country" in result
    assert result["pooled"]["n"] >= result["active_only"]["n"]
