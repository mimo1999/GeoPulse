"""
Purged expanding-window walk-forward backtest over ucdp.country_pit_labels.

Replaces evaluation/backtester.py for the PIT targets: same fold-walking
spirit, different (non-circular) target and metrics. This module owns the
splitting and baseline logic; a real model (Phase A/B) plugs in as a
`predict_fn(train_df, test_df) -> array of predictions` callable and is
scored through the same `run_backtest()` path as the baselines below, so
"does the model beat the baseline" is always an apples-to-apples comparison
against numbers computed by the exact same code.

Fold design
-----------
Monthly grid, run_date in [2015-01-01, 2025-11-01] (131 months). Train on
run_date <= test_date - PURGE_MONTHS, test on run_date == test_date. A
3-month purge matches the longest feature lookback used anywhere in the plan
(quarterly UCDP lag features); it prevents a training row's window from
overlapping a test row's window.

Evaluation window is locked to the plan's scope, 2023-01 through 2025-11 (35
folds) -- but training data is NOT restricted to that window. Everything
back to 2015-01 is available as history for the per-country baseline and the
lagged-UCDP-only model; using more real history than the plan assumed can
only make those baselines stronger, never leak information forward.

Every metric is reported three ways per the plan's requirement: pooled (all
126 countries, most are near-permanently zero), active-only (countries with
ANY UCDP event on record, 2015-2025), and macro (mean of per-country
metrics). The active-only number is the headline -- pooled is inflated by
countries like Iceland that are trivially always "flat".

Baselines, in the order the plan lists them:
    1. global_base_rate   -- training-window class distribution, hard-predict the mode
    2. country_base_rate  -- same, but per-country (falls back to global if no history)
    3. persistence        -- P1: always "flat" (0). This is the plan's own choice,
                              not "predict last month's direction" -- Δ month-to-month
                              is closer to a random walk than direction is to itself.
    4. lagged_ucdp_only   -- HistGradientBoostingClassifier on A3 features derived
                              purely from ucdp.country_pit_labels' own stored history
                              (no GDELT). "The baseline that matters" -- the whole
                              project's claim is that GDELT adds over this.

Usage:
    python -m evaluation.pit_backtest
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

logger = logging.getLogger("evaluation.pit_backtest")

DSN = os.environ.get(
    "GEOPULSE_DSN",
    "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432",
)

PURGE_MONTHS = 3
EVAL_START = date(2023, 1, 1)
EVAL_END = date(2025, 11, 1)
RESULTS_PATH = "evaluation/results/pit_backtest_baselines.json"

P1_CLASSES = [-1, 0, 1]  # down, flat, up
P1_CLASS_NAMES = {-1: "down", 0: "flat", 1: "up"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_country_panel(dsn: str = DSN) -> pd.DataFrame:
    conn = psycopg2.connect(dsn)
    try:
        df = pd.read_sql(
            """SELECT fips2, gw_id, run_date, n_events, deaths_best, deaths_civilians,
                      n_dyads, n_adm1, prev_deaths_best, prev_n_events,
                      escalation_delta, escalation_dir, civ_share,
                      any_violence, minor_conflict, war_intensity, months_since_last_minor
               FROM ucdp.country_pit_labels
               ORDER BY fips2, run_date""",
            conn,
        )
    finally:
        conn.close()
    df["run_date"] = pd.to_datetime(df["run_date"]).dt.date
    return df


def _add_month_index(df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(df["run_date"].unique())
    idx = {d: i for i, d in enumerate(dates)}
    df = df.copy()
    df["month_idx"] = df["run_date"].map(idx)
    return df


# ---------------------------------------------------------------------------
# A3 lagged-UCDP-only features (baseline 4) -- computed purely from history
# already sitting in country_pit_labels, no GDELT, no parquet.
# ---------------------------------------------------------------------------

A3_FEATURES = [
    "log_deaths_prev1", "log_deaths_prev3_mean", "log_deaths_prev12_mean",
    "n_dyads_prev1", "n_adm1_prev1", "civ_share_prev1",
    "months_since_last_minor_prev1", "any_violence_prev1", "minor_conflict_prev1",
]


def build_a3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country trailing history features, shifted so row t only uses
    data known as of t (the row's own current-window fields are NOT used --
    only lagged versions, computed by shifting within each country's series
    sorted by month_idx)."""
    df = df.sort_values(["fips2", "month_idx"]).copy()
    g = df.groupby("fips2")

    log_deaths = np.log1p(df["deaths_best"])
    df["log_deaths_prev1"] = g["deaths_best"].shift(1).pipe(np.log1p)
    log_deaths_shifted = g["deaths_best"].shift(1).pipe(np.log1p)
    df["log_deaths_prev3_mean"] = (
        log_deaths_shifted.groupby(df["fips2"]).rolling(3, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["log_deaths_prev12_mean"] = (
        log_deaths_shifted.groupby(df["fips2"]).rolling(12, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df["n_dyads_prev1"] = g["n_dyads"].shift(1)
    df["n_adm1_prev1"] = g["n_adm1"].shift(1)
    df["civ_share_prev1"] = g["civ_share"].shift(1)
    df["months_since_last_minor_prev1"] = g["months_since_last_minor"].shift(1)
    df["any_violence_prev1"] = g["any_violence"].shift(1).astype(float)
    df["minor_conflict_prev1"] = g["minor_conflict"].shift(1).astype(float)
    return df


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    test_month_idx: int
    test_date: date
    train_cutoff_idx: int  # inclusive


def build_folds(df: pd.DataFrame, eval_start: date = EVAL_START, eval_end: date = EVAL_END,
                 purge_months: int = PURGE_MONTHS) -> list[Fold]:
    dates = sorted(df["run_date"].unique())
    idx_of = {d: i for i, d in enumerate(dates)}
    folds = []
    for d in dates:
        if eval_start <= d <= eval_end:
            i = idx_of[d]
            folds.append(Fold(test_month_idx=i, test_date=d, train_cutoff_idx=i - purge_months))
    return folds


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _mode_class(series: pd.Series, classes: list[int]) -> int:
    if series.empty:
        return 0  # "flat" is the safe default with zero history
    counts = series.value_counts()
    return int(counts.idxmax())


def predict_global_base_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    mode = _mode_class(train["escalation_dir"], P1_CLASSES)
    return np.full(len(test), mode)


def predict_country_base_rate(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_mode = _mode_class(train["escalation_dir"], P1_CLASSES)
    per_country_mode = train.groupby("fips2")["escalation_dir"].agg(lambda s: _mode_class(s, P1_CLASSES))
    return test["fips2"].map(per_country_mode).fillna(global_mode).astype(int).to_numpy()


def predict_persistence_flat(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The plan's own choice for P1: always predict 'flat' (0)."""
    return np.zeros(len(test), dtype=int)


def predict_lagged_ucdp_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline 4 -- 'the baseline that matters'. A3 features only, no GDELT."""
    train_f = train.dropna(subset=["log_deaths_prev1"])
    if train_f["escalation_dir"].nunique() < 2 or len(train_f) < 30:
        # Not enough history yet for a real fit -- fall back to persistence-flat
        # rather than fit garbage on a handful of rows.
        return predict_persistence_flat(train, test)

    X_train = train_f[A3_FEATURES].to_numpy(dtype=float)
    y_train = train_f["escalation_dir"].to_numpy(dtype=int)
    X_test = test[A3_FEATURES].to_numpy(dtype=float)

    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4, random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict(X_test).astype(int)


BASELINES: dict[str, Callable] = {
    "global_base_rate": predict_global_base_rate,
    "country_base_rate": predict_country_base_rate,
    "persistence_flat": predict_persistence_flat,
    "lagged_ucdp_only": predict_lagged_ucdp_only,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    labels = P1_CLASSES
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = float(np.mean(y_true == y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            P1_CLASS_NAMES[c]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, c in enumerate(labels)
        },
        "confusion_matrix": {"labels": [P1_CLASS_NAMES[c] for c in labels], "matrix": cm},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    active_fips: Optional[set[str]] = None,
    eval_start: date = EVAL_START, eval_end: date = EVAL_END,
    purge_months: int = PURGE_MONTHS,
) -> dict:
    """Run one predictor through every fold; return pooled + active-only + macro metrics."""
    folds = build_folds(df, eval_start, eval_end, purge_months)
    y_true_all, y_pred_all, fips_all = [], [], []
    fold_metrics = []

    for fold in folds:
        train = df[df["month_idx"] <= fold.train_cutoff_idx]
        test = df[df["month_idx"] == fold.test_month_idx]
        if test.empty:
            continue
        preds = predict_fn(train, test)
        y_true_all.extend(test["escalation_dir"].tolist())
        y_pred_all.extend(preds.tolist())
        fips_all.extend(test["fips2"].tolist())
        fold_metrics.append({
            "test_date": fold.test_date.isoformat(),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        })

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    fips_all = np.array(fips_all)

    pooled = compute_metrics(y_true_all, y_pred_all)

    if active_fips:
        mask = np.isin(fips_all, list(active_fips))
        active = compute_metrics(y_true_all[mask], y_pred_all[mask])
    else:
        active = pooled

    # Macro-by-country: mean of per-country accuracy/macro-F1, active countries only.
    per_country_acc, per_country_f1 = [], []
    if active_fips:
        for c in active_fips:
            m = fips_all == c
            if m.sum() == 0:
                continue
            per_country_acc.append(float(np.mean(y_true_all[m] == y_pred_all[m])))
            per_country_f1.append(f1_score(y_true_all[m], y_pred_all[m], labels=P1_CLASSES,
                                            average="macro", zero_division=0))
    macro = {
        "n_countries": len(per_country_acc),
        "mean_accuracy": round(float(np.mean(per_country_acc)), 4) if per_country_acc else None,
        "mean_macro_f1": round(float(np.mean(per_country_f1)), 4) if per_country_f1 else None,
    }

    return {
        "n_folds": len(fold_metrics),
        "pooled": pooled,
        "active_only": active,
        "macro_by_country": macro,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Loading country panel ...")
    df = load_country_panel()
    df = _add_month_index(df)
    df = build_a3_features(df)

    # "Active" is scoped to the eval window itself (any event in 2023-2025),
    # matching the plan's ~77-country figure -- NOT lifetime-since-2015
    # activity, which is a much broader, easier population (100 countries)
    # and understates how non-trivial "flat" persistence really is.
    eval_mask = (df["run_date"] >= EVAL_START) & (df["run_date"] <= EVAL_END)
    active_fips = set(df.loc[eval_mask & (df["n_events"] > 0), "fips2"].unique())
    lifetime_active_fips = set(df.loc[df["n_events"] > 0, "fips2"].unique())
    logger.info("Loaded %d rows, %d countries, %d active in eval window (%d active lifetime 2015-2025)",
                len(df), df["fips2"].nunique(), len(active_fips), len(lifetime_active_fips))

    results = {"target": "escalation_dir (P1)", "eval_window": [EVAL_START.isoformat(), EVAL_END.isoformat()],
               "purge_months": PURGE_MONTHS, "n_active_countries": len(active_fips), "baselines": {}}

    for name, fn in BASELINES.items():
        logger.info("Running baseline: %s", name)
        results["baselines"][name] = run_backtest(df, fn, active_fips=active_fips)
        active_acc = results["baselines"][name]["active_only"]["accuracy"]
        active_f1 = results["baselines"][name]["active_only"]["macro_f1"]
        logger.info("  %s: active-only accuracy=%.3f macro_f1=%.3f", name, active_acc, active_f1)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", RESULTS_PATH)

    print(f"\n{'baseline':<22}{'n_folds':>8}{'acc(active)':>14}{'macroF1(active)':>17}{'acc(pooled)':>14}")
    for name, r in results["baselines"].items():
        print(f"{name:<22}{r['n_folds']:>8}{r['active_only']['accuracy']:>14.3f}"
              f"{r['active_only']['macro_f1']:>17.3f}{r['pooled']['accuracy']:>14.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
