"""
FX GATE — does GDELT news signal predict next-month FX moves at all?

THROWAWAY DIAGNOSTIC, NOT PRODUCTION CODE. Run once, record the verdict,
delete or archive. Its only job is to open or close the "pivot the whole
project to economic targets" question on evidence rather than opinion.

=============================================================================
PRE-REGISTERED DECISION RULE  (written before any analysis code below)
=============================================================================
Signal is declared PRESENT only if, under purged walk-forward CV pooled over
the active country set:

    (a) directional accuracy > 0.53          -- i.e. beats a coin flip by >3pp
        OR
    (b) out-of-sample R^2 > 0 vs the random walk (predict 0 change)

AND at least one feature survives Benjamini-Hochberg FDR correction at
q = 0.05 across the full feature x country correlation grid.

Rationale for the correction: the grid is ~40 features x ~11 countries ~= 440
tests. At p<0.05 uncorrected, ~22 will look "significant" by chance alone.
Without a pre-set threshold and FDR control, we will rationalise noise.

VERDICT SEMANTICS
  - FAIL is the expected outcome and is a SUCCESS for the project: it closes
    the econ-target pivot for one day of work instead of several weeks.
  - PASS is surprising. Meese-Rogoff (1983) -- FX models fail to beat a random
    walk out-of-sample at short horizons -- has survived 40+ years of
    replication. A PASS should trigger a leakage audit of THIS script before
    anyone believes it.

Result is committed to evaluation/results/fx_gate.json either way.
=============================================================================

Design
------
  run_date t  = first day of month M
  features    : parquet snapshots dated <= t - 7d (12-month trailing window)
  label       : log(FX at end of M) - log(FX at start of M), i.e. strictly
                inside (t, t+30d]

The 7-day buffer mirrors the main plan's PIT discipline so that a PASS here
would be directly comparable to the conflict-target pipeline.

FX source: Frankfurter (api.frankfurter.app), a free no-key wrapper over ECB
reference rates. ECB publishes ~30 currencies, which caps the country set --
notably excluding Colombia, Egypt, Pakistan, Kenya, Ghana, Nigeria. Countries
included are those with BOTH an ECB-published floating rate AND real GDELT
variation.

Usage:
    python scripts/fx_gate_test.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# FIPS-2 (parquet key) -> ISO-4217 currency published by the ECB.
# Restricted to floating currencies with genuine GDELT event variation.
COUNTRY_FX = {
    "TU": "TRY",   # Turkey
    "MX": "MXN",   # Mexico
    "BR": "BRL",   # Brazil
    "SF": "ZAR",   # South Africa
    "IN": "INR",   # India
    "ID": "IDR",   # Indonesia
    "RP": "PHP",   # Philippines
    "IS": "ILS",   # Israel
    "TH": "THB",   # Thailand
    "PL": "PLN",   # Poland
    "HU": "HUF",   # Hungary
    "KS": "KRW",   # South Korea
    "MY": "MYR",   # Malaysia
    "RO": "RON",   # Romania
}

FEATURE_COLS = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
FEATURE_LABELS = {
    "f0": "protest", "f1": "violence", "f2": "diplo_stress", "f3": "econ_stress",
    "f4": "terror", "f5": "tone_neg", "f6": "goldstein_inv",
}

START = "2023-01-01"
END = "2026-01-31"
PIT_BUFFER_DAYS = 7

DIR_ACC_THRESHOLD = 0.53
FDR_Q = 0.05


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_fx() -> pd.DataFrame:
    """Daily USD-base rates from ECB via Frankfurter. Returns long df."""
    syms = ",".join(sorted(set(COUNTRY_FX.values())))
    url = f"https://api.frankfurter.app/{START}..{END}?base=USD&symbols={syms}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rates = r.json()["rates"]
    rows = [
        {"date": pd.Timestamp(d), "ccy": c, "rate": v}
        for d, per in rates.items() for c, v in per.items()
    ]
    return pd.DataFrame(rows).sort_values("date")


def monthly_fx_returns(fx: pd.DataFrame) -> pd.DataFrame:
    """
    Month-end USD/XXX level -> next-month log return.

    Label for run_date t (= first of month M) is the return realised DURING
    month M, i.e. strictly after t. Computed as
    log(level at last obs in M) - log(level at last obs in M-1).
    """
    fx = fx.copy()
    fx["month"] = fx["date"].values.astype("datetime64[M]")
    me = fx.sort_values("date").groupby(["ccy", "month"], as_index=False).last()
    me = me.sort_values(["ccy", "month"])
    me["prev"] = me.groupby("ccy")["rate"].shift(1)
    # return during month M (rise = local currency depreciating vs USD)
    me["fx_ret"] = np.log(me["rate"] / me["prev"])
    return me.dropna(subset=["fx_ret"])[["ccy", "month", "fx_ret"]]


def load_parquet_features(cache_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(cache_dir, "*_features.parquet")))
    if not files:
        raise RuntimeError(f"no parquet snapshots in {cache_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[df["country"].isin(COUNTRY_FX)].sort_values(["country", "date"])


def build_panel(feat: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """
    PIT join. For each (country, month M): take the latest parquet snapshot
    dated <= (first of M) - 7d, plus trailing derivatives, and attach the
    month-M FX return as the label.
    """
    rows = []
    months = sorted(rets["month"].unique())
    for fips, ccy in COUNTRY_FX.items():
        cf = feat[feat["country"] == fips].sort_values("date")
        cr = rets[rets["ccy"] == ccy].set_index("month")["fx_ret"]
        if cf.empty:
            continue
        for m in months:
            if m not in cr.index:
                continue
            cutoff = pd.Timestamp(m) - pd.Timedelta(days=PIT_BUFFER_DAYS)
            hist = cf[cf["date"] <= cutoff]
            if len(hist) < 14:          # need ~6mo of bi-weekly history
                continue
            cur = hist.iloc[-1]
            rec = {"country": fips, "ccy": ccy, "month": m,
                   "fx_ret": float(cr.loc[m]), "feat_max_date": hist["date"].max()}
            for c in FEATURE_COLS:
                s = hist[c].astype(float)
                rec[f"{c}_lvl"] = float(cur[c])
                rec[f"{c}_d1"] = float(s.iloc[-1] - s.iloc[-2])
                rec[f"{c}_d3"] = float(s.iloc[-1] - s.iloc[-4]) if len(s) >= 4 else 0.0
                tail = s.iloc[-13:]
                mu, sd = tail.mean(), tail.std(ddof=0)
                rec[f"{c}_z"] = float((s.iloc[-1] - mu) / sd) if sd > 1e-9 else 0.0
                rec[f"{c}_vol"] = float(sd)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Return boolean mask of hypotheses surviving BH-FDR at level q."""
    n = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        keep[order[: kmax + 1]] = True
    return keep


def walk_forward(panel: pd.DataFrame, fcols: list[str]) -> dict:
    """Purged expanding-window walk-forward. Returns pooled OOS predictions."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    months = sorted(panel["month"].unique())
    preds, actuals, base = [], [], []
    for i in range(12, len(months) - 1):
        train_m = months[:i]          # purge: skip months[i] entirely
        test_m = months[i + 1]
        tr = panel[panel["month"].isin(train_m)]
        te = panel[panel["month"] == test_m]
        if len(tr) < 50 or te.empty:
            continue
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=150, learning_rate=0.05, random_state=0
        )
        model.fit(tr[fcols].values, tr["fx_ret"].values)
        p = model.predict(te[fcols].values)
        preds.extend(p)
        actuals.extend(te["fx_ret"].values)
        base.extend(np.zeros(len(te)))          # random walk: predict 0
    preds, actuals, base = map(np.asarray, (preds, actuals, base))
    if len(actuals) == 0:
        return {"n": 0}
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_rw = float(np.sum((actuals - base) ** 2))
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
    return {
        "n": int(len(actuals)),
        "dir_acc": dir_acc,
        "oos_r2_vs_rw": float(1 - ss_res / ss_rw) if ss_rw > 0 else float("nan"),
        "rmse_model": float(np.sqrt(ss_res / len(actuals))),
        "rmse_randomwalk": float(np.sqrt(ss_rw / len(actuals))),
    }


def correlation_grid(panel: pd.DataFrame, fcols: list[str]) -> tuple[list, np.ndarray]:
    """Per-country, per-feature Spearman vs next-month return."""
    from scipy.stats import spearmanr
    recs, pv = [], []
    for fips in sorted(panel["country"].unique()):
        sub = panel[panel["country"] == fips]
        if len(sub) < 12:
            continue
        for c in fcols:
            if sub[c].std(ddof=0) < 1e-12:
                continue
            rho, p = spearmanr(sub[c].values, sub["fx_ret"].values)
            if not (np.isnan(rho) or np.isnan(p)):
                recs.append({"country": fips, "feature": c, "rho": float(rho),
                             "p": float(p), "n": int(len(sub))})
                pv.append(p)
    return recs, np.asarray(pv)


# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("FX GATE — pre-registered rule: dir_acc > 0.53 OR oos_r2 > 0,")
    print("          AND >=1 feature surviving BH-FDR q=0.05")
    print("=" * 72)

    fx = fetch_fx()
    rets = monthly_fx_returns(fx)
    print(f"FX: {rets['ccy'].nunique()} currencies, {rets['month'].nunique()} months")

    feat = load_parquet_features("data/real_cache")
    print(f"parquet: {feat['country'].nunique()} matched countries, "
          f"{feat['date'].min().date()} -> {feat['date'].max().date()}")

    panel = build_panel(feat, rets)
    if panel.empty:
        raise SystemExit("empty panel — check country/date overlap")

    fcols = [c for c in panel.columns
             if any(c.startswith(f + "_") for f in FEATURE_COLS)]
    print(f"panel: {len(panel)} country-months, {panel['country'].nunique()} countries, "
          f"{len(fcols)} features")

    # PIT assertion — the claim this whole script rests on
    bad = panel[panel["feat_max_date"] > panel["month"] - pd.Timedelta(days=PIT_BUFFER_DAYS)]
    assert bad.empty, f"PIT VIOLATION in {len(bad)} rows"
    print("PIT check: OK (all features predate run_date - 7d)")

    wf = walk_forward(panel, fcols)
    recs, pv = correlation_grid(panel, fcols)
    n_sig_raw = int((pv < 0.05).sum()) if len(pv) else 0
    keep = benjamini_hochberg(pv, FDR_Q) if len(pv) else np.array([], dtype=bool)
    survivors = [recs[i] for i in np.where(keep)[0]] if len(pv) else []

    print("\n--- walk-forward (purged, expanding) ---")
    for k, v in wf.items():
        print(f"  {k:18} {v}")
    print("\n--- correlation grid ---")
    print(f"  tests run                 {len(pv)}")
    print(f"  p<0.05 uncorrected        {n_sig_raw}  (expected by chance: ~{0.05*len(pv):.0f})")
    print(f"  survive BH-FDR q={FDR_Q}     {len(survivors)}")
    for s in sorted(survivors, key=lambda x: x["p"])[:10]:
        print(f"    {s['country']} {FEATURE_LABELS.get(s['feature'].split('_')[0], '')}"
              f":{s['feature']:16} rho={s['rho']:+.3f} p={s['p']:.4f} n={s['n']}")

    crit_a = wf.get("dir_acc", 0) > DIR_ACC_THRESHOLD
    crit_b = wf.get("oos_r2_vs_rw", -1) > 0
    crit_fdr = len(survivors) > 0
    verdict = "PASS" if ((crit_a or crit_b) and crit_fdr) else "FAIL"

    print("\n" + "=" * 72)
    print(f"  (a) dir_acc > {DIR_ACC_THRESHOLD}          : {crit_a}  ({wf.get('dir_acc'):.4f})")
    print(f"  (b) oos_r2 vs random walk > 0 : {crit_b}  ({wf.get('oos_r2_vs_rw'):.4f})")
    print(f"  FDR survivors >= 1            : {crit_fdr}  ({len(survivors)})")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "FAIL":
        print("  -> econ-target pivot CLOSED on evidence. Proceed with UCDP conflict targets.")
    else:
        print("  -> SURPRISING. Audit this script for leakage before believing it.")
    print("=" * 72)

    out = {
        "verdict": verdict,
        "decision_rule": {
            "dir_acc_threshold": DIR_ACC_THRESHOLD,
            "oos_r2_threshold": 0.0,
            "fdr_q": FDR_Q,
            "pre_registered": True,
        },
        "criteria_met": {"dir_acc": crit_a, "oos_r2": crit_b, "fdr": crit_fdr},
        "walk_forward": wf,
        "correlation_grid": {
            "n_tests": int(len(pv)),
            "n_p_lt_05_uncorrected": n_sig_raw,
            "n_expected_by_chance": float(0.05 * len(pv)),
            "n_survive_bh_fdr": len(survivors),
            "survivors": survivors[:25],
        },
        "panel": {
            "n_country_months": int(len(panel)),
            "n_countries": int(panel["country"].nunique()),
            "countries": sorted(panel["country"].unique().tolist()),
            "n_features": len(fcols),
            "month_min": str(pd.Timestamp(panel["month"].min()).date()),
            "month_max": str(pd.Timestamp(panel["month"].max()).date()),
        },
        "fx_source": "ECB reference rates via api.frankfurter.app",
        "pit_buffer_days": PIT_BUFFER_DAYS,
    }
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/fx_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote evaluation/results/fx_gate.json")


if __name__ == "__main__":
    main()
