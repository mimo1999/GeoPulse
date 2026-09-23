"""
External validation of GeoPulse predictions against UCDP organized-violence data.

Breaks the GDELT-circularity in the existing evaluation by comparing model
predictions against an entirely independent coding source:
  UCDP Organized Violence Country-Year Dataset v26.1 (Sundberg & Melander 2013)

Ground truth:
  - sb_total_deaths_best  (state-based battle deaths, best estimate)
  - ns_total_deaths_best  (non-state conflict deaths)
  - os_total_deaths_best  (one-sided violence/civilian killings)
  - sb_exist              (binary: any state-based armed conflict)

Metrics:
  - Spearman rank correlation: avg predicted risk vs log(total deaths + 1)
  - AUC-ROC: predicted war_probability vs UCDP any-armed-conflict binary label
  - AUC-ROC: predicted risk_score vs high-death binary (> 100 battle deaths/year)
  - Country coverage report: how many matched between backtest and UCDP

Usage::

    python scripts/eval_against_ucdp.py \\
        --ucdp  data/UCDP/organizedviolencecy-261-csv.zip \\
        --backtest evaluation/results/backtest_results.json \\
        --year  2024 \\
        --out   evaluation/results/ucdp_eval.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_ucdp")

# ---------------------------------------------------------------------------
# Country name → FIPS 10-4 mapping
# Covers the most common UCDP country names.  Extend as needed.
# ---------------------------------------------------------------------------

UCDP_NAME_TO_FIPS: dict[str, str] = {
    "Afghanistan": "AF",
    "Albania": "AL",
    "Algeria": "AG",
    "Angola": "AO",
    "Argentina": "AR",
    "Armenia": "AM",
    "Azerbaijan": "AJ",
    "Bangladesh": "BG",
    "Bosnia-Herzegovina": "BK",
    "Brazil": "BR",
    "Burundi": "BY",
    "Cambodia": "CB",
    "Cameroon": "CM",
    "Central African Republic": "CT",
    "Chad": "CD",
    "Chile": "CI",
    "China": "CH",
    "Colombia": "CO",
    "Croatia": "HR",
    "Cuba": "CU",
    "Democratic Republic of Congo": "CG",
    "DR Congo (Zaire)": "CG",
    "Djibouti": "DJ",
    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "ES",
    "Eritrea": "ER",
    "Ethiopia": "ET",
    "Georgia": "GG",
    "Guatemala": "GT",
    "Guinea": "GV",
    "Guinea-Bissau": "PU",
    "Haiti": "HA",
    "Honduras": "HO",
    "India": "IN",
    "Indonesia": "ID",
    "Iran": "IR",
    "Iraq": "IZ",
    "Israel": "IS",
    "Jordan": "JO",
    "Kenya": "KE",
    "Kosovo": "KV",
    "Lebanon": "LE",
    "Liberia": "LI",
    "Libya": "LY",
    "Madagascar": "MA",
    "Malawi": "MI",
    "Mali": "ML",
    "Mexico": "MX",
    "Moldova": "MD",
    "Morocco": "MO",
    "Mozambique": "MZ",
    "Myanmar (Burma)": "BM",
    "Myanmar": "BM",
    "Nepal": "NP",
    "Nicaragua": "NU",
    "Niger": "NG",
    "Nigeria": "NI",
    "North Korea": "KN",
    "Pakistan": "PK",
    "Palestinian State": "WE",
    "Papua New Guinea": "PP",
    "Peru": "PE",
    "Philippines": "RP",
    "Russia (Soviet Union)": "RS",
    "Russia": "RS",
    "Rwanda": "RW",
    "Saudi Arabia": "SA",
    "Senegal": "SG",
    "Sierra Leone": "SL",
    "Somalia": "SO",
    "South Africa": "SF",
    "South Korea": "KS",
    "South Sudan": "OD",
    "Spain": "SP",
    "Sri Lanka": "CE",
    "Sudan": "SU",
    "Syria": "SY",
    "Tajikistan": "TI",
    "Tanzania": "TZ",
    "Thailand": "TH",
    "Togo": "TO",
    "Tunisia": "TS",
    "Turkey/Ottoman Empire": "TU",
    "Turkey": "TU",
    "Uganda": "UG",
    "Ukraine": "UP",
    "United Kingdom": "UK",
    "United States of America": "US",
    "Uzbekistan": "UZ",
    "Venezuela": "VE",
    "Vietnam (South Vietnam)": "VM",
    "Vietnam": "VM",
    "Yemen (North Yemen)": "YM",
    "Yemen": "YM",
    "Yugoslavia": "YI",
    "Zimbabwe": "ZI",
    "Zambia": "ZA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy import stats  # type: ignore[import]
    if len(x) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(x, y)
    return float(rho)


def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUC-ROC (sklearn if available, else manual)."""
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore[import]
        if labels.sum() == 0 or labels.sum() == len(labels):
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except ImportError:
        pass
    # Manual: count concordant pairs
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    concordant = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return concordant / (len(pos) * len(neg))


def _load_ucdp_cy(zip_path: Path, year: int) -> dict[str, dict]:
    """
    Load UCDP Organized Violence Country-Year data for a given year.

    Returns dict keyed by FIPS code:
        {fips: {sb_deaths, ns_deaths, os_deaths, total_deaths,
                any_sb_conflict, high_death, high_oneside}}
    """
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            rows = [r for r in reader if int(r["year"]) == year]

    if not rows:
        logger.warning("No UCDP data found for year %d", year)
        return {}

    logger.info("Loaded %d UCDP CY rows for year %d", len(rows), year)

    ucdp: dict[str, dict] = {}
    skipped = []
    for row in rows:
        name_raw = row["country"].strip()
        fips = UCDP_NAME_TO_FIPS.get(name_raw)
        if fips is None:
            skipped.append(name_raw)
            continue

        def _int(col: str) -> int:
            try:
                return int(row.get(col) or 0)
            except (ValueError, TypeError):
                return 0

        sb   = _int("sb_total_deaths_best")
        ns   = _int("ns_total_deaths_best")
        os   = _int("os_total_deaths_best")
        total = sb + ns + os

        ucdp[fips] = {
            "country_name":    name_raw,
            "sb_deaths":       sb,
            "ns_deaths":       ns,
            "os_deaths":       os,
            "total_deaths":    total,
            "any_sb_conflict": int(row.get("sb_exist", "0") == "1"),
            # Threshold labels for calibration-style evaluation
            "war_label":       int(sb > 100),    # >100 battle deaths = significant armed conflict
            "terror_label":    int(os > 50),     # >50 one-sided killings = mass atrocity/terrorism proxy
            "high_death":      int(total > 200), # high-lethality conflict
        }

    if skipped:
        logger.debug("Unmatched UCDP country names (no FIPS): %s", sorted(skipped))

    return ucdp


def _load_backtest_predictions(bt_path: Path, year: int) -> dict[str, dict]:
    """
    Aggregate backtest fold predictions for a given year by country.

    For each country, compute:
        avg_risk:          mean predicted risk across all folds/horizons in year
        avg_war:           not directly available in fold_log — skipped
        n_predictions:     count of predictions
    """
    with open(bt_path) as f:
        fold_log = json.load(f)["fold_log"]

    # Filter to predictions whose pivot_date falls in the target year
    year_str = str(year)
    year_log = [
        p for p in fold_log
        if p.get("pivot_date", "").startswith(year_str)
    ]
    logger.info("Backtest predictions in year %d: %d", year, len(year_log))

    agg: dict[str, dict] = defaultdict(lambda: {
        "risks": [],
        "actuals": [],
        "errors": [],
    })
    for p in year_log:
        c = p["country"]
        agg[c]["risks"].append(float(p["predicted"]))
        if p.get("actual") is not None:
            agg[c]["actuals"].append(float(p["actual"]))
        if p.get("error") is not None:
            agg[c]["errors"].append(float(p["error"]))

    result = {}
    for c, d in agg.items():
        result[c] = {
            "avg_predicted_risk": float(np.mean(d["risks"])),
            "avg_actual_risk":    float(np.mean(d["actuals"])) if d["actuals"] else None,
            "avg_error":          float(np.mean(d["errors"])) if d["errors"] else None,
            "n_predictions":      len(d["risks"]),
        }
    return result


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    ucdp_zip: Path,
    backtest_json: Path,
    year: int,
    out_path: Path,
) -> dict:
    ucdp = _load_ucdp_cy(ucdp_zip, year)
    preds = _load_backtest_predictions(backtest_json, year)

    # Find countries present in both
    common = sorted(set(ucdp) & set(preds))
    logger.info(
        "Matched %d countries (UCDP: %d, backtest: %d)",
        len(common), len(ucdp), len(preds),
    )

    if not common:
        logger.error("No overlap between UCDP and backtest countries. "
                     "Check UCDP_NAME_TO_FIPS mapping.")
        return {"error": "no_overlap"}

    risk_scores   = np.array([preds[c]["avg_predicted_risk"] for c in common])
    log_deaths    = np.array([np.log1p(ucdp[c]["total_deaths"])   for c in common])
    log_sb_deaths = np.array([np.log1p(ucdp[c]["sb_deaths"])      for c in common])
    any_sb        = np.array([ucdp[c]["any_sb_conflict"]           for c in common])
    war_label     = np.array([ucdp[c]["war_label"]                 for c in common])
    terror_label  = np.array([ucdp[c]["terror_label"]              for c in common])
    high_death    = np.array([ucdp[c]["high_death"]                for c in common])

    # Spearman rank correlation
    try:
        from scipy import stats  # type: ignore[import]
        rho_total, p_total  = stats.spearmanr(risk_scores, log_deaths)
        rho_sb,    p_sb     = stats.spearmanr(risk_scores, log_sb_deaths)
        spearman = {
            "risk_vs_total_deaths_rho": round(float(rho_total), 4),
            "risk_vs_total_deaths_p":   round(float(p_total),   4),
            "risk_vs_sb_deaths_rho":    round(float(rho_sb),    4),
            "risk_vs_sb_deaths_p":      round(float(p_sb),      4),
        }
    except ImportError:
        logger.warning("scipy not available; skipping Spearman correlation")
        spearman = {}

    # AUC-ROC (model score vs UCDP binary label)
    auc_any_sb   = _auc_roc(risk_scores, any_sb)
    auc_war      = _auc_roc(risk_scores, war_label)
    auc_terror   = _auc_roc(risk_scores, terror_label)
    auc_high     = _auc_roc(risk_scores, high_death)

    # Top misses: high UCDP deaths but low predicted risk
    country_detail = sorted([
        {
            "fips":            c,
            "country_name":    ucdp[c]["country_name"],
            "avg_pred_risk":   round(preds[c]["avg_predicted_risk"], 4),
            "ucdp_total_deaths": ucdp[c]["total_deaths"],
            "ucdp_sb_deaths":  ucdp[c]["sb_deaths"],
            "any_sb_conflict": ucdp[c]["any_sb_conflict"],
            "war_label":       ucdp[c]["war_label"],
        }
        for c in common
    ], key=lambda x: -x["ucdp_total_deaths"])

    results = {
        "label_source":  "ucdp_organized_violence_cy_v26.1",
        "evaluation_year": year,
        "n_ucdp_countries":     len(ucdp),
        "n_backtest_countries": len(preds),
        "n_matched":            len(common),
        "ucdp_prevalence": {
            "any_armed_conflict":      int(any_sb.sum()),
            "war_label_gt100_deaths":  int(war_label.sum()),
            "terror_label_gt50_os":    int(terror_label.sum()),
            "high_death_gt200":        int(high_death.sum()),
        },
        "spearman_rank_correlation": spearman,
        "auc_roc": {
            "risk_score_vs_any_armed_conflict": round(auc_any_sb, 4) if not np.isnan(auc_any_sb) else None,
            "risk_score_vs_war_gt100_deaths":   round(auc_war,   4) if not np.isnan(auc_war)    else None,
            "risk_score_vs_terror_os_gt50":     round(auc_terror,4) if not np.isnan(auc_terror) else None,
            "risk_score_vs_high_death_gt200":   round(auc_high,  4) if not np.isnan(auc_high)   else None,
        },
        "top_conflicts_by_deaths": country_detail[:20],
        "interpretation": (
            "AUC-ROC > 0.70 with UCDP labels indicates the model ranks truly conflicted "
            "countries higher, validating discrimination power independent of GDELT. "
            "Spearman rho > 0.50 indicates monotone relationship between predicted risk "
            "and actual organized-violence lethality."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", out_path)

    # Print summary
    print("\n=== UCDP External Validation ===")
    print(f"Year {year} | Matched countries: {len(common)}")
    print(f"AUC-ROC vs any armed conflict : {auc_any_sb:.4f}")
    print(f"AUC-ROC vs war (>100 deaths)  : {auc_war:.4f}")
    if spearman:
        print(f"Spearman rho vs total deaths  : {rho_total:.4f} (p={p_total:.3f})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ucdp",
        default="data/UCDP/organizedviolencecy-261-csv.zip",
        help="Path to UCDP organizedviolencecy zip",
    )
    ap.add_argument(
        "--backtest",
        default="evaluation/results/backtest_results.json",
        help="Path to backtest_results.json from run_backtest.py",
    )
    ap.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year to evaluate (must be in both UCDP and backtest data)",
    )
    ap.add_argument(
        "--out",
        default="evaluation/results/ucdp_eval.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    evaluate(
        ucdp_zip=Path(args.ucdp),
        backtest_json=Path(args.backtest),
        year=args.year,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
