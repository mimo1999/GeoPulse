# -*- coding: utf-8 -*-
"""
Evaluate HybridRiskTransformer on POLECAT event data.

Produces the same metrics as eval_risk_transformer.py:
  AUC-ROC, AUC-PR, MAE, MSE, F1@0.5 per task (instability/war/terrorism/financial)
  + Composite risk skill score + MC-Dropout calibration

POLECAT daily features are resampled to bi-weekly (14-day averages) to match
the temporal resolution the model was trained on. The last 20% of each
country's bi-weekly sequence forms the held-out test set.

Usage:
    python scripts/eval_polecat.py
    python scripts/eval_polecat.py --zip data/POLECAT/dataverse_files.zip
    python scripts/eval_polecat.py --years 2022 2023 2024
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("eval_polecat")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_ZIP        = ROOT / "data" / "POLECAT" / "dataverse_files.zip"
DEFAULT_MODEL_PATH = ROOT / "models" / "real_data_model.pt"
RESULTS_PATH       = ROOT / "evaluation" / "results" / "polecat_eval.json"

TASKS        = ["instability", "war", "terrorism", "financial"]
RISK_WEIGHTS = {"instability": 0.40, "war": 0.30, "terrorism": 0.20, "financial": 0.10}

# Feature order must match training data (parquet f0..f6)
FEAT_COLS = [
    "protest_score",
    "violence_score",
    "diplomatic_stress",
    "economic_stress",
    "terrorism_score",
    "avg_sentiment",
    "avg_goldstein",
]

# Raw PLOVER event counts used for native ground-truth labels.
# These are independent of the feature-engineering formula and derived
# directly from PLOVER's event taxonomy (Scarborough et al. 2023).
NATIVE_COLS = [
    "n_assault",   # ASSAULT events (Material Conflict, PLOVER score -9)
    "n_mobilize",  # MOBILIZE events (Material Conflict, PLOVER score -5.5)
    "n_coerce",    # COERCE events (Material Conflict, PLOVER score -7)
    "n_protest",   # PROTEST events (Material Conflict, PLOVER score -5)
    "n_sanction",  # SANCTION events (Verbal Conflict, PLOVER score -6)
    "n_total",     # total events for normalisation
]

BIWEEKLY_DAYS = 14


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_polecat_daily(
    zip_path: Path,
    years: list[int] | None,
) -> dict[str, dict]:
    """
    Parse POLECAT zip and return per-country daily dicts containing both
    normalised feature arrays (model input) and raw PLOVER event counts
    (for native label computation).

    Returns:
        {fips: {date: {"feat": np.ndarray(7), "native": np.ndarray(6)}}}
    """
    from ingestion.polecat_parser import parse_file

    country_daily: dict[str, dict] = defaultdict(dict)
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
        if years:
            txt_names = [n for n in txt_names if any(str(y) in n for y in years)]

        logger.info("Loading %d POLECAT files ...", len(txt_names))
        for name in txt_names:
            logger.info("  Parsing %s", name)
            with zf.open(name) as raw:
                txt = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                rows = parse_file(txt)
            for (fips, ev_date), row in rows.items():
                feat   = np.array([row[c] for c in FEAT_COLS],   dtype=np.float32)
                native = np.array([row[c] for c in NATIVE_COLS], dtype=np.float32)
                country_daily[fips][ev_date] = {"feat": feat, "native": native}

    logger.info("Loaded %d countries", len(country_daily))
    return country_daily


def _resample_biweekly_avg(daily_vals: dict) -> list[np.ndarray]:
    """
    Average daily arrays into non-overlapping 14-day bins.
    Used for normalised feature ratios (0..1).

    Args:
        daily_vals: {date: np.ndarray}

    Returns:
        List of averaged arrays in chronological order.
    """
    if not daily_vals:
        return []
    import datetime
    dates = sorted(daily_vals.keys())
    delta = datetime.timedelta(days=BIWEEKLY_DAYS)
    bins: list[np.ndarray] = []
    window_start = dates[0]
    while window_start <= dates[-1]:
        window_end = window_start + delta
        chunk = [daily_vals[d] for d in dates if window_start <= d < window_end]
        if chunk:
            bins.append(np.stack(chunk).mean(axis=0))
        window_start = window_end
    return bins


def _resample_biweekly_sum(daily_vals: dict) -> list[np.ndarray]:
    """
    Sum daily arrays into non-overlapping 14-day bins.
    Used for raw integer event counts so totals are preserved.

    Args:
        daily_vals: {date: np.ndarray}

    Returns:
        List of summed arrays in chronological order.
    """
    if not daily_vals:
        return []
    import datetime
    dates = sorted(daily_vals.keys())
    delta = datetime.timedelta(days=BIWEEKLY_DAYS)
    bins: list[np.ndarray] = []
    window_start = dates[0]
    while window_start <= dates[-1]:
        window_end = window_start + delta
        chunk = [daily_vals[d] for d in dates if window_start <= d < window_end]
        if chunk:
            bins.append(np.stack(chunk).sum(axis=0))
        window_start = window_end
    return bins


# ---------------------------------------------------------------------------
# Native PLOVER ground-truth labels
# ---------------------------------------------------------------------------

def _compute_native_plover_labels(native_window: np.ndarray) -> np.ndarray:
    """
    Derive ground-truth labels directly from PLOVER event taxonomy counts.

    This replaces the old proxy-label formula which re-derived labels from
    the same feature-engineering pipeline used to build model inputs —
    creating a circular evaluation that measured POLECAT-on-POLECAT
    consistency rather than real predictive validity.

    native_window: (seq_len, 6) — summed raw PLOVER event counts per
        bi-weekly bin, columns matching NATIVE_COLS:
        [n_assault, n_mobilize, n_coerce, n_protest, n_sanction, n_total]

    Label construction uses PLOVER quad categories (Scarborough et al. 2023):
      instability — Material Conflict fraction (ASSAULT+MOBILIZE+COERCE+PROTEST)
      war         — Severe material conflict fraction (ASSAULT+MOBILIZE)
      terrorism   — ASSAULT fraction (highest-severity Material Conflict, score -9)
      financial   — SANCTION fraction (Verbal Conflict economic coercion, score -6)

    Scaling factors calibrate so that 0.5 threshold corresponds to a
    country where ~25% of events are Material Conflict (instability),
    ~12% are ASSAULT+MOBILIZE (war), ~10% are ASSAULT (terrorism), or
    ~6% are SANCTION (financial) — rates consistent with active-conflict
    countries in POLECAT data.
    """
    # Use the most recent 5 bins (≈10 weeks) as the signal window
    tail = native_window[-min(5, len(native_window)):]

    n_assault  = float(tail[:, 0].sum())
    n_mobilize = float(tail[:, 1].sum())
    n_coerce   = float(tail[:, 2].sum())
    n_protest  = float(tail[:, 3].sum())
    n_sanction = float(tail[:, 4].sum())
    n_total    = float(tail[:, 5].sum())

    if n_total == 0:
        return np.zeros(4, dtype=np.float32)

    # Material Conflict (PLOVER quad=4): ASSAULT, MOBILIZE, COERCE, PROTEST
    mat_conflict_rate = (n_assault + n_mobilize + n_coerce + n_protest) / n_total
    # Severe conflict: only the two highest-intensity Material Conflict types
    severe_rate   = (n_assault + n_mobilize) / n_total
    # Pure ASSAULT rate (most severe, incl. targeted killings, explosives, aerial)
    assault_rate  = n_assault / n_total
    # SANCTION rate (economic / diplomatic coercion)
    sanction_rate = n_sanction / n_total

    instability = float(np.clip(mat_conflict_rate * 2.0, 0.0, 1.0))
    war         = float(np.clip(severe_rate       * 4.0, 0.0, 1.0))
    terrorism   = float(np.clip(assault_rate      * 5.0, 0.0, 1.0))
    financial   = float(np.clip(sanction_rate     * 8.0, 0.0, 1.0))

    return np.array([instability, war, terrorism, financial], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class POLECATWindowDataset(Dataset):
    """
    Sliding-window dataset built from bi-weekly resampled POLECAT features.
    Last test_fraction of each country's series forms the held-out test set.

    Model input windows use normalised feature ratios (FEAT_COLS, 7-dim).
    Ground-truth labels are computed from raw PLOVER event counts (NATIVE_COLS)
    via _compute_native_plover_labels — independent of the feature formula.
    """

    def __init__(
        self,
        zip_path: Path,
        seq_len: int,
        test_fraction: float = 0.20,
        years: list[int] | None = None,
    ):
        country_daily = _load_polecat_daily(zip_path, years)

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        skipped = 0
        for fips, daily in country_daily.items():
            feat_daily   = {d: v["feat"]   for d, v in daily.items()}
            native_daily = {d: v["native"] for d, v in daily.items()}

            feat_bins   = _resample_biweekly_avg(feat_daily)
            native_bins = _resample_biweekly_sum(native_daily)

            if len(feat_bins) != len(native_bins):
                skipped += 1
                continue

            feat_arr   = np.stack(feat_bins)   if feat_bins   else np.empty((0, 7), dtype=np.float32)
            native_arr = np.stack(native_bins) if native_bins else np.empty((0, 6), dtype=np.float32)
            T = len(feat_arr)
            if T < seq_len:
                skipped += 1
                continue

            test_start_idx = max(seq_len - 1, int(T * (1.0 - test_fraction)))
            for end in range(test_start_idx, T):
                feat_window   = feat_arr  [end - seq_len + 1 : end + 1]  # (seq_len, 7)
                native_window = native_arr[end - seq_len + 1 : end + 1]  # (seq_len, 6)
                mask   = (feat_window.sum(axis=1) > 0).astype(np.float32)
                labels = _compute_native_plover_labels(native_window)
                self.samples.append((feat_window, mask, labels))

        logger.info(
            "POLECATWindowDataset: %d test windows from %d countries (%d skipped — too short)",
            len(self.samples), len(country_daily) - skipped, skipped,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        window, mask, labels = self.samples[idx]
        return {
            "features": torch.from_numpy(window),
            "mask":     torch.from_numpy(mask),
            "labels":   torch.from_numpy(labels),
        }


# ---------------------------------------------------------------------------
# Model + inference (mirrors eval_risk_transformer.py)
# ---------------------------------------------------------------------------

def load_model(path: Path, device: str):
    from models.risk_model import HybridRiskTransformer

    ckpt = torch.load(str(path), map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg   = ckpt["config"]
        model = HybridRiskTransformer(**cfg)
        model.load_state_dict(ckpt["state_dict"])
        seq_len = cfg.get("seq_len", 21)
    else:
        pe      = ckpt["pos_enc.pe"]
        seq_len = int(pe.shape[1]) - 10
        d_model = int(pe.shape[2])
        in_feats = int(ckpt["input_proj.0.weight"].shape[1])
        model   = HybridRiskTransformer(num_features=in_feats, d_model=d_model, seq_len=seq_len)
        model.load_state_dict(ckpt)

    model = model.to(device).eval()
    logger.info(
        "Loaded HybridRiskTransformer  params=%d  seq_len=%d",
        model.parameter_count(), seq_len,
    )
    return model, seq_len


def run_inference(model, loader, device):
    all_preds  = {t: [] for t in TASKS}
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask     = batch["mask"].to(device)
            lab      = batch["labels"].cpu().numpy()
            out = model(features, mask)
            for i, t in enumerate(TASKS):
                all_preds[t].append(out[t].cpu().numpy())
            all_labels.append(lab)
    return (
        {t: np.concatenate(v) for t, v in all_preds.items()},
        np.concatenate(all_labels, axis=0),
    )


def run_mc_inference(model, loader, device, n_passes):
    all_preds = {t: [] for t in TASKS}
    all_conf  = []
    all_labels = []
    for batch in loader:
        features = batch["features"].to(device)
        mask     = batch["mask"].to(device)
        out = model.predict_with_confidence(features, mask, n_passes=n_passes)
        for t in TASKS:
            all_preds[t].append(out[t].cpu().numpy())
        all_conf.append(out["confidence"].cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    return (
        {t: np.concatenate(v) for t, v in all_preds.items()},
        np.concatenate(all_conf),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Metrics (identical to eval_risk_transformer.py)
# ---------------------------------------------------------------------------

def _auc_roc(labels, preds, threshold=0.5):
    from sklearn.metrics import roc_auc_score
    binary = (labels >= threshold).astype(int)
    if binary.max() == binary.min():
        return float("nan")
    return float(roc_auc_score(binary, preds))


def _auc_pr(labels, preds, threshold=0.5):
    from sklearn.metrics import average_precision_score
    binary = (labels >= threshold).astype(int)
    if binary.max() == binary.min():
        return float("nan")
    return float(average_precision_score(binary, preds))


def _f1(labels, preds, threshold=0.5):
    from sklearn.metrics import f1_score
    return float(f1_score(
        (labels >= threshold).astype(int),
        (preds  >= threshold).astype(int),
        zero_division=0,
    ))


def _ece(confidences, errors, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = float((errors[mask] < 0.10).mean())
        ece += float(mask.mean()) * abs(bin_conf - bin_acc)
    return ece


def compute_metrics(preds, labels):
    results = {}
    for i, task in enumerate(TASKS):
        p = preds[task]
        l = labels[:, i]
        results[task] = {
            "n":       int(len(p)),
            "auc_roc": round(_auc_roc(l, p), 4),
            "auc_pr":  round(_auc_pr(l, p),  4),
            "mae":     round(float(np.mean(np.abs(p - l))), 4),
            "mse":     round(float(np.mean((p - l) ** 2)),  4),
            "f1_0.5":  round(_f1(l, p, 0.5), 4),
        }
    composite_pred  = sum(RISK_WEIGHTS[t] * preds[t]    for t in TASKS)
    composite_label = sum(RISK_WEIGHTS[t] * labels[:, i] for i, t in enumerate(TASKS))
    baseline_mae    = float(np.mean(np.abs(composite_label.mean() - composite_label)))
    model_mae       = float(np.mean(np.abs(composite_pred - composite_label)))
    skill           = (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0 else 0.0
    results["composite_risk"] = {
        "n":            int(len(composite_pred)),
        "mae":          round(model_mae, 4),
        "mse":          round(float(np.mean((composite_pred - composite_label) ** 2)), 4),
        "baseline_mae": round(baseline_mae, 4),
        "skill_score":  round(skill, 4),
    }
    return results


def print_report(metrics, mc_metrics, n_samples, model_path, zip_path):
    SEP = "=" * 70
    print("\n" + SEP)
    print("  GEOPULSE  --  HYBRID RISK TRANSFORMER  x  POLECAT EVALUATION")
    print(SEP)
    print(f"  Model   : {model_path}")
    print(f"  Data    : {zip_path}  (bi-weekly resampled, last 20% per country)")
    print(f"  Labels  : PLOVER native event counts (ASSAULT/MOBILIZE/COERCE/")
    print(f"            PROTEST/SANCTION rates) — not re-derived from features")
    print(f"  Samples : {n_samples:,}")
    print()
    print("-- Per-Task Metrics " + "-" * 51)
    print(f"  {'Task':<14} {'AUC-ROC':>8} {'AUC-PR':>8} {'MAE':>8} {'MSE':>8} {'F1@0.5':>8}")
    print("  " + "-" * 64)
    for task in TASKS:
        m = metrics[task]
        auc_roc = f"{m['auc_roc']:.4f}" if m['auc_roc'] == m['auc_roc'] else "   n/a"
        auc_pr  = f"{m['auc_pr']:.4f}"  if m['auc_pr']  == m['auc_pr']  else "   n/a"
        print(f"  {task:<14} {auc_roc:>8} {auc_pr:>8} {m['mae']:>8.4f} {m['mse']:>8.4f} {m['f1_0.5']:>8.4f}")
    print()
    print("-- Composite Risk Score " + "-" * 47)
    cr = metrics["composite_risk"]
    sign = "+" if cr["skill_score"] >= 0 else ""
    print(f"  MAE (model)      {cr['mae']:.4f}")
    print(f"  MAE (baseline)   {cr['baseline_mae']:.4f}")
    print(f"  Skill score      {sign}{cr['skill_score']:.1%}  vs naive mean baseline")
    print()
    if mc_metrics:
        print("-- MC-Dropout Confidence Calibration " + "-" * 33)
        print(f"  ECE                        {mc_metrics['ece']:.4f}")
        print(f"  Mean confidence            {mc_metrics['mean_confidence']:.4f}")
        print(f"  Coverage @ |err| < 0.10    {mc_metrics['coverage_0.10']:.1%}")
        print(f"  Coverage @ |err| < 0.20    {mc_metrics['coverage_0.20']:.1%}")
        print()
    avg_auc = np.nanmean([metrics[t]["auc_roc"] for t in TASKS])
    best    = min(TASKS, key=lambda t: metrics[t]["mae"])
    worst   = max(TASKS, key=lambda t: metrics[t]["mae"])
    print("-- Interpretation " + "-" * 52)
    print(f"  Mean AUC-ROC across tasks : {avg_auc:.4f}")
    print(f"  Best task  : {best}  (MAE {metrics[best]['mae']:.4f})")
    print(f"  Worst task : {worst}  (MAE {metrics[worst]['mae']:.4f})")
    if cr["skill_score"] > 0.05:
        print(f"  [+] Model beats naive baseline by {cr['skill_score']:.1%} on POLECAT composite risk.")
    elif cr["skill_score"] > 0:
        print("  [~] Marginal improvement over naive baseline on POLECAT.")
    else:
        print("  [-] Model underperforms naive baseline on POLECAT — domain shift expected.")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridRiskTransformer on POLECAT data")
    parser.add_argument("--zip",         default=str(DEFAULT_ZIP))
    parser.add_argument("--model-path",  default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--years",       nargs="+", type=int, default=None,
                        help="Restrict to these years (default: all)")
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--mc-passes",   type=int, default=30)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--output",      default=str(RESULTS_PATH))
    args = parser.parse_args()

    zip_path   = Path(args.zip)
    model_path = Path(args.model_path)

    if not zip_path.exists():
        logger.error("POLECAT zip not found: %s", zip_path)
        sys.exit(1)
    if not model_path.exists():
        fallback = ROOT / "models" / "real_data_model.pt"
        if fallback.exists():
            logger.warning("Checkpoint not found at %s, falling back to %s", model_path, fallback)
            model_path = fallback
        else:
            logger.error("No checkpoint found.")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, seq_len = load_model(model_path, device)

    logger.info("Building POLECAT test dataset (seq_len=%d, bi-weekly, test_fraction=%.0f%%) ...",
                seq_len, args.test_fraction * 100)
    ds = POLECATWindowDataset(
        zip_path, seq_len=seq_len,
        test_fraction=args.test_fraction,
        years=args.years,
    )
    if len(ds) == 0:
        logger.error("Dataset is empty — check zip path and year filters.")
        sys.exit(1)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Running deterministic inference ...")
    preds, labels = run_inference(model, loader, device)
    metrics = compute_metrics(preds, labels)

    mc_metrics = None
    if args.mc_passes > 0:
        logger.info("Running MC-Dropout (%d passes) ...", args.mc_passes)
        mc_preds, confidences, mc_labels = run_mc_inference(model, loader, device, args.mc_passes)
        composite_pred  = sum(RISK_WEIGHTS[t] * mc_preds[t]    for t in TASKS)
        composite_label = sum(RISK_WEIGHTS[t] * mc_labels[:, i] for i, t in enumerate(TASKS))
        errors          = np.abs(composite_pred - composite_label)
        mc_metrics = {
            "ece":             round(_ece(confidences, errors), 4),
            "mean_confidence": round(float(confidences.mean()), 4),
            "coverage_0.10":   round(float((errors < 0.10).mean()), 4),
            "coverage_0.20":   round(float((errors < 0.20).mean()), 4),
        }
        metrics["mc_calibration"] = mc_metrics

    print_report(metrics, mc_metrics, len(ds), model_path, zip_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path":    str(model_path),
        "data_source":   f"polecat:{zip_path.name}",
        "label_source":  "plover_native_counts",
        "years":         args.years,
        "test_fraction": args.test_fraction,
        "n_samples":     len(ds),
        "metrics":       metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
