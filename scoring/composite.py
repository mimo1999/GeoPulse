"""
The one implementation of the risk formula.

Before this module existed, the same conceptual formula was hand-typed in
five places (scripts/seed_db_from_cache.py, inference/risk_scorer.py,
models/risk_model.py, models/forecaster.py, models/forecaster_dataset.py /
scripts/train_real_data.py) with coefficients that had already drifted apart
between at least two of them, and a confidence value that in one of those
places was simply a hardcoded literal (0.75) rather than computed at all.

Every writer of risk_score/confidence must go through CompositeRiskScorer
instead of re-deriving this arithmetic. Import DEFAULT_SCORER for the common
case of using the project's configured weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

# Repo-root-relative, matching the convention already used by
# backend/main.py's _load_config() and the various scripts/*.py that read
# configs/config.yaml directly.
_CONFIG_PATH = Path("configs/config.yaml")

_FALLBACK_WEIGHTS = {
    "instability": 0.40,
    "war": 0.30,
    "terrorism": 0.20,
    "financial": 0.10,
}


class CompositeRiskScorer:
    """The one implementation of the risk formula. Every writer of
    risk_score/confidence (seed script, RiskScorer, HybridRiskTransformer,
    EscalationForecaster, training label generation) must go through this
    class instead of re-deriving the arithmetic."""

    # A hand-tuned, non-learned formula shouldn't claim more certainty than
    # a real (trained, ideally calibrated) model would need to earn -- this
    # ceiling is a deliberate choice, not an arbitrary one, and callers
    # should not push it upward without also addressing the fact that
    # nothing here is fit against outcomes yet (see scripts/eval_against_ucdp.py).
    MAX_HEURISTIC_CONFIDENCE = 0.6

    # Beyond this many days without a real (non-forward-filled) feature row,
    # heuristic confidence starts decaying toward zero over an equal further
    # window -- stale data means the formula is extrapolating blind.
    STALE_DATA_DAYS = 7

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or self._load_weights_from_config()
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"CompositeRiskScorer weights must sum to 1.0, got {total:.6f}: {self.weights}"
            )

    @staticmethod
    def _load_weights_from_config() -> dict[str, float]:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            weights = cfg.get("model", {}).get("risk_weights")
            if weights:
                return {
                    "instability": float(weights["instability"]),
                    "war": float(weights["war"]),
                    "terrorism": float(weights["terrorism"]),
                    "financial": float(weights["financial"]),
                }
        return dict(_FALLBACK_WEIGHTS)

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def compute_subscores(self, features: dict) -> dict[str, float]:
        """Compute the 4 sub-scores from canonical, pre-normalized inputs.

        All inputs are expected in [0, 1] on a "higher = worse" convention.
        This is the convention `country_daily_features.diplomatic_stress`
        already uses (preprocessing/feature_extractor.py computes it as
        `1 - avg_goldstein_norm`, i.e. higher = more conflictual) -- callers
        sourcing from a different raw representation (e.g. the parquet
        cache's raw goldstein_norm, which is high-for-*cooperative*) are
        responsible for inverting before calling this, not this function.

        Required keys: protest, violence, diplomatic_stress, economic_stress,
        terrorism.
        Optional keys (default 0.0 if absent -- some callers, e.g. the live
        DB-backed heuristic path, may not have every signal available):
        tone_negativity (higher = more negative sentiment/tone),
        conflict_signal (higher = more conflictual, goldstein-derived,
        already inverted to this module's convention).

        Sub-weights mirror the set scripts/seed_db_from_cache.py's
        compute_all() used (0.5/0.5, 0.4/0.4/0.2, x1.2, 0.7/0.3) rather than
        inference/risk_scorer.py's now-retired divergent set (0.6/0.4,
        0.7/0.3, x1.2, x1.1) -- kept as canonical since every existing
        trained/backtested artifact was built against seed_db_from_cache's
        formula shape.
        """
        protest = features["protest"]
        violence = features["violence"]
        diplomatic_stress = features["diplomatic_stress"]
        economic_stress = features["economic_stress"]
        terrorism_raw = features["terrorism"]
        tone_negativity = features.get("tone_negativity", 0.0)
        conflict_signal = features.get("conflict_signal", 0.0)

        instability = min(0.5 * violence + 0.5 * protest, 1.0)
        war = min(0.4 * violence + 0.4 * diplomatic_stress + 0.2 * conflict_signal, 1.0)
        terrorism = min(terrorism_raw * 1.2, 1.0)
        financial = min(0.7 * economic_stress + 0.3 * tone_negativity, 1.0)

        return {
            "instability": instability,
            "war": war,
            "terrorism": terrorism,
            "financial": financial,
        }

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def compute_composite_risk(
        self, instability: float, war: float, terrorism: float, financial: float
    ) -> float:
        """The one place the top-level weighted blend is written."""
        risk = (
            self.weights["instability"] * instability
            + self.weights["war"] * war
            + self.weights["terrorism"] * terrorism
            + self.weights["financial"] * financial
        )
        return min(max(risk, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Confidence (heuristic / non-neural path only)
    # ------------------------------------------------------------------

    def heuristic_confidence(
        self, coverage_fraction: float, data_recency_days: Optional[int] = None
    ) -> float:
        """Real, non-constant confidence for the heuristic path, replacing
        both scripts/seed_db_from_cache.py's hardcoded 0.75 and
        inference/risk_scorer.py's previously-separate `coverage * 0.6`.

        coverage_fraction: fraction of the lookback window with real
        (non-zero / actually-observed, not forward-filled) feature rows --
        sparse data means the formula is extrapolating.
        data_recency_days: days since the most recent real feature row, if
        known. None (the default) skips the recency penalty -- appropriate
        for backfilled historical rows, where "days stale relative to now"
        isn't a meaningful concept for a row that was current when written.
        """
        coverage_fraction = max(0.0, min(1.0, coverage_fraction))
        confidence = coverage_fraction * self.MAX_HEURISTIC_CONFIDENCE

        if data_recency_days is not None and data_recency_days > self.STALE_DATA_DAYS:
            decay = max(
                0.0,
                1.0 - (data_recency_days - self.STALE_DATA_DAYS) / self.STALE_DATA_DAYS,
            )
            confidence *= decay

        return round(confidence, 4)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def score(self, features: dict, coverage_fraction: float,
              data_recency_days: Optional[int] = None) -> dict[str, float]:
        """Sub-scores + composite risk + heuristic confidence in one call,
        for callers that just want the end-to-end heuristic path."""
        sub = self.compute_subscores(features)
        risk_score = self.compute_composite_risk(
            sub["instability"], sub["war"], sub["terrorism"], sub["financial"]
        )
        confidence = self.heuristic_confidence(coverage_fraction, data_recency_days)
        return {**sub, "risk_score": risk_score, "confidence": confidence}


DEFAULT_SCORER = CompositeRiskScorer()
