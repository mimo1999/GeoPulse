"""
Runtime risk scoring engine.

Loads the trained model, fetches the latest country features,
runs inference (with MC Dropout), and returns a structured prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
import torch

from models.risk_model import HybridRiskTransformer
from models.dataset import FEATURE_COLUMNS, NUM_FEATURES
from advisory.rule_engine import AdvisoryEngine, RiskAdvisory
from scoring.composite import DEFAULT_SCORER

logger = logging.getLogger("inference.scorer")


@dataclass
class RiskPrediction:
    country: str
    prediction_date: date
    risk_score: float
    instability: float
    war_probability: float
    terrorism_risk: float
    financial_stress: float
    confidence: float
    trend: str
    advisory: RiskAdvisory
    coverage_tier: str = "unknown"  # GDELT media coverage density for this country

    def to_dict(self) -> dict:
        return {
            "country":         self.country,
            "prediction_date": str(self.prediction_date),
            "risk_score":      round(self.risk_score, 4),
            "instability":     round(self.instability, 4),
            "war_probability": round(self.war_probability, 4),
            "terrorism_risk":  round(self.terrorism_risk, 4),
            "financial_stress":round(self.financial_stress, 4),
            "confidence":      round(self.confidence, 4),
            "trend":           self.trend,
            "coverage_tier":   self.coverage_tier,
            **self.advisory.to_dict(),
        }


class RiskScorer:
    """
    Singleton-style risk scoring engine.

    Usage::

        scorer = RiskScorer(dsn=..., model_path=...)
        pred = scorer.score("Pakistan")
    """

    _FETCH_FEATURES_SQL = """
        SELECT
            feature_date,
            {cols},
            COALESCE(coverage_tier, 'unknown') AS coverage_tier
        FROM country_daily_features
        WHERE country = %s
          AND feature_date >= %s
          AND feature_date <= %s
        ORDER BY feature_date ASC
    """

    _FETCH_TREND_SQL = """
        SELECT AVG(risk_score)
        FROM country_daily_features
        WHERE country = %s
          AND feature_date >= %s
          AND feature_date <= %s
          AND risk_score IS NOT NULL
    """

    _UPSERT_PREDICTION_SQL = """
        INSERT INTO country_risk_predictions (
            country, prediction_time, forecast_horizon,
            risk_score, instability_score, war_probability,
            terrorism_risk, financial_stress, confidence,
            trend, advisory, model_version
        ) VALUES (
            %(country)s, NOW(), 30,
            %(risk_score)s, %(instability)s, %(war_probability)s,
            %(terrorism_risk)s, %(financial_stress)s, %(confidence)s,
            %(trend)s, %(advisory_text)s, %(model_version)s
        )
    """

    def __init__(
        self,
        dsn: str,
        model_path: Optional[str] = None,
        seq_len: int = 90,
        mc_passes: int = 50,
        device: str = "cpu",
    ):
        self._dsn = dsn
        self._seq_len = seq_len
        self._mc_passes = mc_passes
        self._device = device

        # Load model if available, otherwise use heuristic scorer
        self._model: Optional[HybridRiskTransformer] = None
        if model_path and Path(model_path).exists():
            logger.info("Loading model from %s", model_path)
            self._model = HybridRiskTransformer.load(model_path, device)
            logger.info("Model loaded (%d params)", self._model.parameter_count())
        else:
            logger.warning(
                "No model checkpoint found — using heuristic scoring. "
                "Train a model and provide model_path for neural inference."
            )

        self._advisory = AdvisoryEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, country: str, as_of: Optional[date] = None) -> RiskPrediction:
        """
        Score a single country.

        Args:
            country: ISO or FIPS country code (e.g. "PK", "US").
            as_of:   Reference date (default: today).

        Returns:
            RiskPrediction with all scores and advisory.
        """
        if as_of is None:
            as_of = date.today()

        feature_matrix, coverage_tier, last_real_date = self._load_features(country, as_of)
        trend = self._compute_trend(country, as_of)

        if self._model is not None:
            scores = self._neural_score(feature_matrix)
        else:
            recency_days = (as_of - last_real_date).days if last_real_date else None
            scores = self._heuristic_score(feature_matrix, recency_days)

        advisory = self._advisory.generate(
            country=country,
            risk_score=scores["risk_score"],
            confidence=scores["confidence"],
            trend=trend,
            instability=scores["instability"],
            war=scores["war"],
            terrorism=scores["terrorism"],
            financial=scores["financial"],
            feature_scores=self._last_day_features(feature_matrix),
        )

        prediction = RiskPrediction(
            country=country,
            prediction_date=as_of,
            risk_score=scores["risk_score"],
            instability=scores["instability"],
            war_probability=scores["war"],
            terrorism_risk=scores["terrorism"],
            financial_stress=scores["financial"],
            confidence=scores["confidence"],
            trend=trend,
            advisory=advisory,
            coverage_tier=coverage_tier,
        )

        self._persist_prediction(prediction)
        return prediction

    def score_all_countries(self, as_of: Optional[date] = None) -> list[RiskPrediction]:
        """Score all countries that have recent data."""
        if as_of is None:
            as_of = date.today()

        countries = self._fetch_active_countries(as_of)
        logger.info("Scoring %d countries for %s", len(countries), as_of)

        results = []
        for country in countries:
            try:
                pred = self.score(country, as_of)
                results.append(pred)
            except Exception as exc:
                logger.warning("Failed to score %s: %s", country, exc)

        return results

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_features(self, country: str, as_of: date) -> tuple[np.ndarray, str, Optional[date]]:
        """Load (seq_len, num_features) matrix, forward-filled.

        Returns:
            (matrix, coverage_tier, last_real_date) where coverage_tier is
            from the most recent row and last_real_date is the most recent
            date that had an actual (non-forward-filled) row -- used to
            derive heuristic_confidence's staleness penalty.
        """
        start = as_of - timedelta(days=self._seq_len - 1)
        cols = ", ".join(FEATURE_COLUMNS)

        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    self._FETCH_FEATURES_SQL.format(cols=cols),
                    (country, start, as_of),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        # Build date→values dict; last column is coverage_tier (string, not float)
        by_date = {row[0]: (list(row[1:-1]), row[-1]) for row in rows}
        matrix = np.zeros((self._seq_len, NUM_FEATURES), dtype=np.float32)
        coverage_tier = "unknown"
        last_real_date: Optional[date] = None

        last_valid = None
        for i in range(self._seq_len):
            d = start + timedelta(days=i)
            if d in by_date:
                vals, tier = by_date[d]
                vals = [v if v is not None else 0.0 for v in vals]
                matrix[i] = vals
                last_valid = vals
                coverage_tier = tier or "unknown"
                last_real_date = d
            elif last_valid is not None:
                matrix[i] = last_valid

        return matrix, coverage_tier, last_real_date

    def _last_day_features(self, matrix: np.ndarray) -> dict[str, float]:
        last = matrix[-1]
        return {col: float(val) for col, val in zip(FEATURE_COLUMNS, last)}

    def _compute_trend(self, country: str, as_of: date) -> str:
        """Compare recent 7-day avg vs previous 7-day avg risk."""
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                # Recent week
                cur.execute(
                    self._FETCH_TREND_SQL,
                    (country, as_of - timedelta(days=7), as_of),
                )
                recent = cur.fetchone()[0]

                # Prior week
                cur.execute(
                    self._FETCH_TREND_SQL,
                    (country, as_of - timedelta(days=14), as_of - timedelta(days=7)),
                )
                prior = cur.fetchone()[0]
        finally:
            conn.close()

        if recent is None or prior is None:
            return "stable"

        delta = float(recent) - float(prior)
        if delta > 0.03:
            return "increasing"
        if delta < -0.03:
            return "decreasing"
        return "stable"

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def _neural_score(self, matrix: np.ndarray) -> dict[str, float]:
        """Run neural inference with MC Dropout."""
        tensor = torch.from_numpy(matrix).unsqueeze(0)     # (1, T, F)
        mask = torch.ones(1, self._seq_len)                # all valid

        out = self._model.predict_with_confidence(         # type: ignore[union-attr]
            tensor, mask, n_passes=self._mc_passes,
        )

        return {
            "instability": float(out["instability"].item()),
            "war":         float(out["war"].item()),
            "terrorism":   float(out["terrorism"].item()),
            "financial":   float(out["financial"].item()),
            "risk_score":  float(out["risk_score"].item()),
            "confidence":  float(out["confidence"].item()),
        }

    def _heuristic_score(
        self, matrix: np.ndarray, recency_days: Optional[int] = None
    ) -> dict[str, float]:
        """
        Weighted heuristic for when no model is trained yet, delegating to
        scoring.composite.CompositeRiskScorer -- the same implementation
        scripts/seed_db_from_cache.py uses, replacing this method's
        previously-separate (and differently-weighted) formula. Uses the
        last 14 days of features.
        """
        recent = matrix[-14:]
        col_idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

        violence = float(np.mean(recent[:, col_idx["violence_score"]]))
        protest = float(np.mean(recent[:, col_idx["protest_score"]]))
        diplomatic_stress = float(np.mean(recent[:, col_idx["diplomatic_stress"]]))
        terror = float(np.mean(recent[:, col_idx["terrorism_score"]]))
        economic = float(np.mean(recent[:, col_idx["economic_stress"]]))
        # conflict_signal and tone_negativity are intentionally omitted (they
        # default to 0.0 in compute_subscores). country_daily_features.avg_goldstein
        # and avg_sentiment carry different conventions depending on which pipeline
        # wrote the row: the parquet-seeded rows hold percentile-ranked values with
        # goldstein already inverted (higher = more conflictual), live-ingested rows
        # hold (raw+10)/20 (higher = more cooperative), and POLECAT rows hold a
        # cooperation fraction. Rather than guess a sign per row, leave them out.
        sub = DEFAULT_SCORER.compute_subscores({
            "protest": protest,
            "violence": violence,
            "diplomatic_stress": diplomatic_stress,
            "economic_stress": economic,
            "terrorism": terror,
        })
        risk_score = DEFAULT_SCORER.compute_composite_risk(
            sub["instability"], sub["war"], sub["terrorism"], sub["financial"]
        )

        # Confidence is low when data is sparse (many zero rows) and/or stale.
        coverage = float(np.sum(matrix.sum(axis=1) > 0)) / self._seq_len
        confidence = DEFAULT_SCORER.heuristic_confidence(coverage, recency_days)

        return {**sub, "risk_score": risk_score, "confidence": confidence}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_prediction(self, pred: RiskPrediction) -> None:
        try:
            conn = psycopg2.connect(self._dsn)
            with conn, conn.cursor() as cur:
                cur.execute(
                    self._UPSERT_PREDICTION_SQL,
                    {
                        "country":       pred.country,
                        "risk_score":    pred.risk_score,
                        "instability":   pred.instability,
                        "war_probability": pred.war_probability,
                        "terrorism_risk": pred.terrorism_risk,
                        "financial_stress": pred.financial_stress,
                        "confidence":    pred.confidence,
                        "trend":         pred.trend,
                        "advisory_text": pred.advisory.advisory_text,
                        "model_version": "v0.2-unified-heuristic" if self._model is None else "v0.1",
                    },
                )
            conn.close()
        except Exception as exc:
            logger.warning("Failed to persist prediction for %s: %s", pred.country, exc)

    def _fetch_active_countries(self, as_of: date) -> list[str]:
        start = as_of - timedelta(days=7)
        conn = psycopg2.connect(self._dsn)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT country FROM country_daily_features "
                    "WHERE feature_date >= %s AND feature_date <= %s",
                    (start, as_of),
                )
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()
