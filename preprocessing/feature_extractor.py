"""
Country-level daily feature extractor.

Reads raw GDELT events from PostgreSQL and materializes
structured features into country_daily_features.

Features computed per (country, date):
  - protest_score       fraction of protest events (CAMEO 14x)
  - violence_score      fraction of quad_class=4 (material conflict)
  - diplomatic_stress   inverted avg goldstein for diplomatic events
  - economic_stress     fraction of sanction/economic events (CAMEO 15x, 163)
  - terrorism_score     fraction of terror/assault events (CAMEO 18x-20x)
  - avg_sentiment       average AvgTone (normalized -10..+10 → 0..1)
  - avg_goldstein       average GoldsteinScale (normalized → 0..1)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import psycopg2
import psycopg2.extras

from scoring.composite import DEFAULT_SCORER

logger = logging.getLogger("preprocessing.feature_extractor")


# CAMEO event root codes by category
PROTEST_CODES = {14}
VIOLENCE_CODES = {18, 19, 20}
TERROR_CODES   = {18, 19, 20}          # overlap, weighted separately
ECONOMIC_CODES = {15, 16, 163}
SANCTION_CODES = {163}
MILITARY_CODES = {15, 16, 17}
VERBAL_CONF    = {11, 12, 13}


class FeatureExtractor:
    """
    Extracts country-level daily features from raw GDELT events.

    Usage::

        extractor = FeatureExtractor(dsn="postgresql://...")
        extractor.compute_daily_features(date(2024, 3, 15))
    """

    # Single query for ALL countries on a date -- replaces the old
    # one-query-per-country loop (TODO.md P1: "~35k round trips" over a
    # 162-day backfill). GROUP BY happens in Python (see
    # compute_daily_features) rather than in SQL because _compute_features'
    # category logic (protest/violence/terror flags, goldstein normalization)
    # is per-event conditional logic that doesn't translate cleanly into a
    # single aggregate query without duplicating it in two languages: this
    # keeps one source of truth for the feature definitions and still cuts
    # the round trips from ~2N per date to 2 total.
    _FETCH_ALL_SQL = """
        SELECT
            action_geo_country,
            event_root_code,
            quad_class,
            goldstein,
            avg_tone,
            num_mentions,
            num_sources
        FROM gdelt_events
        WHERE event_date = %s
          AND action_geo_country IS NOT NULL
          AND action_geo_country != ''
    """

    # Bulk upsert into feature store -- one round trip for every country on
    # a date, via execute_values, instead of one UPSERT per country.
    _UPSERT_COLUMNS = [
        "country", "feature_date", "total_events", "conflict_events", "cooperation_events",
        "protest_score", "violence_score", "diplomatic_stress",
        "economic_stress", "terrorism_score", "avg_sentiment", "avg_goldstein",
        "risk_score", "confidence", "coverage_tier",
    ]
    _UPSERT_SQL = f"""
        INSERT INTO country_daily_features ({", ".join(_UPSERT_COLUMNS)}, computed_at)
        VALUES %s
        ON CONFLICT (country, feature_date)
        DO UPDATE SET
            total_events        = EXCLUDED.total_events,
            conflict_events     = EXCLUDED.conflict_events,
            cooperation_events  = EXCLUDED.cooperation_events,
            protest_score       = EXCLUDED.protest_score,
            violence_score      = EXCLUDED.violence_score,
            diplomatic_stress   = EXCLUDED.diplomatic_stress,
            economic_stress     = EXCLUDED.economic_stress,
            terrorism_score     = EXCLUDED.terrorism_score,
            avg_sentiment       = EXCLUDED.avg_sentiment,
            avg_goldstein       = EXCLUDED.avg_goldstein,
            risk_score          = EXCLUDED.risk_score,
            confidence          = EXCLUDED.confidence,
            coverage_tier       = EXCLUDED.coverage_tier,
            computed_at         = EXCLUDED.computed_at
    """

    # Kept for backward compatibility (tests, ad-hoc debugging of a single
    # country) -- no longer used by compute_daily_features itself.
    _FETCH_SQL = """
        SELECT
            event_root_code,
            quad_class,
            goldstein,
            avg_tone,
            num_mentions,
            num_sources
        FROM gdelt_events
        WHERE action_geo_country = %s
          AND event_date = %s
    """

    def __init__(self, dsn: str):
        self._dsn = dsn

    def compute_daily_features(
        self,
        target_date: date,
        countries: list[str] | None = None,
    ) -> int:
        """
        Compute and upsert daily features for all (or specified) countries.

        One SELECT for every event on target_date (regardless of country),
        grouped in Python, then one bulk UPSERT -- not one round trip per
        country. A 162-day backfill over ~200 active countries/day used to
        cost ~2 round trips per country per day (~65k); this costs 2 per day
        (~324) regardless of how many countries are active.

        Returns:
            Number of country-date rows written.
        """
        conn = psycopg2.connect(self._dsn)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(self._FETCH_ALL_SQL, (target_date,))
                    rows = cur.fetchall()

                    if not rows:
                        logger.info("No events found for %s", target_date)
                        return 0

                    by_country: dict[str, list[tuple]] = {}
                    for country, *event_fields in rows:
                        by_country.setdefault(country, []).append(tuple(event_fields))

                    if countries:
                        by_country = {c: v for c, v in by_country.items() if c in countries}

                    logger.info(
                        "Computing features for %d countries on %s (%d raw events)",
                        len(by_country), target_date, len(rows),
                    )

                    upsert_rows = []
                    for country, events in by_country.items():
                        feature_row = self._compute_features(country, target_date, events)
                        upsert_rows.append(
                            tuple(feature_row[col] for col in self._UPSERT_COLUMNS) + (self._now(),)
                        )

                    if upsert_rows:
                        psycopg2.extras.execute_values(cur, self._UPSERT_SQL, upsert_rows)

            logger.info(
                "Feature extraction done: %d rows for %s",
                len(by_country), target_date,
            )
            return len(by_country)

        finally:
            conn.close()

    @staticmethod
    def _now():
        from datetime import datetime, timezone
        return datetime.now(timezone.utc)

    def compute_date_range(
        self,
        start: date,
        end: date,
    ) -> dict[date, int]:
        """Compute features for a range of dates. Returns {date: row_count}."""
        results = {}
        current = start
        while current <= end:
            count = self.compute_daily_features(current)
            results[current] = count
            current += timedelta(days=1)
        return results

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        country: str,
        feature_date: date,
        events: list[tuple],
    ) -> dict[str, Any]:
        """
        Compute normalized feature scores from raw event tuples.

        Input rows: (event_root_code, quad_class, goldstein, avg_tone,
                     num_mentions, num_sources)
        """
        n = len(events)

        # Accumulators
        goldsteins: list[float] = []
        tones: list[float] = []
        conflict_count = 0
        coop_count = 0
        protest_count = 0
        violence_count = 0
        terror_count = 0
        economic_count = 0
        total_mentions = 0

        for row in events:
            root_code, quad_class, goldstein, avg_tone, mentions, _ = row

            if goldstein is not None:
                goldsteins.append(float(goldstein))
            if avg_tone is not None:
                tones.append(float(avg_tone))
            if mentions:
                total_mentions += int(mentions)

            # Category flags
            if quad_class in (3, 4):
                conflict_count += 1
            if quad_class in (1, 2):
                coop_count += 1
            if root_code in PROTEST_CODES:
                protest_count += 1
            if quad_class == 4 or root_code in VIOLENCE_CODES:
                violence_count += 1
            if root_code in TERROR_CODES and quad_class == 4:
                terror_count += 1
            if root_code in ECONOMIC_CODES:
                economic_count += 1

        # --- Normalized scores (0–1) ---
        def ratio(count: int) -> float:
            return min(count / n, 1.0)

        avg_goldstein_raw = sum(goldsteins) / len(goldsteins) if goldsteins else 0.0
        avg_tone_raw      = sum(tones) / len(tones) if tones else 0.0

        # Goldstein: -10 to +10 → 0 to 1 (inverted: more negative = higher stress)
        avg_goldstein_norm = (avg_goldstein_raw + 10.0) / 20.0
        diplomatic_stress  = 1.0 - avg_goldstein_norm   # higher = more conflict

        # Tone: roughly -100 to +100 → 0 to 1
        avg_sentiment_norm = (avg_tone_raw + 100.0) / 200.0
        avg_sentiment_norm = max(0.0, min(1.0, avg_sentiment_norm))

        # Coverage tier: proxy for English-language media density.
        # Countries with few daily events are systematically undercovered by GDELT.
        if n < 3:
            coverage_tier = "sparse"
        elif n < 20:
            coverage_tier = "low"
        elif n < 100:
            coverage_tier = "medium"
        else:
            coverage_tier = "high"

        protest_score = ratio(protest_count)
        violence_score = ratio(violence_count)
        economic_score = ratio(economic_count)
        terrorism_score = ratio(terror_count)

        # diplomatic_stress is already 1-avg_goldstein_norm (higher=worse),
        # so it doubles directly as the war subscore's conflict_signal term
        # -- both are the same underlying quantity in this pipeline.
        # tone_negativity is deliberately omitted (left at compute_subscores'
        # default 0.0): this module's avg_sentiment = (avg_tone+100)/200 is
        # higher-for-*positive*-tone, the opposite sign convention from the
        # tone_neg (higher-for-negative-tone) that scripts/seed_db_from_cache.py
        # feeds into the same slot for parquet-backfilled rows -- a real,
        # unresolved cross-writer inconsistency in what this column means,
        # not something to paper over by guessing a sign here.
        sub = DEFAULT_SCORER.compute_subscores({
            "protest": protest_score,
            "violence": violence_score,
            "diplomatic_stress": diplomatic_stress,
            "economic_stress": economic_score,
            "terrorism": terrorism_score,
            "conflict_signal": diplomatic_stress,
        })
        risk_score = DEFAULT_SCORER.compute_composite_risk(
            sub["instability"], sub["war"], sub["terrorism"], sub["financial"]
        )
        # No rolling window at this call site (one day's events only) -- use
        # the same event-count threshold that already defines coverage_tier's
        # "high" bucket (n>=100) as a continuous coverage_fraction proxy.
        confidence = DEFAULT_SCORER.heuristic_confidence(min(n / 100, 1.0))

        return {
            "country":             country,
            "feature_date":        feature_date,
            "total_events":        n,
            "conflict_events":     conflict_count,
            "cooperation_events":  coop_count,
            "protest_score":       round(protest_score, 6),
            "violence_score":      round(violence_score, 6),
            "diplomatic_stress":   round(diplomatic_stress, 6),
            "economic_stress":     round(economic_score, 6),
            "terrorism_score":     round(terrorism_score, 6),
            "avg_sentiment":       round(avg_sentiment_norm, 6),
            "avg_goldstein":       round(avg_goldstein_norm, 6),
            "risk_score":          round(risk_score, 6),
            "confidence":          confidence,
            "coverage_tier":       coverage_tier,
        }
