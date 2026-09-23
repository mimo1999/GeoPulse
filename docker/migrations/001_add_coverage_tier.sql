-- Migration: add coverage_tier column to country_daily_features
-- Reflects GDELT English-language media density; used to expose
-- prediction accuracy stratified by coverage bias in backtest reports.

ALTER TABLE country_daily_features
    ADD COLUMN IF NOT EXISTS coverage_tier TEXT
        CHECK (coverage_tier IN ('high', 'medium', 'low', 'sparse', 'unknown'));

-- Back-fill existing rows from total_events heuristic
UPDATE country_daily_features
SET coverage_tier = CASE
    WHEN total_events >= 100 THEN 'high'
    WHEN total_events >= 20  THEN 'medium'
    WHEN total_events >= 3   THEN 'low'
    ELSE                          'sparse'
END
WHERE coverage_tier IS NULL;
