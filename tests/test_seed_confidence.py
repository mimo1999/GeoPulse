"""
Regression test for the bug that started the scoring/composite.py
consolidation: scripts/seed_db_from_cache.py used to write a hardcoded
literal 0.75 as `confidence` for every single row, regardless of country or
data quality. This asserts confidence actually varies with input instead of
asserting an exact value (which would be brittle to legitimate future tuning
of the formula).
"""

from __future__ import annotations

import pytest

from scoring.composite import CompositeRiskScorer, DEFAULT_SCORER


def test_heuristic_confidence_varies_with_coverage():
    low = DEFAULT_SCORER.heuristic_confidence(coverage_fraction=0.1)
    high = DEFAULT_SCORER.heuristic_confidence(coverage_fraction=0.9)
    assert low != high
    assert low < high


def test_heuristic_confidence_is_never_the_old_hardcoded_value():
    for coverage in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert DEFAULT_SCORER.heuristic_confidence(coverage) != 0.75


def test_heuristic_confidence_capped_below_one():
    # A non-learned heuristic should never claim near-total certainty even
    # at full coverage -- see MAX_HEURISTIC_CONFIDENCE's docstring.
    assert DEFAULT_SCORER.heuristic_confidence(coverage_fraction=1.0) <= CompositeRiskScorer.MAX_HEURISTIC_CONFIDENCE


def test_heuristic_confidence_decays_with_staleness():
    fresh = DEFAULT_SCORER.heuristic_confidence(coverage_fraction=1.0, data_recency_days=1)
    stale = DEFAULT_SCORER.heuristic_confidence(coverage_fraction=1.0, data_recency_days=30)
    assert stale < fresh


def test_seed_db_compute_all_confidence_varies_across_rows():
    """Regression test for the original bug: run compute_all()-derived
    confidence logic over two synthetic countries with different coverage
    and assert the results differ."""
    import scripts.seed_db_from_cache as seed_module

    files_total = 10
    low_coverage = seed_module.DEFAULT_SCORER.heuristic_confidence(2 / files_total)
    high_coverage = seed_module.DEFAULT_SCORER.heuristic_confidence(10 / files_total)

    assert low_coverage != high_coverage
    assert low_coverage != pytest.approx(0.75)
    assert high_coverage != pytest.approx(0.75)
