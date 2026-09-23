"""
Regression tests for scoring/composite.py -- the single implementation of
the risk formula, consolidated from what were previously 5-6 independently
hand-typed (and, in two cases, actually different) copies. These tests exist
to catch exactly the kind of drift/duplication that motivated the
consolidation, and the hardcoded-confidence bug that started it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from scoring.composite import CompositeRiskScorer, DEFAULT_SCORER
from models.risk_model import HybridRiskTransformer
from models.forecaster import EscalationForecaster

REPO_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_FEATURES = {
    "protest": 0.3,
    "violence": 0.6,
    "diplomatic_stress": 0.4,
    "economic_stress": 0.2,
    "terrorism": 0.1,
    "tone_negativity": 0.5,
    "conflict_signal": 0.7,
}


def test_weights_sum_to_one():
    assert abs(sum(DEFAULT_SCORER.weights.values()) - 1.0) < 1e-9


def test_constructor_rejects_weights_not_summing_to_one():
    with pytest.raises(ValueError):
        CompositeRiskScorer({"instability": 0.5, "war": 0.5, "terrorism": 0.5, "financial": 0.5})


def test_composite_risk_matches_manual_calculation():
    sub = DEFAULT_SCORER.compute_subscores(SYNTHETIC_FEATURES)
    risk = DEFAULT_SCORER.compute_composite_risk(
        sub["instability"], sub["war"], sub["terrorism"], sub["financial"]
    )
    expected = (
        0.40 * sub["instability"]
        + 0.30 * sub["war"]
        + 0.20 * sub["terrorism"]
        + 0.10 * sub["financial"]
    )
    assert risk == pytest.approx(expected)


def test_composite_risk_clamped_to_unit_interval():
    # Sub-scores are themselves clamped to [0,1] by compute_subscores, and
    # weights sum to 1, so this should never exceed 1 -- assert the
    # invariant directly rather than trusting the arithmetic blindly.
    sub = DEFAULT_SCORER.compute_subscores({
        "protest": 1.0, "violence": 1.0, "diplomatic_stress": 1.0,
        "economic_stress": 1.0, "terrorism": 1.0,
        "tone_negativity": 1.0, "conflict_signal": 1.0,
    })
    risk = DEFAULT_SCORER.compute_composite_risk(
        sub["instability"], sub["war"], sub["terrorism"], sub["financial"]
    )
    assert 0.0 <= risk <= 1.0


def test_hybrid_risk_transformer_uses_the_same_weights():
    """The neural model's composite blend must use the exact same weights as
    the heuristic path -- this is the test that would have caught the
    original 3-formula drift if the model's RISK_WEIGHTS had been edited
    independently of scoring/composite.py."""
    assert HybridRiskTransformer.RISK_WEIGHTS == DEFAULT_SCORER.weights


def test_hybrid_risk_transformer_forward_matches_composite_formula():
    model = HybridRiskTransformer(num_features=7, d_model=32, num_heads=2, num_layers=1, seq_len=10)
    model.eval()
    features = torch.rand(2, 10, 7)
    mask = torch.ones(2, 10)
    with torch.no_grad():
        out = model(features, mask)

    # compute_composite_risk's min()/max() are for Python scalars, not
    # batched tensors (see models/risk_model.py's comment on why forward()
    # uses .clamp() directly) -- replicate the same clamp here rather than
    # calling compute_composite_risk on tensors.
    w = DEFAULT_SCORER.weights
    expected = (
        w["instability"] * out["instability"]
        + w["war"] * out["war"]
        + w["terrorism"] * out["terrorism"]
        + w["financial"] * out["financial"]
    ).clamp(0.0, 1.0)
    assert torch.allclose(out["risk_score"], expected, atol=1e-6)


def test_escalation_forecaster_uses_the_same_weights():
    assert EscalationForecaster.RISK_WEIGHTS == DEFAULT_SCORER.weights


def test_seed_script_and_risk_scorer_use_the_same_scorer_instance():
    """Both call sites must delegate to scoring.composite.DEFAULT_SCORER
    rather than re-implementing the formula -- this is what makes drift
    between them structurally impossible now, rather than merely coincidentally
    absent."""
    import scripts.seed_db_from_cache as seed_module
    import inference.risk_scorer as risk_scorer_module

    assert seed_module.DEFAULT_SCORER is DEFAULT_SCORER
    assert risk_scorer_module.DEFAULT_SCORER is DEFAULT_SCORER


def test_seed_script_compute_all_uses_composite_formula():
    import scripts.seed_db_from_cache as seed_module

    # f0..f6 per scripts/train_real_data.py's FEATURE_NAMES ordering.
    row = {"f0": 0.3, "f1": 0.6, "f2": 0.4, "f3": 0.2, "f4": 0.1, "f5": 0.5, "f6": 0.3}
    result = seed_module.compute_all(row)
    risk = result[-1]

    sub = DEFAULT_SCORER.compute_subscores({
        "protest": 0.3, "violence": 0.6, "diplomatic_stress": 0.4,
        "economic_stress": 0.2, "terrorism": 0.1,
        "tone_negativity": 0.5, "conflict_signal": 0.3,
    })
    expected_risk = DEFAULT_SCORER.compute_composite_risk(
        sub["instability"], sub["war"], sub["terrorism"], sub["financial"]
    )
    assert risk == pytest.approx(expected_risk)


NO_HARDCODED_FORMULA_PATTERN = re.compile(
    r"0\.4\d?\s*\*\s*(instability|self\.RISK_WEIGHTS)"
)
ALLOWED_FILES = {
    REPO_ROOT / "scoring" / "composite.py",
    REPO_ROOT / "tests" / "test_composite_formula.py",
}


def test_no_reintroduced_hardcoded_composite_weights():
    """Guard against someone re-inlining `0.40 * instability + ...` instead
    of importing CompositeRiskScorer -- this is the test that would catch
    the original problem (5-6 independently hand-typed copies) coming back."""
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if path in ALLOWED_FILES:
            continue
        if any(part in {".git", "__pycache__", "node_modules"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if NO_HARDCODED_FORMULA_PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, f"Hardcoded composite-weight arithmetic reintroduced in: {offenders}"
