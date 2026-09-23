"""Tests for data/cameo_codes.py -- the CAMEO -> interaction-type mapping
that maps CAMEO event codes to interpretable interaction
types."""

from __future__ import annotations

from data.cameo_codes import (
    CAMEO_ROOT_NAMES,
    CONSULT_ROOT_CODE,
    interaction_type,
)


def test_all_20_root_codes_named():
    assert set(CAMEO_ROOT_NAMES.keys()) == set(range(1, 21))


def test_consult_is_its_own_category():
    assert interaction_type(root_code=CONSULT_ROOT_CODE) == "consultation"
    assert CAMEO_ROOT_NAMES[CONSULT_ROOT_CODE] == "Consult"


def test_cooperation_roots():
    for root in (1, 2, 3, 5, 6, 7, 8, 9):  # verbal + material cooperation, excluding Consult
        assert interaction_type(root_code=root) == "cooperation", root


def test_conflict_roots():
    for root in range(10, 21):  # verbal + material conflict, incl. mass violence
        assert interaction_type(root_code=root) == "conflict", root


def test_unknown_root_code_returns_none():
    assert interaction_type(root_code=99) is None
    assert interaction_type(root_code=0) is None


def test_quad_class_fallback_when_no_root_code():
    assert interaction_type(root_code=None, quad_class=1) == "cooperation"
    assert interaction_type(root_code=None, quad_class=2) == "cooperation"
    assert interaction_type(root_code=None, quad_class=3) == "conflict"
    assert interaction_type(root_code=None, quad_class=4) == "conflict"


def test_quad_class_alone_cannot_isolate_consultation():
    """Documents a real limitation: quad_class 1 covers both Consult (04) and
    the other verbal-cooperation roots, so quad_class alone can never yield
    'consultation' -- only a root_code can."""
    assert interaction_type(root_code=None, quad_class=1) != "consultation"


def test_no_inputs_returns_none():
    assert interaction_type(root_code=None, quad_class=None) is None
