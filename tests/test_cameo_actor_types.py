"""Tests for data/cameo_actor_types.py."""

from __future__ import annotations

from data.cameo_actor_types import is_bare_role_code


def test_single_role_code():
    assert is_bare_role_code("GOV") is True
    assert is_bare_role_code("MIL") is True


def test_concatenated_role_codes():
    assert is_bare_role_code("GOVMIL") is True
    assert is_bare_role_code("GOVCOPMIL") is True


def test_country_prefixed_code_is_not_bare():
    # USAGOV isn't literally a country+role concatenation this module
    # understands (that's actor1_country's job) -- "USA" isn't a known
    # CAMEO Actor Type code, so the whole thing correctly fails the check.
    assert is_bare_role_code("USAGOV") is False


def test_non_multiple_of_three_is_not_bare():
    assert is_bare_role_code("GO") is False
    assert is_bare_role_code("GOVM") is False


def test_empty_or_none():
    assert is_bare_role_code("") is False
    assert is_bare_role_code(None) is False


def test_unknown_code_is_not_bare():
    assert is_bare_role_code("XYZ") is False
    assert is_bare_role_code("GOVXYZ") is False
