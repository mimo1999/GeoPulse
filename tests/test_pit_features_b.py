"""
Integration tests for preprocessing/pit_features_b.py. Skipped if Postgres
isn't reachable, same pattern as tests/test_pit_labels.py. Assumes the
2023-2025 raw GDELT backfill (Step 5) has been run.

build_phase_b_panel() cross-joins all 126 countries even for a single
run_date (it's a set-based query, not a per-country loop -- see the
module's own docstring on the anti-N+1 discipline), so a single call
already costs a couple of minutes against the full 47.8M-row backfill.
All tests here share ONE module-scoped panel build rather than one per
test function.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

psycopg2 = pytest.importorskip("psycopg2")

from preprocessing.pit_features_b import B1_CODES, build_phase_b_panel

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"
RUN_DATE = date(2024, 6, 1)


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg2.connect(DSN, connect_timeout=3)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc}); skipping Phase B feature tests")
    yield c
    c.close()


@pytest.fixture(scope="module")
def panel(conn):
    return build_phase_b_panel(RUN_DATE, RUN_DATE)


@pytest.fixture(scope="module")
def syria_row(panel):
    row = panel[panel["fips2"] == "SY"]
    if row.empty or row.iloc[0]["n_events_b"] == 0:
        pytest.skip("no Syria events in this window to check against")
    return row.iloc[0]


def test_panel_shape(panel):
    assert len(panel) > 0
    assert "fips2" in panel.columns and "run_date" in panel.columns
    b1_cols = [c for c in panel.columns if c.startswith("b1_")]
    # share + log_count + log_count_delta per code
    assert len(b1_cols) == len(B1_CODES) * 3


def test_pit_boundary_current_window(conn, syria_row):
    """The current window for run_date t must never include events dated
    on or after t - 7d. Recompute Syria's event count directly with the
    exact boundary and compare to the panel's own n_events_b."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM gdelt_events
               WHERE action_geo_country = 'SY'
                 AND event_date > %s - INTERVAL '37 days'
                 AND event_date <= %s - INTERVAL '7 days'""",
            (RUN_DATE, RUN_DATE),
        )
        expected = cur.fetchone()[0]
    assert int(syria_row["n_events_b"]) == expected

    # This equality check IS the PIT boundary test: `expected` was computed
    # with the correct t-7d upper bound. If the production query instead
    # used, say, <= t (leaking up to the run date itself), n_events_b would
    # include those extra rows and the two counts would diverge.
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM gdelt_events
               WHERE action_geo_country = 'SY'
                 AND event_date > %s - INTERVAL '37 days'
                 AND event_date <= %s""",
            (RUN_DATE, RUN_DATE),
        )
        leaky_count = cur.fetchone()[0]
    assert leaky_count > expected, "test fixture has no distinguishing power on this date"


def test_b1_log_count_matches_raw(conn, syria_row):
    """Hand-check one B1 code's count against a direct query."""
    code = 190  # fight (high-volume, should be present in most active countries)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT COUNT(*) FROM gdelt_events
                WHERE action_geo_country = 'SY' AND event_base_code = {code}
                  AND event_date > %s - INTERVAL '37 days'
                  AND event_date <= %s - INTERVAL '7 days'""",
            (RUN_DATE, RUN_DATE),
        )
        expected_count = cur.fetchone()[0]

    expected_log = np.log1p(expected_count)
    actual_log = syria_row[f"b1_{code}_log_count"]
    assert abs(actual_log - expected_log) < 1e-9


def test_no_negative_shares_or_entropy(panel):
    """Shares must be in [0, 1]; entropy must be >= 0 where defined."""
    share_cols = [c for c in panel.columns if c.endswith("_share")]
    for col in share_cols:
        vals = panel[col].dropna()
        if len(vals):
            assert (vals >= -1e-9).all() and (vals <= 1 + 1e-9).all(), col

    entropy = panel["b5_adm1_entropy"].dropna()
    if len(entropy):
        assert (entropy >= -1e-9).all()
