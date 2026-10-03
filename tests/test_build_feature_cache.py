import numpy as np
import pandas as pd

from scripts import build_feature_cache as b
from scripts.seed_db_from_cache import trailing_mean


def _events(country, n, quad):
    return pd.DataFrame({
        "action_geo_country_code": [country] * n, "actor1_country_code": ["XXX"] * n,
        "num_mentions": [1] * n, "quad_class": [quad] * n, "goldstein_scale": [-5.0] * n,
        "avg_tone": [-3.0] * n, "event_base_code": ["190"] * n, "event_root_code": ["19"] * n,
    })


def test_events_without_geo_country_are_dropped_not_keyed_by_actor():
    df = _events("UP", 10, 4)
    df.loc[:4, "action_geo_country_code"] = None
    features, volumes = b.aggregate_day(df, min_events=1)
    assert set(features) == {"UP"} and volumes["violence"]["UP"] == 5


def test_volume_lifts_a_large_country_above_a_small_one_with_the_same_share():
    big, small = _events("RS", 400, 4), _events("EK", 8, 4)
    features, volumes = b.aggregate_day(pd.concat([big, small]), min_events=1)
    ranked = b.percentile_normalize_day({**features, **{f"C{i}": features["RS"] * 0.5 for i in range(5)}},
                                        volumes)
    assert ranked["RS"][1] > ranked["EK"][1]


def test_trailing_mean_uses_only_the_window_and_present_snapshots():
    assert trailing_mean({0: 0.9, 1: 0.3, 3: 0.6}, 3) == np.mean([0.3, 0.6])
    assert trailing_mean({5: 0.4}, 5) == 0.4
