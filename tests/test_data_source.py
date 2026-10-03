import pandas as pd

from streamlit_app import data_source


def _no_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(data_source, "_backend_up", False)
    monkeypatch.setattr(data_source, "ASSETS_DIR", tmp_path)
    data_source._csv.cache_clear()


def test_heatmap_falls_back_to_csv(monkeypatch, tmp_path):
    _no_backend(monkeypatch, tmp_path)
    pd.DataFrame({"country": ["UP"], "risk_score": [0.5]}).to_csv(tmp_path / "heatmap.csv", index=False)
    assert data_source.heatmap()["country"].tolist() == ["UP"]


def test_timeline_csv_is_filtered_and_anchored_to_latest_date(monkeypatch, tmp_path):
    _no_backend(monkeypatch, tmp_path)
    dates = pd.date_range("2026-01-01", periods=100).strftime("%Y-%m-%d")
    rows = [{"country": c, "feature_date": d, "risk_score": 0.1} for c in ("UP", "WE") for d in dates]
    pd.DataFrame(rows).to_csv(tmp_path / "timelines.csv", index=False)

    out = data_source.timeline("UP", days=30)
    assert "country" not in out and out["feature_date"].max() == dates[-1]
    assert len(out) == 31  # latest date minus 30 days, inclusive


def test_backend_probe_failure_means_csv(monkeypatch):
    monkeypatch.setattr(data_source, "_backend_up", None)
    monkeypatch.setattr(data_source, "BACKEND_URL", "http://127.0.0.1:9")
    assert data_source.backend_available() is False


# ---------------------------------------------------------------------------
# Global Intelligence (static path)
# ---------------------------------------------------------------------------

def _gi_assets(monkeypatch, tmp_path):
    _no_backend(monkeypatch, tmp_path)
    (tmp_path / "meta.json").write_text(
        '{"gi_latest_event_date": "2026-10-02", "gi_default_since": "2026-07-04", "windows": [30, 90]}'
    )
    cols = ["country_iso3", "event_date", "total_events", "cooperation", "consultation", "conflict",
            "intensity_n", "intensity_sum", "mentions"]
    pd.DataFrame([
        ["UKR", "2026-08-15", 10, 5, 2, 3, 10, -20.0, 100],
        ["UKR", "2026-09-20", 20, 10, 4, 6, 20, -10.0, 300],
        ["FRA", "2026-09-20", 5, 5, 0, 0, 0, 0.0, 50],
    ], columns=cols).to_csv(tmp_path / "gi_daily.csv", index=False)
    pd.DataFrame({"iso3": ["UKR", "UKR"], "counterpart": ["RUS", "USA"], "n": [9, 4], "rank": [1, 2]}
                 ).to_csv(tmp_path / "gi_counterparts.csv", index=False)
    pd.DataFrame([
        {"window": 30, "scope": "", "event_id": 1, "event_date": "2026-09-20", "country_iso3": "UKR",
         "interaction_type": "conflict", "num_mentions": 7, "intensity": -9.0, "source_url": "u",
         "country_day_z_score": 4.2, "tags": "Where media is looking|What no one saw coming"},
        {"window": 90, "scope": "", "event_id": 2, "event_date": "2026-08-15", "country_iso3": "UKR",
         "interaction_type": "conflict", "num_mentions": 3, "intensity": -5.0, "source_url": "v",
         "country_day_z_score": None, "tags": "Where people are looking"},
    ]).to_csv(tmp_path / "gi_events.csv", index=False)


def test_gi_heatmap_aggregates_the_window(monkeypatch, tmp_path):
    _gi_assets(monkeypatch, tmp_path)
    out = data_source.gi_heatmap("2026-09-01").set_index("iso3")
    assert out.loc["UKR", "total_events"] == 20 and out.loc["UKR", "conflict_share"] == 0.3
    assert out.loc["UKR", "avg_intensity"] == -0.5
    assert pd.isna(out.loc["FRA", "avg_intensity"])  # no intensity readings, not zero


def test_gi_country_summary_matches_the_backend_shape(monkeypatch, tmp_path):
    _gi_assets(monkeypatch, tmp_path)
    s = data_source.gi_country_summary("UKR", "2026-08-01")
    assert s["total_events"] == 30 and s["interaction_mix"] == {"cooperation": 15, "consultation": 6, "conflict": 9}
    assert s["monthly_volume"] == [["2026-08-01", 10], ["2026-09-01", 20]]
    assert s["top_counterparts"] == [["RUS", 9], ["USA", 4]]
    assert data_source.gi_country_summary("UKR", "2026-10-01") is None


def test_gi_highlighted_uses_closest_window_and_splits_tags(monkeypatch, tmp_path):
    _gi_assets(monkeypatch, tmp_path)
    ev = data_source.gi_highlighted("2026-09-02", None)  # 30 days before the latest event
    assert [e["event_id"] for e in ev] == [1]
    assert ev[0]["tags"] == ["Where media is looking", "What no one saw coming"]
    assert data_source.gi_highlighted("2026-09-02", "FRA") == []


def test_spillover_missing_correlation_is_none(monkeypatch, tmp_path):
    _no_backend(monkeypatch, tmp_path)
    pd.DataFrame({"country": ["CA"], "neighbor": ["IND"], "spillover_weight": [0.3], "risk_correlation": [None]}
                 ).to_csv(tmp_path / "spillover.csv", index=False)
    assert data_source.spillover("CA")[0]["risk_correlation"] is None
