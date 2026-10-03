"""Data access for the dashboard: live backend when reachable, else the files in assets/.

On GitHub Pages (stlite / Pyodide) there is no backend, so the asset path always runs.
Both paths return the same shapes; the asset path reproduces the backend's per-window
results from the rollups written by scripts/export_static_assets.py.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
ASSETS_DIR = Path(os.getenv("GEOPULSE_ASSETS", Path(__file__).resolve().parent.parent / "assets"))
IN_BROWSER = sys.platform == "emscripten"

# Large files are fetched on first use in the browser instead of being mounted at startup.
_REMOTE = {"gi_daily.csv", "gi_events.csv", "gi_counterparts.csv"}

_backend_up: bool | None = None


def backend_available() -> bool:
    global _backend_up
    if _backend_up is None:
        if IN_BROWSER:
            _backend_up = False
        else:
            import requests
            try:
                _backend_up = requests.get(f"{BACKEND_URL}/health", timeout=2).ok
            except requests.RequestException:
                _backend_up = False
    return _backend_up


def _get(path: str, timeout: int = 60, **params):
    import requests
    resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@lru_cache(maxsize=None)
def _csv(name: str) -> pd.DataFrame:
    if IN_BROWSER and name in _REMOTE:
        from pyodide.http import open_url
        from static_config import BASE_URL  # written into the page by scripts/build_static_site.py
        return pd.read_csv(open_url(f"{BASE_URL}assets/{name}"))
    return pd.read_csv(ASSETS_DIR / name)


def meta() -> dict:
    """Static builds only: dates the assets were exported for."""
    path = ASSETS_DIR / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _latest() -> date:
    return date.fromisoformat(meta()["gi_latest_event_date"])


# ---------------------------------------------------------------------------
# Home / Drilldown
# ---------------------------------------------------------------------------

def heatmap() -> pd.DataFrame:
    if backend_available():
        return pd.DataFrame(_get("/global/heatmap", timeout=10)["countries"])
    return _csv("heatmap.csv")


def timeline(country: str, days: int = 90) -> pd.DataFrame:
    if backend_available():
        return pd.DataFrame(_get(f"/country/{country}/timeline", timeout=10, days=days)["timeline"])
    df = _csv("timelines.csv")
    df = df[df["country"] == country].drop(columns="country")
    if df.empty:
        return df
    # Same anchoring as the backend: window ends at the country's latest data, not today.
    dates = pd.to_datetime(df["feature_date"])
    return df[dates >= dates.max() - pd.Timedelta(days=days)].reset_index(drop=True)


def spillover(country: str, top_n: int = 6) -> list[dict]:
    if backend_available():
        return _get(f"/country/{country}/spillover", timeout=10, top_n=top_n)["neighbors"]
    df = _csv("spillover.csv")
    df = df[df["country"] == country].head(top_n).drop(columns="country").astype(object)
    return df.where(df.notna(), None).to_dict("records")


# ---------------------------------------------------------------------------
# Global Intelligence
# ---------------------------------------------------------------------------

def gi_windows() -> list[int]:
    """Preset windows (days back from the latest event) offered when there is no backend."""
    return meta().get("windows", [30, 90, 180, 365])


def gi_default_since() -> date:
    if backend_available():
        return date.fromisoformat(_get("/intelligence/meta")["default_since"])
    return date.fromisoformat(meta()["gi_default_since"])


def gi_since_for_window(days: int) -> date:
    return _latest() - timedelta(days=days)


def gi_countries() -> list[str]:
    if backend_available():
        return _get("/intelligence/countries")
    return _csv("gi_countries.csv")["iso3"].tolist()


def gi_heatmap(since: str) -> pd.DataFrame:
    if backend_available():
        return pd.DataFrame(_get("/intelligence/heatmap", since=since))
    d = _csv("gi_daily.csv")
    d = d[d["event_date"] >= since]
    g = d.groupby("country_iso3").sum(numeric_only=True)
    return pd.DataFrame({
        "iso3": g.index,
        "total_events": g["total_events"].to_numpy(),
        "conflict_events": g["conflict"].to_numpy(),
        "conflict_share": (g["conflict"] / g["total_events"]).to_numpy(),
        "avg_intensity": (g["intensity_sum"] / g["intensity_n"].where(g["intensity_n"] > 0)).to_numpy(),
        "total_mentions": g["mentions"].to_numpy(),
    })


def gi_highlighted(since: str, iso3: str | None) -> list[dict]:
    if backend_available():
        params = {"since": since, **({"iso3": iso3} if iso3 else {})}
        return _get("/intelligence/events/highlighted", **params)
    # Static feeds are exported per preset window; use the closest one.
    days = (_latest() - date.fromisoformat(since)).days
    window = min(gi_windows(), key=lambda w: abs(w - days))
    ev = _csv("gi_events.csv")
    ev = ev[(ev["window"] == window) & (ev["scope"].fillna("") == (iso3 or ""))]
    out = []
    for r in ev.to_dict("records"):
        z = r["country_day_z_score"]
        out.append({
            **r,
            "country_day_z_score": None if pd.isna(z) else z,
            "num_mentions": None if pd.isna(r["num_mentions"]) else int(r["num_mentions"]),
            "intensity": None if pd.isna(r["intensity"]) else r["intensity"],
            "tags": r["tags"].split("|"),
        })
    return out


def gi_country_summary(iso3: str, since: str, include_by_type: bool = False) -> dict | None:
    if backend_available():
        import requests
        resp = requests.get(
            f"{BACKEND_URL}/intelligence/country/{iso3}/summary",
            params={"since": since, "include_by_type": include_by_type},
            timeout=150 if include_by_type else 60,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    d = _csv("gi_daily.csv")
    d = d[(d["country_iso3"] == iso3) & (d["event_date"] >= since)]
    mix = {"cooperation": int(d["cooperation"].sum()), "consultation": int(d["consultation"].sum()),
           "conflict": int(d["conflict"].sum())}
    total = sum(mix.values())
    if total == 0:
        return None
    months = pd.to_datetime(d["event_date"]).dt.to_period("M").dt.to_timestamp()
    monthly = d.groupby(months)["total_events"].sum()
    cp = _csv("gi_counterparts.csv")
    cp = cp[cp["iso3"] == iso3].sort_values("rank")
    return {
        "iso3": iso3,
        "total_events": total,
        "interaction_mix": mix,
        "monthly_volume": [[m.date().isoformat(), int(n)] for m, n in monthly.items()],
        "top_counterparts": [[r.counterpart, int(r.n)] for r in cp.itertuples()],
        "top_counterparts_by_type": {},
    }
