"""
Global intelligence API -- country
summaries, highlighted events, activity heatmap, all backed by graph.*
(Postgres). Independent of the parked PIT/risk-scoring routes in
backend/main.py -- different data, different tables, mounted separately.

Routes:
    GET /intelligence/countries                 Countries with any activity
    GET /intelligence/country/{iso3}/summary    Country-level summary
    GET /intelligence/events/highlighted        Highlighted events feed
    GET /intelligence/heatmap                   Country activity choropleth data
    GET /intelligence/heatmap/timeseries        Weekly per-country activity series
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from preprocessing.activity_heatmap import country_activity, country_activity_timeseries
from preprocessing.country_summary import (
    build_country_summary,
    latest_resolvable_event_date,
    list_countries_with_activity,
)
from preprocessing.highlighted_events import get_highlighted_events

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

DEFAULT_LOOKBACK_DAYS = 90


def _load_dsn() -> str:
    cfg_path = "configs/config.yaml"
    db = {}
    try:
        with open(cfg_path) as f:
            db = yaml.safe_load(f).get("database", {})
    except FileNotFoundError:
        pass
    return os.getenv(
        "DATABASE_SYNC_URL",
        f"postgresql://{db.get('user','gldt')}:{db.get('password','gldt_secret')}@"
        f"{db.get('host','localhost')}:{db.get('port',5432)}/{db.get('name','gdelt_risk')}",
    )


def _get_conn():
    return psycopg2.connect(_load_dsn())


def _default_since(conn) -> date:
    """Anchored to the data's own latest resolvable date, never wall-clock
    date.today() -- see latest_resolvable_event_date()'s docstring for the
    "No data available" bug this fixes."""
    latest = latest_resolvable_event_date(conn)
    return (latest - timedelta(days=DEFAULT_LOOKBACK_DAYS)) if latest else date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CountrySummaryResponse(BaseModel):
    iso3: str
    total_events: int
    interaction_mix: dict[str, int]
    monthly_volume: list[tuple[date, int]]
    top_counterparts: list[tuple[str, int]]
    top_counterparts_by_type: dict[str, list[tuple[str, int]]]


class HighlightedEventResponse(BaseModel):
    event_id: int
    event_date: date
    country_iso3: Optional[str]
    interaction_type: Optional[str]
    num_mentions: Optional[int]
    intensity: Optional[float]
    source_url: Optional[str]
    country_day_z_score: Optional[float]
    tags: list[str]


class CountryActivityResponse(BaseModel):
    iso3: str
    total_events: int
    conflict_events: int
    conflict_share: Optional[float]
    avg_intensity: Optional[float]
    total_mentions: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/meta")
def meta():
    """Data-range metadata for building sensible UI defaults -- in
    particular, the latest date with resolvable (located) events, so a
    caller can build a "last N days" window anchored to the data instead
    of wall-clock date.today() (see _default_since's docstring)."""
    conn = _get_conn()
    try:
        latest = latest_resolvable_event_date(conn)
        default_since = (latest - timedelta(days=DEFAULT_LOOKBACK_DAYS)) if latest else None
        return {
            "latest_resolvable_event_date": latest,
            "default_since": default_since,
        }
    finally:
        conn.close()


@router.get("/countries", response_model=list[str])
def countries():
    """All ISO3 codes with at least one located event -- the valid input
    set for /country/{iso3}/summary."""
    conn = _get_conn()
    try:
        return list_countries_with_activity(conn)
    finally:
        conn.close()


@router.get("/country/{iso3}/summary", response_model=CountrySummaryResponse)
def country_summary_endpoint(
    iso3: str,
    since: Optional[date] = Query(None, description=f"Defaults to {DEFAULT_LOOKBACK_DAYS} days ago"),
    n_counterparts: int = Query(10, ge=1, le=50),
    include_by_type: bool = Query(
        False,
        description="Counterparts broken down by interaction type. Slow (~50s extra, "
                     "3 expensive queries) and frequently empty -- opt-in only.",
    ),
):
    iso3 = iso3.upper()
    conn = _get_conn()
    try:
        s = build_country_summary(conn, iso3, since=since or _default_since(conn),
                                   n_counterparts=n_counterparts, include_by_type=include_by_type)
        if s.total_events == 0:
            raise HTTPException(status_code=404, detail=f"No events found for country {iso3}")
        return CountrySummaryResponse(
            iso3=s.iso3, total_events=s.total_events, interaction_mix=s.interaction_mix,
            monthly_volume=s.monthly_volume, top_counterparts=s.top_counterparts,
            top_counterparts_by_type=s.top_counterparts_by_type,
        )
    finally:
        conn.close()


@router.get("/events/highlighted", response_model=list[HighlightedEventResponse])
def highlighted_events_endpoint(
    iso3: Optional[str] = Query(None, description="Restrict to one country (ISO3)"),
    since: Optional[date] = Query(None, description=f"Defaults to {DEFAULT_LOOKBACK_DAYS} days ago"),
    limit_per_dimension: int = Query(20, ge=1, le=100),
):
    conn = _get_conn()
    try:
        events = get_highlighted_events(
            conn, since=since or _default_since(conn),
            iso3=iso3.upper() if iso3 else None,
            limit_per_dimension=limit_per_dimension,
        )
        return [
            HighlightedEventResponse(
                event_id=e.event_id, event_date=e.event_date, country_iso3=e.country_iso3,
                interaction_type=e.interaction_type, num_mentions=e.num_mentions,
                intensity=e.intensity, source_url=e.source_url,
                country_day_z_score=e.country_day_z_score, tags=e.tags,
            )
            for e in events
        ]
    finally:
        conn.close()


@router.get("/heatmap", response_model=list[CountryActivityResponse])
def heatmap_endpoint(
    since: Optional[date] = Query(None, description=f"Defaults to {DEFAULT_LOOKBACK_DAYS} days ago"),
    until: Optional[date] = None,
):
    conn = _get_conn()
    try:
        rows = country_activity(conn, since=since or _default_since(conn), until=until)
        return [
            CountryActivityResponse(
                iso3=r.iso3, total_events=r.total_events, conflict_events=r.conflict_events,
                conflict_share=r.conflict_share, avg_intensity=r.avg_intensity,
                total_mentions=r.total_mentions,
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/heatmap/timeseries")
def heatmap_timeseries_endpoint(
    since: Optional[date] = Query(None, description=f"Defaults to {DEFAULT_LOOKBACK_DAYS} days ago"),
):
    conn = _get_conn()
    try:
        rows = country_activity_timeseries(conn, since=since or _default_since(conn))
        return [{"iso3": iso3, "week": week, "n": n} for iso3, week, n in rows]
    finally:
        conn.close()
