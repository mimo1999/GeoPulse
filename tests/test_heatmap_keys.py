import pytest

psycopg2 = pytest.importorskip("psycopg2")
from fastapi.testclient import TestClient

DSN = "dbname=gdelt_risk user=gldt password=gldt_secret host=localhost port=5432"


def test_global_heatmap_lists_only_fips_country_keys():
    try:
        psycopg2.connect(DSN, connect_timeout=3).close()
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc})")
    from backend.main import app

    with TestClient(app) as client:
        countries = client.get("/global/heatmap").json()["countries"]
    assert countries
    assert all(len(c["country"]) == 2 for c in countries)
