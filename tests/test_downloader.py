"""
Tests for ingestion/gdelt_downloader.py.

Covers the bug found during the Phase B backfill (2026-09-12): a failed
download (retries exhausted) used to make stream_csv_rows return an empty
generator, which ingestion_pipeline.ingest_date could not distinguish from a
genuinely quiet day -- it logged status="success" with 0 events. 12 of 1,096
dates in the 2023-2025 backfill were silently wrong this way before it was
caught by a direct COUNT(DISTINCT event_date) sanity check against the
requested range, not by anything the pipeline itself reported.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ingestion.gdelt_downloader import GDELTDownloader, GDELTDownloadError


def test_stream_csv_rows_raises_on_download_failure():
    downloader = GDELTDownloader()
    with patch.object(downloader, "_download_with_retry", return_value=None):
        with pytest.raises(GDELTDownloadError):
            list(downloader.stream_csv_rows(__import__("datetime").date(2024, 1, 1)))


def test_stream_csv_rows_yields_rows_on_success():
    downloader = GDELTDownloader()
    # A minimal valid GDELT v1 zip: one tab-separated row, 58 columns.
    import io
    import zipfile

    row = "\t".join(["1"] * 58) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("20240101.export.CSV", row)
    raw_bytes = buf.getvalue()

    with patch.object(downloader, "_download_with_retry", return_value=raw_bytes):
        chunks = list(downloader.stream_csv_rows(__import__("datetime").date(2024, 1, 1)))
    assert len(chunks) == 1
    assert len(chunks[0]) == 1
