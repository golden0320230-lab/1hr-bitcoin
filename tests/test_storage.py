"""Tests for the DuckDB storage layer."""

from __future__ import annotations

from pathlib import Path

from app.schemas import BTCCandle, KalshiMarket, NewsArticle
from app.services.storage import DuckDBStorage


def test_storage_initializes_all_expected_tables(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")

    try:
        tables = {
            row[0]
            for row in storage.connection.execute("SHOW TABLES").fetchall()
        }
    finally:
        storage.close()

    assert tables >= {
        "backtest_results",
        "btc_candles",
        "market_snapshots",
        "markets",
        "model_metadata",
        "news_articles",
        "news_scores",
        "prediction_runs",
    }


def test_insert_articles_deduplicates_by_url(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    article = NewsArticle(
        title="Bitcoin rallies after ETF flow update",
        url="https://example.com/bitcoin-rally",
        source="Example News",
        published_at="2026-03-19T20:00:00Z",
        summary="BTC climbed after a fresh ETF headline.",
    )

    try:
        inserted_first = storage.insert_articles([article])
        inserted_second = storage.insert_articles([article])

        assert inserted_first == 1
        assert inserted_second == 0
        assert storage.count_rows("news_articles") == 1
        assert len(storage.list_articles()) == 1
    finally:
        storage.close()


def test_insert_candles_deduplicates_by_timestamp_timeframe_and_source(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    candle = BTCCandle(
        source="coinbase",
        timeframe="1m",
        timestamp="2026-03-19T20:00:00Z",
        open=72_000,
        high=72_050,
        low=71_990,
        close=72_010,
        volume=12.5,
    )

    try:
        inserted_first = storage.insert_candles([candle])
        inserted_second = storage.insert_candles([candle])

        assert inserted_first == 1
        assert inserted_second == 0
        assert storage.count_rows("btc_candles") == 1
        assert len(storage.list_candles(source="coinbase", timeframe="1m")) == 1
    finally:
        storage.close()


def test_upsert_market_replaces_existing_market_row(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    first_market = KalshiMarket(
        ticker="BTC-1HR-72500",
        title="Will Bitcoin close above $72,500 at 4pm ET?",
        direction="ABOVE",
        threshold=72_500,
        expires_at="2026-03-19T20:00:00Z",
    )
    updated_market = KalshiMarket(
        ticker="BTC-1HR-72500",
        title="Will Bitcoin close above $72,750 at 4pm ET?",
        direction="ABOVE",
        threshold=72_750,
        expires_at="2026-03-19T20:00:00Z",
    )

    try:
        storage.upsert_market(first_market)
        storage.upsert_market(updated_market)
        market = storage.get_market("BTC-1HR-72500")

        assert market is not None
        assert market.threshold == 72_750
        assert storage.count_rows("markets") == 1
    finally:
        storage.close()
