"""Tests for historical backtesting and baseline comparisons."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import sin
from pathlib import Path

from app.schemas import BTCCandle
from app.services.backtest import BacktestService
from app.services.storage import DuckDBStorage
from app.services.training import TrainingDatasetBuilder


def _make_backtest_candles() -> list[BTCCandle]:
    start = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    candles: list[BTCCandle] = []
    for index in range(180):
        close = 84_500 + int(220 * sin(index / 4)) + ((index % 8) - 4) * 12
        candles.append(
            BTCCandle(
                source="coinbase",
                timeframe="5m",
                timestamp=start + timedelta(minutes=index * 5),
                open=close - 15,
                high=close + 20,
                low=close - 20,
                close=close,
                volume=25 + index,
            )
        )
    return candles


def test_backtest_service_returns_model_and_required_baselines(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    builder = TrainingDatasetBuilder(
        dataset_path=tmp_path / "dataset.csv",
        strike_increment=100,
        history_minutes=60,
    )
    dataset = builder.build_dataset(_make_backtest_candles(), horizon_minutes=60)

    try:
        result = BacktestService(storage).run(dataset)

        assert result.model_name in {"logistic_regression", "gradient_boosting"}
        assert result.num_samples > 1
        assert len(result.baselines) == 4
        assert {metric.name for metric in result.baselines} == {
            "kalshi_follow",
            "spot_vs_strike",
            "momentum",
            "no_news",
        }
        assert storage.count_rows("backtest_results") == 1
    finally:
        storage.close()
