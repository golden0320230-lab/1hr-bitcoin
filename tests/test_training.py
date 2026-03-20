"""Tests for historical training dataset generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.schemas import BTCCandle
from app.services.storage import DuckDBStorage
from app.services.training import ModelTrainer, TrainingDatasetBuilder


def _make_candles(closes: list[float]) -> list[BTCCandle]:
    start = datetime(2026, 3, 18, 12, 0, tzinfo=UTC)
    candles: list[BTCCandle] = []
    for index, close in enumerate(closes):
        timestamp = start + timedelta(minutes=index * 5)
        candles.append(
            BTCCandle(
                source="coinbase",
                timeframe="5m",
                timestamp=timestamp,
                open=close - 0.5,
                high=close + 0.5,
                low=close - 1.0,
                close=close,
                volume=10 + index,
            )
        )
    return candles


def test_training_dataset_uses_expiry_threshold_label_semantics(tmp_path: Path) -> None:
    closes = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        118,
        117,
        116,
        115,
        114,
        113,
        112,
        111,
        110,
        109,
        108,
        107,
        106,
        105,
        104,
        103,
        102,
        101,
        100,
        99,
    ]
    builder = TrainingDatasetBuilder(
        dataset_path=tmp_path / "training_dataset.csv",
        strike_increment=10,
        history_minutes=60,
    )

    dataset = builder.build_dataset(_make_candles(closes), horizon_minutes=60)

    assert not dataset.empty
    first_row = dataset.iloc[0]
    assert first_row["threshold"] == 110
    assert first_row["label"] == 1

    bearish_rows = dataset.loc[dataset["threshold"] == 120]
    assert not bearish_rows.empty
    assert bool((bearish_rows["settlement_price"] < 120).all())
    assert bool((bearish_rows["label"] == 0).all())

    saved = builder.save_dataset(dataset)
    reloaded = builder.load_dataset(saved.path)

    assert saved.path.exists()
    assert saved.row_count == len(dataset)
    assert list(reloaded.columns) == list(dataset.columns)


def test_training_dataset_includes_feature_columns_and_schema_version(tmp_path: Path) -> None:
    closes = [100 + (index * 0.5) for index in range(30)]
    builder = TrainingDatasetBuilder(
        dataset_path=tmp_path / "dataset.csv",
        strike_increment=5,
        history_minutes=60,
    )

    dataset = builder.build_dataset(_make_candles(closes), horizon_minutes=60)

    assert "feature_schema_version" in dataset.columns
    assert "market_implied_probability" in dataset.columns
    assert "minutes_to_expiry" in dataset.columns
    assert set(dataset["feature_schema_version"]) == {"1.0.0"}


def test_model_trainer_saves_artifact_and_metadata(tmp_path: Path) -> None:
    storage = DuckDBStorage(tmp_path / "kalshi_btc.duckdb")
    builder = TrainingDatasetBuilder(
        dataset_path=tmp_path / "dataset.csv",
        strike_increment=10,
        history_minutes=60,
    )
    dataset = builder.build_dataset(
        _make_candles(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                118,
                117,
                116,
                115,
                114,
                113,
                112,
                111,
                110,
                109,
                108,
                107,
                106,
                105,
                104,
                103,
                102,
                101,
                100,
                99,
            ]
        ),
        horizon_minutes=60,
    )

    try:
        result = ModelTrainer(storage).train(dataset, artifact_path=tmp_path / "baseline.pkl")

        assert result.artifact_path.exists()
        assert result.model_name in {"logistic_regression", "gradient_boosting"}
        assert storage.count_rows("model_metadata") == 1
    finally:
        storage.close()


def test_model_artifact_can_be_loaded_for_inference(tmp_path: Path) -> None:
    builder = TrainingDatasetBuilder(
        dataset_path=tmp_path / "dataset.csv",
        strike_increment=10,
        history_minutes=60,
    )
    dataset = builder.build_dataset(
        _make_candles(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                118,
                117,
                116,
                115,
                114,
                113,
                112,
                111,
                110,
                109,
                108,
                107,
                106,
                105,
                104,
                103,
                102,
                101,
                100,
                99,
            ]
        ),
        horizon_minutes=60,
    )
    trainer = ModelTrainer()
    result = trainer.train(dataset, artifact_path=tmp_path / "baseline.pkl")

    artifact = trainer.load_artifact(result.artifact_path)
    probabilities = trainer.predict_dataset_probabilities(artifact, dataset.head(2))

    assert len(probabilities) == 2
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
