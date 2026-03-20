"""DuckDB-backed local storage helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import duckdb
import orjson

from app.schemas import (
    ArticleSentimentScore,
    BacktestResult,
    BTCCandle,
    CandleTimeframe,
    KalshiMarket,
    MarketSnapshot,
    NewsArticle,
    PredictionResult,
)


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None

    return orjson.dumps(value).decode("utf-8")


def _json_loads(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None

    return cast(dict[str, Any], orjson.loads(value))


class DuckDBStorage:
    """Persistent local storage for normalized project data."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.db_path))
        self.initialize()

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS markets (
                ticker VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                event_ticker VARCHAR,
                event_title VARCHAR,
                direction VARCHAR NOT NULL,
                threshold DOUBLE NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                status VARCHAR NOT NULL,
                rules VARCHAR,
                market_url VARCHAR,
                raw_payload VARCHAR
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                ticker VARCHAR NOT NULL,
                captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
                yes_price DOUBLE,
                no_price DOUBLE,
                yes_bid DOUBLE,
                yes_ask DOUBLE,
                no_bid DOUBLE,
                no_ask DOUBLE,
                volume DOUBLE,
                open_interest BIGINT,
                raw_payload VARCHAR,
                UNIQUE (ticker, captured_at)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS btc_candles (
                source VARCHAR NOT NULL,
                product_id VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE,
                raw_payload VARCHAR,
                UNIQUE (source, timeframe, timestamp)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                url VARCHAR PRIMARY KEY,
                title VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                published_at TIMESTAMP WITH TIME ZONE NOT NULL,
                summary VARCHAR,
                content_fingerprint VARCHAR,
                raw_payload VARCHAR
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS news_scores (
                article_url VARCHAR NOT NULL,
                model_name VARCHAR NOT NULL,
                scored_at TIMESTAMP WITH TIME ZONE NOT NULL,
                market_call VARCHAR NOT NULL DEFAULT 'NEUTRAL',
                sentiment VARCHAR NOT NULL,
                relevance DOUBLE NOT NULL,
                impact_horizon_minutes INTEGER NOT NULL,
                impact_score DOUBLE NOT NULL,
                confidence DOUBLE NOT NULL,
                reason VARCHAR NOT NULL,
                raw_response VARCHAR,
                UNIQUE (article_url, model_name, scored_at)
            )
            """
        )
        self.connection.execute(
            """
            ALTER TABLE news_scores
            ADD COLUMN IF NOT EXISTS market_call VARCHAR DEFAULT 'NEUTRAL'
            """
        )
        self.connection.execute(
            """
            UPDATE news_scores
            SET market_call = 'NEUTRAL'
            WHERE market_call IS NULL
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                run_id VARCHAR PRIMARY KEY,
                generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                market_ticker VARCHAR NOT NULL,
                label VARCHAR NOT NULL,
                probability DOUBLE NOT NULL,
                confidence VARCHAR NOT NULL,
                degraded BOOLEAN NOT NULL,
                disclaimer VARCHAR NOT NULL,
                drivers_json VARCHAR NOT NULL,
                warnings_json VARCHAR NOT NULL,
                feature_vector_json VARCHAR,
                market_json VARCHAR,
                market_snapshot_json VARCHAR,
                prediction_json VARCHAR NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metadata (
                metadata_id VARCHAR PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                feature_schema_version VARCHAR NOT NULL,
                artifact_path VARCHAR,
                training_window_start TIMESTAMP WITH TIME ZONE,
                training_window_end TIMESTAMP WITH TIME ZONE,
                metrics_json VARCHAR,
                raw_json VARCHAR NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                result_id VARCHAR PRIMARY KEY,
                generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                window_start TIMESTAMP WITH TIME ZONE NOT NULL,
                window_end TIMESTAMP WITH TIME ZONE NOT NULL,
                model_name VARCHAR NOT NULL,
                num_samples INTEGER NOT NULL,
                accuracy DOUBLE NOT NULL,
                log_loss DOUBLE NOT NULL,
                brier_score DOUBLE,
                baselines_json VARCHAR NOT NULL,
                notes_json VARCHAR NOT NULL,
                result_json VARCHAR NOT NULL
            )
            """
        )

    def count_rows(self, table_name: str) -> int:
        row = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return 0 if row is None else int(row[0])

    def upsert_market(self, market: KalshiMarket) -> None:
        self.connection.execute("DELETE FROM markets WHERE ticker = ?", [market.ticker])
        self.connection.execute(
            """
            INSERT INTO markets (
                ticker,
                title,
                event_ticker,
                event_title,
                direction,
                threshold,
                expires_at,
                status,
                rules,
                market_url,
                raw_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                market.ticker,
                market.title,
                market.event_ticker,
                market.event_title,
                market.direction,
                market.threshold,
                market.expires_at,
                market.status,
                market.rules,
                str(market.market_url) if market.market_url is not None else None,
                _json_dumps(market.raw_payload),
            ],
        )

    def get_market(self, ticker: str) -> KalshiMarket | None:
        row = self.connection.execute(
            """
            SELECT
                ticker,
                title,
                event_ticker,
                event_title,
                direction,
                threshold,
                expires_at,
                status,
                rules,
                market_url,
                raw_payload
            FROM markets
            WHERE ticker = ?
            """,
            [ticker],
        ).fetchone()
        if row is None:
            return None

        return KalshiMarket(
            ticker=row[0],
            title=row[1],
            event_ticker=row[2],
            event_title=row[3],
            direction=row[4],
            threshold=row[5],
            expires_at=row[6],
            status=row[7],
            rules=row[8],
            market_url=row[9],
            raw_payload=_json_loads(row[10]),
        )

    def insert_market_snapshot(self, snapshot: MarketSnapshot) -> int:
        before_count = self.count_rows("market_snapshots")
        self.connection.execute(
            """
            INSERT INTO market_snapshots (
                ticker,
                captured_at,
                yes_price,
                no_price,
                yes_bid,
                yes_ask,
                no_bid,
                no_ask,
                volume,
                open_interest,
                raw_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [
                snapshot.ticker,
                snapshot.captured_at,
                snapshot.yes_price,
                snapshot.no_price,
                snapshot.yes_bid,
                snapshot.yes_ask,
                snapshot.no_bid,
                snapshot.no_ask,
                snapshot.volume,
                snapshot.open_interest,
                _json_dumps(snapshot.raw_payload),
            ],
        )
        return self.count_rows("market_snapshots") - before_count

    def list_market_snapshots(self, ticker: str, limit: int = 20) -> list[MarketSnapshot]:
        rows = self.connection.execute(
            """
            SELECT
                ticker,
                captured_at,
                yes_price,
                no_price,
                yes_bid,
                yes_ask,
                no_bid,
                no_ask,
                volume,
                open_interest,
                raw_payload
            FROM market_snapshots
            WHERE ticker = ?
            ORDER BY captured_at DESC
            LIMIT ?
            """,
            [ticker, limit],
        ).fetchall()
        return [
            MarketSnapshot(
                ticker=row[0],
                captured_at=row[1],
                yes_price=row[2],
                no_price=row[3],
                yes_bid=row[4],
                yes_ask=row[5],
                no_bid=row[6],
                no_ask=row[7],
                volume=row[8],
                open_interest=row[9],
                raw_payload=_json_loads(row[10]),
            )
            for row in rows
        ]

    def insert_candles(self, candles: Iterable[BTCCandle]) -> int:
        before_count = self.count_rows("btc_candles")
        rows = [
            [
                candle.source,
                candle.product_id,
                candle.timeframe,
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                _json_dumps(candle.raw_payload),
            ]
            for candle in candles
        ]
        if not rows:
            return 0

        self.connection.executemany(
            """
            INSERT INTO btc_candles (
                source,
                product_id,
                timeframe,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                raw_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            rows,
        )
        return self.count_rows("btc_candles") - before_count

    def list_candles(
        self,
        *,
        source: str | None = None,
        timeframe: CandleTimeframe | None = None,
        limit: int = 500,
    ) -> list[BTCCandle]:
        filters: list[str] = []
        params: list[Any] = []

        if source is not None:
            filters.append("source = ?")
            params.append(source)

        if timeframe is not None:
            filters.append("timeframe = ?")
            params.append(timeframe)

        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"

        query = f"""
            SELECT
                source,
                product_id,
                timeframe,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                raw_payload
            FROM btc_candles
            {where_clause}
            ORDER BY timestamp ASC
            LIMIT ?
        """
        params.append(limit)

        rows = self.connection.execute(query, params).fetchall()
        return [
            BTCCandle(
                source=row[0],
                product_id=row[1],
                timeframe=row[2],
                timestamp=row[3],
                open=row[4],
                high=row[5],
                low=row[6],
                close=row[7],
                volume=row[8],
                raw_payload=_json_loads(row[9]),
            )
            for row in rows
        ]

    def list_recent_candles(
        self,
        *,
        source: str | None = None,
        timeframe: CandleTimeframe | None = None,
        limit: int = 500,
    ) -> list[BTCCandle]:
        filters: list[str] = []
        params: list[Any] = []

        if source is not None:
            filters.append("source = ?")
            params.append(source)

        if timeframe is not None:
            filters.append("timeframe = ?")
            params.append(timeframe)

        where_clause = ""
        if filters:
            where_clause = f"WHERE {' AND '.join(filters)}"

        query = f"""
            SELECT
                source,
                product_id,
                timeframe,
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                raw_payload
            FROM btc_candles
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.connection.execute(query, params).fetchall()
        candles = [
            BTCCandle(
                source=row[0],
                product_id=row[1],
                timeframe=row[2],
                timestamp=row[3],
                open=row[4],
                high=row[5],
                low=row[6],
                close=row[7],
                volume=row[8],
                raw_payload=_json_loads(row[9]),
            )
            for row in rows
        ]
        return list(reversed(candles))

    def insert_articles(self, articles: Iterable[NewsArticle]) -> int:
        before_count = self.count_rows("news_articles")
        rows = [
            [
                str(article.url),
                article.title,
                article.source,
                article.published_at,
                article.summary,
                article.content_fingerprint,
                _json_dumps(article.raw_payload),
            ]
            for article in articles
        ]
        if not rows:
            return 0

        self.connection.executemany(
            """
            INSERT INTO news_articles (
                url,
                title,
                source,
                published_at,
                summary,
                content_fingerprint,
                raw_payload
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            rows,
        )
        return self.count_rows("news_articles") - before_count

    def list_articles(self, limit: int = 100) -> list[NewsArticle]:
        rows = self.connection.execute(
            """
            SELECT
                title,
                url,
                source,
                published_at,
                summary,
                content_fingerprint,
                raw_payload
            FROM news_articles
            ORDER BY published_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [
            NewsArticle(
                title=row[0],
                url=row[1],
                source=row[2],
                published_at=row[3],
                summary=row[4],
                content_fingerprint=row[5],
                raw_payload=_json_loads(row[6]),
            )
            for row in rows
        ]

    def insert_news_scores(self, scores: Iterable[ArticleSentimentScore]) -> int:
        before_count = self.count_rows("news_scores")
        rows = [
            [
                str(score.article_url),
                score.model_name,
                score.scored_at,
                score.market_call,
                score.sentiment,
                score.relevance,
                score.impact_horizon_minutes,
                score.impact_score,
                score.confidence,
                score.reason,
                _json_dumps(score.raw_response),
            ]
            for score in scores
        ]
        if not rows:
            return 0

        self.connection.executemany(
            """
            INSERT INTO news_scores (
                article_url,
                model_name,
                scored_at,
                market_call,
                sentiment,
                relevance,
                impact_horizon_minutes,
                impact_score,
                confidence,
                reason,
                raw_response
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            rows,
        )
        return self.count_rows("news_scores") - before_count

    def insert_prediction_run(self, prediction: PredictionResult) -> None:
        payload = prediction.model_dump(mode="json")
        self.connection.execute(
            "DELETE FROM prediction_runs WHERE run_id = ?",
            [str(prediction.run_id)],
        )
        self.connection.execute(
            """
            INSERT INTO prediction_runs (
                run_id,
                generated_at,
                market_ticker,
                label,
                probability,
                confidence,
                degraded,
                disclaimer,
                drivers_json,
                warnings_json,
                feature_vector_json,
                market_json,
                market_snapshot_json,
                prediction_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(prediction.run_id),
                prediction.generated_at,
                prediction.market_ticker,
                prediction.label,
                prediction.probability,
                prediction.confidence,
                prediction.degraded,
                prediction.disclaimer,
                _json_dumps(prediction.drivers) or "[]",
                _json_dumps(prediction.warnings) or "[]",
                _json_dumps(payload.get("feature_vector")),
                _json_dumps(payload.get("market")),
                _json_dumps(payload.get("market_snapshot")),
                _json_dumps(payload) or "{}",
            ],
        )

    def get_prediction_run(self, run_id: str) -> PredictionResult | None:
        row = self.connection.execute(
            """
            SELECT prediction_json
            FROM prediction_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            return None

        payload = orjson.loads(row[0])
        return PredictionResult.model_validate(payload)

    def get_latest_prediction_run(self) -> PredictionResult | None:
        row = self.connection.execute(
            """
            SELECT prediction_json
            FROM prediction_runs
            ORDER BY generated_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None

        payload = orjson.loads(row[0])
        return PredictionResult.model_validate(payload)

    def insert_model_metadata(
        self,
        *,
        metadata_id: str,
        model_name: str,
        feature_schema_version: str,
        created_at: Any,
        artifact_path: str | None = None,
        training_window_start: Any | None = None,
        training_window_end: Any | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "metadata_id": metadata_id,
            "model_name": model_name,
            "feature_schema_version": feature_schema_version,
            "created_at": created_at,
            "artifact_path": artifact_path,
            "training_window_start": training_window_start,
            "training_window_end": training_window_end,
            "metrics": metrics,
        }
        self.connection.execute("DELETE FROM model_metadata WHERE metadata_id = ?", [metadata_id])
        self.connection.execute(
            """
            INSERT INTO model_metadata (
                metadata_id,
                model_name,
                created_at,
                feature_schema_version,
                artifact_path,
                training_window_start,
                training_window_end,
                metrics_json,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                metadata_id,
                model_name,
                created_at,
                feature_schema_version,
                artifact_path,
                training_window_start,
                training_window_end,
                _json_dumps(metrics),
                _json_dumps(payload) or "{}",
            ],
        )

    def insert_backtest_result(self, result: BacktestResult) -> str:
        result_id = f"{result.model_name}:{result.generated_at.isoformat()}"
        payload = result.model_dump(mode="json")
        self.connection.execute("DELETE FROM backtest_results WHERE result_id = ?", [result_id])
        self.connection.execute(
            """
            INSERT INTO backtest_results (
                result_id,
                generated_at,
                window_start,
                window_end,
                model_name,
                num_samples,
                accuracy,
                log_loss,
                brier_score,
                baselines_json,
                notes_json,
                result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result_id,
                result.generated_at,
                result.window_start,
                result.window_end,
                result.model_name,
                result.num_samples,
                result.accuracy,
                result.log_loss,
                result.brier_score,
                _json_dumps(payload.get("baselines")) or "[]",
                _json_dumps(result.notes) or "[]",
                _json_dumps(payload) or "{}",
            ],
        )
        return result_id
