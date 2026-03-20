"""Shared domain models for the Kalshi BTC CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, HttpUrl, model_validator

from app.constants import RESEARCH_DISCLAIMER

MarketDirection = Literal["ABOVE", "BELOW"]
MarketStatus = Literal["open", "closed", "settled", "unknown"]
CandleTimeframe = Literal["1m", "5m", "15m", "1h"]
SentimentLabel = Literal["bullish", "bearish", "neutral"]
ConfidenceBucket = Literal["low", "medium", "high"]


def _normalize_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, int | float):
        dt = datetime.fromtimestamp(value, tz=UTC)
    else:
        raise TypeError("Expected a datetime, ISO 8601 string, or Unix timestamp.")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)

    return dt.astimezone(UTC)


UtcDateTime = Annotated[datetime, BeforeValidator(_normalize_datetime)]


class SchemaBase(BaseModel):
    """Base schema configuration shared by all domain models."""

    model_config = ConfigDict(extra="forbid")


class KalshiMarket(SchemaBase):
    """Normalized Kalshi market metadata."""

    ticker: str = Field(min_length=1)
    title: str = Field(min_length=1)
    event_ticker: str | None = None
    event_title: str | None = None
    direction: MarketDirection
    threshold: float = Field(gt=0)
    expires_at: UtcDateTime
    status: MarketStatus = "open"
    rules: str | None = None
    market_url: HttpUrl | None = None
    raw_payload: dict[str, Any] | None = None


class MarketSnapshot(SchemaBase):
    """Latest pricing state for a Kalshi market."""

    ticker: str = Field(min_length=1)
    captured_at: UtcDateTime
    yes_price: float | None = Field(default=None, ge=0, le=1)
    no_price: float | None = Field(default=None, ge=0, le=1)
    yes_bid: float | None = Field(default=None, ge=0, le=1)
    yes_ask: float | None = Field(default=None, ge=0, le=1)
    no_bid: float | None = Field(default=None, ge=0, le=1)
    no_ask: float | None = Field(default=None, ge=0, le=1)
    volume: float | None = Field(default=None, ge=0)
    open_interest: int | None = Field(default=None, ge=0)
    raw_payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_snapshot_prices(self) -> MarketSnapshot:
        if all(
            price is None
            for price in (
                self.yes_price,
                self.no_price,
                self.yes_bid,
                self.yes_ask,
                self.no_bid,
                self.no_ask,
            )
        ):
            raise ValueError("Market snapshots must include at least one price field.")

        if self.yes_bid is not None and self.yes_ask is not None and self.yes_bid > self.yes_ask:
            raise ValueError("yes_bid cannot be greater than yes_ask.")

        if self.no_bid is not None and self.no_ask is not None and self.no_bid > self.no_ask:
            raise ValueError("no_bid cannot be greater than no_ask.")

        return self


class BTCCandle(SchemaBase):
    """Normalized BTC OHLCV candle row."""

    source: str = Field(min_length=1)
    product_id: str = Field(default="BTC-USD", min_length=1)
    timeframe: CandleTimeframe
    timestamp: UtcDateTime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)
    raw_payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_ohlc(self) -> BTCCandle:
        highest_price = max(self.open, self.close, self.low)
        lowest_price = min(self.open, self.close, self.high)

        if self.high < highest_price:
            raise ValueError("high must be at least the max of open, low, and close.")

        if self.low > lowest_price:
            raise ValueError("low must be at most the min of open, high, and close.")

        if self.low > self.high:
            raise ValueError("low cannot be greater than high.")

        return self


class NewsArticle(SchemaBase):
    """Normalized news article metadata."""

    title: str = Field(min_length=1)
    url: HttpUrl
    source: str = Field(min_length=1)
    published_at: UtcDateTime
    summary: str | None = None
    content_fingerprint: str | None = None
    raw_payload: dict[str, Any] | None = None


class ArticleSentimentScore(SchemaBase):
    """Structured KimiClaw sentiment output for a single article."""

    article_url: HttpUrl
    model_name: str = Field(min_length=1)
    scored_at: UtcDateTime
    sentiment: SentimentLabel
    relevance: float = Field(ge=0, le=1)
    impact_horizon_minutes: int = Field(gt=0, le=240)
    impact_score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    raw_response: dict[str, Any] | None = None


class FeatureVector(SchemaBase):
    """Model-ready feature values derived from market, price, and news inputs."""

    schema_version: str = Field(default="1.0.0", min_length=1)
    generated_at: UtcDateTime
    market_ticker: str = Field(min_length=1)
    spot_price: float = Field(gt=0)
    strike_price: float = Field(gt=0)
    distance_to_strike: float
    distance_to_strike_pct: float
    kalshi_yes_price: float = Field(ge=0, le=1)
    kalshi_no_price: float = Field(ge=0, le=1)
    market_implied_probability: float = Field(ge=0, le=1)
    spread: float = Field(ge=0, le=1)
    orderbook_imbalance: float | None = Field(default=None, ge=-1, le=1)
    return_5m: float
    return_15m: float
    return_30m: float
    return_60m: float
    realized_vol_15m: float = Field(ge=0)
    realized_vol_60m: float = Field(ge=0)
    ma_deviation: float
    momentum_slope: float
    rsi: float = Field(ge=0, le=100)
    minutes_to_expiry: int = Field(ge=0)
    news_weighted_impact: float = Field(default=0, ge=-1, le=1)
    news_weighted_bullish: float = Field(default=0, ge=0)
    news_weighted_bearish: float = Field(default=0, ge=0)
    high_confidence_article_count: int = Field(default=0, ge=0)
    breaking_news: bool = False


class PredictionResult(SchemaBase):
    """Prediction output returned by the CLI pipeline."""

    run_id: UUID = Field(default_factory=uuid4)
    generated_at: UtcDateTime
    market_ticker: str = Field(min_length=1)
    label: MarketDirection
    probability: float = Field(ge=0, le=1)
    confidence: ConfidenceBucket
    drivers: list[str] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER, min_length=1)
    feature_vector: FeatureVector | None = None
    market: KalshiMarket | None = None
    market_snapshot: MarketSnapshot | None = None


class BacktestMetric(SchemaBase):
    """Per-strategy evaluation metrics."""

    name: str = Field(min_length=1)
    accuracy: float = Field(ge=0, le=1)
    log_loss: float | None = Field(default=None, ge=0)
    brier_score: float | None = Field(default=None, ge=0, le=1)


class BacktestResult(SchemaBase):
    """Historical evaluation output for the predictor and baselines."""

    generated_at: UtcDateTime
    window_start: UtcDateTime
    window_end: UtcDateTime
    model_name: str = Field(min_length=1)
    num_samples: int = Field(gt=0)
    accuracy: float = Field(ge=0, le=1)
    log_loss: float = Field(ge=0)
    brier_score: float | None = Field(default=None, ge=0, le=1)
    baselines: list[BacktestMetric] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window_order(self) -> BacktestResult:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start.")

        return self

