"""Typer CLI entrypoint for the Kalshi BTC tool."""

from __future__ import annotations

import logging as stdlib_logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import orjson
import typer
from rich.console import Console
from rich.table import Table

from app import __version__
from app.config import Settings, get_settings
from app.constants import APP_NAME, CLI_NAME, RESEARCH_DISCLAIMER
from app.logging import configure_logging, get_logger
from app.schemas import ArticleSentimentScore, BTCCandle, FeatureVector, NewsArticle
from app.services.backtest import BacktestService
from app.services.coinbase import CoinbaseClient, CoinbaseServiceError
from app.services.explain import ExplainService
from app.services.features import FeatureBuilder
from app.services.kalshi import KalshiClient, KalshiServiceError
from app.services.kimiclaw import KimiClawClient, KimiClawServiceError
from app.services.news import NewsClient, NewsServiceError
from app.services.predictor import Predictor
from app.services.storage import DuckDBStorage
from app.services.training import ModelTrainer, TrainingDatasetBuilder

app = typer.Typer(
    name=CLI_NAME,
    help="Research-only CLI for the Kalshi BTC 1-hour predictor.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


@dataclass(slots=True)
class Runtime:
    """Per-command runtime state stored on the Typer context."""

    settings: Settings
    logger: stdlib_logging.Logger
    storage: DuckDBStorage


def _build_runtime(log_level_override: str | None = None) -> Runtime:
    settings = get_settings()
    if log_level_override is not None:
        settings = settings.with_overrides(log_level=log_level_override)
    settings.ensure_runtime_directories()
    configure_logging(settings.log_level)
    return Runtime(
        settings=settings,
        logger=get_logger(),
        storage=DuckDBStorage(settings.db_path),
    )


def _get_runtime(ctx: typer.Context) -> Runtime:
    runtime = ctx.obj
    if isinstance(runtime, Runtime):
        return runtime
    raise RuntimeError("CLI context was not initialized.")


def _json_echo(payload: Mapping[str, object]) -> None:
    typer.echo(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))


def _is_kimiclaw_configured(settings: Settings) -> bool:
    return all(
        [
            settings.kimiclaw_base_url != "https://replace-me.example.com",
            settings.kimiclaw_api_key.get_secret_value() != "replace-me",
            settings.kimiclaw_model != "replace-me",
        ]
    )


def _render_market(market_payload: Mapping[str, object]) -> None:
    threshold = cast(float, market_payload["threshold"])
    yes_price = cast(float, market_payload["yes_price"])
    no_price = cast(float, market_payload["no_price"])
    table = Table(show_header=False, box=None)
    table.add_row("Ticker", str(market_payload["ticker"]))
    table.add_row("Question", str(market_payload["title"]))
    table.add_row("Threshold", f"${threshold:,.2f}")
    table.add_row("Direction", str(market_payload["direction"]))
    table.add_row("Expiry", str(market_payload["expires_at"]))
    table.add_row("Yes", f"{yes_price:.2%}")
    table.add_row("No", f"{no_price:.2%}")
    table.add_row("Implied prior", f"{yes_price:.2%}")
    console.print(table)


def _market_payload(
    market_ticker: str,
    title: str,
    threshold: float,
    direction: str,
    expires_at: str,
    yes_price: float,
    no_price: float,
) -> dict[str, object]:
    return cast(
        dict[str, object],
        {
            "ticker": market_ticker,
            "title": title,
            "threshold": threshold,
            "direction": direction,
            "expires_at": expires_at,
            "yes_price": yes_price,
            "no_price": no_price,
        },
    )


def _render_warnings(warnings: list[str]) -> None:
    if not warnings:
        return

    console.print("Warnings:")
    for warning in warnings:
        console.print(f"- {warning}")


def _kimiclaw_fallback_used(scores: list[ArticleSentimentScore]) -> bool:
    return any(score.raw_response == {"fallback": True} for score in scores)


def _score_articles_with_warnings(
    runtime: Runtime,
    articles: list[NewsArticle],
) -> tuple[list[ArticleSentimentScore], list[str]]:
    if not articles:
        return [], ["No recent BTC-related articles found; using neutral news contribution."]

    if not _is_kimiclaw_configured(runtime.settings):
        return [], ["KimiClaw is not configured; using neutral news contribution."]

    with KimiClawClient(
        base_url=runtime.settings.kimiclaw_base_url,
        api_key=runtime.settings.kimiclaw_api_key.get_secret_value(),
        model_name=runtime.settings.kimiclaw_model,
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as client:
        try:
            scores = client.score_articles(articles)
        except KimiClawServiceError:
            return [], ["KimiClaw unavailable; using neutral news contribution."]

    warnings: list[str] = []
    if _kimiclaw_fallback_used(scores):
        warnings.append("KimiClaw unavailable or invalid output; using neutral news contribution.")
    return scores, warnings


def _load_cached_candles(storage: DuckDBStorage) -> list[BTCCandle]:
    for timeframe in ("1m", "5m", "15m"):
        candles = storage.list_recent_candles(source="coinbase", timeframe=timeframe, limit=90)
        if candles:
            return candles
    return []


def _load_price_model_probability(
    runtime: Runtime,
    features: FeatureVector,
) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    model_path = runtime.settings.model_path
    if not model_path.exists():
        warnings.append("Model artifact missing; using heuristic price model.")
        return None, warnings

    try:
        artifact = ModelTrainer.load_artifact(model_path)
        probability = ModelTrainer().predict_feature_probability(artifact, features)
    except (OSError, ValueError, KeyError, TypeError):
        warnings.append("Model artifact could not be loaded; using heuristic price model.")
        return None, warnings

    return probability, warnings


@app.callback()
def root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed CLI version and exit.",
        is_eager=True,
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Override the configured log level for the current command.",
    ),
) -> None:
    """Initialize common runtime state for all commands."""

    if version:
        typer.echo(f"{APP_NAME} {__version__}")
        raise typer.Exit()

    ctx.obj = _build_runtime(log_level)


@app.command()
def market(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Show the live Kalshi BTC hourly market."""

    runtime = _get_runtime(ctx)
    try:
        with KalshiClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as client:
            discovered = client.get_live_btc_hourly_market()
    except KalshiServiceError:
        payload = {
            "market": None,
            "warnings": ["Kalshi market discovery failed."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("Kalshi market discovery failed.")
            console.print(RESEARCH_DISCLAIMER)
        return

    if discovered is None:
        no_market_payload: dict[str, object] = {
            "market": None,
            "warnings": ["No live BTC hourly Kalshi market found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(no_market_payload)
        else:
            console.print("No live BTC hourly Kalshi market found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    market_model, snapshot = discovered
    market_payload = _market_payload(
        market_ticker=market_model.ticker,
        title=market_model.title,
        threshold=market_model.threshold,
        direction=market_model.direction,
        expires_at=market_model.expires_at.isoformat(),
        yes_price=snapshot.yes_price or 0.0,
        no_price=snapshot.no_price or 0.0,
    )

    if json_output:
        _json_echo(
            {
                "market": market_payload,
                "warnings": [],
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
    else:
        _render_market(market_payload)
        console.print(RESEARCH_DISCLAIMER)


@app.command()
def news(
    ctx: typer.Context,
    limit: int | None = typer.Option(None, "--limit", min=1, help="Maximum articles to return."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    score: bool = typer.Option(True, "--score/--no-score", help="Score articles with KimiClaw."),
) -> None:
    """Fetch and display recent BTC-related news."""

    runtime = _get_runtime(ctx)
    effective_limit = limit or runtime.settings.news_article_limit
    warnings: list[str] = []

    try:
        with NewsClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as client:
            articles = client.fetch_recent_articles(limit=effective_limit, lookback_hours=24)
    except NewsServiceError:
        articles = []
        warnings.append("News fetch failed; returning no recent articles.")

    scores: list[ArticleSentimentScore] = []
    if score and articles:
        scores, score_warnings = _score_articles_with_warnings(runtime, articles)
        warnings.extend(score_warnings)

    if json_output:
        _json_echo(
            {
                "articles": [article.model_dump(mode="json") for article in articles],
                "scores": [score.model_dump(mode="json") for score in scores],
                "warnings": warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    table = Table(title="BTC News")
    table.add_column("Published")
    table.add_column("Source")
    table.add_column("Title")
    table.add_column("Sentiment")
    score_by_url = {str(score.article_url): score.sentiment for score in scores}

    for article in articles:
        table.add_row(
            article.published_at.strftime("%Y-%m-%d %H:%M"),
            article.source,
            article.title,
            score_by_url.get(str(article.url), "unscored"),
        )

    console.print(table)
    _render_warnings(warnings)
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def predict(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    news_limit: int | None = typer.Option(None, "--news-limit", min=1, help="Article limit."),
    no_news: bool = typer.Option(False, "--no-news", help="Disable news ingestion and scoring."),
    verbose: bool = typer.Option(False, "--verbose", help="Show extra pipeline details."),
    save_run: bool = typer.Option(False, "--save-run", help="Persist the prediction to DuckDB."),
) -> None:
    """Run the research-only ABOVE or BELOW prediction flow."""

    runtime = _get_runtime(ctx)
    warnings: list[str] = []

    try:
        with KalshiClient(
            storage=runtime.storage,
            timeout_seconds=runtime.settings.http_timeout_seconds,
        ) as kalshi_client:
            discovered = kalshi_client.get_live_btc_hourly_market()
    except KalshiServiceError:
        payload = {
            "prediction": None,
            "warnings": ["Kalshi market discovery failed."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("Kalshi market discovery failed.")
            console.print(RESEARCH_DISCLAIMER)
        return

    if discovered is None:
        payload = {
            "prediction": None,
            "warnings": ["No live BTC hourly Kalshi market found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("No live BTC hourly Kalshi market found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    market_model, snapshot = discovered

    spot_price: float | None = None
    candles: list[BTCCandle] = []
    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            spot_price = coinbase_client.get_spot_price()
        except CoinbaseServiceError:
            warnings.append(
                "BTC spot fetch failed; using latest available candle close if possible."
            )

        try:
            candles = coinbase_client.get_candles(lookback_minutes=90, timeframe="1m")
        except CoinbaseServiceError:
            warnings.append("BTC candle fetch failed; using cached BTC candles if available.")

    if not candles:
        candles = _load_cached_candles(runtime.storage)
        if candles:
            warnings.append("Using cached BTC candles for feature generation.")

    if spot_price is None and candles:
        spot_price = candles[-1].close
        warnings.append("Using latest cached candle close as BTC spot price.")

    if spot_price is None or not candles:
        error_warnings = warnings + ["BTC fetch failed and no cached data was available."]
        payload = {
            "prediction": None,
            "warnings": error_warnings,
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            for warning in error_warnings:
                console.print(warning)
            console.print(RESEARCH_DISCLAIMER)
        return

    articles = []
    scores: list[ArticleSentimentScore] = []
    if not no_news:
        try:
            with NewsClient(
                storage=runtime.storage,
                timeout_seconds=runtime.settings.http_timeout_seconds,
            ) as news_client:
                articles = news_client.fetch_recent_articles(
                    limit=news_limit or runtime.settings.news_article_limit,
                    lookback_hours=24,
                )
        except NewsServiceError:
            warnings.append("News fetch failed; using neutral news contribution.")
            articles = []

        if articles:
            scores, score_warnings = _score_articles_with_warnings(runtime, articles)
            warnings.extend(score_warnings)
        elif "News fetch failed; using neutral news contribution." not in warnings:
            warnings.append(
                "No recent BTC-related articles found; using neutral news contribution."
            )
    else:
        warnings.append("News scoring disabled; using neutral news contribution.")

    features = FeatureBuilder().build_feature_vector(
        market=market_model,
        snapshot=snapshot,
        spot_price=spot_price,
        candles=candles,
        news_scores=scores,
        generated_at=snapshot.captured_at,
    )
    price_model_probability, model_warnings = _load_price_model_probability(runtime, features)
    warnings.extend(model_warnings)
    result = Predictor().predict(
        market=market_model,
        snapshot=snapshot,
        features=features,
        price_model_probability=price_model_probability,
    )

    if warnings:
        result = result.model_copy(
            update={
                "warnings": result.warnings
                + [warning for warning in warnings if warning not in result.warnings],
                "degraded": True,
            }
        )

    if save_run:
        runtime.storage.insert_prediction_run(result)

    if json_output:
        _json_echo(
            {
                "prediction": result.model_dump(mode="json"),
                "articles": [article.model_dump(mode="json") for article in articles],
                "warnings": result.warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    console.print(f"Live market: {market_model.title}")
    console.print(f"Kalshi ticker: {market_model.ticker}")
    console.print(f"BTC spot: ${spot_price:,.2f}")
    console.print(f"Distance to strike: {features.distance_to_strike:+.2f}")
    console.print(f"Time to expiry: {features.minutes_to_expiry}m")
    console.print()
    console.print(f"Prediction: {result.label}")
    console.print(f"Probability: {result.probability:.2%}")
    console.print(f"Confidence: {result.confidence}")
    console.print("Top drivers:")
    for driver in result.drivers:
        console.print(f"- {driver}")

    _render_warnings(result.warnings)
    if verbose:
        console.print(f"Articles considered: {len(articles)}")

    console.print(RESEARCH_DISCLAIMER)


@app.command()
def train(
    ctx: typer.Context,
    days: int = typer.Option(90, "--days", min=7, help="Historical window for training."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Train or refresh the local baseline model."""

    runtime = _get_runtime(ctx)
    dataset_path = runtime.settings.db_path.parent / "training_dataset.csv"
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)

    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            candles = coinbase_client.get_candles_range(
                start_at=start_time,
                end_at=end_time,
                timeframe="5m",
            )
        except CoinbaseServiceError as exc:
            console.print(f"Unable to fetch training candles: {exc}")
            console.print(RESEARCH_DISCLAIMER)
            raise typer.Exit(code=1) from exc

    dataset_builder = TrainingDatasetBuilder(dataset_path=dataset_path, strike_increment=100.0)
    dataset = dataset_builder.build_dataset(candles, horizon_minutes=60, step_candles=1)
    saved_dataset = dataset_builder.save_dataset(dataset)

    try:
        result = ModelTrainer(runtime.storage).train(
            dataset,
            artifact_path=runtime.settings.model_path,
        )
    except ValueError as exc:
        console.print(f"Unable to train model: {exc}")
        console.print(RESEARCH_DISCLAIMER)
        raise typer.Exit(code=1) from exc

    if json_output:
        _json_echo(
            {
                "training": {
                    "model_name": result.model_name,
                    "artifact_path": str(result.artifact_path),
                    "dataset_path": str(saved_dataset.path),
                    "dataset_rows": result.dataset_rows,
                    "feature_schema_version": result.feature_schema_version,
                    "training_window_start": result.training_window_start.isoformat(),
                    "training_window_end": result.training_window_end.isoformat(),
                    "metrics": result.metrics,
                },
                "warnings": [],
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    console.print(f"Training rows: {result.dataset_rows}")
    console.print(f"Dataset saved: {saved_dataset.path}")
    console.print(f"Selected model: {result.model_name}")
    console.print(f"Artifact saved: {result.artifact_path}")
    console.print(
        "Validation metrics: "
        f"accuracy {result.metrics[result.model_name]['accuracy']:.2%}, "
        f"log loss {result.metrics[result.model_name]['log_loss']:.4f}, "
        f"Brier {result.metrics[result.model_name]['brier_score']:.4f}"
    )
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def backtest(
    ctx: typer.Context,
    days: int = typer.Option(30, "--days", min=7, help="Historical window for evaluation."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Run historical evaluations against archived windows."""

    runtime = _get_runtime(ctx)
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)

    with CoinbaseClient(
        storage=runtime.storage,
        timeout_seconds=runtime.settings.http_timeout_seconds,
    ) as coinbase_client:
        try:
            candles = coinbase_client.get_candles_range(
                start_at=start_time,
                end_at=end_time,
                timeframe="5m",
            )
        except CoinbaseServiceError as exc:
            console.print(f"Unable to fetch backtest candles: {exc}")
            console.print(RESEARCH_DISCLAIMER)
            raise typer.Exit(code=1) from exc

    dataset = TrainingDatasetBuilder(strike_increment=100.0).build_dataset(
        candles,
        horizon_minutes=60,
    )

    try:
        result = BacktestService(runtime.storage).run(dataset)
    except ValueError as exc:
        console.print(f"Unable to run backtest: {exc}")
        console.print(RESEARCH_DISCLAIMER)
        raise typer.Exit(code=1) from exc

    if json_output:
        _json_echo(
            {
                "backtest": result.model_dump(mode="json"),
                "warnings": [],
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    console.print(f"Backtest model: {result.model_name}")
    console.print(f"Backtest samples: {result.num_samples}")
    console.print(f"Accuracy: {result.accuracy:.2%}")
    console.print(f"Log loss: {result.log_loss:.4f}")
    if result.brier_score is not None:
        console.print(f"Brier score: {result.brier_score:.4f}")
    console.print("Baselines:")
    for baseline in result.baselines:
        console.print(
            f"- {baseline.name}: accuracy {baseline.accuracy:.2%}, "
            f"log loss {baseline.log_loss:.4f}, "
            f"Brier {baseline.brier_score:.4f}"
        )
    console.print(RESEARCH_DISCLAIMER)


@app.command()
def explain(
    ctx: typer.Context,
    last: bool = typer.Option(
        True,
        "--last/--no-last",
        help="Show the latest saved prediction run.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Inspect the latest saved prediction run."""

    runtime = _get_runtime(ctx)
    if not last:
        raise typer.BadParameter("Only --last is currently supported.", param_hint="--last")

    service = ExplainService(runtime.storage)
    prediction = service.get_last_prediction()

    if prediction is None:
        payload = {
            "prediction": None,
            "warnings": ["No saved prediction run found."],
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        if json_output:
            _json_echo(payload)
        else:
            console.print("No saved prediction run found.")
            console.print(RESEARCH_DISCLAIMER)
        return

    if json_output:
        _json_echo(
            {
                "prediction": prediction.model_dump(mode="json"),
                "warnings": prediction.warnings,
                "disclaimer": RESEARCH_DISCLAIMER,
            }
        )
        return

    market = prediction.market
    features = prediction.feature_vector
    if market is not None:
        console.print(f"Live market: {market.title}")
        console.print(f"Kalshi ticker: {market.ticker}")
    console.print(f"Prediction: {prediction.label}")
    console.print(f"Probability: {prediction.probability:.2%}")
    console.print(f"Confidence: {prediction.confidence}")
    console.print("Main drivers:")
    for driver in prediction.drivers:
        console.print(f"- {driver}")
    if features is not None:
        console.print("Feature values:")
        console.print(f"- Spot price: ${features.spot_price:,.2f}")
        console.print(f"- Strike price: ${features.strike_price:,.2f}")
        console.print(f"- Distance to strike: {features.distance_to_strike:+.2f}")
        console.print(f"- Market implied probability: {features.market_implied_probability:.2%}")
        console.print(f"- Return 15m: {features.return_15m:+.2%}")
        console.print(f"- Return 60m: {features.return_60m:+.2%}")
        console.print(f"- News weighted impact: {features.news_weighted_impact:+.2f}")
        console.print(f"- Minutes to expiry: {features.minutes_to_expiry}")
    _render_warnings(prediction.warnings)
    console.print(RESEARCH_DISCLAIMER)


def main() -> None:
    """Run the CLI application."""

    app()
