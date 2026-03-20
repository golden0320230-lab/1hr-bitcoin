"""Shared constants for the CLI scaffold."""

from pathlib import Path

APP_NAME = "Kalshi BTC 1-Hour Predictor CLI"
CLI_NAME = "btc-kalshi"
APP_LOGGER_NAME = "btc_kalshi"
DEFAULT_DB_PATH = Path("data/kalshi_btc.duckdb")
DEFAULT_MODEL_PATH = Path("models/baseline.pkl")
DEFAULT_NEWS_ARTICLE_LIMIT = 20
DEFAULT_HTTP_TIMEOUT_SECONDS = 20.0
RESEARCH_DISCLAIMER = "Research only. Not financial advice. No trade execution performed."
PLACEHOLDER_MESSAGE = "This command is scaffolded in issue 1 and will be implemented later."

