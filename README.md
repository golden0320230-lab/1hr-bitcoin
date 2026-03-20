# Kalshi BTC 15-Minute Predictor CLI

Research-only, local-first CLI for the live Kalshi Bitcoin 15-minute up/down market.

The tool:
- discovers the current live Kalshi BTC 15-minute market
- fetches current BTC spot and recent candles from Coinbase
- ingests recent BTC-related news from Google News RSS and GDELT
- optionally scores articles with a hosted KimiClaw endpoint
- blends market prior, price action, and news into an `ABOVE` or `BELOW` research prediction
- saves local history, training datasets, model artifacts, and backtest summaries

It does not place trades, connect to brokerage execution flows, or provide financial advice.

## Research-only boundary

Every command is scoped to research and debugging:
- no order placement
- no trade execution
- no position sizing
- no buy/sell recommendations
- local storage only, except the optional hosted KimiClaw sentiment call

The CLI always prints or returns the disclaimer:

`Research only. Not financial advice. No trade execution performed.`

## Architecture

The current project is organized around small service layers:
- `app/services/kalshi.py`: live BTC 15-minute Kalshi market discovery and pricing snapshots
- `app/services/coinbase.py`: public BTC spot, recent candles, and chunked historical candle fetches
- `app/services/news.py`: Google News RSS and GDELT ingestion with deduplication and freshness filtering
- `app/services/kimiclaw.py`: hosted structured article scoring with neutral fallback
- `app/services/features.py`: market, price, and news feature generation
- `app/services/predictor.py`: hybrid research predictor and explanation drivers
- `app/services/training.py`: historical dataset generation, sklearn training, and model artifact loading
- `app/services/backtest.py`: holdout backtesting plus naive baselines
- `app/services/storage.py`: DuckDB persistence for markets, candles, news, predictions, metadata, and backtests

Local outputs are written under:
- `data/kalshi_btc.duckdb`
- `data/training_dataset.csv`
- `models/baseline.pkl`

## Requirements

- Python 3.11+
- `uv` recommended for dependency and environment management
- internet access for live market/data/news fetches
- optional KimiClaw endpoint credentials if you want article scoring

## Setup

Preferred workflow with `uv`:

```bash
uv sync --extra dev
cp .env.example .env
uv run btc-kalshi --help
```

Windows PowerShell alternative:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
btc-kalshi --help
```

## Configuration

Environment variables live in `.env`. The minimum set is documented in `.env.example`.

Important variables:
- `DB_PATH`: DuckDB file for cached data, prediction history, model metadata, and backtests
- `MODEL_PATH`: local sklearn artifact used by `predict` when present
- `NEWS_ARTICLE_LIMIT`: default article count for `news` and `predict`
- `KIMICLAW_BASE_URL`, `KIMICLAW_API_KEY`, `KIMICLAW_MODEL`: optional hosted scoring config
- `HTTP_TIMEOUT_SECONDS`: shared timeout used by all external clients

KimiClaw is optional. If it is not configured or returns invalid output, the CLI falls back to neutral news scoring and surfaces a warning.

## Commands

Show the live Kalshi BTC 15-minute market:

```bash
uv run btc-kalshi market
uv run btc-kalshi market --json
```

Fetch recent BTC news:

```bash
uv run btc-kalshi news --limit 10
uv run btc-kalshi news --limit 10 --json
uv run btc-kalshi news --limit 10 --no-score
```

Run the research prediction:

```bash
uv run btc-kalshi predict
uv run btc-kalshi predict --json
uv run btc-kalshi predict --news-limit 20
uv run btc-kalshi predict --no-news
uv run btc-kalshi predict --verbose
uv run btc-kalshi predict --save-run
```

Train the local baseline model artifact:

```bash
uv run btc-kalshi train --days 90
uv run btc-kalshi train --days 90 --json
```

Run a historical backtest:

```bash
uv run btc-kalshi backtest --days 30
uv run btc-kalshi backtest --days 30 --json
```

Inspect the latest saved prediction:

```bash
uv run btc-kalshi explain --last
```

## Reliability and degraded behavior

The CLI is designed to fail safely:
- if Kalshi market discovery fails, the command exits cleanly with a warning
- if Coinbase live fetches fail, `predict` falls back to cached candles when available
- if no recent news is found, the predictor uses neutral news features
- if KimiClaw fails, article scores fall back to neutral and the warning is surfaced
- if `MODEL_PATH` is missing, `predict` falls back to the heuristic price model

## Training and backtesting notes

The training/backtest flow uses public BTC candles and generates historical examples aligned to the 15-minute market question:
- prediction time `t`
- target price `X`
- expiry `E = t + 15 minutes`
- label = whether BTC settles at or above `X` at `E`

Because public historical Kalshi 15-minute orderbook data is not used in this baseline, the training dataset synthesizes the market framing around public BTC candles. That keeps the live question semantics aligned, but it is still a baseline approximation rather than a full historical replay of Kalshi microstructure.

Backtest baselines currently include:
- `kalshi_follow`
- `spot_vs_strike`
- `momentum`
- `no_news`

## Data sources

- Kalshi public market endpoints
- Coinbase public spot and candles
- Google News RSS
- GDELT Doc API
- optional hosted KimiClaw endpoint

## Limitations

- The tool depends on live third-party APIs and can degrade when they fail.
- News scoring quality depends on the external KimiClaw endpoint when configured.
- Training and backtesting use a synthetic 15-minute-question framing from public BTC candles, not archived live Kalshi orderbook history.
- The predictor is a research baseline. It is not calibrated for production trading.
- No execution, brokerage, or automation layer exists in this repo by design.

## Developer workflow

Install hooks and run the local quality gate:

```bash
pre-commit install
pytest
ruff check .
mypy app
```

## Repository layout

```text
app/
  cli.py
  config.py
  constants.py
  logging.py
  schemas.py
  services/
  prompts/
  utils/
data/
models/
tests/
```
