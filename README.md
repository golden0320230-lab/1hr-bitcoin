# Kalshi BTC 1-Hour Predictor CLI

Research-only, local-first CLI scaffold for the live Kalshi Bitcoin 1-hour market workflow.

This repository currently contains the issue 1 bootstrap:
- Python packaging and dependency metadata
- a Typer CLI entrypoint
- typed environment-based configuration
- reusable logging setup
- lint, type-check, test, and pre-commit tooling

The command surface is in place, but all domain commands are placeholders until later issues are implemented.

## Research-only scope

This project does not place trades, connect to execution APIs, or provide financial advice.

## Requirements

- Python 3.11+
- `uv` for the preferred workflow
- `pre-commit` for local hooks

## Quick start with `uv`

```bash
uv sync --extra dev
uv run btc-kalshi --help
uv run btc-kalshi market --help
uv run pytest
uv run ruff check .
uv run mypy app
```

## Alternative local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
btc-kalshi --help
```

## Environment

Copy `.env.example` to `.env` and adjust values as needed:

```powershell
Copy-Item .env.example .env
```

## Available commands

```bash
btc-kalshi --help
btc-kalshi market
btc-kalshi predict
btc-kalshi news
btc-kalshi train
btc-kalshi backtest
btc-kalshi explain
```

## Tooling

```bash
pytest
ruff check .
mypy app
pre-commit install
```

## Repository layout

```text
app/
  cli.py
  config.py
  constants.py
  logging.py
  services/
  utils/
data/
models/
tests/
```

