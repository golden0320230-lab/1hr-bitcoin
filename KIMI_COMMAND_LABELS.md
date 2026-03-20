# Kimi Command Labels

Use these labels as shorthand for the CLI commands below.

## `run market`

Purpose: run the full workflow and return the market call.

Command:

```powershell
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" predict --json --save-run
```

Optional reviewer flags:

```powershell
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" predict --json --save-run --reviewer codex --reviewer-model gpt-5
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" predict --json --save-run --reviewer claude --reviewer-model claude-sonnet-4-6
```

Notes:
- This runs Kalshi market discovery, Coinbase price data, and news ingestion.
- The prediction label is returned as `ABOVE` or `BELOW`.
- For the 15-minute market, treat `ABOVE` as `UP` and `BELOW` as `DOWN`.
- If reviewer scoring is enabled, the output also includes a clean reviewer news call in `news_review_summary.market_call`.
- `--save-run` stores the run so `Explain run` can inspect it afterward.
- `--reviewer` can be `kimiclaw`, `codex`, or `claude`.

## `get info`

Purpose: get market info only.

Command:

```powershell
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" market --json
```

Read these fields from the output:
- condition: use `market.direction` + `market.threshold`
- up price: `market.yes_price`
- down price: `market.no_price`
- expiration time: `market.expires_at`

Note:
- If you need "how long until expiry", compute it from the current time and `market.expires_at`.

## `Explain run`

Purpose: inspect the saved run's drivers, warnings, and feature values.

Command:

```powershell
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" explain --last --json
```

Notes:
- This requires a previously saved run.
- The easiest way to create that saved run is `run market`.
- Read `prediction.drivers`, `prediction.warnings`, and `prediction.feature_vector`.

## `Monitor market`

Purpose: continuously monitor the live 15-minute Kalshi market until the watched side reaches a target price.

Command:

```powershell
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" monitor up 0.65 --poll-seconds 20 --max-checks 180 --json
```

Example for the down side:

```powershell
.\.venv\Scripts\python.exe -c "from app.cli import app; app()" monitor down 65 --poll-seconds 20 --max-checks 180 --json
```

Notes:
- `up` watches the Kalshi yes price.
- `down` watches the Kalshi no price.
- `target_price` accepts either `0-1` dollars or `1-100` cents.
- This command only polls Kalshi and is intentionally rate-limited by `--poll-seconds`.

## Usage Order

Recommended sequence:
1. `get info`
2. `run market`
3. `Explain run`
4. `Monitor market`

Run commands sequentially, not in parallel, to avoid DuckDB file locks.
