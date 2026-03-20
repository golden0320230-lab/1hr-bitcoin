# Kalshi BTC 1-Hour Predictor CLI — Codex Build Spec

## 1. Objective
Build a **CLI-only**, **research-only**, **local-first** Bitcoin prediction tool for the **live Kalshi Bitcoin 1-hour market**.

The tool must:
- discover the current live Kalshi BTC hourly market
- extract the market condition in normalized form: **Bitcoin above or below $X at expiry time**
- fetch current BTC price and recent historical BTC candle data
- ingest fresh Bitcoin-related news articles
- send relevant news article summaries to a hosted KimiClaw endpoint for structured sentiment scoring
- combine market prior, price action, and news sentiment into a binary research prediction: **ABOVE** or **BELOW**
- never place trades
- never connect to brokerage or execution APIs
- never present output as financial advice

This project is a **local CLI app** with **local storage and local inference orchestration**, except for the optional hosted KimiClaw sentiment call.

---

## 2. Product Boundaries

### In scope
- CLI interface only
- Local execution on one machine
- Local cache/database
- Pull-only market/data/news ingestion
- Hosted KimiClaw sentiment scoring for article impact classification
- Historical backtesting against archived hourly windows
- Human-readable terminal output and machine-readable JSON export

### Out of scope
- Trade execution
- Browser UI
- Mobile app
- API server for external clients
- Position sizing
- Profit optimization logic
- Arbitrage engine
- Social media scraping in v1
- Autonomous background daemon in v1

---

## 3. Non-Negotiable Requirements
1. The app must be **CLI only**. No web frontend.
2. The app must run **locally** and store data locally.
3. The app must use **free APIs that are usable in the US**.
4. The app must use the **latest live Kalshi BTC 1-hour market** instead of a hardcoded ticker.
5. The app must make only a **research prediction** and never place trades.
6. The app must show a clear disclaimer that output is not trading or financial advice.
7. The app must ingest **recent BTC news** and score it through KimiClaw.
8. KimiClaw scoring must return **strict structured JSON** only.
9. The system must work even if KimiClaw is unavailable by falling back to a **neutral news score** and warning the user.
10. All network dependencies must be wrapped in retry, timeout, and graceful-failure logic.

---

## 4. Key Design Decision
Although the app is local-first, the user explicitly wants to use hosted KimiClaw for news classification. Treat this as an **allowed remote scoring dependency**.

Interpret the requirement like this:
- the CLI runs locally
- all data collection, caching, feature engineering, prediction, and output happen locally
- news sentiment is optionally enriched via a hosted KimiClaw endpoint
- if the KimiClaw endpoint fails, the rest of the predictor still works

Do **not** redesign this into a fully cloud-hosted app.

---

## 5. Recommended Technology Stack
Use the following stack unless there is a strong implementation reason to deviate:

- **Python 3.11+**
- **uv** for environment and dependency management
- **Typer** for CLI
- **httpx** for HTTP clients
- **pydantic** for config and schema validation
- **DuckDB** for local analytical storage
- **SQLite** acceptable if easier, but prefer DuckDB for backtesting and feature work
- **pandas** for data manipulation
- **scikit-learn** for baseline models
- **numpy** for numeric ops
- **feedparser** for RSS ingestion
- **BeautifulSoup4** for article cleanup
- **rich** for terminal formatting
- **tenacity** for retries
- **orjson** for fast JSON IO
- **pytest** for tests
- **ruff** for linting
- **mypy** for type checking
- **pre-commit** for local checks

Avoid heavyweight ML frameworks in v1. No PyTorch or TensorFlow in the initial version.

---

## 6. External Data Sources

### 6.1 Kalshi
Purpose:
- discover the active BTC hourly market
- fetch market metadata
- fetch market pricing and optionally orderbook state

What the implementation must support:
- event and market discovery
- normalize market title/rules into a threshold and expiry timestamp
- fetch latest yes/no pricing
- fetch orderbook or recent market data if helpful

### 6.2 BTC spot and candles
Primary source:
- Coinbase public market data

Fallback source:
- optional secondary BTC source if Coinbase temporarily fails

What the implementation must support:
- current BTC-USD spot
- recent candles in at least 1m, 5m, 15m, and 1h granularity
- local caching of fetched candles

### 6.3 News
Primary sources:
- Google News RSS searches for Bitcoin-related terms
- GDELT article search for broader coverage

What the implementation must support:
- fetching recent article metadata
- normalizing title, source, url, published timestamp, summary/snippet
- de-duplication by URL and content fingerprint
- relevance filtering before KimiClaw scoring

### 6.4 KimiClaw
Purpose:
- classify each relevant article for likely short-horizon BTC impact

Treat KimiClaw as an OpenAI-compatible or custom HTTP endpoint configured by environment variables. The code must abstract it behind a dedicated service so it can be swapped later.

---

## 7. CLI Commands
Implement the following commands.

### 7.1 `predict`
Primary user command.

Behavior:
- discover live Kalshi BTC hourly market
- pull current BTC spot
- pull recent BTC candles
- pull recent BTC news
- score relevant articles via KimiClaw
- build features
- run predictor
- print result

Example:
```bash
btc-kalshi predict
```

Optional flags:
```bash
btc-kalshi predict --json
btc-kalshi predict --news-limit 20
btc-kalshi predict --no-news
btc-kalshi predict --verbose
btc-kalshi predict --save-run
```

### 7.2 `market`
Show current live Kalshi BTC hourly market only.

Example:
```bash
btc-kalshi market
```

Output must include:
- ticker
- event title
- threshold/strike
- market type above/below
- expiry time
- yes price
- no price
- implied prior

### 7.3 `news`
Fetch and display the latest BTC-related news and optional KimiClaw sentiment results.

Example:
```bash
btc-kalshi news --limit 10
```

### 7.4 `backtest`
Run historical evaluation against archived windows.

Example:
```bash
btc-kalshi backtest --days 30
```

### 7.5 `train`
Train or refresh the local baseline model.

Example:
```bash
btc-kalshi train --days 90
```

### 7.6 `explain`
Show a breakdown of the last prediction run and top feature contributions.

Example:
```bash
btc-kalshi explain --last
```

---

## 8. Core Prediction Strategy
Use a **hybrid predictor** with three signal families.

### 8.1 Signal A: Kalshi prior
Derive a market-implied baseline from live yes/no pricing.

Features:
- yes price
- no price
- mid-price if both available
- spread
- optional orderbook imbalance

Purpose:
- represent crowd-implied probability
- provide a baseline that the model can compare against

### 8.2 Signal B: BTC price-action features
This is the strongest v1 signal family.

Must include features for:
- spot price
- distance to strike in dollars
- distance to strike in percent
- return over last 5m
- return over last 15m
- return over last 30m
- return over last 60m
- realized volatility over 15m
- realized volatility over 60m
- rolling high/low breakout context
- moving average deviation
- RSI or similarly simple oscillator
- candle momentum slope
- volume change if available
- time remaining to expiry in minutes

### 8.3 Signal C: News pressure
Build a single aggregated news score from scored articles.

Per-article fields from KimiClaw must include:
- sentiment: bullish, bearish, neutral
- relevance score: 0.0 to 1.0
- impact score: -1.0 to 1.0
- confidence: 0.0 to 1.0
- impact horizon minutes
- concise reason

Aggregate into:
- weighted average impact
- weighted bullish count
- weighted bearish count
- count of high-confidence relevant articles in last 60m
- breaking-news flag if multiple high-relevance articles cluster in short time

### 8.4 Final decision
Blend the three signal families into a final research prediction.

Initial implementation can be:
- logistic regression model trained on engineered features
- plus a deterministic post-processing blend with Kalshi prior and news score

Suggested starting blend:
- 45% market prior
- 40% price model probability
- 15% news pressure

This blend must be configurable, not hardcoded forever.

Final output:
- label: ABOVE or BELOW
- probability: 0.0 to 1.0
- confidence bucket: low / medium / high
- explanation bullets

---

## 9. Historical Label Definition
This is critical.

Do **not** train the model on generic next-candle up/down labels.

Instead, every training row must align to the prediction-market question:
- at prediction time `t`, with threshold `X` and expiry `E`, the label is `1` if BTC settlement price at expiry is **above or equal to X** according to the normalized market condition, else `0`

The model must be trained on the same question the CLI is answering.

---

## 10. News Scoring Contract
Create a dedicated KimiClaw client that sends a structured prompt and expects strict JSON.

### 10.1 KimiClaw prompt requirements
The prompt must instruct the model to:
- evaluate whether the article is relevant to BTC over the next 1 hour
- classify likely directional pressure: bullish, bearish, neutral
- estimate likely price impact magnitude
- avoid overreacting to generic or stale stories
- return JSON only

### 10.2 Required JSON schema
```json
{
  "sentiment": "bullish",
  "relevance": 0.82,
  "impact_horizon_minutes": 45,
  "impact_score": 0.35,
  "confidence": 0.74,
  "reason": "ETF inflow headline may support near-term bullish sentiment."
}
```

### 10.3 Fallback behavior
If KimiClaw fails or returns invalid JSON:
- log the failure
- mark article as unscored
- continue pipeline
- aggregate news as neutral if all scoring fails
- surface warning in CLI output

---

## 11. Output Requirements
Default terminal output must be compact and explainable.

Example structure:

```text
Live market: Bitcoin price today at 4pm EDT? $72,250 or above
Kalshi ticker: BTC-...
BTC spot: $72,118.42
Distance to strike: -$131.58
Time to expiry: 18m

Prediction: BELOW
Probability: 0.61
Confidence: medium

Signal summary:
- Kalshi implied prior: YES 0.54
- Price model: ABOVE 0.41
- News pressure: -0.08 (slightly bearish)

Top drivers:
- BTC remains below strike with weak short-term momentum
- 15m volatility elevated without breakout confirmation
- Recent news flow mildly bearish but not dominant

Research only. Not financial advice. No trade execution performed.
```

When `--json` is used, output a structured JSON payload.

---

## 12. Data Storage Requirements
Use DuckDB with a simple local file database.

Suggested tables:
- `markets`
- `market_snapshots`
- `btc_candles`
- `news_articles`
- `news_scores`
- `prediction_runs`
- `model_metadata`
- `backtest_results`

Requirements:
- idempotent inserts where practical
- unique constraints for URLs and candle timestamps
- timestamps stored in UTC
- preserve raw source payloads in JSON columns when useful

---

## 13. Project Structure
Create the repo with this structure:

```text
kalshi-btc-cli/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  .pre-commit-config.yaml
  app/
    __init__.py
    cli.py
    config.py
    logging.py
    schemas.py
    constants.py
    services/
      __init__.py
      kalshi.py
      coinbase.py
      news.py
      kimiclaw.py
      storage.py
      features.py
      predictor.py
      training.py
      backtest.py
      explain.py
    prompts/
      news_impact_prompt.txt
    utils/
      __init__.py
      time.py
      text.py
      math_utils.py
      retries.py
  data/
    .gitkeep
  models/
    .gitkeep
  tests/
    test_cli.py
    test_kalshi.py
    test_coinbase.py
    test_news.py
    test_kimiclaw.py
    test_features.py
    test_predictor.py
    test_backtest.py
```

---

## 14. Configuration
Use environment variables and a strongly typed config object.

Minimum `.env.example`:
```env
APP_ENV=dev
LOG_LEVEL=INFO
DB_PATH=./data/kalshi_btc.duckdb
MODEL_PATH=./models/baseline.pkl
NEWS_ARTICLE_LIMIT=20
KIMICLAW_BASE_URL=https://replace-me.example.com
KIMICLAW_API_KEY=replace-me
KIMICLAW_MODEL=replace-me
HTTP_TIMEOUT_SECONDS=20
```

Rules:
- fail fast on invalid config
- do not hardcode secrets
- allow CLI flags to override config where appropriate

---

## 15. Implementation Phases
Build in the following order.

### Phase 1: Project bootstrap
Deliverables:
- repo scaffold
- dependency setup with `uv`
- Typer CLI entrypoint
- config and logging
- lint/test tooling

Acceptance criteria:
- `btc-kalshi --help` works
- `pytest` runs
- `ruff check .` passes
- `mypy app` passes

### Phase 2: Kalshi live market discovery
Deliverables:
- discover active BTC hourly market
- normalize title and threshold
- fetch live market prices

Acceptance criteria:
- `btc-kalshi market` returns the active hourly BTC market
- parsed threshold matches market question semantics
- errors are handled gracefully when no market is live

### Phase 3: BTC data ingestion
Deliverables:
- current spot fetch
- candle fetch for multiple timeframes
- local caching

Acceptance criteria:
- current BTC spot can be fetched
- last 60 minutes of 1-minute data can be retrieved and stored
- duplicate inserts are avoided

### Phase 4: News ingestion
Deliverables:
- RSS ingestion
- GDELT ingestion
- normalization and de-duplication

Acceptance criteria:
- news command returns fresh BTC-related articles
- duplicate URLs are not stored twice
- stale articles outside configured lookback are excluded

### Phase 5: KimiClaw scoring
Deliverables:
- client abstraction
- prompt template
- structured JSON validation
- fallback behavior

Acceptance criteria:
- valid KimiClaw responses are parsed into typed models
- invalid responses are rejected and logged
- predictor still runs if KimiClaw is down

### Phase 6: Feature engineering and predictor
Deliverables:
- market prior features
- price action features
- news aggregate features
- baseline model inference

Acceptance criteria:
- `predict` produces ABOVE/BELOW with probability and explanation
- feature vector is logged in debug mode
- missing feature families degrade gracefully

### Phase 7: Backtesting and training
Deliverables:
- training dataset builder
- backtest runner
- metrics output

Acceptance criteria:
- train command persists a model artifact
- backtest command prints accuracy, log loss, and baseline comparison
- at least one naive baseline is implemented

### Phase 8: Polish and docs
Deliverables:
- README
- examples
- JSON export
- explain command

Acceptance criteria:
- first-time local setup is documented end-to-end
- README includes warning that tool is not for trade execution

---

## 16. Naive Baselines for Evaluation
Implement these baselines so the model has something meaningful to beat:

1. **Kalshi-follow baseline**
   - predict ABOVE when market-implied yes probability >= 0.5
2. **Spot-vs-strike baseline**
   - predict ABOVE when current spot >= strike
3. **Momentum baseline**
   - predict direction from recent 15m return
4. **No-news baseline**
   - same as full model but with news score forced neutral

Backtest reports must compare the full model against these baselines.

---

## 17. Model Requirements
Start with two simple models:
- logistic regression
- gradient boosting classifier

Choose the default model by validation performance and stability.

Model requirements:
- calibrate probabilities if useful
- persist model artifact locally
- save feature schema version
- fail clearly if model artifact is missing or stale

Do not build a black-box deep model in v1.

---

## 18. Explainability Requirements
The predictor must surface why it made a call.

At minimum, show:
- market prior contribution
- price model contribution
- news pressure contribution
- top 3 directional drivers

For linear models, coefficients can be mapped into a simple explanation.
For non-linear models, use feature importances or simple local contribution approximations.

---

## 19. Reliability and Error Handling
All external calls must include:
- timeout
- retry with backoff
- typed response validation
- logging with enough context to debug failures

The CLI must not crash on common external failures. It should exit cleanly with an understandable error message.

Required graceful-degradation cases:
- Kalshi discovery fails
- no active hourly BTC market exists
- Coinbase candle request partially fails
- news source returns empty results
- KimiClaw is unavailable
- model file missing

---

## 20. Security and Safety Controls
Even though this is a local tool, implement sensible controls.

Requirements:
- never store secrets in source code
- sanitize article text before sending to KimiClaw
- limit maximum article length sent to the model
- strip scripts/HTML
- reject untrusted local file execution
- never execute any code from article content
- never expose trade or execution flows
- always print a research-only disclaimer

---

## 21. Testing Requirements
Minimum tests required:

### Unit tests
- threshold parsing from Kalshi market titles
- candle normalization
- news de-duplication
- KimiClaw response validation
- feature generation
- blend logic

### Integration tests
- live market discovery with mocked Kalshi payloads
- live candle fetch with mocked Coinbase payloads
- end-to-end predict flow with mocked news and KimiClaw responses

### CLI tests
- `--help`
- `market`
- `predict --json`
- `news`
- `backtest`

Use fixtures for representative API responses.

---

## 22. Developer Experience Requirements
The repo must be pleasant for iterative coding.

Required:
- `make` or simple script aliases for common tasks
- `uv sync`
- `pytest`
- `ruff check .`
- `mypy app`
- `pre-commit install`

Optional but recommended:
- `justfile` or `Makefile`

---

## 23. README Requirements
The README must include:
- project purpose
- clear disclaimer
- architecture summary
- setup steps
- environment variables
- CLI examples
- model limitations
- data sources
- future improvements

---

## 24. Codex Execution Rules
Codex must follow these rules while building:

1. Do not add a frontend.
2. Do not add trade execution code.
3. Do not add unsupported paid-only dependencies.
4. Prefer small, typed, testable functions.
5. Keep services isolated behind interfaces.
6. Add tests alongside each feature.
7. Do not leave TODO placeholders for core logic.
8. Use real parsing and validation, not fragile regex-only hacks where structured data exists.
9. Keep output explainable.
10. Make the app usable from the CLI as early as possible.

---

## 25. Definition of Done
The project is done when all of the following are true:

- a user can install the project locally
- `btc-kalshi market` shows the active live BTC hourly Kalshi market
- `btc-kalshi predict` produces a research-only ABOVE/BELOW prediction for that live market
- the prediction uses live Kalshi market data, BTC price history, and recent news
- relevant news is scored by KimiClaw via structured JSON
- the system degrades gracefully when KimiClaw is unavailable
- `btc-kalshi backtest` works on historical windows
- there is a saved local model artifact
- tests pass
- lint/type checks pass
- README documents setup and limitations clearly

---

## 26. First Implementation Target
Codex should prioritize getting the following working first:

1. project scaffold
2. `market` command
3. `predict` command with market + BTC price only
4. news ingestion
5. KimiClaw scoring
6. baseline predictor blend
7. backtest and training

The earliest usable version should already answer:
- what is the live Kalshi BTC 1-hour question right now?
- what is BTC trading at right now?
- based on current features, is BTC more likely to finish above or below the threshold?

---

## 27. Future Enhancements After MVP
Do not implement these until the MVP is stable, but design for them:
- optional WebSocket mode for live refresh
- source reliability weighting for news publishers
- article clustering by event topic
- historical feature store improvements
- SHAP-based explainability
- alert mode for manual human review
- local model replacement for KimiClaw sentiment
- optional Discord message formatting for personal notifications without trading language

---

## 28. Final Instruction to Codex
Build this as a **clean, production-style local CLI application** with strong typing, strong validation, graceful degradation, and explainable output.

Bias toward:
- correctness
- clarity
- maintainability
- safe research-only behavior

Do not bias toward:
- overengineering
- excessive abstraction before it is needed
- autonomous trading logic
- flashy output over reliability

