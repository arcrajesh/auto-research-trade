# autoresearch-trade

Autonomous research for trading strategies, following Karpathy's autoresearch
pattern: fixed environment + single mutable file + human-edited `program.md` +
single scalar metric + keep/discard loop.

## Layout

| Trading project          | LLM project (original) | Role                                |
|--------------------------|------------------------|-------------------------------------|
| `src/data_pipeline.py`   | `prepare.py`           | Fixed data loading, features, eval  |
| `src/backtest.py`        | eval in `prepare.py`   | Fixed metric & harness              |
| `src/strategy.py`        | `train.py`             | **Mutable** — agent edits this      |
| `src/program.md`         | `program.md`           | Human-edited agent instructions     |

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (from the repo root)
uv sync
```

---

## Step-by-Step: Running the Project

### Step 1 — Verify the data pipeline

Download and cache OHLCV data for the default ticker (AAPL):

```bash
uv run src/data_pipeline.py
```

Expected output:

```
       train:  2516 bars  (2010-01-04 .. 2019-12-31)
  validation:   756 bars  (2020-01-02 .. 2022-12-30)
        test:   503 bars  (2023-01-03 .. 2025-01-02)

Feature columns: ['open', 'high', 'low', 'close', 'volume', 'returns', 'sma_10', 'sma_50', 'atr_14', 'volatility_20', 'rsi_14']
Data pipeline OK.
```

Data is cached in `~/.cache/autoresearch_trade/` so subsequent runs are instant.

### Step 2 — Run the baseline backtest

Run the current strategy on the **validation split** (2020–2022):

```bash
uv run src/backtest.py
```

Expected output:

```
Running backtest on validation split (ticker=AAPL) ...

---
composite_score:   1.032682
sharpe:            1.032682
annualized_return: 0.060555
max_drawdown:      0.065482
mar:               0.924759
win_rate:          0.5000
avg_rr:            3.2433
avg_holding_days:  19.7
trade_count:       10
turnover:          2.9508
```

You can also run it as a module:

```bash
uv run python -m src.backtest
```

### Step 3 — Understand the metrics

| Metric | Meaning |
|--------|---------|
| `composite_score` | **Primary optimisation target.** `sharpe` if `max_dd ≤ 25%`, else `sharpe - 10*(max_dd - 0.25)`. Higher = better. |
| `sharpe` | Annualised Sharpe ratio (risk-free = 0). |
| `annualized_return` | Geometric annualised return. |
| `max_drawdown` | Largest peak-to-trough decline (as a fraction, e.g. 0.15 = 15%). |
| `mar` | MAR ratio = annualised return / max drawdown. |
| `win_rate` | Fraction of trades that were profitable. |
| `avg_rr` | Average reward-to-risk ratio (avg win / avg loss). |
| `avg_holding_days` | Mean number of bars a position is held. |
| `trade_count` | Total closed trades in the split. |
| `turnover` | Annualised dollar turnover as a multiple of initial capital. |

---

## Step-by-Step: Finding a New Strategy

The autoresearch loop lets you (or an AI agent) iterate on `src/strategy.py` to find better strategies.

### Step 1 — Create an experiment branch

```bash
git checkout -b autoresearch/jun30
```

### Step 2 — Run the baseline (first experiment)

```bash
uv run src/backtest.py > run.log 2>&1
grep "^composite_score:" run.log
```

Record the baseline result.

### Step 3 — Edit `src/strategy.py`

Modify the strategy — change indicators, entry/exit logic, parameters, position sizing, or risk rules. For example, add an RSI filter:

```python
# In the next() method, before the buy signal:
if not self.position:
    if self.crossover[0] > 0 and self.data.close[0] > self.sma_slow[0]:
        # Only buy when RSI < 70 (not overbought)
        rsi = ...  # add RSI indicator in __init__
        if rsi[0] < 70:
            ...
```

### Step 4 — Commit and test

```bash
git add src/strategy.py
git commit -m "add RSI overbought filter"
uv run src/backtest.py > run.log 2>&1
grep "^composite_score:\|^sharpe:\|^max_drawdown:\|^trade_count:" run.log
```

### Step 5 — Keep or discard

- **If `composite_score` improved** (higher) → keep the commit, advance the branch.
- **If equal or worse** → revert: `git reset --hard HEAD~1`

### Step 6 — Log results

Append to `results.tsv` (do NOT commit this file):

```
commit	composite_score	sharpe	max_drawdown	trade_count	status	description
a1b2c3d	1.032682	1.032682	0.065482	10	keep	baseline
b2c3d4e	1.150000	1.150000	0.072000	12	keep	add RSI < 70 filter
```

### Step 7 — Repeat

Go back to Step 3. The loop continues until you're satisfied or an agent is stopped.

---

## Step-by-Step: Backtesting Across Multiple Stocks

The harness supports any ticker available on Yahoo Finance. Here's how to evaluate your strategy across a portfolio of stocks.

### Option A — Quick shell loop

```bash
# Define your stock list
TICKERS="AAPL MSFT GOOGL AMZN TSLA META NVDA"

# Run backtest for each ticker on the validation split
for TICKER in $TICKERS; do
    echo "=== $TICKER ==="
    uv run python -c "
from src.backtest import run_backtest, print_summary
from src.strategy import TradingStrategy
metrics, score = run_backtest(TradingStrategy, ticker='$TICKER', split='validation')
print_summary(metrics, score)
"
    echo ""
done
```

### Option B — Python script for structured comparison

Create a script (e.g. `run_portfolio.py` in the repo root):

```python
"""Backtest the current strategy across multiple tickers and summarise."""
import sys
sys.path.insert(0, ".")

from src.backtest import run_backtest, print_summary
from src.strategy import TradingStrategy

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
SPLIT = "validation"  # or "test" for final evaluation

results = []
for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"  {ticker} ({SPLIT})")
    print(f"{'='*60}")
    try:
        metrics, score = run_backtest(TradingStrategy, ticker=ticker, split=SPLIT)
        print_summary(metrics, score)
        results.append({"ticker": ticker, **metrics})
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"ticker": ticker, "composite_score": None})

# --- Summary table ---
print(f"\n{'='*60}")
print("  PORTFOLIO SUMMARY")
print(f"{'='*60}")
print(f"{'Ticker':<8} {'Score':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
print("-" * 45)
for r in results:
    if r.get("composite_score") is not None:
        print(f"{r['ticker']:<8} {r['composite_score']:>10.4f} {r['sharpe']:>8.4f} "
              f"{r['max_drawdown']:>8.4f} {r['trade_count']:>7}")
    else:
        print(f"{r['ticker']:<8} {'ERROR':>10}")

# Average composite score across successful tickers
scores = [r["composite_score"] for r in results if r.get("composite_score") is not None]
if scores:
    print(f"\nAverage composite_score: {sum(scores)/len(scores):.4f}")
    print(f"Median composite_score:  {sorted(scores)[len(scores)//2]:.4f}")
```

Run it:

```bash
uv run python run_portfolio.py
```

### Option C — Compare on the test split (final evaluation)

Once your strategy is mature, evaluate on the held-out **test** split:

```bash
uv run python -c "
from src.backtest import run_backtest, print_summary
from src.strategy import TradingStrategy
metrics, score = run_backtest(TradingStrategy, ticker='AAPL', split='test')
print_summary(metrics, score)
"
```

---

## Full Worked Example

Here's an end-to-end session showing the autoresearch workflow:

```bash
# 1. Setup
cd auto-research-trade
uv sync
git checkout -b autoresearch/example

# 2. Establish baseline
uv run src/backtest.py > run.log 2>&1
grep "^composite_score:" run.log
# → composite_score:   1.032682

# 3. Try an idea: tighten the stop-loss (1.5x ATR instead of 2x)
sed -i 's/ATR_RISK_MULT = 2.0/ATR_RISK_MULT = 1.5/' src/strategy.py
git add src/strategy.py && git commit -m "tighten stop to 1.5x ATR"

# 4. Run experiment
uv run src/backtest.py > run.log 2>&1
grep "^composite_score:" run.log
# → composite_score:   0.890000  (worse!)

# 5. Discard — revert to baseline
git reset --hard HEAD~1

# 6. Try another idea: widen the slow SMA to 60
sed -i 's/SLOW_PERIOD = 50/SLOW_PERIOD = 60/' src/strategy.py
git add src/strategy.py && git commit -m "widen slow SMA to 60"

# 7. Run experiment
uv run src/backtest.py > run.log 2>&1
grep "^composite_score:" run.log
# → composite_score:   1.150000  (better!)

# 8. Keep! The branch has advanced.

# 9. Evaluate across multiple stocks
for T in AAPL MSFT GOOGL; do
    echo "--- $T ---"
    uv run python -c "
from src.backtest import run_backtest, print_summary
from src.strategy import TradingStrategy
m, s = run_backtest(TradingStrategy, ticker='$T', split='validation')
print(f'composite_score: {s:.4f}')
"
done
```

---

## Data splits

| Split      | Date range                | Purpose |
|------------|---------------------------|---------|
| Train      | 2010-01-01 → 2019-12-31   | Feature exploration, hypothesis generation |
| Validation | 2020-01-01 → 2022-12-31   | **Optimisation target** — the agent loop runs here |
| Test       | 2023-01-01 → 2025-12-31   | Held-out final evaluation (do not optimise on this) |

---

## Key Design Principles

1. **Single file to modify** — Only `src/strategy.py` changes. Everything else is frozen.
2. **No lookahead bias** — Signals use only past/present data. The `compute_features()` helper and backtrader indicators enforce this.
3. **Realistic execution** — Commission (0.05%), slippage (5 bps), and position limits mirror real brokerage conditions.
4. **One scalar metric** — `composite_score` is the sole optimisation target. It rewards high Sharpe and penalises excessive drawdown.
5. **Deterministic** — Same strategy + same data = same results. No randomness in the backtest.
6. **Cached data** — Downloads are cached in `~/.cache/autoresearch_trade/`. Delete the cache to force a re-download.

---

## Available Tickers

Any ticker supported by Yahoo Finance works. Common examples:

| Category | Tickers |
|----------|---------|
| US Large Cap | AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, BRK-B |
| US Indices (ETFs) | SPY, QQQ, IWM, DIA |
| Sectors | XLF, XLE, XLK, XLV, XBI |
| International | EWJ (Japan), FXI (China), EWZ (Brazil) |
| Commodities | GLD (Gold), SLV (Silver), USO (Oil) |
| Crypto | BTC-USD, ETH-USD |
| Bonds | TLT, IEF, SHY |

---

## Step-by-Step: Live Trading with Alpaca API

The `src/live.py` module lets you deploy your backtested strategy to paper or live markets via [Alpaca](https://alpaca.markets/).

### Step 1 — Get Alpaca API keys

1. Sign up at [https://alpaca.markets](https://alpaca.markets) (free).
2. Go to **Paper Trading** → **API Keys** → Generate a new key pair.
3. Set environment variables:

```bash
export ALPACA_API_KEY="your-api-key-here"
export ALPACA_SECRET_KEY="your-secret-key-here"
```

> **Never commit these keys!** Use a `.env` file or your shell profile.

### Step 2 — Dry run (see signals without placing orders)

```bash
uv run src/live.py --ticker AAPL --dry-run
```

Example output:

```
============================================================
  autoresearch-trade LIVE (PAPER)
  2025-06-30 16:05:00
  Account equity: $100,000.00
  Tickers: AAPL
  MODE: DRY RUN (no orders will be placed)
============================================================

[AAPL]
  Signal: BUY | Close: $195.20 | SMA(10): $193.50 | SMA(50): $190.80 | ATR: $3.42
  Reason: SMA(10) crossed above SMA(50)
  [AAPL] DRY RUN: Would BUY 292 shares @ ~$195.20
           Stop-loss: $188.36
```

### Step 3 — Paper trade (simulated orders, no real money)

```bash
# Single ticker
uv run src/live.py --ticker AAPL

# Multiple tickers
uv run src/live.py --ticker AAPL MSFT GOOGL TSLA NVDA
```

### Step 4 — Run continuously (daily at market close)

```bash
uv run src/live.py --ticker AAPL MSFT --loop
```

This will:
1. Compute signals at ~4:05 PM ET each market day.
2. Execute BUY/SELL/HOLD for each ticker.
3. Sleep until the next market close.
4. Repeat forever (Ctrl+C to stop).

### Step 5 — Go live (real money)

```bash
# Switch to live API keys first!
export ALPACA_API_KEY="your-LIVE-key"
export ALPACA_SECRET_KEY="your-LIVE-secret"

uv run src/live.py --ticker AAPL --live
```

> You get a 5-second countdown warning before live orders are placed.

### How it works (architecture)

```
┌─────────────────────────────────────────────────────────┐
│                    src/live.py                           │
├─────────────────────────────────────────────────────────┤
│  1. Fetch latest bars (yfinance)                        │
│  2. Compute indicators (same as strategy.py)            │
│     - SMA(10), SMA(50) crossover detection              │
│     - ATR(14) for position sizing & stop-loss           │
│  3. Generate signal: BUY / SELL / HOLD                  │
│  4. Check current Alpaca position                       │
│  5. Execute via Alpaca Trading API                      │
│     - Market orders with day time-in-force              │
│     - Position sizing: 2% equity risk per trade         │
│     - Trailing stop based on ATR                        │
└─────────────────────────────────────────────────────────┘
```

### CLI reference

```
uv run src/live.py [OPTIONS]

Options:
  --ticker TICKER [TICKER ...]   Symbols to trade (default: AAPL)
  --live                         Use real money (default: paper)
  --dry-run                      Show signals without placing orders
  --loop                         Run continuously at market close
```

### Example: Full workflow from research to live

```bash
# 1. Backtest and iterate (research phase)
git checkout -b autoresearch/my-strategy
# ... edit strategy.py, run backtest, keep/discard ...

# 2. Validate on test split (out-of-sample)
uv run python -c "
from src.backtest import run_backtest, print_summary
from src.strategy import TradingStrategy
for t in ['AAPL', 'MSFT', 'GOOGL']:
    m, s = run_backtest(TradingStrategy, ticker=t, split='test')
    print(f'{t}: score={s:.4f}, sharpe={m[\"sharpe\"]:.4f}, dd={m[\"max_drawdown\"]:.4f}')
"

# 3. Paper trade for a week to verify execution
export ALPACA_API_KEY="pk_paper_xxx"
export ALPACA_SECRET_KEY="sk_paper_xxx"
uv run src/live.py --ticker AAPL MSFT GOOGL --loop

# 4. Review paper results in Alpaca dashboard
#    https://app.alpaca.markets/paper/dashboard/overview

# 5. Go live when confident
export ALPACA_API_KEY="pk_live_xxx"
export ALPACA_SECRET_KEY="sk_live_xxx"
uv run src/live.py --ticker AAPL MSFT GOOGL --live --loop
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the repo root: `cd auto-research-trade && uv run src/backtest.py` |
| `RuntimeError: yfinance returned no data` | Check ticker spelling; ensure internet access; delete cache and retry |
| `No data for TICKER in split ...` | The ticker may not have data for the full date range (e.g. IPO after 2010) |
| Very few trades | Widen the signal sensitivity or shorten the slow period |
| `max_drawdown > 0.25` penalty | Add tighter stop-losses or reduce position size |
