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

## How it works

1. **`data_pipeline.py`** downloads daily OHLCV via `yfinance` (default: AAPL),
   caches to `~/.cache/autoresearch_trade/`, and provides `load_data(ticker, split)`
   plus `compute_features(df)` for common indicators (returns, SMAs, ATR, RSI).
   It also exposes the frozen evaluation entry-point `evaluate()`.

2. **`backtest.py`** runs a `backtrader` simulation with realistic execution costs
   (commission, slippage, position-size limits) and computes metrics: annualized
   return, Sharpe, max drawdown, MAR, win rate, average R:R, holding period,
   trade count, turnover. These collapse into **one composite scalar**:

   ```
   composite_score = sharpe                               if max_dd <= 0.25
   composite_score = sharpe - 10 * (max_dd - 0.25)        otherwise
   ```

   Higher is better.

3. **`strategy.py`** is the ONLY file the agent edits. It implements a
   `backtrader.Strategy` subclass with clearly separated signal generation,
   position sizing, and risk management.

4. **`program.md`** is the human-edited constitution that tells the agent what
   it can/cannot do, the metric to optimise, and the keep/discard loop protocol.

## Quick start

```bash
# Install dependencies
uv sync

# Run the baseline strategy on the validation split
uv run src/backtest.py
```

## Data splits

| Split      | Date range                |
|------------|---------------------------|
| Train      | 2010-01-01 → 2019-12-31   |
| Validation | 2020-01-01 → 2022-12-31   |
| Test       | 2023-01-01 → 2025-12-31   |

The agent optimises on validation. Test is held out for final evaluation.
