"""
Fixed backtesting harness for autoresearch-trade.
Uses backtrader with realistic execution costs.

Analogous to the evaluation logic in prepare.py (LLM project).

The composite score and metric computation here are FIXED — do not modify.

Usage:
    python -m src.backtest          # run current strategy on validation split
    uv run src/backtest.py          # same, via uv
"""

import os
import sys

# Ensure the project root is on sys.path so `src.*` imports work
# regardless of how the script is invoked.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math
import datetime as dt

import numpy as np
import pandas as pd
import backtrader as bt

from src.data_pipeline import load_data, DEFAULT_TICKER

# ---------------------------------------------------------------------------
# Constants (fixed)
# ---------------------------------------------------------------------------

COMMISSION_PCT = 0.0005      # 0.05 %
SLIPPAGE_PCT = 0.0005        # 5 bps
INITIAL_CASH = 100_000.0
MAX_POSITION_PCT = 0.95      # max fraction of portfolio in a single position

# Composite-score parameters
MAX_DD_THRESHOLD = 0.25      # hard penalty kicks in above 25 % drawdown

# ---------------------------------------------------------------------------
# Backtrader helpers
# ---------------------------------------------------------------------------


class _PandasData(bt.feeds.PandasData):
    """Adapter so backtrader can consume our DataFrame directly."""
    params = (
        ("datetime", None),   # use the index
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class _TradeLogger(bt.Analyzer):
    """Collects per-trade stats needed for win-rate, R:R, holding period."""

    def __init__(self):
        super().__init__()
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                "pnl": trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "barlen": trade.barlen,
                "size": abs(trade.size),
            })

    def get_analysis(self):
        return self.trades


class _Turnover(bt.Analyzer):
    """Tracks total absolute dollar turnover."""

    def __init__(self):
        super().__init__()
        self.total_turnover = 0.0

    def notify_order(self, order):
        if order.status == order.Completed:
            self.total_turnover += abs(order.executed.size * order.executed.price)

    def get_analysis(self):
        return {"total_turnover": self.total_turnover}

# ---------------------------------------------------------------------------
# Metric computation (FIXED — do not change)
# ---------------------------------------------------------------------------


def _compute_metrics(cerebro_result, initial_cash: float, num_days: int):
    """Derive all required metrics from a completed backtrader run.

    Returns ``(metrics_dict, composite_score)``.
    """
    strat = cerebro_result[0]
    broker = strat.broker

    final_value = broker.getvalue()
    total_return = (final_value / initial_cash) - 1.0
    years = max(num_days / 252.0, 1e-9)
    ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0

    # --- Drawdown from the built-in DrawDown analyser ---
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.max.drawdown / 100.0 if dd_analysis.max.drawdown else 0.0

    # --- Sharpe (annualised, risk-free = 0) ---
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get("sharperatio", None)
    if sharpe is None:
        sharpe = 0.0

    # --- MAR ratio ---
    mar = ann_return / max_dd if max_dd > 1e-9 else float("inf")

    # --- Trade-level stats ---
    trades = strat.analyzers.tradelogger.get_analysis()
    trade_count = len(trades)
    if trade_count > 0:
        wins = [t for t in trades if t["pnlcomm"] > 0]
        losses = [t for t in trades if t["pnlcomm"] <= 0]
        win_rate = len(wins) / trade_count
        avg_win = np.mean([t["pnlcomm"] for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t["pnlcomm"] for t in losses])) if losses else 1e-9
        avg_rr = avg_win / avg_loss if avg_loss > 1e-9 else float("inf")
        avg_holding = np.mean([t["barlen"] for t in trades])
    else:
        win_rate = 0.0
        avg_rr = 0.0
        avg_holding = 0.0

    # --- Turnover (annualised as fraction of initial capital) ---
    turnover_raw = strat.analyzers.turnover.get_analysis()["total_turnover"]
    turnover = (turnover_raw / initial_cash) / years

    # --- Composite score (FIXED) ---
    # Higher is better.  Sharpe is the primary signal.
    # Hard penalty when max drawdown exceeds threshold.
    if max_dd <= MAX_DD_THRESHOLD:
        composite_score = sharpe
    else:
        composite_score = sharpe - 10.0 * (max_dd - MAX_DD_THRESHOLD)

    metrics = {
        "composite_score": round(composite_score, 6),
        "sharpe": round(sharpe, 6),
        "annualized_return": round(ann_return, 6),
        "max_drawdown": round(max_dd, 6),
        "mar": round(mar, 6),
        "win_rate": round(win_rate, 4),
        "avg_rr": round(avg_rr, 4),
        "avg_holding_days": round(avg_holding, 1),
        "trade_count": trade_count,
        "turnover": round(turnover, 4),
    }
    return metrics, composite_score

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(strategy_cls, ticker: str = DEFAULT_TICKER,
                 split: str = "validation"):
    """Execute *strategy_cls* (a ``backtrader.Strategy`` subclass) on the
    requested data split with realistic execution costs.

    Returns ``(metrics_dict, composite_score)``.
    """
    df = load_data(ticker=ticker, split=split)
    num_days = len(df)

    cerebro = bt.Cerebro()

    # Feed
    data_feed = _PandasData(dataname=df)
    cerebro.adddata(data_feed)

    # Strategy
    cerebro.addstrategy(strategy_cls)

    # Broker settings
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION_PCT)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=MAX_POSITION_PCT * 100)

    # Slippage
    cerebro.broker.set_slippage_perc(SLIPPAGE_PCT)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        timeframe=bt.TimeFrame.Days, annualize=True,
                        riskfreerate=0.0)
    cerebro.addanalyzer(_TradeLogger, _name="tradelogger")
    cerebro.addanalyzer(_Turnover, _name="turnover")

    result = cerebro.run()
    return _compute_metrics(result, INITIAL_CASH, num_days)


def print_summary(metrics: dict, composite_score: float):
    """Print the standard summary block (mirrors program.md output format)."""
    print("---")
    print(f"composite_score:   {composite_score:.6f}")
    print(f"sharpe:            {metrics['sharpe']:.6f}")
    print(f"annualized_return: {metrics['annualized_return']:.6f}")
    print(f"max_drawdown:      {metrics['max_drawdown']:.6f}")
    print(f"mar:               {metrics['mar']:.6f}")
    print(f"win_rate:          {metrics['win_rate']:.4f}")
    print(f"avg_rr:            {metrics['avg_rr']:.4f}")
    print(f"avg_holding_days:  {metrics['avg_holding_days']:.1f}")
    print(f"trade_count:       {metrics['trade_count']}")
    print(f"turnover:          {metrics['turnover']:.4f}")


# ---------------------------------------------------------------------------
# Main — run the current strategy on the validation split
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.strategy import TradingStrategy

    print(f"Running backtest on validation split (ticker={DEFAULT_TICKER}) ...")
    metrics, score = run_backtest(TradingStrategy, ticker=DEFAULT_TICKER,
                                  split="validation")
    print()
    print_summary(metrics, score)
