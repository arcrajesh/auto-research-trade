"""
Trading strategy — THE ONLY FILE THE AGENT EDITS.
Analogous to train.py in the LLM autoresearch project.

Implements a backtrader.Strategy subclass consumed by src/backtest.py.
The agent iterates on this file to maximise the composite score.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import backtrader as bt
import numpy as np

from src.data_pipeline import compute_features  # frozen utilities

# ---------------------------------------------------------------------------
# Baseline strategy: SMA crossover + ATR-based sizing + stop-loss
# ---------------------------------------------------------------------------

# --- Parameters (agent may tune these) ---
FAST_PERIOD = 10
SLOW_PERIOD = 50
ATR_PERIOD = 14
ATR_RISK_MULT = 2.0      # stop-loss distance = ATR * multiplier
RISK_PER_TRADE = 0.02    # risk 2 % of equity per trade


class TradingStrategy(bt.Strategy):
    """Moving-average crossover with ATR-based volatility position sizing
    and a trailing stop-loss.

    Sections:
        1. Signal generation  (SMA crossover)
        2. Position sizing    (ATR-scaled, risk-budget based)
        3. Risk management    (trailing stop)
    """

    params = dict(
        fast_period=FAST_PERIOD,
        slow_period=SLOW_PERIOD,
        atr_period=ATR_PERIOD,
        atr_risk_mult=ATR_RISK_MULT,
        risk_per_trade=RISK_PER_TRADE,
    )

    # ------------------------------------------------------------------ init
    def __init__(self):
        # 1. Signal generation — SMA crossover
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

        # ATR for position sizing and stop-loss
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # Internal state
        self.stop_price = None
        self.order = None

    # --------------------------------------------------------- order mgmt
    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Margin,
                            order.Rejected):
            self.order = None

    # ------------------------------------------------------------------ next
    def next(self):
        if self.order:
            return  # wait for pending order

        atr_val = self.atr[0]
        if atr_val <= 0:
            return  # guard against degenerate ATR

        # --- 3. Risk management: trailing stop ---
        if self.position:
            if self.position.size > 0:
                new_stop = self.data.close[0] - self.p.atr_risk_mult * atr_val
                if self.stop_price is None or new_stop > self.stop_price:
                    self.stop_price = new_stop
                if self.data.close[0] <= self.stop_price:
                    self.order = self.close()
                    self.stop_price = None
                    return

        # --- 1. Signal generation ---
        if not self.position:
            if self.crossover[0] > 0:  # fast crosses above slow → buy
                # --- 2. Position sizing ---
                risk_amount = self.broker.getvalue() * self.p.risk_per_trade
                stop_dist = self.p.atr_risk_mult * atr_val
                shares = int(risk_amount / stop_dist) if stop_dist > 0 else 0
                if shares > 0:
                    self.order = self.buy(size=shares)
                    self.stop_price = self.data.close[0] - stop_dist

        elif self.position.size > 0:
            if self.crossover[0] < 0:  # fast crosses below slow → sell
                self.order = self.close()
                self.stop_price = None
