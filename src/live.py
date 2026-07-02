"""
Live trading execution via Alpaca API.

Applies the same strategy logic from src/strategy.py to live/paper markets.
Runs once per invocation (designed for daily cron) or in a continuous loop.

Usage:
    # Paper trading (default):
    uv run src/live.py --ticker AAPL

    # Live trading (real money):
    uv run src/live.py --ticker AAPL --live

    # Multiple tickers:
    uv run src/live.py --ticker AAPL MSFT GOOGL

    # Continuous mode (runs every market day at close):
    uv run src/live.py --ticker AAPL --loop

Environment variables required:
    ALPACA_API_KEY        Your Alpaca API key
    ALPACA_SECRET_KEY     Your Alpaca secret key

Set these in your shell or in a .env file (never commit secrets!).
"""

import os
import sys
import time
import argparse
import datetime as dt
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    GetAssetsRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.common.exceptions import APIError

from src.data_pipeline import load_data, compute_features, DEFAULT_TICKER

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Strategy parameters (must match src/strategy.py)
FAST_PERIOD = 10
SLOW_PERIOD = 50
ATR_PERIOD = 14
ATR_RISK_MULT = 2.0
RISK_PER_TRADE = 0.02

# Live trading settings
LOOKBACK_DAYS = 200  # bars of history needed for indicators
MAX_POSITION_PCT = 0.95  # max portfolio fraction per ticker (multi-ticker: divided)

# ---------------------------------------------------------------------------
# Alpaca client setup
# ---------------------------------------------------------------------------


def get_client(paper: bool = True) -> TradingClient:
    """Create an authenticated Alpaca TradingClient."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        print("  For paper trading: use your paper account keys from https://app.alpaca.markets/paper/dashboard/overview")
        print("  For live trading:  use your live account keys from https://app.alpaca.markets/live/dashboard/overview")
        sys.exit(1)

    client = TradingClient(api_key, secret_key, paper=paper)
    return client


# ---------------------------------------------------------------------------
# Market data (uses yfinance for simplicity; Alpaca data API is an alternative)
# ---------------------------------------------------------------------------


def get_latest_bars(ticker: str, lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Download recent daily bars for signal computation.

    Uses yfinance for simplicity. In production, you might prefer
    Alpaca's market data API for lower latency.
    """
    end = dt.date.today()
    # Fetch extra days to account for weekends/holidays
    start = end - dt.timedelta(days=int(lookback * 1.5))

    raw = yf.download(ticker, start=str(start), end=str(end + dt.timedelta(days=1)),
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No recent data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.dropna()
    return df.tail(lookback)


# ---------------------------------------------------------------------------
# Signal generation (mirrors strategy.py logic)
# ---------------------------------------------------------------------------


def compute_signal(df: pd.DataFrame) -> dict:
    """Compute the current trading signal from the latest bars.

    Returns a dict with:
        signal: "BUY", "SELL", or "HOLD"
        sma_fast: current fast SMA value
        sma_slow: current slow SMA value
        atr: current ATR value
        close: latest close price
        stop_price: suggested stop-loss price (for BUY signals)
    """
    if len(df) < SLOW_PERIOD + 1:
        return {"signal": "HOLD", "reason": "insufficient data"}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # SMA crossover
    sma_fast = pd.Series(close).rolling(FAST_PERIOD).mean().values
    sma_slow = pd.Series(close).rolling(SLOW_PERIOD).mean().values

    # ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = pd.Series(tr).rolling(ATR_PERIOD).mean().values

    # Current values
    current_fast = sma_fast[-1]
    current_slow = sma_slow[-1]
    prev_fast = sma_fast[-2]
    prev_slow = sma_slow[-2]
    current_atr = atr[-1]
    current_close = close[-1]

    # Crossover detection
    cross_up = (prev_fast <= prev_slow) and (current_fast > current_slow)
    cross_down = (prev_fast >= prev_slow) and (current_fast < current_slow)

    result = {
        "sma_fast": round(current_fast, 2),
        "sma_slow": round(current_slow, 2),
        "atr": round(current_atr, 2),
        "close": round(current_close, 2),
    }

    if cross_up:
        stop_price = current_close - ATR_RISK_MULT * current_atr
        result.update({
            "signal": "BUY",
            "reason": f"SMA({FAST_PERIOD}) crossed above SMA({SLOW_PERIOD})",
            "stop_price": round(stop_price, 2),
        })
    elif cross_down:
        result.update({
            "signal": "SELL",
            "reason": f"SMA({FAST_PERIOD}) crossed below SMA({SLOW_PERIOD})",
        })
    else:
        trend = "bullish" if current_fast > current_slow else "bearish"
        result.update({
            "signal": "HOLD",
            "reason": f"no crossover (trend: {trend})",
        })

    return result


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def compute_position_size(client: TradingClient, ticker: str,
                          atr: float, num_tickers: int = 1) -> int:
    """Compute number of shares to buy based on risk budget."""
    account = client.get_account()
    equity = float(account.equity)

    # Risk budget per trade
    risk_amount = equity * RISK_PER_TRADE
    stop_distance = ATR_RISK_MULT * atr

    if stop_distance <= 0:
        return 0

    shares = int(risk_amount / stop_distance)

    # Cap at max position size (divided across tickers)
    max_dollar = (equity * MAX_POSITION_PCT) / num_tickers
    price = atr * 10  # rough estimate; actual price used below
    # Re-fetch actual price
    bars = get_latest_bars(ticker, lookback=5)
    if not bars.empty:
        price = bars["close"].iloc[-1]
    max_shares = int(max_dollar / price) if price > 0 else 0

    return min(shares, max_shares)


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------


def execute_signal(client: TradingClient, ticker: str, signal: dict,
                   num_tickers: int = 1, dry_run: bool = False):
    """Execute a trading signal via Alpaca.

    Args:
        client: Authenticated TradingClient.
        ticker: Stock symbol.
        signal: Output from compute_signal().
        num_tickers: Total tickers being traded (for position sizing).
        dry_run: If True, print what would happen without executing.
    """
    action = signal["signal"]

    # Check current position
    try:
        position = client.get_open_position(ticker)
        has_position = True
        current_qty = int(float(position.qty))
    except APIError:
        has_position = False
        current_qty = 0

    if action == "BUY" and not has_position:
        shares = compute_position_size(client, ticker, signal["atr"], num_tickers)
        if shares <= 0:
            print(f"  [{ticker}] BUY signal but position size = 0 (skip)")
            return

        if dry_run:
            print(f"  [{ticker}] DRY RUN: Would BUY {shares} shares @ ~${signal['close']}")
            print(f"           Stop-loss: ${signal.get('stop_price', 'N/A')}")
            return

        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_data=order_data)
        print(f"  [{ticker}] BUY {shares} shares — order {order.id} ({order.status})")

    elif action == "SELL" and has_position:
        if dry_run:
            print(f"  [{ticker}] DRY RUN: Would SELL {current_qty} shares (close position)")
            return

        client.close_position(ticker)
        print(f"  [{ticker}] SELL (closed position of {current_qty} shares)")

    elif action == "HOLD":
        print(f"  [{ticker}] HOLD — {signal['reason']}")
        if has_position:
            # Check trailing stop
            stop_price = signal["close"] - ATR_RISK_MULT * signal["atr"]
            if signal["close"] <= stop_price:
                if dry_run:
                    print(f"  [{ticker}] DRY RUN: Would SELL (trailing stop hit)")
                    return
                client.close_position(ticker)
                print(f"  [{ticker}] TRAILING STOP triggered — closed {current_qty} shares")

    else:
        if action == "BUY" and has_position:
            print(f"  [{ticker}] Already in position ({current_qty} shares), skip BUY")
        elif action == "SELL" and not has_position:
            print(f"  [{ticker}] No position to sell, skip")


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------


def run_once(tickers: list[str], paper: bool = True, dry_run: bool = False):
    """Run one iteration of signal generation + execution for all tickers."""
    client = get_client(paper=paper)
    account = client.get_account()

    print(f"\n{'='*60}")
    print(f"  autoresearch-trade LIVE {'(PAPER)' if paper else '(REAL)'}")
    print(f"  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Account equity: ${float(account.equity):,.2f}")
    print(f"  Tickers: {', '.join(tickers)}")
    if dry_run:
        print(f"  MODE: DRY RUN (no orders will be placed)")
    print(f"{'='*60}\n")

    for ticker in tickers:
        print(f"[{ticker}]")
        try:
            df = get_latest_bars(ticker)
            signal = compute_signal(df)
            print(f"  Signal: {signal['signal']} | Close: ${signal['close']} | "
                  f"SMA({FAST_PERIOD}): ${signal['sma_fast']} | "
                  f"SMA({SLOW_PERIOD}): ${signal['sma_slow']} | "
                  f"ATR: ${signal['atr']}")
            print(f"  Reason: {signal['reason']}")
            execute_signal(client, ticker, signal, num_tickers=len(tickers),
                           dry_run=dry_run)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


def wait_for_next_market_close():
    """Sleep until ~5 min after market close (4:05 PM ET).

    Simplified: waits until 16:05 ET on the next weekday.
    For production, use the Alpaca calendar API.
    """
    import pytz
    et = pytz.timezone("US/Eastern")

    while True:
        now = dt.datetime.now(et)
        # Target: 4:05 PM ET today (or next weekday)
        target = now.replace(hour=16, minute=5, second=0, microsecond=0)

        if now >= target or now.weekday() >= 5:
            # Already past close today, or weekend — advance to next weekday
            days_ahead = 1
            if now.weekday() == 4:  # Friday after close → Monday
                days_ahead = 3
            elif now.weekday() == 5:  # Saturday → Monday
                days_ahead = 2
            elif now.weekday() == 6:  # Sunday → Monday
                days_ahead = 1
            target += dt.timedelta(days=days_ahead)

        sleep_seconds = (target - now).total_seconds()
        if sleep_seconds > 0:
            print(f"Sleeping until {target.strftime('%Y-%m-%d %H:%M %Z')} "
                  f"({sleep_seconds/3600:.1f} hours)...")
            time.sleep(sleep_seconds)
            return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Live/paper trading using the autoresearch-trade strategy via Alpaca API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no orders, just signals):
  uv run src/live.py --ticker AAPL --dry-run

  # Paper trading single stock:
  uv run src/live.py --ticker AAPL

  # Paper trading multiple stocks:
  uv run src/live.py --ticker AAPL MSFT GOOGL TSLA

  # Live trading (real money!):
  uv run src/live.py --ticker AAPL --live

  # Continuous mode (runs daily at market close):
  uv run src/live.py --ticker AAPL MSFT --loop

Environment:
  ALPACA_API_KEY      Your Alpaca API key
  ALPACA_SECRET_KEY   Your Alpaca secret key

Get keys at: https://app.alpaca.markets/paper/dashboard/overview
""",
    )
    parser.add_argument("--ticker", nargs="+", default=[DEFAULT_TICKER],
                        help="One or more ticker symbols (default: AAPL)")
    parser.add_argument("--live", action="store_true",
                        help="Use LIVE trading (real money). Default is paper trading.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute signals but don't place any orders.")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously, executing once per market day at close.")

    args = parser.parse_args()
    paper = not args.live

    if args.live and not args.dry_run:
        print("\n⚠️  WARNING: You are about to trade with REAL MONEY!")
        print("    Press Ctrl+C within 5 seconds to abort...\n")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    if args.loop:
        print("Starting continuous trading loop...")
        print(f"  Tickers: {', '.join(args.ticker)}")
        print(f"  Mode: {'PAPER' if paper else 'LIVE'}")
        print(f"  Press Ctrl+C to stop.\n")
        while True:
            try:
                run_once(args.ticker, paper=paper, dry_run=args.dry_run)
                wait_for_next_market_close()
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
    else:
        run_once(args.ticker, paper=paper, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
