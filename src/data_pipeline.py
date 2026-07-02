"""
Fixed data pipeline for autoresearch-trade experiments.
Downloads OHLCV data via yfinance, computes features, and provides
the frozen evaluation harness.

Analogous to prepare.py in the LLM autoresearch project.

Usage:
    from src.data_pipeline import load_data, compute_features, evaluate
"""

import os
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # seconds per experiment (wall clock)

# Date ranges for train / validation / test splits
TRAIN_START = "2010-01-01"
TRAIN_END = "2019-12-31"
VALIDATION_START = "2020-01-01"
VALIDATION_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2025-12-31"

DEFAULT_TICKER = "AAPL"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_trade")

SPLIT_RANGES = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "test": (TEST_START, TEST_END),
}

# ---------------------------------------------------------------------------
# Data download & caching
# ---------------------------------------------------------------------------


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker.upper()}_daily.parquet")


def _download(ticker: str) -> pd.DataFrame:
    """Download full daily OHLCV history and cache to disk."""
    path = _cache_path(ticker)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df

    print(f"[data_pipeline] Downloading {ticker} daily OHLCV via yfinance ...")
    raw = yf.download(ticker, start="2005-01-01", end=TEST_END, auto_adjust=True,
                      progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # Flatten multi-level columns if present (yfinance >= 0.2.36)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.dropna()
    df.to_parquet(path)
    print(f"[data_pipeline] Cached {len(df)} bars -> {path}")
    return df


def load_data(ticker: str = DEFAULT_TICKER, split: str = "train") -> pd.DataFrame:
    """Return a clean DataFrame for the requested split.

    Columns: open, high, low, close, volume (all numeric).
    Index: DatetimeIndex named 'date'.
    """
    if split not in SPLIT_RANGES:
        raise ValueError(f"Unknown split {split!r}; choose from {list(SPLIT_RANGES)}")
    start, end = SPLIT_RANGES[split]
    df = _download(ticker)
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    mask = (df.index >= start) & (df.index <= end)
    out = df.loc[mask].copy()
    if out.empty:
        raise RuntimeError(f"No data for {ticker} in split {split} ({start}..{end})")
    return out

# ---------------------------------------------------------------------------
# Feature helpers (no lookahead — uses only past / present bars)
# ---------------------------------------------------------------------------


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute common technical indicators.  All signals use only past/present
    data (shifted where necessary to avoid lookahead).

    Adds columns: returns, sma_10, sma_50, atr_14, volatility_20, rsi_14.
    Returns a *copy* — the original DataFrame is not modified.
    """
    out = df.copy()

    # Daily log-returns (available at the close of each bar)
    out["returns"] = np.log(out["close"] / out["close"].shift(1))

    # Simple moving averages (use close up to and including today)
    out["sma_10"] = out["close"].rolling(window=10, min_periods=10).mean()
    out["sma_50"] = out["close"].rolling(window=50, min_periods=50).mean()

    # ATR-14 (Average True Range)
    high = out["high"]
    low = out["low"]
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(window=14, min_periods=14).mean()

    # 20-day realised volatility (annualised, from daily log-returns)
    out["volatility_20"] = out["returns"].rolling(window=20, min_periods=20).std() * np.sqrt(252)

    # RSI-14
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return out

# ---------------------------------------------------------------------------
# Evaluation harness — DO NOT CHANGE — fixed metric
# ---------------------------------------------------------------------------


def evaluate(strategy_cls, ticker: str = DEFAULT_TICKER, split: str = "validation"):
    """Run the backtest for *strategy_cls* on *split* and return
    ``(metrics_dict, composite_score)``.

    This is the FIXED evaluation entry-point.  The agent must NOT modify it.
    It delegates to ``src.backtest.run_backtest`` which performs the actual
    backtrader simulation and metric computation.
    """
    from src.backtest import run_backtest  # local import to avoid circular deps
    return run_backtest(strategy_cls, ticker=ticker, split=split)


# ---------------------------------------------------------------------------
# Main (quick sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for split in ("train", "validation", "test"):
        df = load_data(split=split)
        print(f"{split:>12s}: {len(df):>5d} bars  ({df.index[0].date()} .. {df.index[-1].date()})")
    feat = compute_features(load_data(split="train"))
    print(f"\nFeature columns: {list(feat.columns)}")
    print("Data pipeline OK.")
