"""Price data, causal features, time-ordered splits, and train-only normalization.

Daily prices come from Yahoo Finance's public chart endpoint, which returns the raw
close and a dividend-and-split adjusted close for every trading day. Only the adjusted
close is used, so a day's return includes the dividend paid that day. Stooq, the other
source named in the Part 1 plan, now sits behind a JavaScript browser check that a
script cannot pass, so Yahoo is the source here.

Every feature for day t is built from closes on or before t, and the reward for a
decision made at the close of day t is the return from t to t+1. The data is cut into
time blocks (training, validation, test, and an earlier out-of-sample window) so that
both the decision day and the return day of every step fall inside one block. The
statistics that standardize the features are fit on the training block alone and then
applied unchanged to every other block.
"""

import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")


def _check_ticker(ticker):
    """Reject anything that is not a plain ticker symbol, so a crafted value cannot
    traverse the filesystem through the CSV path or be injected into the request URL."""
    if not _TICKER_RE.match(ticker or ""):
        raise ValueError("invalid ticker %r: use only letters, digits, '.' and '-'" % (ticker,))
    return ticker

# Decision-date ranges for each block (inclusive). The earlier window is a stress test
# that reaches back to the 2008 crash, a regime the training block never saw.
BLOCKS = {
    "earlier": ("2005-01-01", "2009-12-31"),
    "train": ("2010-01-01", "2018-12-31"),
    "val": ("2019-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2023-12-31"),
}
DOWNLOAD_START = "2003-01-01"   # leaves two years of warm-up before the earlier window
DOWNLOAD_END = "2024-01-01"

MA_SHORT, MA_LONG = 20, 100     # moving-average ratio used as a state feature
RSI_DAYS = 14
VOL_DAYS = 20


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(ticker):
    """Fetch daily raw and adjusted closes for a ticker and save them to data/raw."""
    _check_ticker(ticker)
    p1 = int(datetime.strptime(DOWNLOAD_START, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.strptime(DOWNLOAD_END, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d"
           "&interval=1d&events=div%%2Csplit&includeAdjustedClose=true" % (ticker, p1, p2))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    res = payload["chart"]["result"][0]
    stamps = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York")
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": stamps.strftime("%Y-%m-%d"),
        "close": quote["close"],
        "adj_close": res["indicators"]["adjclose"][0]["adjclose"],
        "volume": quote["volume"],
    }).dropna().drop_duplicates("date").reset_index(drop=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, "%s.csv" % ticker)
    df.to_csv(path, index=False)
    return path, len(df)


def load_prices(ticker):
    """Read the saved daily prices for a ticker, downloading them first if absent."""
    _check_ticker(ticker)
    path = os.path.join(RAW_DIR, "%s.csv" % ticker)
    if not os.path.exists(path):
        download(ticker)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _rsi(close, days):
    """Wilder's relative strength index, rescaled from [0, 100] to [-1, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / days, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / days, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    return (rsi.fillna(100.0) - 50.0) / 50.0


def build_features(prices, window):
    """Return a frame of causal features plus the next-day return each decision earns.

    Columns ret_0 .. ret_{window-1} hold the most recent daily log-returns (ret_0 is
    day t itself), followed by the log ratio of the short to the long moving average,
    the rescaled RSI, and the annualized realized volatility. next_ret is the simple
    return from the close of t to the close of t+1 and is never part of the state.
    Rows without a full look-back are dropped.
    """
    close = prices["adj_close"].astype(float)
    logret = np.log(close).diff()
    feats = {"ret_%d" % k: logret.shift(k) for k in range(window)}
    feats["ma_ratio"] = np.log(close.rolling(MA_SHORT).mean() / close.rolling(MA_LONG).mean())
    feats["rsi"] = _rsi(close, RSI_DAYS)
    feats["vol"] = logret.rolling(VOL_DAYS).std() * np.sqrt(252.0)
    df = pd.DataFrame(feats)
    df.insert(0, "date", prices["date"])
    df["close"] = close
    df["next_ret"] = close.shift(-1) / close - 1.0
    # Slow moving averages for the crossover baseline (not used by the agent).
    df["sma50"] = close.rolling(50).mean()
    df["sma200"] = close.rolling(200).mean()
    return df.dropna(subset=[c for c in df.columns if c not in ("next_ret",)]).reset_index(drop=True)


def feature_columns(window):
    return ["ret_%d" % k for k in range(window)] + ["ma_ratio", "rsi", "vol"]


# ---------------------------------------------------------------------------
# Splits and normalization
# ---------------------------------------------------------------------------
def block_frame(df, name):
    """Rows whose date lies in the named block, keeping only steps whose return day is
    also inside the block: the last day of a block has no in-block next return, so it
    is the terminal row and carries no decision of its own."""
    lo, hi = BLOCKS[name]
    sub = df[(df["date"] >= lo) & (df["date"] <= hi)].reset_index(drop=True)
    sub.loc[sub.index[-1], "next_ret"] = np.nan
    return sub


def fit_normalizer(train_df, cols):
    """Per-column mean and standard deviation from the training block only."""
    x = train_df[cols].to_numpy(dtype=np.float64)
    return {"cols": cols, "mean": x.mean(axis=0).tolist(), "std": (x.std(axis=0) + 1e-8).tolist()}


def apply_normalizer(df, norm, clip=5.0):
    x = df[norm["cols"]].to_numpy(dtype=np.float64)
    z = (x - np.array(norm["mean"])) / np.array(norm["std"])
    return np.clip(z, -clip, clip).astype(np.float32)


def make_blocks(ticker, window):
    """Build every block for a ticker as arrays the environment can step through.

    Returns (blocks, norm). Each block holds the standardized features, the next-day
    returns (NaN on the terminal row), dates, closes, and the 50/200-day averages for
    the crossover baseline. The normalizer is fit on the training block alone.
    """
    df = build_features(load_prices(ticker), window)
    cols = feature_columns(window)
    norm = fit_normalizer(block_frame(df, "train").iloc[:-1], cols)
    blocks = {}
    for name in BLOCKS:
        sub = block_frame(df, name)
        blocks[name] = {
            "x": apply_normalizer(sub, norm),
            "next_ret": sub["next_ret"].to_numpy(dtype=np.float64),
            "dates": sub["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "close": sub["close"].to_numpy(dtype=np.float64),
            "sma50": sub["sma50"].to_numpy(dtype=np.float64),
            "sma200": sub["sma200"].to_numpy(dtype=np.float64),
        }
    return blocks, norm


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download daily prices and report block sizes.")
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    args = ap.parse_args()
    for t in args.tickers:
        path, n = download(t)
        blocks, _ = make_blocks(t, window=20)
        sizes = ", ".join("%s %d (%s to %s)" % (k, len(b["dates"]), b["dates"][0], b["dates"][-1])
                          for k, b in blocks.items())
        print("%s: %d days saved to %s | %s" % (t, n, os.path.relpath(path, ROOT), sizes))
