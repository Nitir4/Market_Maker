# ============================================================
# src/data_loader.py — Pulls price data and saves to CSV
# ============================================================

import yfinance as yf
import pandas as pd
import os
from config import DATA_PATH, END_DATE, PERIOD, START_DATE, TICKER


def _download_prices():
    print(f"Downloading {TICKER} data from yfinance...")
    print(f"Date range: {START_DATE} to {END_DATE} (end exclusive in yfinance)")
    raw = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        period=None,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty or "Close" not in raw:
        raise ValueError("yfinance returned no usable close data for the requested range.")

    prices = raw["Close"].squeeze()
    if prices.empty:
        raise ValueError("Downloaded close series is empty.")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    prices.to_frame(name="Close").to_csv(DATA_PATH)
    print(f"Data saved to {DATA_PATH}")
    return prices


def load_data():
    """
    Load price data. If CSV already exists, load from disk.
    Otherwise pull from yfinance and save to disk.
    Returns a pandas Series of closing prices.
    """

    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

        # 🔴 FIX: Clean column names
        df.columns = df.columns.str.strip()

        # 🔴 FIX: Handle different possible column names
        if 'Close' in df.columns:
            prices = df['Close']
        elif 'Adj Close' in df.columns:
            prices = df['Adj Close']
        else:
            raise ValueError(f"Close column not found. Columns available: {df.columns}")

        requested_start = pd.Timestamp(START_DATE)
        requested_end = pd.Timestamp(END_DATE)
        if prices.index.min() > requested_start or prices.index.max() < requested_end - pd.Timedelta(days=3):
            print("Cached data does not cover the requested date range. Refreshing from yfinance...")
            try:
                prices = _download_prices()
            except Exception as exc:
                print(f"Refresh failed: {exc}")
                print("Keeping the existing cached file instead of overwriting with an empty download.")

    else:
        prices = _download_prices()

    # Ensure Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()

    # Drop missing values
    prices = prices.dropna()

    print(f"Loaded {len(prices)} trading days of data for {TICKER}")
    return prices
