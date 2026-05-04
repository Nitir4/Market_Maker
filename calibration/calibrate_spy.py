import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate dt and sigma from Yahoo Finance daily OHLCV data."
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker to download from Yahoo Finance.")
    parser.add_argument("--period", default="5y", help="Yahoo Finance period, e.g. 5y.")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window for volatility estimation.",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Trading days per year for annualization.",
    )
    return parser.parse_args()


def download_prices(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker={ticker!r}, period={period!r}.")

    # Keep the calibration scoped to columns relevant to this project.
    prices = data.loc[:, ["Open", "Close"]].copy()
    prices = prices.dropna()
    if prices.empty:
        raise ValueError("Downloaded data has no usable Open/Close rows.")
    return prices


def compute_returns(prices):
    returns = prices.copy()
    returns["close_to_close_log_return"] = np.log(
        returns["Close"] / returns["Close"].shift(1)
    )
    returns["open_to_close_log_return"] = np.log(returns["Close"] / returns["Open"])
    returns["overnight_log_return"] = np.log(
        returns["Open"] / returns["Close"].shift(1)
    )
    return returns.dropna()


def compute_dt(n_steps, trading_days):
    return {
        "n_steps": int(n_steps),
        "dt_normalized": 1.0 / float(n_steps),
        "dt_trading_days": 1.0,
        "dt_year_fraction": 1.0 / float(trading_days),
    }


def compute_sigma(returns, window, trading_days):
    close_returns = returns["close_to_close_log_return"]
    rolling_daily = close_returns.rolling(window=window).std().dropna()
    rolling_annualized = rolling_daily * np.sqrt(trading_days)

    summary = {
        "window": int(window),
        "daily_sigma_close_to_close": float(close_returns.std()),
        "annualized_sigma_close_to_close": float(close_returns.std() * np.sqrt(trading_days)),
        "daily_sigma_open_to_close": float(returns["open_to_close_log_return"].std()),
        "annualized_sigma_open_to_close": float(
            returns["open_to_close_log_return"].std() * np.sqrt(trading_days)
        ),
        "daily_sigma_overnight": float(returns["overnight_log_return"].std()),
        "annualized_sigma_overnight": float(
            returns["overnight_log_return"].std() * np.sqrt(trading_days)
        ),
        "rolling_daily_sigma_mean": float(rolling_daily.mean()),
        "rolling_daily_sigma_median": float(rolling_daily.median()),
        "rolling_annualized_sigma_mean": float(rolling_annualized.mean()),
        "rolling_annualized_sigma_median": float(rolling_annualized.median()),
    }

    rolling = pd.DataFrame(
        {
            "rolling_daily_sigma": rolling_daily,
            "rolling_annualized_sigma": rolling_annualized,
        }
    )
    return summary, rolling


def build_summary(ticker, period, trading_days, dt_info, sigma_info):
    return {
        "ticker": ticker,
        "period": period,
        "trading_days_per_year": trading_days,
        "dt": dt_info,
        "sigma": sigma_info,
        "notes": [
            "The main project currently uses rolling close-to-close daily volatility directly.",
            "The paper's kappa and market-spread calibration cannot be inferred from daily OHLCV alone.",
            "If you keep T=1 and use one step per row, dt_normalized is the quantity used by the current simulator.",
        ],
    }


def write_text_summary(path, summary):
    lines = [
        "Calibration Summary",
        "===================",
        f"Ticker: {summary['ticker']}",
        f"Period: {summary['period']}",
        f"Trading days/year: {summary['trading_days_per_year']}",
        "",
        "dt",
        "--",
        f"n_steps: {summary['dt']['n_steps']}",
        f"dt_normalized: {summary['dt']['dt_normalized']:.8f}",
        f"dt_trading_days: {summary['dt']['dt_trading_days']:.2f}",
        f"dt_year_fraction: {summary['dt']['dt_year_fraction']:.8f}",
        "",
        "sigma",
        "-----",
        f"daily_sigma_close_to_close: {summary['sigma']['daily_sigma_close_to_close']:.8f}",
        f"annualized_sigma_close_to_close: {summary['sigma']['annualized_sigma_close_to_close']:.8f}",
        f"daily_sigma_open_to_close: {summary['sigma']['daily_sigma_open_to_close']:.8f}",
        f"annualized_sigma_open_to_close: {summary['sigma']['annualized_sigma_open_to_close']:.8f}",
        f"daily_sigma_overnight: {summary['sigma']['daily_sigma_overnight']:.8f}",
        f"annualized_sigma_overnight: {summary['sigma']['annualized_sigma_overnight']:.8f}",
        f"rolling_daily_sigma_mean: {summary['sigma']['rolling_daily_sigma_mean']:.8f}",
        f"rolling_daily_sigma_median: {summary['sigma']['rolling_daily_sigma_median']:.8f}",
        f"rolling_annualized_sigma_mean: {summary['sigma']['rolling_annualized_sigma_mean']:.8f}",
        f"rolling_annualized_sigma_median: {summary['sigma']['rolling_annualized_sigma_median']:.8f}",
        "",
        "notes",
        "-----",
    ]
    lines.extend(f"- {note}" for note in summary["notes"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_prices(args.ticker, args.period)
    returns = compute_returns(prices)
    dt_info = compute_dt(len(returns), args.trading_days)
    sigma_info, rolling = compute_sigma(returns, args.window, args.trading_days)
    summary = build_summary(args.ticker, args.period, args.trading_days, dt_info, sigma_info)

    prices.to_csv(OUTPUT_DIR / "raw_prices.csv")
    returns.to_csv(OUTPUT_DIR / "returns.csv")
    rolling.to_csv(OUTPUT_DIR / "rolling_volatility.csv")
    (OUTPUT_DIR / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_text_summary(OUTPUT_DIR / "calibration_summary.txt", summary)

    print(f"Saved calibration outputs to {OUTPUT_DIR}")
    print(f"dt_normalized = {dt_info['dt_normalized']:.8f}")
    print(
        "daily_sigma_close_to_close = "
        f"{sigma_info['daily_sigma_close_to_close']:.8f}"
    )
    print(
        "annualized_sigma_close_to_close = "
        f"{sigma_info['annualized_sigma_close_to_close']:.8f}"
    )


if __name__ == "__main__":
    main()
