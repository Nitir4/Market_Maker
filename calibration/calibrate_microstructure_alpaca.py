import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "alpaca_microstructure"
BASE_URL = "https://data.alpaca.markets/v2/stocks"
DEFAULT_SYMBOL = "SPY"
REGULAR_OPEN = "09:30:00"
REGULAR_CLOSE = "16:00:00"


@dataclass
class AlpacaCredentials:
    key_id: str
    secret_key: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate SPY microstructure proxies M and kappa from Alpaca historical quotes and trades."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Stock symbol, default SPY.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD.")
    parser.add_argument("--feed", default="iex", help="Alpaca feed to use, default iex.")
    parser.add_argument("--limit", type=int, default=10000, help="API page size, max 10000.")
    parser.add_argument("--sleep-seconds", type=float, default=0.15, help="Pause between API pages.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each Alpaca API request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for transient API/network failures.",
    )
    parser.add_argument(
        "--max-pages-per-day",
        type=int,
        default=None,
        help="Optional cap on pages fetched per endpoint per day for faster validation runs.",
    )
    parser.add_argument("--tick-size", type=float, default=0.01, help="Assumed tick size for SPY.")
    parser.add_argument(
        "--max-delta-ticks",
        type=int,
        default=10,
        help="Largest synthetic quote distance in ticks for kappa estimation.",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=1.0,
        help="Forward time horizon used to detect synthetic fills.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=25,
        help="Use every Nth quote for kappa estimation to control runtime.",
    )
    return parser.parse_args()


def load_credentials() -> AlpacaCredentials:
    key_id = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")
    if not key_id or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )
    return AlpacaCredentials(key_id=key_id, secret_key=secret_key)


def daterange(start_date: datetime, end_date: datetime) -> Iterable[datetime]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def is_non_trading_day(day: datetime) -> bool:
    """
    Skip weekends up front. This avoids unnecessary requests on days where
    U.S. equity markets are definitely closed.
    """
    return day.weekday() >= 5


def build_request(url: str, credentials: AlpacaCredentials) -> Request:
    headers = {
        "APCA-API-KEY-ID": credentials.key_id,
        "APCA-API-SECRET-KEY": credentials.secret_key,
    }
    return Request(url, headers=headers)


def fetch_paginated_records(
    endpoint: str,
    symbol: str,
    start_iso: str,
    end_iso: str,
    feed: str,
    limit: int,
    credentials: AlpacaCredentials,
    sleep_seconds: float,
    request_timeout: float,
    max_retries: int,
    max_pages: Optional[int] = None,
) -> List[Dict]:
    records: List[Dict] = []
    page_token: Optional[str] = None
    page_count = 0

    while True:
        params = {
            "start": start_iso,
            "end": end_iso,
            "limit": limit,
            "feed": feed,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        url = f"{BASE_URL}/{symbol}/{endpoint}?{urlencode(params)}"
        request = build_request(url, credentials)
        payload = None
        for attempt in range(1, max_retries + 1):
            try:
                with urlopen(request, timeout=request_timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                # Retry only for rate limits and transient server-side failures.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == max_retries:
                    raise
                wait_seconds = min(60.0, sleep_seconds * (2 ** attempt))
                print(
                    f"  retrying {endpoint} page {page_count + 1} after HTTP {exc.code} "
                    f"(attempt {attempt}/{max_retries}, wait {wait_seconds:.2f}s)",
                    flush=True,
                )
                time.sleep(wait_seconds)
            except URLError:
                if attempt == max_retries:
                    raise
                wait_seconds = min(60.0, sleep_seconds * (2 ** attempt))
                print(
                    f"  retrying {endpoint} page {page_count + 1} after network error "
                    f"(attempt {attempt}/{max_retries}, wait {wait_seconds:.2f}s)",
                    flush=True,
                )
                time.sleep(wait_seconds)

        if payload is None:
            raise RuntimeError(f"Failed to fetch {endpoint} page after retries.")

        batch = payload.get(endpoint) or []
        records.extend(batch)
        page_count += 1
        print(
            f"  fetched {endpoint} page {page_count} with {len(batch)} rows",
            flush=True,
        )
        page_token = payload.get("next_page_token")
        if max_pages is not None and page_count >= max_pages:
            break
        if not page_token:
            break
        time.sleep(sleep_seconds)

    return records


def fetch_day(
    symbol,
    day,
    feed,
    limit,
    credentials,
    sleep_seconds,
    request_timeout,
    max_retries,
    max_pages_per_day=None,
):
    start_dt = datetime.combine(day.date(), datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)
    start_iso = start_dt.isoformat().replace("+00:00", "Z")
    end_iso = end_dt.isoformat().replace("+00:00", "Z")

    quotes = fetch_paginated_records(
        endpoint="quotes",
        symbol=symbol,
        start_iso=start_iso,
        end_iso=end_iso,
        feed=feed,
        limit=limit,
        credentials=credentials,
        sleep_seconds=sleep_seconds,
        request_timeout=request_timeout,
        max_retries=max_retries,
        max_pages=max_pages_per_day,
    )
    trades = fetch_paginated_records(
        endpoint="trades",
        symbol=symbol,
        start_iso=start_iso,
        end_iso=end_iso,
        feed=feed,
        limit=limit,
        credentials=credentials,
        sleep_seconds=sleep_seconds,
        request_timeout=request_timeout,
        max_retries=max_retries,
        max_pages=max_pages_per_day,
    )
    return quotes, trades


def records_to_frames(quotes: List[Dict], trades: List[Dict]):
    quotes_df = pd.DataFrame(quotes)
    trades_df = pd.DataFrame(trades)

    if not quotes_df.empty:
        quotes_df["timestamp"] = pd.to_datetime(quotes_df["t"], utc=True, format="ISO8601")
        quotes_df["bid_price"] = quotes_df["bp"].astype(float)
        quotes_df["ask_price"] = quotes_df["ap"].astype(float)
        quotes_df["bid_size"] = quotes_df["bs"].astype(float)
        quotes_df["ask_size"] = quotes_df["as"].astype(float)
        quotes_df["spread"] = quotes_df["ask_price"] - quotes_df["bid_price"]
        quotes_df["mid_price"] = (quotes_df["ask_price"] + quotes_df["bid_price"]) / 2.0
        quotes_df = quotes_df[
            ["timestamp", "bid_price", "ask_price", "bid_size", "ask_size", "spread", "mid_price"]
        ].sort_values("timestamp")

    if not trades_df.empty:
        trades_df["timestamp"] = pd.to_datetime(trades_df["t"], utc=True, format="ISO8601")
        trades_df["price"] = trades_df["p"].astype(float)
        trades_df["size"] = trades_df["s"].astype(float)
        trades_df = trades_df[["timestamp", "price", "size"]].sort_values("timestamp")

    return quotes_df, trades_df


def filter_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    eastern = df["timestamp"].dt.tz_convert("America/New_York")
    mask = (
        (eastern.dt.strftime("%H:%M:%S") >= REGULAR_OPEN)
        & (eastern.dt.strftime("%H:%M:%S") <= REGULAR_CLOSE)
    )
    return df.loc[mask].copy()


def compute_dt_sigma_from_quotes(quotes_df: pd.DataFrame) -> Dict:
    if quotes_df.empty:
        return {}

    mids = quotes_df[["timestamp", "mid_price"]].drop_duplicates("timestamp").copy()
    mids["log_return"] = np.log(mids["mid_price"] / mids["mid_price"].shift(1))
    mids = mids.dropna()

    if mids.empty:
        return {}

    step_seconds = mids["timestamp"].diff().dropna().dt.total_seconds()
    sigma_per_step = float(mids["log_return"].std()) if not mids["log_return"].empty else float("nan")

    return {
        "n_mid_updates": int(len(mids)),
        "median_dt_seconds": float(step_seconds.median()) if not step_seconds.empty else float("nan"),
        "mean_dt_seconds": float(step_seconds.mean()) if not step_seconds.empty else float("nan"),
        "sigma_mid_log_return_per_step": sigma_per_step,
    }


def summarize_m(quotes_df: pd.DataFrame) -> Dict:
    if quotes_df.empty:
        return {}

    spread = quotes_df["spread"]
    mid = quotes_df["mid_price"]
    spread_bps = 10000.0 * spread / mid

    return {
        "symbol": DEFAULT_SYMBOL,
        "n_quotes": int(len(quotes_df)),
        "mean_spread": float(spread.mean()),
        "median_spread": float(spread.median()),
        "spread_p25": float(spread.quantile(0.25)),
        "spread_p75": float(spread.quantile(0.75)),
        "mean_spread_bps": float(spread_bps.mean()),
        "median_spread_bps": float(spread_bps.median()),
    }


def estimate_kappa(quotes_df, trades_df, tick_size, max_delta_ticks, horizon_seconds, sample_every):
    if quotes_df.empty or trades_df.empty:
        return pd.DataFrame(), {}

    quotes = quotes_df.iloc[::sample_every].copy()
    trades = trades_df.copy()
    trade_times = trades["timestamp"].astype("int64").to_numpy()
    trade_prices = trades["price"].to_numpy()
    horizon_ns = int(horizon_seconds * 1_000_000_000)

    rows = []
    for tick_count in range(1, max_delta_ticks + 1):
        delta = tick_count * tick_size
        bid_fills = 0
        ask_fills = 0
        total = 0

        for _, quote in quotes.iterrows():
            quote_time_ns = int(quote["timestamp"].value)
            end_time_ns = quote_time_ns + horizon_ns

            left = np.searchsorted(trade_times, quote_time_ns, side="right")
            right = np.searchsorted(trade_times, end_time_ns, side="right")
            if right <= left:
                continue

            window_prices = trade_prices[left:right]
            mid = float(quote["mid_price"])
            synthetic_bid = mid - delta
            synthetic_ask = mid + delta

            bid_fill = np.any(window_prices <= synthetic_bid)
            ask_fill = np.any(window_prices >= synthetic_ask)

            bid_fills += int(bid_fill)
            ask_fills += int(ask_fill)
            total += 1

        if total == 0:
            continue

        bid_prob = bid_fills / total
        ask_prob = ask_fills / total
        avg_prob = (bid_prob + ask_prob) / 2.0

        if avg_prob >= 1.0:
            lambda_hat = float("inf")
        elif avg_prob <= 0.0:
            lambda_hat = 0.0
        else:
            lambda_hat = -math.log(1.0 - avg_prob) / horizon_seconds

        rows.append(
            {
                "delta_ticks": tick_count,
                "delta_dollars": delta,
                "sample_count": total,
                "bid_fill_prob": bid_prob,
                "ask_fill_prob": ask_prob,
                "avg_fill_prob": avg_prob,
                "lambda_hat": lambda_hat,
            }
        )

    delta_df = pd.DataFrame(rows)
    fit = {}
    valid = delta_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lambda_hat"])
    valid = valid[valid["lambda_hat"] > 0].copy()

    if len(valid) >= 2:
        x = valid["delta_dollars"].to_numpy()
        y = np.log(valid["lambda_hat"].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        fit = {
            "log_lambda_intercept": float(intercept),
            "kappa": float(-slope),
            "a_scale": float(np.exp(intercept)),
            "fit_points": int(len(valid)),
            "horizon_seconds": float(horizon_seconds),
            "sample_every": int(sample_every),
        }

    return delta_df, fit


def write_summary(path: Path, symbol: str, feed: str, m_summary: Dict, dt_sigma: Dict, kappa_fit: Dict):
    lines = [
        "Alpaca Microstructure Calibration",
        "================================",
        f"Symbol: {symbol}",
        f"Feed: {feed}",
        "",
        "M summary",
        "---------",
    ]
    for key, value in m_summary.items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "dt and sigma summary",
            "--------------------",
        ]
    )
    for key, value in dt_sigma.items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "kappa fit",
            "---------",
        ]
    )
    if kappa_fit:
        for key, value in kappa_fit.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("Not enough valid delta buckets to fit kappa.")

    lines.extend(
        [
            "",
            "caveats",
            "-------",
            "- This calibration is SPY-specific only if you run it with symbol SPY.",
            "- Alpaca free historical market data uses the IEX feed, not full NBBO.",
            "- kappa is estimated from a synthetic fill approximation with no queue model.",
            "- Treat these results as simulator inputs, not universal market constants.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.symbol.upper() != DEFAULT_SYMBOL:
        raise ValueError("This sandbox is scoped to SPY. Use --symbol SPY.")

    credentials = load_credentials()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    all_quotes = []
    all_trades = []
    for day in daterange(start_date, end_date):
        if is_non_trading_day(day):
            print(f"Skipping non-trading day {day.date().isoformat()}...", flush=True)
            continue

        print(f"Processing {day.date().isoformat()} for {args.symbol}...", flush=True)
        quotes, trades = fetch_day(
            symbol=args.symbol,
            day=day,
            feed=args.feed,
            limit=args.limit,
            credentials=credentials,
            sleep_seconds=args.sleep_seconds,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            max_pages_per_day=args.max_pages_per_day,
        )
        day_quotes_df, day_trades_df = records_to_frames(quotes, trades)
        day_quotes_df = filter_regular_hours(day_quotes_df)
        day_trades_df = filter_regular_hours(day_trades_df)
        print(
            f"  kept {len(day_quotes_df)} regular-hours quotes and {len(day_trades_df)} trades",
            flush=True,
        )

        if not day_quotes_df.empty:
            day_quotes_df["date"] = day.date().isoformat()
            all_quotes.append(day_quotes_df)
        if not day_trades_df.empty:
            day_trades_df["date"] = day.date().isoformat()
            all_trades.append(day_trades_df)

    quotes_df = pd.concat(all_quotes, ignore_index=True) if all_quotes else pd.DataFrame()
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    if quotes_df.empty:
        raise RuntimeError("No quote data downloaded for SPY in the requested range.")

    quotes_df.to_csv(OUTPUT_DIR / "quotes.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    daily_m = (
        quotes_df.groupby("date")
        .agg(
            mean_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            mean_mid=("mid_price", "mean"),
            quote_count=("spread", "size"),
        )
        .reset_index()
    )
    daily_m.to_csv(OUTPUT_DIR / "daily_m_summary.csv", index=False)

    m_summary = summarize_m(quotes_df)
    dt_sigma = compute_dt_sigma_from_quotes(quotes_df)
    delta_df, kappa_fit = estimate_kappa(
        quotes_df=quotes_df,
        trades_df=trades_df,
        tick_size=args.tick_size,
        max_delta_ticks=args.max_delta_ticks,
        horizon_seconds=args.horizon_seconds,
        sample_every=args.sample_every,
    )

    delta_df.to_csv(OUTPUT_DIR / "delta_fill_summary.csv", index=False)
    (OUTPUT_DIR / "kappa_fit.json").write_text(json.dumps(kappa_fit, indent=2), encoding="utf-8")
    write_summary(
        OUTPUT_DIR / "microstructure_summary.txt",
        symbol=args.symbol,
        feed=args.feed,
        m_summary=m_summary,
        dt_sigma=dt_sigma,
        kappa_fit=kappa_fit,
    )

    print(f"Saved Alpaca microstructure outputs to {OUTPUT_DIR}")
    print(f"Symbol = {args.symbol}")
    print(f"Feed = {args.feed}")
    if m_summary:
        print(f"Median spread proxy M = {m_summary['median_spread']:.6f}")
    if kappa_fit:
        print(f"Estimated kappa = {kappa_fit['kappa']:.6f}")
    else:
        print("Estimated kappa = unavailable from current sample")


if __name__ == "__main__":
    main()
