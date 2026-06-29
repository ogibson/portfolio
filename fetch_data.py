"""
Download monthly historical returns for all tickers and cache as Parquet.
Universe is built from ETF holdings via get_holdings.py.
"""

import yfinance as yf
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from get_holdings import build_universe

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "returns.parquet"

END   = date.today().strftime("%Y-%m-%d")
START = (date.today() - relativedelta(years=5)).strftime("%Y-%m-%d")

BATCH = 100
MIN_COVERAGE = 0.8  # drop assets missing more than 20% of periods


def download_prices(tickers: list[str]) -> pd.DataFrame:
    all_prices = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        print(f"Downloading batch {i // BATCH + 1}/{(len(tickers) - 1) // BATCH + 1} ({len(batch)} tickers)...")
        raw = yf.download(batch, start=START, end=END, interval="1mo", auto_adjust=True, progress=False)
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": batch[0]})
        all_prices.append(prices)
    if not all_prices:
        raise RuntimeError("No data downloaded.")
    return pd.concat(all_prices, axis=1)


def fetch_and_cache():
    tickers = build_universe()
    print(f"\nFetching monthly prices for {len(tickers)} tickers  ({START} → {END})...")
    prices = download_prices(tickers)

    # Drop assets with insufficient history
    min_rows = int(len(prices) * MIN_COVERAGE)
    prices = prices.dropna(thresh=min_rows, axis=1)
    print(f"Kept {prices.shape[1]} assets after coverage filter ({MIN_COVERAGE:.0%} threshold).")

    returns = prices.pct_change().dropna(how="all")
    returns = returns.dropna(axis=1, how="all")

    # Drop assets with implausible weekly returns (>50% in any single week)
    # This catches stock-split data glitches and obvious yfinance errors
    extreme = returns.abs().max() > 0.5
    n_extreme = extreme.sum()
    if n_extreme:
        print(f"Dropping {n_extreme} assets with extreme returns (>50% in one week).")
        returns = returns.loc[:, ~extreme]

    returns.to_parquet(CACHE_FILE)
    print(f"Saved returns to {CACHE_FILE}  ({returns.shape[0]} months x {returns.shape[1]} assets)")
    return returns


def load_returns() -> pd.DataFrame:
    if CACHE_FILE.exists():
        print(f"Loading cached returns from {CACHE_FILE}...")
        return pd.read_parquet(CACHE_FILE)
    return fetch_and_cache()


if __name__ == "__main__":
    fetch_and_cache()
