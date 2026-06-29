"""
Fetch individual stock holdings from ETFs.
Sources:
  - S&P 500: Wikipedia (full 500 constituents)
  - All other ETFs: yfinance funds_data.top_holdings (top 10 per ETF, free limit)

Note: yfinance only returns top 10 holdings per ETF. For full holdings,
a paid data provider (Refinitiv, Bloomberg, Tiingo) would be needed.
For a research project this gives ~500 S&P stocks + ~150 international stocks.
"""

import requests
import pandas as pd
import yfinance as yf
import time
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HOLDINGS_CACHE = DATA_DIR / "holdings.parquet"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# ETFs whose individual stock holdings we want
TARGET_ETFS = [
    # Asia-Pacific
    "EWJ",   # Japan
    "EWY",   # South Korea
    "EWT",   # Taiwan
    "EWH",   # Hong Kong
    "EWA",   # Australia
    "EWS",   # Singapore
    "EWM",   # Malaysia
    "INDA",  # India
    "EPHE",  # Philippines
    "EIDO",  # Indonesia
    "THD",   # Thailand
    "FXI",   # China large cap
    "MCHI",  # China MSCI
    "AAXJ",  # All Asia ex Japan
    # Latin America
    "EWZ",   # Brazil
    "EWW",   # Mexico
    "ECH",   # Chile
    "EPU",   # Peru
    "GXG",   # Colombia
    "ILF",   # Latin America 40
    "ARGT",  # Argentina
    # Europe
    "EWG",   # Germany
    "EWU",   # UK
    "EWQ",   # France
    "EWI",   # Italy
    "EWP",   # Spain
    "EWN",   # Netherlands
    "EWK",   # Belgium
    "EWD",   # Sweden
    "EWL",   # Switzerland
    # Broad international
    "EFA",   # Developed markets
    "VEA",   # Vanguard developed
    "VWO",   # Vanguard emerging
    "VPL",   # Vanguard Pacific
]


def _fetch_wiki_tables(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(pd.io.common.StringIO(resp.text))


def _find_ticker_column(df, candidates):
    """Find the first matching column (case-insensitive substring match)."""
    for col in df.columns:
        col_str = str(col).lower()
        for cand in candidates:
            if cand.lower() in col_str:
                return df[col]
    return None


def fetch_sp500() -> list[str]:
    print("Fetching S&P 500...")
    tables = _fetch_wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    tickers = tables[0]["Symbol"].astype(str).tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    print(f"  S&P 500: {len(tickers)}")
    return tickers


def fetch_nasdaq100() -> list[str]:
    print("Fetching NASDAQ 100...")
    tables = _fetch_wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
    for t in tables:
        col = _find_ticker_column(t, ["Ticker", "Symbol"])
        if col is not None and 80 < len(col) < 150:
            tickers = [str(x).replace(".", "-") for x in col.dropna()]
            print(f"  NASDAQ 100: {len(tickers)}")
            return tickers
    print("  NASDAQ 100: failed")
    return []


def _fetch_with_suffix(name: str, url: str, suffix: str, candidates: list[str],
                       min_n: int, max_n: int) -> list[str]:
    print(f"Fetching {name}...")
    try:
        tables = _fetch_wiki_tables(url)
        for t in tables:
            col = _find_ticker_column(t, candidates)
            if col is not None and min_n < len(col) < max_n:
                raw = [str(x).strip() for x in col.dropna()]
                # numeric tickers (Japan/Korea) need zero-padding sometimes
                tickers = [t if t.endswith(suffix) else f"{t}{suffix}" for t in raw if t and t.lower() != "nan"]
                print(f"  {name}: {len(tickers)}")
                return tickers
        print(f"  {name}: no matching table")
        return []
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
        return []


def fetch_ftse100() -> list[str]:
    return _fetch_with_suffix(
        "FTSE 100", "https://en.wikipedia.org/wiki/FTSE_100_Index",
        ".L", ["EPIC", "Ticker", "Symbol"], 80, 150)


def fetch_dax() -> list[str]:
    return _fetch_with_suffix(
        "DAX 40", "https://en.wikipedia.org/wiki/DAX",
        ".DE", ["Ticker", "Symbol"], 30, 60)


def fetch_nikkei225() -> list[str]:
    print("Fetching Nikkei 225...")
    try:
        tables = _fetch_wiki_tables("https://en.wikipedia.org/wiki/Nikkei_225")
        for t in tables:
            col = _find_ticker_column(t, ["Code", "Ticker", "Symbol"])
            if col is not None and 150 < len(col) < 300:
                raw = [str(x).strip() for x in col.dropna()]
                tickers = []
                for r in raw:
                    digits = "".join(c for c in r if c.isdigit())
                    if digits:
                        tickers.append(f"{digits}.T")
                print(f"  Nikkei 225: {len(tickers)}")
                return tickers
        print("  Nikkei 225: no matching table")
        return []
    except Exception as e:
        print(f"  Nikkei 225: FAILED ({e})")
        return []


def fetch_hang_seng() -> list[str]:
    print("Fetching Hang Seng...")
    try:
        tables = _fetch_wiki_tables("https://en.wikipedia.org/wiki/Hang_Seng_Index")
        for t in tables:
            col = _find_ticker_column(t, ["Ticker", "Symbol", "Code", "Stock"])
            if col is not None and 50 < len(col) < 120:
                raw = [str(x).strip() for x in col.dropna()]
                # HK tickers need 4-digit padding + .HK
                tickers = []
                for r in raw:
                    digits = "".join(c for c in r if c.isdigit())
                    if digits:
                        tickers.append(f"{int(digits):04d}.HK")
                print(f"  Hang Seng: {len(tickers)}")
                return tickers
        print("  Hang Seng: no matching table")
        return []
    except Exception as e:
        print(f"  Hang Seng: FAILED ({e})")
        return []


def fetch_nifty50() -> list[str]:
    return _fetch_with_suffix(
        "Nifty 50", "https://en.wikipedia.org/wiki/NIFTY_50",
        ".NS", ["Symbol", "Ticker"], 40, 70)


def fetch_bovespa() -> list[str]:
    return _fetch_with_suffix(
        "Bovespa", "https://en.wikipedia.org/wiki/Ibovespa",
        ".SA", ["Ticker", "Symbol", "Code"], 50, 120)


def fetch_russell1000() -> list[str]:
    print("Fetching Russell 1000...")
    try:
        tables = _fetch_wiki_tables("https://en.wikipedia.org/wiki/Russell_1000_Index")
        for t in tables:
            col = _find_ticker_column(t, ["Ticker", "Symbol"])
            if col is not None and 800 < len(col) < 1200:
                tickers = [str(x).replace(".", "-") for x in col.dropna()]
                print(f"  Russell 1000: {len(tickers)}")
                return tickers
        print("  Russell 1000: no matching table")
        return []
    except Exception as e:
        print(f"  Russell 1000: FAILED ({e})")
        return []


def fetch_ftse250() -> list[str]:
    return _fetch_with_suffix(
        "FTSE 250", "https://en.wikipedia.org/wiki/FTSE_250_Index",
        ".L", ["EPIC", "Ticker", "Symbol"], 200, 300)


def fetch_cac40() -> list[str]:
    return _fetch_with_suffix(
        "CAC 40", "https://en.wikipedia.org/wiki/CAC_40",
        ".PA", ["Ticker", "Symbol"], 30, 60)


def fetch_smi() -> list[str]:
    return _fetch_with_suffix(
        "SMI (Swiss)", "https://en.wikipedia.org/wiki/Swiss_Market_Index",
        ".SW", ["Ticker", "Symbol"], 15, 30)


def fetch_aex() -> list[str]:
    return _fetch_with_suffix(
        "AEX (Netherlands)", "https://en.wikipedia.org/wiki/AEX_index",
        ".AS", ["Ticker", "Symbol"], 20, 35)


def fetch_omxs30() -> list[str]:
    return _fetch_with_suffix(
        "OMXS30 (Sweden)", "https://en.wikipedia.org/wiki/OMX_Stockholm_30",
        ".ST", ["Ticker", "Symbol"], 25, 40)


def fetch_tsx60() -> list[str]:
    return _fetch_with_suffix(
        "TSX 60 (Canada)", "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        ".TO", ["Ticker", "Symbol"], 50, 80)


def fetch_asx200() -> list[str]:
    return _fetch_with_suffix(
        "ASX 200", "https://en.wikipedia.org/wiki/S%26P/ASX_200",
        ".AX", ["Code", "Ticker", "Symbol"], 150, 250)


def fetch_kospi200() -> list[str]:
    # KOSPI 200 on Wikipedia uses numeric codes
    print("Fetching KOSPI 200...")
    try:
        tables = _fetch_wiki_tables("https://en.wikipedia.org/wiki/KOSPI_200")
        for t in tables:
            col = _find_ticker_column(t, ["Ticker", "Symbol", "Code"])
            if col is not None and 100 < len(col) < 250:
                raw = [str(x).strip() for x in col.dropna()]
                tickers = [f"{int(''.join(c for c in r if c.isdigit())):06d}.KS"
                           for r in raw if any(c.isdigit() for c in r)]
                print(f"  KOSPI 200: {len(tickers)}")
                return tickers
        print("  KOSPI 200: no matching table")
        return []
    except Exception as e:
        print(f"  KOSPI 200: FAILED ({e})")
        return []


def fetch_etf_holdings(etf: str) -> list[str]:
    try:
        t = yf.Ticker(etf)
        holdings = t.funds_data.top_holdings
        if holdings is None or holdings.empty:
            print(f"  {etf}: no holdings data")
            return []
        tickers = holdings.index.tolist()
        print(f"  {etf}: {len(tickers)} holdings")
        return tickers
    except Exception as e:
        print(f"  {etf}: FAILED ({e})")
        return []


def build_universe(force: bool = False) -> list[str]:
    if HOLDINGS_CACHE.exists() and not force:
        print(f"Loading cached holdings from {HOLDINGS_CACHE}...")
        df = pd.read_parquet(HOLDINGS_CACHE)
        tickers = df["ticker"].tolist()
        print(f"Total universe: {len(tickers)} tickers")
        return tickers

    all_tickers: set[str] = set()

    # Index constituents (full lists from Wikipedia)
    for fn in [fetch_sp500, fetch_russell1000, fetch_nasdaq100,
               fetch_ftse100, fetch_ftse250, fetch_dax, fetch_cac40,
               fetch_smi, fetch_aex, fetch_omxs30, fetch_tsx60,
               fetch_asx200, fetch_nikkei225, fetch_hang_seng,
               fetch_kospi200, fetch_nifty50, fetch_bovespa]:
        all_tickers.update(fn())
        time.sleep(0.5)

    # Top holdings from each international ETF
    print(f"\nFetching top holdings from {len(TARGET_ETFS)} ETFs...")
    for etf in TARGET_ETFS:
        tickers = fetch_etf_holdings(etf)
        all_tickers.update(tickers)
        time.sleep(0.5)

    # Remove obvious non-tickers (cash lines, derivatives, empty strings)
    cleaned = sorted(
        t for t in all_tickers
        if t and len(t) <= 12 and t not in ("-", "—", "CASH", "USD")
    )

    print(f"\nTotal unique tickers: {len(cleaned)}")
    pd.DataFrame({"ticker": cleaned}).to_parquet(HOLDINGS_CACHE)
    print(f"Saved to {HOLDINGS_CACHE}")
    return cleaned


if __name__ == "__main__":
    tickers = build_universe(force=True)
    print("\nSample:", tickers[:20])
