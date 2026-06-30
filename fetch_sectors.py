"""
Descarga sector e industria para todos los activos del universo via yfinance.
Cachea en data/sectors.parquet. Se ejecuta una sola vez (o cuando se quiera refrescar).
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from fetch_data import load_returns
from explorer_lib import drop_benchmarks
import time

OUT = Path("data/sectors.parquet")


def fetch_one(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "sector": info.get("sector") or "Desconocido",
            "industry": info.get("industry") or "Desconocido",
        }
    except Exception:
        return {"ticker": ticker, "sector": "Desconocido", "industry": "Desconocido"}


def main(refresh: bool = False):
    returns = load_returns()
    tickers = sorted(drop_benchmarks(returns).columns.tolist())

    existing = {}
    if OUT.exists() and not refresh:
        df = pd.read_parquet(OUT)
        existing = df.set_index("ticker").to_dict("index")
        print(f"Cargados {len(existing)} sectores cacheados.")

    rows = []
    todo = [t for t in tickers if t not in existing]
    print(f"A descargar: {len(todo)} activos (de {len(tickers)} totales).")

    start = time.time()
    for i, t in enumerate(todo, 1):
        rows.append(fetch_one(t))
        if i % 50 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (len(todo) - i) / rate
            print(f"  {i}/{len(todo)}  ({rate:.1f}/s, ETA {eta/60:.1f}min)")

    # Merge nuevo + cacheado
    new_df = pd.DataFrame(rows)
    if existing:
        cached_df = pd.DataFrame([{"ticker": k, **v} for k, v in existing.items()])
        out = pd.concat([cached_df, new_df], ignore_index=True).drop_duplicates(subset=["ticker"], keep="last")
    else:
        out = new_df

    out.to_parquet(OUT)
    print(f"\nGuardado {len(out)} sectores en {OUT}.")
    print("\nDistribución de sectores:")
    print(out["sector"].value_counts())


if __name__ == "__main__":
    import sys
    refresh = "--refresh" in sys.argv
    main(refresh=refresh)
