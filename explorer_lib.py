"""
Helper library used by the explorer notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from pypfopt import EfficientFrontier, risk_models, expected_returns
from fetch_data import load_returns


# Map ticker suffix → region
SUFFIX_REGION = {
    ".HK": "Hong Kong / China",
    ".T":  "Japan",
    ".KS": "South Korea",
    ".TW": "Taiwan",
    ".SI": "Singapore",
    ".JK": "Indonesia",
    ".KL": "Malaysia",
    ".AX": "Australia",
    ".NS": "India",
    ".BO": "India",
    ".L":  "United Kingdom",
    ".DE": "Germany",
    ".PA": "France",
    ".MI": "Italy",
    ".MC": "Spain",
    ".SW": "Switzerland",
    ".AS": "Netherlands",
    ".ST": "Sweden",
    ".BR": "Belgium",
    ".TO": "Canada",
    ".SA": "Brazil",
    ".MX": "Mexico",
    ".SN": "Chile",
    ".CL": "Chile",
    ".LM": "Peru",
}

REGION_GROUP = {
    "United States": "Americas",
    "Canada":         "Americas",
    "Brazil":         "Americas",
    "Mexico":         "Americas",
    "Chile":          "Americas",
    "Peru":           "Americas",
    "United Kingdom": "Europe",
    "Germany":        "Europe",
    "France":         "Europe",
    "Italy":          "Europe",
    "Spain":          "Europe",
    "Switzerland":    "Europe",
    "Netherlands":    "Europe",
    "Sweden":         "Europe",
    "Belgium":        "Europe",
    "Japan":          "Asia-Pacific",
    "South Korea":    "Asia-Pacific",
    "Taiwan":         "Asia-Pacific",
    "Hong Kong / China": "Asia-Pacific",
    "Singapore":      "Asia-Pacific",
    "Indonesia":      "Asia-Pacific",
    "Malaysia":       "Asia-Pacific",
    "Australia":      "Asia-Pacific",
    "India":          "Asia-Pacific",
}


UNIVERSE_SOURCES = [
    ("S&P 500",       "503 US large-cap stocks"),
    ("Russell 1000",  "1003 US large+mid-cap stocks"),
    ("NASDAQ 100",    "101 US tech-heavy large-caps"),
    ("FTSE 100",      "100 UK large-caps"),
    ("FTSE 250",      "250 UK mid-caps"),
    ("DAX 40",        "40 German blue chips"),
    ("CAC 40",        "40 French blue chips"),
    ("SMI",           "20 Swiss large-caps"),
    ("AEX",           "25 Dutch large-caps"),
    ("OMXS 30",       "30 Swedish large-caps"),
    ("TSX 60",        "59 Canadian large-caps"),
    ("ASX 200",       "200 Australian large-caps"),
    ("Hang Seng",     "85 Hong Kong large-caps"),
    ("KOSPI 200",     "200 South Korean stocks"),
    ("Nifty 50",      "50 Indian large-caps"),
    ("iShares ETFs (×24)", "Top 10 holdings from EWJ, EWZ, EWW, INDA, FXI, EFA, etc."),
    ("Vanguard ETFs (×3)", "Top 10 holdings from VEA, VWO, VPL"),
]


def ticker_region(ticker: str) -> str:
    for suffix, region in SUFFIX_REGION.items():
        if ticker.endswith(suffix):
            return region
    return "United States"


def asset_max_drawdown(returns: pd.Series) -> float:
    """Calcula el drawdown máximo (negativo) de una serie de retornos."""
    eq = (1 + returns.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1
    return float(dd.min())


def filter_universe(returns: pd.DataFrame,
                    excluded: list = None,
                    min_sharpe: float = None,
                    regions: list = None,
                    max_volatility: float = None,
                    sectors: list = None,
                    max_drawdown: float = None) -> pd.DataFrame:
    """Apply pre-clustering filters to the universe."""
    df = returns.copy()
    # Keep only columns with at least 90% coverage; fill remaining NaN with 0
    coverage = df.notna().mean()
    df = df.loc[:, coverage >= 0.9].fillna(0.0)
    if excluded:
        df = df.drop(columns=[c for c in excluded if c in df.columns])
    if regions:
        keep = [c for c in df.columns if ticker_region(c) in regions]
        df = df[keep]
    if sectors:
        keep = [c for c in df.columns if ticker_sector(c) in sectors]
        df = df[keep]
    if min_sharpe is not None:
        ann_ret = df.mean() * 12
        ann_vol = df.std() * np.sqrt(12)
        sharpe = (ann_ret - 0.04) / ann_vol
        keep = sharpe[sharpe >= min_sharpe].index
        df = df[keep]
    if max_volatility is not None:
        ann_vol = df.std() * np.sqrt(12)
        keep = ann_vol[ann_vol <= max_volatility].index
        df = df[keep]
    if max_drawdown is not None:
        # max_drawdown es negativo (ej -0.30 = -30%). Mantener solo los menos malos.
        dds = df.apply(asset_max_drawdown)
        keep = dds[dds >= max_drawdown].index
        df = df[keep]
    return df


def cluster_and_pick(returns: pd.DataFrame, n_clusters: int):
    corr = returns.corr()
    distance = 1 - corr
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters = {}
    for ticker, label in zip(returns.columns, labels):
        clusters.setdefault(int(label), []).append(ticker)

    reps = []
    for tickers in clusters.values():
        sub = returns[tickers]
        sharpes = (sub.mean() * 12 - 0.04) / (sub.std() * np.sqrt(12))
        reps.append(sharpes.idxmax())
    return clusters, reps, Z


def optimize_portfolio(returns: pd.DataFrame,
                       n_clusters: int = 50,
                       min_weight: float = 0.0,
                       max_weight: float = 1.0,
                       pinned: list = None,
                       excluded: list = None,
                       min_sharpe: float = None,
                       regions: list = None,
                       max_volatility: float = None,
                       sectors: list = None,
                       max_drawdown: float = None,
                       risk_free: float = 0.04):
    """Full pipeline: filter → cluster → select representatives → max Sharpe."""
    pinned = pinned or []
    excluded = list(excluded or []) + [c for c in BENCHMARK_TICKERS if c in returns.columns]

    # Filter universe (but always keep pinned)
    filtered = filter_universe(returns, excluded=excluded, min_sharpe=min_sharpe,
                               regions=regions, max_volatility=max_volatility,
                               sectors=sectors, max_drawdown=max_drawdown)
    for p in pinned:
        if p in returns.columns and p not in filtered.columns:
            filtered[p] = returns[p]

    if filtered.shape[1] < n_clusters:
        raise ValueError(f"Filtered universe has only {filtered.shape[1]} assets, need >= {n_clusters}.")

    clusters, reps, Z = cluster_and_pick(filtered, n_clusters)

    # Build ticker → cluster_id map so we can replace each cluster's auto-rep
    # with any pinned asset that belongs to the same cluster.
    ticker_to_cluster = {}
    for cid, tickers in clusters.items():
        for t in tickers:
            ticker_to_cluster[t] = cid

    # Map cluster_id → chosen representative (start with auto-picked reps)
    cluster_rep = {}
    for r in reps:
        cluster_rep[ticker_to_cluster[r]] = r

    # For each pinned asset present in the filtered universe, override the
    # representative of its cluster. If two pinned assets fall in the same
    # cluster, the latter wins (a small ambiguity; rare in practice).
    pinned_in = [p for p in pinned if p in filtered.columns]
    for p in pinned_in:
        cid = ticker_to_cluster[p]
        cluster_rep[cid] = p

    final_assets = list(cluster_rep.values())

    sub = filtered[final_assets]
    mu = expected_returns.mean_historical_return(sub, returns_data=True, frequency=12)
    S  = risk_models.CovarianceShrinkage(sub, returns_data=True, frequency=12).ledoit_wolf()
    n  = len(final_assets)
    upper = max(max_weight, 1.5 / n)
    # Per-asset bounds: pinned assets get a floor of max(min_weight, 1%) so the
    # optimizer can't zero them out. Non-pinned assets can go to min_weight.
    pinned_floor = max(min_weight, 0.01)
    bounds = [
        (pinned_floor if a in pinned_in else min_weight, upper)
        for a in final_assets
    ]
    ef = EfficientFrontier(mu, S, weight_bounds=bounds)
    ef.max_sharpe(risk_free_rate=risk_free)
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free)

    return {
        "weights": pd.Series(weights),
        "performance": {"annual_return": perf[0], "annual_vol": perf[1], "sharpe": perf[2]},
        "linkage": Z,
        "clusters": clusters,
        "filtered_returns": filtered,
        "selected_returns": sub,
    }


def asset_metrics(returns: pd.DataFrame, weights: pd.Series, risk_free: float = 0.04) -> pd.DataFrame:
    """Per-asset metrics table for the chosen portfolio."""
    sub = returns[weights.index]
    annual_return = sub.mean() * 12
    annual_vol    = sub.std() * np.sqrt(12)
    sharpe        = (annual_return - risk_free) / annual_vol
    region        = [ticker_region(t) for t in weights.index]
    sector        = [ticker_sector(t) for t in weights.index]
    df = pd.DataFrame({
        "weight": weights,
        "annual_return": annual_return,
        "volatility": annual_vol,
        "sharpe": sharpe,
        "region": region,
        "sector": sector,
    })
    return df.sort_values("weight", ascending=False)


def equity_curve(returns_series: pd.Series) -> pd.Series:
    return (1 + returns_series).cumprod()


BENCHMARK_TICKERS = ("SPY", "ACWI", "AGG", "EFA")


_SECTORS_CACHE = None
def load_sectors() -> dict:
    """Cargar el mapa ticker → sector. Devuelve {} si no existe el archivo."""
    global _SECTORS_CACHE
    if _SECTORS_CACHE is not None:
        return _SECTORS_CACHE
    from pathlib import Path
    p = Path("data/sectors.parquet")
    if not p.exists():
        _SECTORS_CACHE = {}
        return _SECTORS_CACHE
    df = pd.read_parquet(p)
    _SECTORS_CACHE = dict(zip(df["ticker"], df["sector"]))
    return _SECTORS_CACHE


def ticker_sector(ticker: str) -> str:
    return load_sectors().get(ticker, "Desconocido")


def get_benchmark(returns: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    return returns[ticker].dropna() if ticker in returns.columns else pd.Series(dtype=float)


def drop_benchmarks(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.drop(columns=[c for c in BENCHMARK_TICKERS if c in returns.columns])


def perf_stats(returns: pd.Series, risk_free: float = 0.04, periods_per_year: int = 12) -> dict:
    """Compute standard performance stats on a return series."""
    if len(returns) == 0:
        return {}
    ann_r = (1 + returns.mean()) ** periods_per_year - 1
    ann_v = returns.std() * np.sqrt(periods_per_year)
    sh    = (ann_r - risk_free) / ann_v if ann_v > 0 else 0
    eq    = (1 + returns).cumprod()
    dd    = (eq / eq.cummax() - 1).min()
    total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    return {
        "annual_return": ann_r,
        "annual_vol": ann_v,
        "sharpe": sh,
        "max_drawdown": dd,
        "total_return": total,
    }


def backtest_strategy(returns: pd.DataFrame, n_clusters: int = 50,
                      train_months: int = 36, rebal_months: int = 3,
                      min_weight: float = 0.0, max_weight: float = 1.0,
                      pinned: list = None, excluded: list = None,
                      min_sharpe: float = None, regions: list = None,
                      max_volatility: float = None,
                      sectors: list = None,
                      max_drawdown: float = None):
    """Walk-forward backtest. Returns a Series of monthly portfolio returns."""
    portfolio_returns = []
    start_idx = train_months
    while start_idx + rebal_months <= len(returns):
        win = returns.iloc[start_idx - train_months: start_idx]
        # Keep columns with at least 90% coverage in the training window
        coverage = win.notna().mean()
        keep = coverage[coverage >= 0.9].index
        train = win[keep].fillna(0.0)
        test  = returns.iloc[start_idx: start_idx + rebal_months]
        try:
            res = optimize_portfolio(train, n_clusters=n_clusters,
                                     min_weight=min_weight, max_weight=max_weight,
                                     pinned=pinned, excluded=excluded,
                                     min_sharpe=min_sharpe, regions=regions,
                                     max_volatility=max_volatility,
                                     sectors=sectors, max_drawdown=max_drawdown)
            w = res["weights"]
            w = w[w > 0]
            common = w.index.intersection(test.columns)
            period_returns = (test[common].fillna(0.0) * w[common]).sum(axis=1)
            portfolio_returns.append(period_returns)
        except Exception as e:
            print(f"  skip period at {returns.index[start_idx].date()}: {e}")
        start_idx += rebal_months
    return pd.concat(portfolio_returns) if portfolio_returns else pd.Series(dtype=float)


def buy_and_hold_backtest(returns: pd.DataFrame, train_months: int = 36,
                          hold_months: int = 24, **kwargs) -> pd.Series:
    """Train on the first `train_months`, then hold those weights for `hold_months`.
    No rebalancing. Returns the monthly portfolio returns over the hold period.
    Any extra kwargs are forwarded to optimize_portfolio.
    """
    if train_months + hold_months > len(returns):
        hold_months = len(returns) - train_months
    train = returns.iloc[:train_months]
    test  = returns.iloc[train_months: train_months + hold_months]

    res = optimize_portfolio(train, **kwargs)
    w = res["weights"]
    w = w[w > 0]
    common = w.index.intersection(test.columns)
    period_returns = (test[common].fillna(0.0) * w[common]).sum(axis=1)
    return period_returns
