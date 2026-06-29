"""
Generate all static assets for the Story page.
Run this once after data updates: `python3 precompute.py`
Outputs go to data/story/ (charts + JSON metrics).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

from fetch_data import load_returns
from explorer_lib import (
    UNIVERSE_SOURCES, REGION_GROUP, ticker_region,
    BENCHMARK_TICKERS, drop_benchmarks, get_benchmark, perf_stats,
    optimize_portfolio, backtest_strategy,
)

OUT_DIR = Path("data/story")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=0.9)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")


def precompute_universe(returns):
    print("\n[universe]")
    stocks = drop_benchmarks(returns)
    regions = pd.Series([ticker_region(t) for t in stocks.columns]).value_counts()
    groups = regions.groupby(lambda r: REGION_GROUP.get(r, "Other")).sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    regions.head(15).plot(kind="barh", ax=axes[0], color="#4a6fa5")
    axes[0].set_title("Assets by country (top 15)")
    axes[0].invert_yaxis()
    axes[1].pie(groups, labels=groups.index, autopct="%1.0f%%", startangle=90,
                colors=sns.color_palette("Set2"))
    axes[1].set_title("Regional split")
    plt.tight_layout()
    save(fig, "universe")
    return stocks


def precompute_correlation_heatmap(stocks):
    print("\n[correlation heatmap]")
    # Subsample for visibility
    sample = stocks.sample(n=min(200, stocks.shape[1]), axis=1, random_state=42)
    corr = sample.corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"label": "Correlation"})
    ax.set_title(f"Correlation matrix of {sample.shape[1]} randomly sampled assets\n(sea of red = lots of co-movement = need for clustering)")
    plt.tight_layout()
    save(fig, "correlation_full")


def precompute_eigenvalue_spectrum(stocks):
    print("\n[eigenvalue spectrum]")
    corr = stocks.corr().values
    eigvals = np.sort(np.linalg.eigvalsh(corr))[::-1]
    # Marchenko-Pastur theoretical bounds
    T, N = stocks.shape
    q = T / N
    lam_max = (1 + np.sqrt(1/q)) ** 2 if q < 1 else None
    lam_min = (1 - np.sqrt(1/q)) ** 2 if q < 1 else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(eigvals)), eigvals, width=1.0, color="#4a6fa5", alpha=0.7)
    if lam_max:
        ax.axhline(lam_max, color="red", linestyle="--", label=f"MP upper bound ≈ {lam_max:.1f}")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalue spectrum of the correlation matrix  ({N} assets, {T} observations)\nMost eigenvalues are within the random-noise band → matrix is fragile")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    save(fig, "eigenvalues")


def precompute_dendrogram(stocks):
    print("\n[dendrogram]")
    sample = stocks.sample(n=min(150, stocks.shape[1]), axis=1, random_state=42)
    corr = sample.corr()
    distance = 1 - corr
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="ward")

    for n_clusters in (5, 20, 50):
        fig, ax = plt.subplots(figsize=(14, 5))
        dendrogram(Z, labels=list(sample.columns), ax=ax, leaf_font_size=5,
                   no_labels=True,
                   color_threshold=Z[-(n_clusters - 1), 2])
        ax.set_title(f"Hierarchical clustering of 150 sampled assets — cut into {n_clusters} clusters")
        plt.tight_layout()
        save(fig, f"dendrogram_n{n_clusters}")


def precompute_n_sensitivity(returns):
    print("\n[N-clusters sensitivity]")
    results = []
    for n in [5, 10, 20, 30, 50, 75, 100]:
        try:
            bt = backtest_strategy(returns, n_clusters=n, train_months=36, rebal_months=3)
            stats = perf_stats(bt)
            results.append({"n": n, **stats})
            print(f"  n={n}: realized Sharpe {stats.get('sharpe', 0):.2f}")
        except Exception as e:
            print(f"  n={n}: FAILED ({e})")

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["n"], df["sharpe"], "o-", linewidth=2, color="#4a6fa5", markersize=10)
    ax.set_xlabel("Number of clusters (N)")
    ax.set_ylabel("Realized OOS Sharpe")
    ax.set_title("Sensitivity: realized Sharpe vs portfolio size\nFinding the sweet spot for diversification")
    ax.axhline(0, color="grey", linewidth=0.5)
    plt.tight_layout()
    save(fig, "n_sensitivity")
    df.to_json(OUT_DIR / "n_sensitivity.json", orient="records")


def precompute_main_backtest(returns):
    print("\n[main backtest with N=50]")
    bt = backtest_strategy(returns, n_clusters=50, train_months=36, rebal_months=3)
    bt.to_frame("ret").to_parquet(OUT_DIR / "backtest_returns.parquet")
    spy = get_benchmark(returns, "SPY")
    spy_oos = spy.loc[bt.index].dropna()

    strat_stats = perf_stats(bt)
    spy_stats   = perf_stats(spy_oos)

    metrics = {
        "strategy": {k: float(v) for k, v in strat_stats.items()},
        "spy":      {k: float(v) for k, v in spy_stats.items()},
        "n_periods": len(bt),
        "start": str(bt.index[0].date()),
        "end":   str(bt.index[-1].date()),
    }
    with open(OUT_DIR / "headline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  metrics:", metrics)

    # === Hero equity curve ===
    eq_s = (1 + bt).cumprod()
    eq_b = (1 + spy_oos).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq_s.index, eq_s.values, label=f"Clustering strategy  (Sharpe {strat_stats['sharpe']:.2f})",
            linewidth=2.5, color="#2c4a7a")
    ax.plot(eq_b.index, eq_b.values, label=f"SPY benchmark  (Sharpe {spy_stats['sharpe']:.2f})",
            linewidth=2.5, color="#a55c3f", alpha=0.85)
    ax.set_title("Out-of-sample equity curve  ($1 invested at start)")
    ax.set_ylabel("Cumulative value")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "hero_equity")

    # === Drawdown underwater ===
    dd_s = eq_s / eq_s.cummax() - 1
    dd_b = eq_b / eq_b.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(dd_s.index, dd_s.values, 0, color="#2c4a7a", alpha=0.5, label="Strategy")
    ax.fill_between(dd_b.index, dd_b.values, 0, color="#a55c3f", alpha=0.4, label="SPY")
    ax.set_title("Drawdown (underwater chart) — worst peak-to-trough during OOS period")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "drawdown")

    # === Rolling 12-month Sharpe ===
    window = 12
    if len(bt) >= window:
        rolling_sh = ((bt.rolling(window).mean() * 12 - 0.04) /
                      (bt.rolling(window).std() * np.sqrt(12)))
        rolling_sh_spy = ((spy_oos.rolling(window).mean() * 12 - 0.04) /
                          (spy_oos.rolling(window).std() * np.sqrt(12)))
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(rolling_sh.index, rolling_sh.values, label="Strategy", linewidth=2, color="#2c4a7a")
        ax.plot(rolling_sh_spy.index, rolling_sh_spy.values, label="SPY", linewidth=2, color="#a55c3f", alpha=0.8)
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title("Rolling 12-month Sharpe ratio")
        ax.set_ylabel("Sharpe")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        save(fig, "rolling_sharpe")

    # === Cumulative outperformance vs SPY ===
    diff = eq_s - eq_b
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(diff.index, diff.values, 0, where=(diff.values >= 0), color="#3a7a3a", alpha=0.5, label="Outperforming")
    ax.fill_between(diff.index, diff.values, 0, where=(diff.values < 0), color="#a55c3f", alpha=0.5, label="Underperforming")
    ax.set_title("Cumulative outperformance vs SPY")
    ax.set_ylabel("Strategy − SPY (dollars per $1 invested)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "outperformance")

    return bt


def precompute_in_vs_out_sample(returns):
    print("\n[in-sample vs out-of-sample bias-variance]")
    from explorer_lib import optimize_portfolio
    train_months = 36
    rebal_months = 3
    records = []
    start_idx = train_months
    while start_idx + rebal_months <= len(returns):
        win = returns.iloc[start_idx - train_months: start_idx]
        coverage = win.notna().mean()
        train = win.loc[:, coverage >= 0.9].fillna(0.0)
        test  = returns.iloc[start_idx: start_idx + rebal_months]
        try:
            res = optimize_portfolio(train, n_clusters=50)
            w = res["weights"][res["weights"] > 0]
            in_sample_sharpe = res["performance"]["sharpe"]
            common = w.index.intersection(test.columns)
            oos_ret = (test[common].fillna(0.0) * w[common]).sum(axis=1)
            oos_ann = (1 + oos_ret.mean()) ** 12 - 1
            oos_vol = oos_ret.std() * np.sqrt(12)
            oos_sh  = (oos_ann - 0.04) / oos_vol if oos_vol > 0 else 0
            records.append({
                "date": str(test.index[0].date()),
                "in_sample": in_sample_sharpe,
                "out_of_sample": oos_sh,
            })
        except Exception as e:
            print(f"  skip {e}")
        start_idx += rebal_months

    df = pd.DataFrame(records)
    df.to_json(OUT_DIR / "bias_variance.json", orient="records")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    width = 0.4
    ax.bar(x - width/2, df["in_sample"], width, label="In-sample Sharpe (predicted)", color="#a55c3f")
    ax.bar(x + width/2, df["out_of_sample"], width, label="Out-of-sample Sharpe (realized)", color="#2c4a7a")
    ax.set_xticks(x)
    ax.set_xticklabels(df["date"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("The estimation gap: what the optimizer promised vs what actually happened")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "bias_variance")


def main():
    returns = load_returns()
    print(f"Loaded {returns.shape[0]} months × {returns.shape[1]} assets")
    stocks = precompute_universe(returns)
    precompute_correlation_heatmap(stocks)
    precompute_eigenvalue_spectrum(stocks)
    precompute_dendrogram(stocks)
    precompute_main_backtest(returns)
    precompute_in_vs_out_sample(returns)
    precompute_n_sensitivity(returns)
    print("\nAll story assets generated in data/story/")


if __name__ == "__main__":
    main()
