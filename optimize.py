"""
Cluster assets by return correlation, then run max-Sharpe optimization
on one representative per cluster.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt


def cluster_assets(returns: pd.DataFrame, n_clusters: int) -> dict[str, list[str]]:
    """Return {cluster_id: [tickers]} using hierarchical clustering on correlation distance."""
    corr = returns.corr()
    distance = 1 - corr
    # squareform expects a condensed distance matrix
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    clusters: dict[int, list[str]] = {}
    for ticker, label in zip(returns.columns, labels):
        clusters.setdefault(int(label), []).append(ticker)
    return clusters, Z


def pick_representatives(clusters: dict[int, list[str]], returns: pd.DataFrame) -> list[str]:
    """From each cluster, pick the asset with the highest individual Sharpe ratio."""
    reps = []
    for tickers in clusters.values():
        sub = returns[tickers]
        sharpes = sub.mean() / sub.std() * np.sqrt(252)
        reps.append(sharpes.idxmax())
    return reps


def max_sharpe_portfolio(returns: pd.DataFrame, risk_free: float = 0.04, min_weight: float = 0.0):
    """Run max-Sharpe optimization on the given returns DataFrame.

    min_weight: floor for each asset (e.g. 0.02 forces at least 2% in every asset).
                Set to 0.0 to let the optimizer zero out assets freely.
    """
    mu = expected_returns.mean_historical_return(returns, returns_data=True, frequency=12)
    S = risk_models.CovarianceShrinkage(returns, returns_data=True, frequency=12).ledoit_wolf()
    n = len(returns.columns)
    weight_bounds = (min_weight, 1.0) if min_weight == 0.0 else (min_weight, 1.0 / n * 3)
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef.max_sharpe(risk_free_rate=risk_free)
    weights = ef.clean_weights()
    performance = ef.portfolio_performance(risk_free_rate=risk_free, verbose=True)
    return weights, performance


def plot_dendrogram(Z, tickers: list[str], n_clusters: int):
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, labels=tickers, ax=ax, leaf_rotation=90, leaf_font_size=7,
               color_threshold=Z[-(n_clusters - 1), 2])
    ax.set_title(f"Asset Clustering Dendrogram  (N={n_clusters} clusters)")
    plt.tight_layout()
    plt.savefig("data/dendrogram.png", dpi=150)
    print("Dendrogram saved to data/dendrogram.png")


def run(n_clusters: int = 10, risk_free: float = 0.04):
    from fetch_data import load_returns

    returns = load_returns()
    print(f"\nUniverse: {returns.shape[1]} assets, {returns.shape[0]} trading days")

    print(f"\nClustering into {n_clusters} groups...")
    clusters, Z = cluster_assets(returns, n_clusters)
    for cid, tickers in clusters.items():
        print(f"  Cluster {cid:2d}: {tickers}")

    reps = pick_representatives(clusters, returns)
    print(f"\nRepresentatives selected: {reps}")

    print("\nRunning max-Sharpe optimization...")
    # Min weight es el parametro de minimo por cluster
    weights, perf = max_sharpe_portfolio(returns[reps], risk_free=risk_free, min_weight=0.0)

    print("\nFinal portfolio weights:")
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        if w > 0.001:
            print(f"  {ticker:8s}  {w:.2%}")

    plot_dendrogram(Z, list(returns.columns), n_clusters)
    return weights, perf


if __name__ == "__main__":
    run(n_clusters=50)
