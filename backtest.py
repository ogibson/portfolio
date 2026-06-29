"""
Walk-forward out-of-sample backtest of the cluster + max-Sharpe strategy.

At each rebalance date:
  1. Use the prior TRAIN_WEEKS as the training window
  2. Cluster, pick representatives, run max-Sharpe → get weights
  3. Hold those weights for the next REBAL_WEEKS (out-of-sample)
  4. Record the realized portfolio returns
  5. Move forward and repeat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fetch_data import load_returns
from optimize import cluster_assets, pick_representatives, max_sharpe_portfolio

TRAIN_WEEKS = 36    # 3 years of training (months)
REBAL_WEEKS = 3     # rebalance every quarter (3 months)
N_CLUSTERS = 50
RISK_FREE = 0.04
MIN_WEIGHT = 0.0


def walk_forward(returns: pd.DataFrame):
    """Run the walk-forward backtest. Returns a Series of weekly portfolio returns."""
    portfolio_returns = []
    portfolio_dates = []
    rebalance_log = []

    start_idx = TRAIN_WEEKS
    end_idx = len(returns)

    period = 0
    while start_idx + REBAL_WEEKS <= end_idx:
        train = returns.iloc[start_idx - TRAIN_WEEKS : start_idx].dropna(axis=1)
        test  = returns.iloc[start_idx : start_idx + REBAL_WEEKS]

        # Some assets in test may be missing — restrict to overlap
        common = train.columns.intersection(test.columns)
        train = train[common]
        test  = test[common].fillna(0.0)

        period += 1
        print(f"\nPeriod {period}: train {train.index[0].date()} → {train.index[-1].date()}  "
              f"({train.shape[1]} assets) | test {test.index[0].date()} → {test.index[-1].date()}")

        # Cluster + optimize on training data only
        try:
            clusters, _ = cluster_assets(train, N_CLUSTERS)
            reps = pick_representatives(clusters, train)
            weights, perf = max_sharpe_portfolio(train[reps], risk_free=RISK_FREE,
                                                 min_weight=MIN_WEIGHT)
        except Exception as e:
            print(f"  optimization failed: {e}")
            start_idx += REBAL_WEEKS
            continue

        # Apply weights to test period
        w = pd.Series(weights)
        w = w[w > 0]
        test_returns = test[w.index]
        period_returns = (test_returns * w).sum(axis=1)

        portfolio_returns.append(period_returns)
        portfolio_dates.append(test.index)
        rebalance_log.append({
            "period": period,
            "rebalance_date": test.index[0],
            "n_assets": int((w > 0.001).sum()),
            "train_sharpe": perf[2],
            "period_return": float((1 + period_returns).prod() - 1),
        })

        print(f"  weights on {(w > 0.001).sum()} assets | in-sample Sharpe {perf[2]:.2f} | "
              f"realized {period}-week return {(1 + period_returns).prod() - 1:.2%}")

        start_idx += REBAL_WEEKS

    all_returns = pd.concat(portfolio_returns)
    return all_returns, pd.DataFrame(rebalance_log)


def summarize(returns: pd.Series, name: str = "Portfolio"):
    annual_return = (1 + returns.mean()) ** 12 - 1
    annual_vol    = returns.std() * np.sqrt(12)
    sharpe        = (annual_return - RISK_FREE) / annual_vol
    equity        = (1 + returns).cumprod()
    drawdown      = equity / equity.cummax() - 1
    max_dd        = drawdown.min()
    total_return  = equity.iloc[-1] - 1

    print(f"\n=== {name} ===")
    print(f"  Total return:     {total_return:.2%}")
    print(f"  Annualized return:{annual_return:.2%}")
    print(f"  Annualized vol:   {annual_vol:.2%}")
    print(f"  Realized Sharpe:  {sharpe:.2f}")
    print(f"  Max drawdown:     {max_dd:.2%}")
    return {"equity": equity, "drawdown": drawdown, "sharpe": sharpe,
            "annual_return": annual_return, "annual_vol": annual_vol,
            "max_dd": max_dd}


def plot_results(portfolio_eq: pd.Series, benchmark_eq: pd.Series = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(portfolio_eq.index, portfolio_eq.values, label="Cluster Max-Sharpe", linewidth=2)
    if benchmark_eq is not None:
        ax.plot(benchmark_eq.index, benchmark_eq.values, label="SPY", linewidth=2, alpha=0.7)
    ax.set_title("Out-of-Sample Equity Curve")
    ax.set_ylabel("Cumulative return (1 = starting capital)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/backtest_equity.png", dpi=150)
    print("\nEquity curve saved to data/backtest_equity.png")


def run():
    returns = load_returns()
    print(f"Loaded {returns.shape[0]} weeks × {returns.shape[1]} assets")
    print(f"Train window: {TRAIN_WEEKS} weeks | Rebalance every: {REBAL_WEEKS} weeks | "
          f"Clusters: {N_CLUSTERS}")

    portfolio_returns, log = walk_forward(returns)
    print("\n" + "=" * 60)
    port_stats = summarize(portfolio_returns, "Cluster Max-Sharpe (out-of-sample)")

    # Benchmark: SPY over the same out-of-sample period
    spy = returns["SPY"] if "SPY" in returns.columns else None
    bench_stats = None
    if spy is not None:
        spy_oos = spy.loc[portfolio_returns.index].dropna()
        bench_stats = summarize(spy_oos, "SPY benchmark (same period)")
        plot_results(port_stats["equity"], bench_stats["equity"])
    else:
        plot_results(port_stats["equity"])

    print("\n--- Rebalance log ---")
    print(log.to_string(index=False))
    return portfolio_returns, log


if __name__ == "__main__":
    run()
