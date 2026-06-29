# Portfolio Explorer

Interactive notebook that builds globally diversified equity portfolios by clustering ~2,000 stocks and running max-Sharpe optimization.

## How to run (Google Colab — recommended for sharing)

1. Click the **Open in Colab** link (or open `portfolio_explorer.ipynb` from the shared Google Drive folder via right-click → **Open with → Google Colaboratory**).
2. In Google Drive, right-click the shared **Portfolio** folder → **Add shortcut to Drive** → place it directly under *My Drive*. This step is required so the notebook can find the data files.
3. In Colab, click **Runtime → Run all**. The first cell will:
   - Install dependencies
   - Mount your Google Drive
   - Set the working directory to the Portfolio folder
4. Play with the levers in the **Controls** section and click **Optimize**.

## How to run locally

```bash
pip install yfinance PyPortfolioOpt seaborn ipywidgets lxml python-dateutil jupyter
cd Portfolio
python3 -m notebook portfolio_explorer.ipynb
```

## What's in the box

| File | Purpose |
|---|---|
| `portfolio_explorer.ipynb` | Interactive notebook (main entry point) |
| `explorer_lib.py` | Pipeline functions: filter, cluster, optimize, backtest |
| `fetch_data.py` | Downloads monthly price data via yfinance |
| `get_holdings.py` | Builds ticker universe from S&P 500, FTSE 100, DAX, Nikkei holdings, etc. |
| `optimize.py` | Standalone optimizer (CLI) |
| `backtest.py` | Standalone walk-forward backtest (CLI) |
| `data/returns.parquet` | Pre-cached monthly returns (~2,000 assets × 60 months) |
| `data/holdings.parquet` | Pre-cached ticker universe |

## Methodology in one paragraph

The strategy assembles ~2,000 global stocks from S&P 500, Russell 1000, NASDAQ 100, FTSE 100/250, DAX, CAC 40, SMI, AEX, OMXS 30, TSX 60, ASX 200, Hang Seng, KOSPI 200, Nifty 50, and top holdings from 30+ international ETFs. At each rebalance it computes the return-correlation matrix on the full universe, runs hierarchical (Ward) clustering to group similar stocks into N clusters, picks the highest-individual-Sharpe representative from each cluster, then runs Markowitz max-Sharpe optimization on those N representatives using **Ledoit-Wolf shrinkage** for the covariance matrix. Out-of-sample performance is measured by a quarterly walk-forward backtest with a 36-month rolling training window.

## Levers

- **# clusters** — portfolio size (5-100)
- **Min weight** — floor per holding
- **Max weight** — cap per holding
- **Min individual Sharpe** — drop weak assets before clustering
- **Pinned tickers** — force-include specific stocks
- **Excluded tickers** — exclude specific stocks
- **Regions** — restrict the universe to chosen countries
- **Run backtest** — toggle the walk-forward out-of-sample test
