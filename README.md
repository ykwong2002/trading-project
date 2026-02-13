# RLVR: News-Driven Market Prediction

A **keyword-specialized reinforcement learning with verifiable rewards (RLVR)** system for next-day market prediction. It turns 30-day rolling financial news into sentiment features (via a shared FinBERT backbone), combines them with technical indicators (MA20, MA50, RSI), and trains per-asset decision heads with **verified** next-day market returns—no human feedback.

- **Context**: 30-day news → FinBERT sentiment (3-d) + MA20, MA50, RSI  
- **Action**: Long / short / no trade (one decision per day)  
- **Reward**: Actual next-day close-to-close return, scaled  
- **Architecture**: Shared encoder + one decision head per asset (e.g. Gold, NVIDIA)

---

## What each file does

| File | Role |
|------|------|
| **config.py** | Paths (`data/`, `models/`), asset list and yfinance tickers, FinBERT model name, `NEWS_WINDOW_DAYS` (30), `ACTION_SPACE_SIZE` (3), `DEFAULT_CASH` (100,000 USD). |
| **data/download_yfinance.py** | Fetches past 10 years (configurable) of daily OHLCV from yfinance for given tickers; saves to `data/{ticker}_daily.csv`. |
| **indicators.py** | **`preprocess_daily(csv_path)`**: loads a daily OHLCV CSV, adds MA20, MA50, RSI, drops NaN; returns a DataFrame used by the bandit env and backtest. Also has `load_and_preprocess_data()` for legacy hourly data. |
| **news_loader.py** | **`get_news_for_date(date, asset_key, window_days=30)`**: returns the list of text snippets (e.g. headlines) for the last 30 days for that asset. Uses `data/news.csv` if present (columns: `date`, `asset_key`, `text`); otherwise returns a stub so the pipeline runs without real news. |
| **sentiment.py** | **`texts_to_sentiment_vector(texts)`**: shared FinBERT backbone (ProsusAI/finbert); maps a list of texts to a 3-d vector (positive, negative, neutral) aggregated over the window. |
| **bandit_env.py** | **`NewsBanditEnv`**: Gymnasium env. One step per day; observation = sentiment + ma_20, ma_50, rsi; action = 0 (no trade), 1 (long), 2 (short); reward = verified next-day return. |
| **policies.py** | **`SharedEncoder`**, **`DecisionHead`**, **`RLVRPolicy`** (shared encoder + per-asset heads), and **`bandit_loss()`** (REINFORCE-style) for training. |
| **train_rlvr.py** | Training script: loads preprocessed daily data per asset, (optionally) precomputes sentiment for all dates, loops over dates and assets, samples actions from the policy, gets reward from next-day return, updates policy with bandit loss. Saves `models/rlvr_policy.pt`. |
| **backtest.py** | Backtesting.py-based backtest: loads the saved policy, builds a custom `Strategy` that uses the RL policy (long/short/no trade) with context = sentiment + MA20, MA50, RSI. Runs with `cash=100_000` and prints stats + equity plot. |
| **requirements.txt** | Python dependencies: numpy, pandas, pandas-ta, gymnasium, shimmy, stable-baselines3, tensorboard, matplotlib, torch, transformers, yfinance, backtesting. |

---

## Setup and run

### 1. Environment

```bash
cd /path/to/trading-project
pip install -r requirements.txt
```

### 2. Download data (10 years of daily OHLCV)

```bash
python -m data.download_yfinance --tickers GLD NVDA --years 10
```

Output: `data/GLD_daily.csv`, `data/NVDA_daily.csv`.

### 3. Train the RLVR policy

```bash
python train_rlvr.py --epochs 3 --lr 1e-3
```

Uses `config.ASSETS` and data under `data/`. Saves `models/rlvr_policy.pt`. First run downloads FinBERT; use `--no-sentiment-cache` to compute sentiment on the fly (slower).

### 4. Backtest (USD 100,000)

```bash
python backtest.py --asset gold --cash 100000
# or
python backtest.py --asset nvidia --cash 100000
```

Uses the saved policy and preprocessed 10-year data; opens an interactive equity plot and prints summary statistics.

---

## Backtest results (example)

Running `backtest.py --asset gold --cash 100000` on the included 10-year GLD data (with a policy trained for a few epochs) produces stats in this vein (exact numbers depend on the trained policy and data):

- **Start / End**: First and last date in the preprocessed series (e.g. 2016–04–26 to 2026–02–12).  
- **Equity Final**: Ending equity in USD (e.g. ~\$379k if the strategy held a long position over a strong bull period).  
- **Return [%]**: Total return over the period.  
- **Buy & Hold Return [%]**: Benchmark return for holding the asset.  
- **Max. Drawdown [%]**: Largest peak-to-trough decline.  
- **Sharpe Ratio**, **Sortino Ratio**, **# Trades**, **Win Rate**, etc.: Standard Backtesting.py metrics.

The script uses **Backtesting.py** with `cash=100_000` and `finalize_trades=True`; the plot shows equity over time starting at \$100,000.

---

## How to test on unseen data

### Option A: Hold-out time period (unseen dates)

1. **Train on a subset of history**  
   - Restrict the dates used in training (e.g. only up to 2022) by filtering the preprocessed DataFrames in `train_rlvr.py` or by using CSVs that contain only that range.  
   - Save the policy as usual.

2. **Backtest on a later period**  
   - Use a CSV that contains only the hold-out period (e.g. 2023–2026), or pass a date-filtered DataFrame in `backtest.run_backtest()`.  
   - Run:  
     `python backtest.py --asset gold --cash 100000 --policy models/rlvr_policy.pt`  
   - The backtest script loads whatever data you point it at; use a separate file or filter in code so the backtest range does not overlap with training.

### Option B: New ticker (new asset)

1. **Download data for the new ticker**  
   ```bash
   python -m data.download_yfinance --tickers AAPL --years 10
   ```  
   This creates `data/AAPL_daily.csv`.

2. **Add the asset to config**  
   In `config.py`, add to `ASSETS`, e.g.:  
   `("apple", "AAPL")`.

3. **Retrain**  
   Run `python train_rlvr.py --epochs 3`. The policy has one head per asset; the new asset gets a new head. The saved `rlvr_policy.pt` will include the new asset.

4. **Backtest the new asset**  
   Extend `backtest.py` (or add a CLI option) so the asset key maps to the new ticker (e.g. `--asset apple` using ticker `AAPL`). Then:  
   `python backtest.py --asset apple --cash 100000`  
   This runs the policy on the new ticker’s preprocessed data (unseen in the sense of a new symbol; dates can overlap with training if you used the same 10-year window).

### Option C: Fully unseen CSV (new symbol or new range)

1. **Put the CSV in `data/`**  
   Format: first column = date, then `Open`, `High`, `Low`, `Close`, and optionally `Volume`. Same format as the yfinance output.

2. **Preprocess**  
   The backtest and env use `indicators.preprocess_daily(csv_path)` to add MA20, MA50, RSI. So pass the path to your CSV where the code expects a preprocessed frame (e.g. in `backtest.run_backtest()`).

3. **Run backtest**  
   - If the new CSV is for an **existing** asset key (e.g. you have a different or longer history for gold): change `backtest.py` (or add an argument) to use that CSV path for that asset, then run `python backtest.py --asset gold --cash 100000`.  
   - If the new CSV is for a **new** asset: add the asset and ticker to `config.ASSETS`, add the ticker to the download step (or skip download and point the code to your CSV), retrain so the policy has a head for that asset, then backtest that asset as in Option B.

### Option D: New ticker without retraining (policy has no head for it)

The current design has **one head per asset**. If you do **not** add the new ticker to `ASSETS` and retrain, there is no head for it, so you cannot run the existing `backtest.py` for that ticker without code changes (e.g. adding a “generic” head or training at least one epoch with the new asset). So for a truly new symbol, use Option B (add asset → retrain → backtest).

---

## Optional: Real news data

By default, `get_news_for_date()` returns stub text when `data/news.csv` is missing. To use real news:

1. Create **`data/news.csv`** with columns: **`date`**, **`asset_key`**, **`text`** (or **`headline`**).  
2. Ensure dates and `asset_key` values match what you use in training and backtest (e.g. `"gold"`, `"nvidia"`).  
3. Re-run training and backtest; sentiment will be computed from this file instead of the stub.

You can also replace the implementation in `news_loader.py` with an API (e.g. NewsAPI, Alpha Vantage); keep the same function signature and use environment variables or config for API keys.
