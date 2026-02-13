"""
Backtest the trained RLVR policy over 10 years of data using the Backtesting.py library.
Starting equity USD 100,000; plot total equity via bt.plot() and print bt.stats.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from backtesting import Backtest, Strategy

from config import MODEL_DIR, DATA_DIR, DEFAULT_CASH, ASSETS
from indicators import preprocess_daily
from news_loader import get_news_for_date
from sentiment import texts_to_sentiment_vector
from policies import RLVRPolicy


OBS_DIM = 6
NEWS_WINDOW_DAYS = 30


def load_policy(path=None):
    path = path or (MODEL_DIR / "rlvr_policy.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    asset_keys = ckpt["asset_keys"]
    policy = RLVRPolicy(OBS_DIM, asset_keys)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    return policy, asset_keys


def make_rl_strategy(policy_path, asset_key, sentiment_cache=None):
    """
    Returns a Strategy subclass that uses the trained RL policy (long/short/no trade)
    with context = sentiment + ma_20, ma_50, rsi. cash=100_000 set in Backtest().
    """
    sentiment_cache = sentiment_cache or {}

    class RLStrategy(Strategy):
        def init(self):
            self.policy, self.asset_keys = load_policy(policy_path)
            self.asset_key = asset_key
            self._bar = 0
            # Precompute sentiment for every bar date so next() is fast
            self._sentiment_vecs = []
            for i in range(len(self.data.df)):
                date = self.data.df.index[i]
                if hasattr(date, "to_pydatetime"):
                    dt = date.to_pydatetime()
                else:
                    dt = date
                key = (date, asset_key)
                if key in sentiment_cache:
                    self._sentiment_vecs.append(sentiment_cache[key])
                else:
                    texts = get_news_for_date(dt, asset_key, window_days=NEWS_WINDOW_DAYS)
                    vec = texts_to_sentiment_vector(texts)
                    self._sentiment_vecs.append(vec)
            self._sentiment_vecs = np.array(self._sentiment_vecs, dtype=np.float32)

        def next(self):
            i = self._bar
            self._bar += 1
            if i >= len(self._sentiment_vecs):
                return
            row = self.data.df.iloc[i]
            sent = self._sentiment_vecs[i]
            tech = np.array([float(row.get("ma_20", 0)), float(row.get("ma_50", 0)), float(row.get("rsi", 50))], dtype=np.float32)
            obs = np.concatenate([sent, tech])
            action, _ = self.policy.get_action(obs, self.asset_key, deterministic=True)
            if action == 1:
                if not self.position:
                    self.buy()
            elif action == 2:
                if not self.position:
                    self.sell()
            # action == 0: no trade

    return RLStrategy


def run_backtest(
    asset_key: str,
    ticker: str,
    data_path: Path = None,
    policy_path: Path = None,
    cash: float = DEFAULT_CASH,
):
    data_path = data_path or (DATA_DIR / f"{ticker}_daily.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Run download first: {data_path}")
    df = preprocess_daily(str(data_path))
    # Backtesting.py expects Open, High, Low, Close (preprocess_daily keeps these)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Precompute sentiment for full series (for Strategy.init)
    sentiment_cache = {}
    for date in df.index:
        dt = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
        texts = get_news_for_date(dt, asset_key, window_days=NEWS_WINDOW_DAYS)
        sentiment_cache[(date, asset_key)] = np.array(texts_to_sentiment_vector(texts), dtype=np.float32)

    StrategyClass = make_rl_strategy(policy_path or MODEL_DIR / "rlvr_policy.pt", asset_key, sentiment_cache)
    bt = Backtest(df, StrategyClass, cash=cash, trade_on_close=True, finalize_trades=True)
    stats = bt.run()
    print(stats)
    bt.plot()
    return bt, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backtest RLVR policy with Backtesting.py")
    parser.add_argument("--asset", default="gold", choices=[a[0] for a in ASSETS], help="Asset key")
    parser.add_argument("--cash", type=float, default=DEFAULT_CASH, help="Starting cash (USD)")
    parser.add_argument("--policy", type=Path, default=MODEL_DIR / "rlvr_policy.pt", help="Path to policy checkpoint")
    args = parser.parse_args()
    ticker = dict(ASSETS).get(args.asset, "GLD")
    run_backtest(args.asset, ticker, cash=args.cash, policy_path=args.policy)


if __name__ == "__main__":
    main()
