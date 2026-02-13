"""
Train RLVR: loop over dates and assets; shared backbone + per-asset heads;
reward = verified next-day return. Saves policy to config.MODEL_DIR.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ASSETS, DATA_DIR, MODEL_DIR, NEWS_WINDOW_DAYS
from indicators import preprocess_daily
from news_loader import get_news_for_date
from sentiment import texts_to_sentiment_vector
from policies import RLVRPolicy, bandit_loss


OBS_DIM = 6  # sentiment (3) + ma_20, ma_50, rsi (3)


def precompute_sentiment_cache(asset_keys, date_index, window_days=NEWS_WINDOW_DAYS):
    """Build (date, asset_key) -> sentiment vector for all dates and assets."""
    from datetime import datetime
    cache = {}
    for date in date_index:
        if hasattr(date, "to_pydatetime"):
            dt = date.to_pydatetime()
        else:
            dt = date
        for asset in asset_keys:
            texts = get_news_for_date(dt, asset, window_days=window_days)
            vec = texts_to_sentiment_vector(texts)
            cache[(date, asset)] = vec
    return cache


def run_training(
    asset_keys,
    data_paths,
    epochs=3,
    lr=1e-3,
    device=None,
    use_sentiment_cache=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Load preprocessed data per asset
    dfs = {}
    for key, ticker in asset_keys:
        path = data_paths.get(key, DATA_DIR / f"{ticker}_daily.csv")
        if not Path(path).exists():
            raise FileNotFoundError(f"Run download first: {path}")
        dfs[key] = preprocess_daily(str(path))

    # Joint date range (dates where we have next day for all assets)
    index_sets = [set(df.index[:-1]) for df in dfs.values()]
    common_dates = sorted(index_sets[0].intersection(*index_sets[1:]))
    if not common_dates:
        raise ValueError("No overlapping dates across assets")

    # Precompute sentiment for all (date, asset)
    if use_sentiment_cache:
        sentiment_cache = precompute_sentiment_cache(
            [a[0] for a in asset_keys],
            pd.DatetimeIndex(common_dates),
        )
    else:
        sentiment_cache = {}

    def get_obs(date, asset_key):
        df = dfs[asset_key]
        row = df.loc[date]
        if (date, asset_key) in sentiment_cache:
            sent = sentiment_cache[(date, asset_key)]
        else:
            from datetime import datetime
            dt = date.to_pydatetime() if hasattr(date, "to_pydatetime") else date
            texts = get_news_for_date(dt, asset_key, window_days=NEWS_WINDOW_DAYS)
            sent = texts_to_sentiment_vector(texts)
        tech = np.array([float(row["ma_20"]), float(row["ma_50"]), float(row["rsi"])], dtype=np.float32)
        return np.concatenate([np.asarray(sent, dtype=np.float32), tech])

    def next_day_return(asset_key, date):
        df = dfs[asset_key]
        idx = df.index.get_loc(date)
        if idx >= len(df) - 1:
            return 0.0
        close_now = float(df.iloc[idx]["Close"])
        close_next = float(df.iloc[idx + 1]["Close"])
        return (close_next - close_now) / close_now if close_now else 0.0

    policy = RLVRPolicy(OBS_DIM, [a[0] for a in asset_keys]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    reward_scale = 100.0

    for epoch in range(epochs):
        total_loss = 0.0
        n_updates = 0
        for i, date in enumerate(common_dates):  # next_day_return uses current date's next row
            for asset_key in [a[0] for a in asset_keys]:
                obs = get_obs(date, asset_key)
                with torch.no_grad():
                    logits = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device), asset_key)
                    probs = torch.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                ret = next_day_return(asset_key, date)
                if action == 0:
                    reward = 0.0
                elif action == 1:
                    reward = ret * reward_scale
                else:
                    reward = -ret * reward_scale

                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                loss = bandit_loss(policy, obs_t, asset_key, action, reward)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_updates += 1
        print(f"Epoch {epoch + 1}/{epochs} loss={total_loss / max(n_updates, 1):.4f} updates={n_updates}")

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "asset_keys": [a[0] for a in asset_keys]}, MODEL_DIR / "rlvr_policy.pt")
    print(f"Saved policy to {MODEL_DIR / 'rlvr_policy.pt'}")
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-sentiment-cache", action="store_true", help="Compute sentiment on the fly (slower)")
    args = parser.parse_args()
    asset_keys = ASSETS
    data_paths = {key: DATA_DIR / f"{ticker}_daily.csv" for key, ticker in asset_keys}
    run_training(
        asset_keys,
        data_paths,
        epochs=args.epochs,
        lr=args.lr,
        use_sentiment_cache=not args.no_sentiment_cache,
    )


if __name__ == "__main__":
    main()
