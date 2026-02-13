"""
Contextual bandit environment for next-day market prediction.
One step per day; observation = sentiment vector + technical indicators (MA20, MA50, RSI);
action = long / short / no_trade; reward = verified next-day return.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from typing import Optional, Callable, Dict, Any

from news_loader import get_news_for_date
from sentiment import texts_to_sentiment_vector
from config import NEWS_WINDOW_DAYS, ACTION_SPACE_SIZE


def _default_sentiment(date: datetime, asset_key: str) -> np.ndarray:
    texts = get_news_for_date(date, asset_key, window_days=NEWS_WINDOW_DAYS)
    return texts_to_sentiment_vector(texts)


class NewsBanditEnv(gym.Env):
    """
    Daily bandit: at each step observe (sentiment + ma_20, ma_50, rsi) for current day,
    take action (0=no_trade, 1=long, 2=short), receive reward from next day's realized return.
    """

    metadata = {"render_mode": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        asset_key: str = "gold",
        get_sentiment: Optional[Callable[[datetime, str], np.ndarray]] = None,
        reward_scale: float = 100.0,
    ):
        super().__init__()
        self.df = df.reset_index()
        if "Date" not in self.df.columns and self.df.shape[1] > 0:
            self.df = self.df.rename(columns={self.df.columns[0]: "Date"})
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.n_days = len(self.df)
        self.asset_key = asset_key
        self.get_sentiment = get_sentiment or _default_sentiment
        self.reward_scale = reward_scale

        # Observation: sentiment (3) + ma_20, ma_50, rsi (3) = 6
        self._sentiment_dim = 3
        self._tech_dim = 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._sentiment_dim + self._tech_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)  # 0 no_trade, 1 long, 2 short

        self.current_step = 0
        self._current_date = None

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        self._current_date = row["Date"]
        if isinstance(self._current_date, pd.Timestamp):
            self._current_date = self._current_date.to_pydatetime()
        sent = self.get_sentiment(self._current_date, self.asset_key)
        tech = np.array([
            float(row.get("ma_20", 0)),
            float(row.get("ma_50", 0)),
            float(row.get("rsi", 50)),
        ], dtype=np.float32)
        obs = np.concatenate([np.asarray(sent, dtype=np.float32), tech])
        return obs

    def _next_day_return(self) -> float:
        """Close-to-close return from current_step to current_step+1."""
        if self.current_step >= self.n_days - 1:
            return 0.0
        close_now = float(self.df.iloc[self.current_step]["Close"])
        close_next = float(self.df.iloc[self.current_step + 1]["Close"])
        if close_now <= 0:
            return 0.0
        return (close_next - close_now) / close_now

    def step(self, action: int):
        # Reward from *next* day's return (verified)
        ret = self._next_day_return()
        if action == 0:  # no trade
            reward = 0.0
        elif action == 1:  # long
            reward = ret * self.reward_scale
        else:  # short
            reward = -ret * self.reward_scale

        self.current_step += 1
        terminated = self.current_step >= self.n_days - 1
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape[0], dtype=np.float32)
        info = {"return": ret, "action": action}
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_observation()
        info = {}
        return obs, info
