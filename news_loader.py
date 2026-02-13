"""
News loader interface: returns 30-day rolling text for a given date and asset/keyword.
Implementations: CSV of (date, asset_key, text) or stub; no API keys in code.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Optional: path to CSV with columns [date, asset_key, text] or [date, asset_key, headline]
NEWS_CSV_PATH = Path(__file__).resolve().parent / "data" / "news.csv"


def get_news_for_date(
    date: datetime,
    asset_key: str,
    window_days: int = 30,
    news_csv_path: Optional[Path] = None,
) -> List[str]:
    """
    Return list of text snippets (headlines or body) for the last `window_days` days
    ending on `date`, for the given `asset_key`. Used to build sentiment context.

    If no CSV exists or no rows match, returns a stub list (e.g. one placeholder string)
    so the pipeline still runs; replace with real data or API later.
    """
    path = news_csv_path or NEWS_CSV_PATH
    if not path.exists():
        return _stub_news(asset_key, date, window_days)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    end = pd.Timestamp(date)
    start = end - timedelta(days=window_days)
    mask = (df["date"] >= start) & (df["date"] <= end)
    if "asset_key" in df.columns:
        mask = mask & (df["asset_key"].astype(str).str.lower() == asset_key.lower())
    rows = df.loc[mask]
    text_col = "text" if "text" in df.columns else "headline"
    if text_col not in df.columns:
        return _stub_news(asset_key, date, window_days)
    texts = rows[text_col].dropna().astype(str).tolist()
    if not texts:
        return _stub_news(asset_key, date, window_days)
    return texts


def _stub_news(asset_key: str, date: datetime, window_days: int) -> List[str]:
    """Placeholder when no news data is available."""
    return [f"Placeholder news for {asset_key} as of {date.date()} (window={window_days}d). Add data/news.csv or connect an API."]
