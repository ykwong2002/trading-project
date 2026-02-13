import pandas as pd
import pandas_ta as ta


def preprocess_daily(csv_path: str) -> pd.DataFrame:
    """
    Load a daily OHLCV CSV and add technical indicators (MA20, MA50, RSI).
    Drops rows with NaN (e.g. warmup for MA50). Returns DataFrame with date index
    and columns: Open, High, Low, Close, Volume, ma_20, ma_50, rsi.
    """
    df = pd.read_csv(csv_path)
    # Assume first column is date/datetime
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "Date"
    df.sort_index(inplace=True)
    # Standardize column names to capitalize
    rename_map = {c: c.capitalize() for c in df.columns if isinstance(c, str) and c.lower() in ("open", "high", "low", "close", "volume") and c != c.capitalize()}
    if rename_map:
        df = df.rename(columns=rename_map)
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df.dropna(inplace=True)
    return df