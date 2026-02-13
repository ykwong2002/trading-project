"""
Download past 10 years of daily OHLCV from yfinance per asset; save to data/.
Usage: python -m data.download_yfinance [--tickers TICKER1 TICKER2 ...]
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# Default tickers if not passed
DEFAULT_TICKERS = ["GLD", "NVDA"]
DATA_DIR = Path(__file__).resolve().parent
YEARS_BACK = 10


def download_asset(ticker: str, years: int = YEARS_BACK, out_dir: Path = DATA_DIR) -> Path:
    """Fetch daily OHLCV for the past `years` years; save to out_dir/{ticker}_daily.csv."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty or len(df) < 2:
        raise ValueError(f"No or insufficient data for {ticker}")
    # yfinance can return MultiIndex columns; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # Normalize column names: yfinance may return Datetime/Date and Open, High, Low, Close, Volume
    if "Datetime" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    # Keep Date + OHLC; drop Adj Close; include Volume if present
    drop_cols = [c for c in df.columns if "Adj" in str(c)]
    df = df.drop(columns=drop_cols, errors="ignore")
    # Normalize OHLCV to capitalized (yfinance usually returns Open, High, etc.)
    for name in ["open", "high", "low", "close", "volume"]:
        if name in [c.lower() for c in df.columns] and name.capitalize() not in df.columns:
            old = [c for c in df.columns if c.lower() == name][0]
            df = df.rename(columns={old: name.capitalize()})
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    required = ["Open", "High", "Low", "Close"]
    cols = ["Date"] + [c for c in required if c in df.columns]
    if "Volume" in df.columns:
        cols.append("Volume")
    df = df[[c for c in cols if c in df.columns]]
    out_path = out_dir / f"{ticker}_daily.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download 10 years of daily data from yfinance")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to download (e.g. GLD NVDA)")
    parser.add_argument("--years", type=int, default=YEARS_BACK, help="Years of history")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Output directory")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        try:
            path = download_asset(ticker, years=args.years, out_dir=args.out_dir)
            print(f"Saved {path}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")


if __name__ == "__main__":
    main()
