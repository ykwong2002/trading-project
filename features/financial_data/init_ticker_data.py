from pathlib import Path
import os

import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[1] # Features folder
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)

def initialise_ticker_data(ticker: str):
    df = yf.download(
        ticker,
        start="2000-01-01",
        auto_adjust=True # Adjust prices for splits and dividends
    )
    df.to_csv(DATA_DIR / f"{ticker}.csv")

if __name__ == "__main__":
    initialise_ticker_data("GOLD") # Edit this to the ticker you want to initialise


