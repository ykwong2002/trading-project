import pandas as pd
import pandas_ta as ta

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Loads EURUSD data from CSV and preprocess it by adding technical indicators.
    Expects these columns: [GMT Time, O, H, L, C, V]
    """
    df = pd.read_csv(csv_path, parse_dates=True, index_col='Gmt time')

    # Sort data by date just in case
    df.sort_index(inplace=True)

    # Technical indicators
    df['rsi_14'] = ta.rsi(df['Close'], length=14)
    df['ma_20'] = ta.sma(df['Close'], length=20)
    df['ma_50'] = ta.sma(df['Close'], length=50)
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    df['ma_20_slope'] = df['ma_20'].diff()

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df
