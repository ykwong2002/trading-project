import yfinance as yf

def get_daily_data(Ticker: str):
    """
    Get the daily data for a given ticker.

    Args:
        :Ticker: The ticker to get the daily data for.

    Returns:
        Daily Data: A pandas dataframe with the daily data for the given ticker.
    """
    stock = yf.Ticker(Ticker)
    data = stock.info
    return data

if __name__ == "__main__":
    print(get_daily_data("AAPL"))