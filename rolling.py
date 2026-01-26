from daily import get_day_sentiment
from datetime import datetime, timedelta

def get_rolling_sentiment(keyword: str, date: str):
    """
    Get the rolling sentiment of a keyword for the last 5 days.

    Args:
        :keyword: The keyword ot search for.
        :date: The date to consolidate the past 5 days of sentiment for.

    Returns:
        Rolling Sentiment: The weighted rolling sentiment of the keyword for the last 5 days.
    """
    scores = []
    for days_ago in range(1, 6):
        prior_date_str = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Get the sentiment of the keyword for the prior date
        sentiment = get_day_sentiment(keyword, prior_date_str, prior_date_str)
        
        # Append weighted score to list
        # Closer dates are weighted more heavily
        scores.append(sentiment * (6 - days_ago))
        print(f"Score for {prior_date_str}: {sentiment * (6 - days_ago)}")
    
    print(f"Weighted Scores for {keyword} on {date} over the past 5 days: {scores}")
    # Calculate the weighted rolling sentiment
    weighted_rolling_sentiment = sum(scores) / len(scores)
    return weighted_rolling_sentiment

if __name__ == "__main__":
    print(get_rolling_sentiment("nvidia", "2026-01-20"))