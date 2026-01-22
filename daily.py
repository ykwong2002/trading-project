import requests
import os
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

pipe = pipeline("text-classification", model="ProsusAI/finbert")

def get_day_sentiment(keyword: str, from_date: str, to_date: str) -> float:
    """
    Get the sentiment of a keyword for a given day.

    Args:
        :keyword: The keyword to search for.
        :from_date: The start date to search from.
        :to_date: The end date to search to.

    Returns:
        Sentiment: The sentiment of the keyword for the given day. (can be positive, negative, or neutral)
        Sentiment Score: The score of the sentiment of the keyword for the given day. (0.0 to 1.0)
    """
    API_KEY = os.getenv('API_KEY')
    POS_THRESHOLD = 0.15
    NEG_THRESHOLD = -0.15

    total_score = 0
    num_articles = 0

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={keyword}&"
        f"from={from_date}&"
        f"to={to_date}&"
        "sortBy=popularity&"
        f"apiKey={API_KEY}"
    )

    # Initialize total score and number of articles
    total_score = 0
    num_articles = 0

    response = requests.get(url)

    articles = response.json()['articles']
    #Filter articles to only include articles that contain keyword in title or description
    articles = [article for article in articles if keyword.lower() in article['title'].lower() or keyword.lower() in article['description'].lower()]

    for i, article in enumerate(articles):
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Link: {article['url']}")
        print(f"Date: {article['publishedAt']}")

        # Get sentiment of article
        sentiment = pipe(article['content'])[0]
        print(f"Sentiment: {sentiment['label']}, Sentiment Score: {sentiment['score']}")
        print("-" * 50)

        if sentiment['label'] == "neutral":
            continue # Ignore neutral articles
        elif sentiment['label'] == "positive":
            total_score += sentiment['score']
        elif sentiment['label'] == "negative":
            total_score -= sentiment['score']

        num_articles += 1

    final_score = total_score / num_articles if num_articles > 0 else 0
    print(f"Overall Sentiment: {"Positive" if final_score >= POS_THRESHOLD else "Negative" if final_score <= NEG_THRESHOLD else "Neutral"} {final_score}")
    return final_score