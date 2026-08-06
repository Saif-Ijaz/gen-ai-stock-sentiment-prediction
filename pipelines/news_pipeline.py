# pipelines/news_pipeline.py
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import pandas as pd
from loguru import logger

NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # keep as fallback

COMPANY_QUERY_MAP = {
    "AAPL": "Apple stock",
    "TSLA": "Tesla stock",
    "GOOGL": "Google Alphabet stock",
    "MSFT": "Microsoft stock",
    "AMZN":  "Amazon stock",
    "NVDA":  "Nvidia stock",
    "META":  "Meta Facebook stock",
    "NFLX":  "Netflix stock",
    "JPM":   "JPMorgan Chase stock",
    "V":     "Visa stock",
}

def fetch_news_newsdata(ticker: str) -> pd.DataFrame:
    """
    Fetch news using NewsData.io — free, production-ready.
    """
    query = COMPANY_QUERY_MAP.get(ticker, ticker)

    url = "https://newsdata.io/api/1/news"
    params = {
        "q": query,
        "language": "en",
        "category": "business,technology",
        "apikey": NEWSDATA_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = data.get("results", [])

        if not articles:
            logger.info(f"No news articles found for {ticker} via NewsData.io")
            return pd.DataFrame()

        rows = []
        for a in articles:
            title = a.get("title", "") or ""
            description = a.get("description", "") or ""

            # FIX: Combine title + description as usable content
            # Free tier content field says "ONLY AVAILABLE IN PAID PLANS"
            combined_content = f"{title}. {description}".strip(". ")

            rows.append({
                "publishedAt": a.get("pubDate"),
                "title": title,
                "description": description,
                "content": combined_content,
            })

        df = pd.DataFrame(rows)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True, errors="coerce")

        logger.info(f"Fetched {len(df)} articles for {ticker} via NewsData.io")
        return df

    except Exception as e:
        logger.error(f"NewsData.io fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_news_newsapi(ticker: str) -> pd.DataFrame:
    """
    Fallback: original NewsAPI (works on localhost only).
    """
    query = COMPANY_QUERY_MAP.get(ticker, ticker)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
        return df[["publishedAt", "title", "description", "content"]]

    except Exception as e:
        logger.error(f"NewsAPI fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_news(ticker: str) -> pd.DataFrame:
    """
    Main entry point — tries NewsData.io first, falls back to NewsAPI.
    """
    # Try NewsData.io first (production-ready, free)
    if NEWSDATA_API_KEY:
        df = fetch_news_newsdata(ticker)
        if not df.empty:
            return df
        logger.warning(f"NewsData.io returned empty for {ticker}, trying fallback...")

    # Fallback to NewsAPI (localhost only)
    if NEWS_API_KEY:
        logger.info(f"Falling back to NewsAPI for {ticker}")
        return fetch_news_newsapi(ticker)

    logger.error("No news API key found! Add NEWSDATA_API_KEY to .env")
    return pd.DataFrame()