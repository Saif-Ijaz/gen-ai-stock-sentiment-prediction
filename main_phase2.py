from pipelines.news_pipeline import fetch_news
from sentiment.sentiment_pipeline import score_news_sentiment
from rag.ingest import ingest_news

ticker = "AAPL"

# 1. Fetch live news
news_df = fetch_news(ticker)

# 2. Sentiment analysis
sentiment_df = score_news_sentiment(news_df)
print(sentiment_df.head())

# 3. Store in vector DB (RAG)
ingest_news(news_df)

print("Phase 2 ingestion complete.")
