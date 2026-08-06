from pipelines.stock_pipeline import fetch_stock_data
from pipelines.news_pipeline import fetch_news
from sentiment.sentiment_pipeline import score_news_sentiment
from indicators.technicals import add_technical_indicators
from pipelines.alignment import align_news_with_stock
from features.feature_engineering import build_features
from models.train import train_xgb
from rag.retriever import hybrid_retrieve
from rag.query_expansion import expand_financial_query

ticker = "AAPL"

# --- Data ---
stock_df = add_technical_indicators(fetch_stock_data(ticker))
news_df = fetch_news(ticker)
sentiment_df = score_news_sentiment(news_df)
aligned_df = align_news_with_stock(stock_df, news_df)

# --- Features ---
X, y = build_features(stock_df, sentiment_df, aligned_df)
print(stock_df.columns)

# --- Train Model ---
model, acc = train_xgb(X, y)
print("XGBoost Accuracy:", acc)

# --- RAG Retrieval ---
query = expand_financial_query(ticker, "market outlook")
docs = hybrid_retrieve(query)

print("Retrieved News Context:")
for d in docs[:3]:
    print("-", d["text"][:120])
