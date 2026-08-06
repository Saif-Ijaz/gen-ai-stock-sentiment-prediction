import joblib

from pipelines.stock_pipeline import fetch_stock_data
from pipelines.news_pipeline import fetch_news
from indicators.technicals import add_technical_indicators
from sentiment.sentiment_pipeline import score_news_sentiment
from pipelines.alignment import align_news_with_stock
from features.feature_engineering import build_features

from models.retrain import retrain_model
from inference.predict_with_rag import predict_with_rag

from monitoring.metrics import log_model_performance, log_rag_usage


def main():
    ticker = "AAPL"

    # -------------------------
    # 1. DATA INGESTION
    # -------------------------
    stock_df = fetch_stock_data(ticker)
    stock_df = add_technical_indicators(stock_df)

    news_df = fetch_news(ticker)
    sentiment_df = score_news_sentiment(news_df)

    aligned_df = align_news_with_stock(stock_df, news_df)

    # -------------------------
    # 2. FEATURE ENGINEERING
    # -------------------------
    X, y = build_features(stock_df, sentiment_df, aligned_df)

    # -------------------------
    # 3. MODEL RETRAINING
    # -------------------------
    retrain_acc = retrain_model(X, y)

    bundle = joblib.load("models/model.pkl")
    model = bundle["model"]
    metadata = bundle["metadata"]

    # 🔍 MODEL MONITORING
    log_model_performance(metadata)

    # -------------------------
    # 4. INFERENCE + RAG
    # -------------------------
    X_latest = X.tail(1)

    prediction, prob, docs = predict_with_rag(
        model=model,
        X_latest=X_latest,
        ticker=ticker
    )

    print("\nPrediction:", "UP" if prediction == 1 else "DOWN")
    print("Confidence:", round(prob, 4))

    # 🔍 RAG MONITORING
    log_rag_usage(len(docs))

    if docs:
        print("\nTop RAG Context:")
        for d in docs[:2]:
            print("-", d["text"][:120])
    else:
        print("\nTop RAG Context: None (no relevant recent news)")


if __name__ == "__main__":
    main()
