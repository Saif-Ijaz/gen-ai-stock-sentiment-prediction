from sentiment.finbert_model import finbert_predict
import pandas as pd

def score_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    sentiments = []

    for _, row in news_df.iterrows():
        text = f"{row['title']} {row.get('description', '')}"

        sentiment, confidence, scores = finbert_predict(text)

        sentiments.append({
            "title": row["title"],
            "publishedAt": row["publishedAt"],
            "sentiment": sentiment,
            "confidence": confidence,
            "positive": scores["positive"],
            "neutral": scores["neutral"],
            "negative": scores["negative"]
        })

    return pd.DataFrame(sentiments)
