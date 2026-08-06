import pandas as pd


def align_news_with_stock(stock_df, news_df, window_minutes=30):
    """
    Align news articles to stock candles safely.
    Fully robust against index/column inconsistencies.
    """

    if stock_df.empty:
        return stock_df

    stock_df = stock_df.copy()

    # --- Ensure datetime column exists ---
    if "datetime" not in stock_df.columns:
        stock_df = stock_df.reset_index()
        stock_df.rename(columns={"index": "datetime"}, inplace=True)

    # Convert stock datetime properly
    stock_df["datetime"] = pd.to_datetime(stock_df["datetime"], utc=True, errors="coerce")

    # --- If no news, attach empty columns ---
    if news_df.empty or "publishedAt" not in news_df.columns:
        stock_df["news_count"] = 0
        stock_df["news_titles"] = [[] for _ in range(len(stock_df))]
        return stock_df

    news_df = news_df.copy()
    news_df["publishedAt"] = pd.to_datetime(
        news_df["publishedAt"], utc=True, errors="coerce"
    )

    news_counts = []
    news_titles = []

    for stock_time in stock_df["datetime"]:

        # Ensure scalar Timestamp
        if pd.isna(stock_time):
            news_counts.append(0)
            news_titles.append([])
            continue

        lower_bound = stock_time - pd.Timedelta(minutes=window_minutes)

        relevant_news = news_df[
            (news_df["publishedAt"] <= stock_time) &
            (news_df["publishedAt"] >= lower_bound)
        ]

        news_counts.append(len(relevant_news))
        news_titles.append(relevant_news["title"].tolist())

    stock_df["news_count"] = news_counts
    stock_df["news_titles"] = news_titles

    return stock_df