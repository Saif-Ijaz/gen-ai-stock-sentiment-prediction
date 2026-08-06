def expand_financial_query(ticker, base_query):
    expansions = [
        f"{ticker} earnings",
        f"{ticker} stock movement",
        f"{ticker} analyst rating",
        f"{ticker} revenue forecast",
        base_query
    ]
    return " OR ".join(expansions)
