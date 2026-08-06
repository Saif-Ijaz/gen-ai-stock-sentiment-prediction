import numpy as np
from rag.retriever import hybrid_retrieve
from rag.query_expansion import expand_financial_query

def predict_with_rag(model, X_latest, ticker):
    prob = model.predict_proba(X_latest)[0][1]
    prediction = int(prob > 0.5)

    query = expand_financial_query(ticker, "market outlook")
    docs = hybrid_retrieve(query)

    return prediction, prob, docs
