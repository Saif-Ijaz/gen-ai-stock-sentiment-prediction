def prediction_prompt(ticker, prediction, prob, news_context):
    return f"""
You are a financial analyst AI.

Stock: {ticker}
Model Prediction: {"UP" if prediction == 1 else "DOWN"}
Model Confidence: {prob:.2f}

Relevant News Context:
{news_context}

Explain in clear financial terms why this prediction makes sense.
Focus on earnings, analyst sentiment, and market momentum.
"""
