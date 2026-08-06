import plotly.graph_objects as go


def price_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price"
    ))

    fig.update_layout(
        title="Stock Price (5-minute)",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400
    )

    return fig
