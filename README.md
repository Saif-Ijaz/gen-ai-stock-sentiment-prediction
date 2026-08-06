# GenAI Financial Sentiment & Stock Prediction

An AI-powered financial analysis platform that combines market data, financial-news sentiment, technical indicators, machine learning, retrieval-augmented generation (RAG), trading signals, and historical backtesting in an interactive Streamlit dashboard.

> **Disclaimer:** This project is for educational and research purposes only. It does not provide financial advice or guarantee investment returns.

## Features

- Live market-price display for selected stocks.
- Financial-news collection and **FinBERT** sentiment analysis.
- XGBoost-based short-term market-direction prediction.
- Technical indicators including RSI, MACD, Bollinger Bands, ADX, EMA crossover, volume ratio, and stochastic oscillator.
- Multiple trading strategies, including RSI, MACD, Bollinger Bands, EMA crossover, trend following, volume confirmation, and combined strategies.
- Strategy backtesting with portfolio equity curve, total return, win rate, max drawdown, profit factor, and Sharpe ratio.
- RAG-based analysis of relevant financial news.
- Interactive Streamlit dashboard with Plotly visualizations.

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Machine Learning:** XGBoost, Scikit-learn
- **NLP / Sentiment Analysis:** FinBERT, Hugging Face Transformers, PyTorch
- **RAG:** LangChain, ChromaDB, Sentence Transformers
- **Market Data:** yfinance, Polygon client
- **Data and Visualization:** Pandas, NumPy, Plotly

## Project Structure

```text
├── app/                 # Streamlit dashboard, API, live signals, backtesting
├── config/              # Environment-based application settings
├── features/            # Feature engineering for machine learning
├── indicators/          # Technical-indicator calculations
├── inference/           # Model inference and RAG prediction workflow
├── models/              # XGBoost model, training, tuning, and retraining code
├── pipelines/           # News, stock-data, and alignment pipelines
├── rag/                 # Embeddings, ingestion, vector store, and retrieval
├── sentiment/           # FinBERT sentiment analysis
├── utils/               # Logging and validation utilities
├── requirements.txt     # Python dependencies
└── main.py              # Training / pipeline entry point
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Saif-Ijaz/gen-ai-stock-sentiment-prediction.git
cd gen-ai-stock-sentiment-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows PowerShell**

```powershell
venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a file named `.env` in the project root. Do not upload this file to GitHub.

```env
NEWS_API_KEY=your_newsapi_key
NEWSDATA_API_KEY=your_newsdata_key
GROQ_API_KEY=your_groq_api_key
FMP_API_KEY=your_fmp_api_key
```

Only configure the keys required by the services you use. The application reads them from `config/settings.py`.

### 5. Run the dashboard

```bash
streamlit run app/dashboard.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Model and Prediction Workflow

1. Download historical OHLCV market data.
2. Compute technical features such as RSI, moving averages, MACD, ATR, and volume ratios.
3. Collect and analyse relevant financial news with FinBERT.
4. Combine technical and sentiment features for XGBoost direction prediction.
5. Display the model outlook, strategy signal, price chart, news sentiment, and key drivers.
6. Replay historical data to evaluate trading strategies through backtesting.

## Backtesting Metrics

- **Total Return:** Overall portfolio gain or loss.
- **Win Rate:** Percentage of completed trades that are profitable.
- **Max Drawdown:** Largest fall from a previous portfolio peak.
- **Profit Factor:** Gross profit divided by gross loss.
- **Sharpe Ratio:** Return relative to portfolio volatility.

## Deployment

This Streamlit project can be deployed on **Streamlit Community Cloud**.

- Repository: `Saif-Ijaz/gen-ai-stock-sentiment-prediction`
- Branch: `main`
- Main file path: `app/dashboard.py`

Add required API keys through the platform's secrets or environment-variable settings. Never commit keys or `.env` files to the repository.

## Limitations

- Stock-market behaviour is uncertain; the model is not a guarantee of future price movement.
- Historical backtests do not guarantee live-trading performance.
- Results can be affected by data availability, market regime, transaction costs, slippage, and news quality.
- The project is intended for analysis and learning, not automated investment execution.

## Author

**Saif Ijaz**  
Data Scientist | Generative AI, Machine Learning & NLP Developer
