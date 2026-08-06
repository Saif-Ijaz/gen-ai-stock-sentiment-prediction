import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

TIMEZONE = "US/Eastern"
STOCK_INTERVAL = "5m"
