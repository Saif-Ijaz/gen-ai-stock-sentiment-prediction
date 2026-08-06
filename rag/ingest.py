# rag/ingest.py
import hashlib
from loguru import logger
from pipelines.news_pipeline import fetch_news
from rag.vector_store import collection


def ingest_news(ticker: str):
    """
    Fetch news and store into ChromaDB for RAG retrieval.
    Skips duplicates using content hash as ID.
    """
    df = fetch_news(ticker)

    if df.empty:
        logger.warning(f"No news to ingest for {ticker}")
        return 0

    documents = []
    metadatas = []
    ids = []
    seen_ids = set()  # FIX: deduplicate within this batch

    for _, row in df.iterrows():
        text = str(row.get("content", "")).strip()

        if not text or len(text) < 20:
            continue

        doc_id = hashlib.md5(text.encode()).hexdigest()

        # Skip duplicates within this batch
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        # Skip if already in ChromaDB
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        documents.append(text)
        metadatas.append({
            "ticker": ticker,
            "publishedAt": str(row.get("publishedAt", "")),
            "title": str(row.get("title", "")),
        })
        ids.append(doc_id)

    if not documents:
        logger.info(f"No new articles to ingest for {ticker} (all duplicates)")
        return 0

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Ingested {len(documents)} new articles for {ticker} into ChromaDB")
    return len(documents)


def ingest_all():
    """Ingest news for all supported tickers"""
    tickers = ["AAPL", "TSLA", "MSFT", "GOOGL"]
    total = 0
    for ticker in tickers:
        total += ingest_news(ticker)
    logger.info(f"Total ingested: {total} articles")
    return total