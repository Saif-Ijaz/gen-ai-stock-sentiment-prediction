# rag/vector_store.py
import chromadb

# FIX: Use PersistentClient instead of deprecated Settings approach
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="financial_news"
)