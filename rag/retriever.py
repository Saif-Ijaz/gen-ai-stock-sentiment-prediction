from rag.vector_store import collection

def hybrid_retrieve(query, top_k=5, since_minutes=120):
    """
    Hybrid semantic + keyword retrieval with temporal filtering
    """

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({
            "text": doc,
            "publishedAt": meta.get("publishedAt")
        })

    return docs
