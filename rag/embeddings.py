from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]):
    return embedding_model.encode(texts).tolist()
