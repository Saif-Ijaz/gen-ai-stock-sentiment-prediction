def log_model_performance(metadata):
    print("Model trained at:", metadata["trained_at"])
    print("Validation accuracy:", metadata["accuracy"])

def log_rag_usage(num_docs):
    print("RAG documents used:", num_docs)
