from src.phase2.embeddings.chroma_ingest import ingest_documents, get_client
from src.phase2.rag.loaders import split_article

sample_article = 'Apple announces new product. Market reacts positively. This is a test article to index.'
docs = []
metas = []
chunks = split_article(sample_article)
for i,c in enumerate(chunks):
    docs.append({'id':f'test_{i}', 'content':c})
    metas.append({'ticker':'AAPL', 'published_at':'2025-01-01T00:00:00Z', 'url':'http://example.com'})

ingest_documents(docs, metas, collection_name='test_news')
client = get_client()
coll = client.get_collection('test_news')
print('Collection size check:', len(coll.get()['ids']))

