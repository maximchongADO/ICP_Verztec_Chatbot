from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)
global_db = FAISS.load_local(
    "faiss_index3",
    embedding_model,
    allow_dangerous_deserialization=True
)
query = "How do I import e-invoice data from Xtranet to ABSS?"
results = global_db.similarity_search_with_score(query, k=10)

print('\n--- SIMILARITY SEARCH RESULTS ---')
for doc, score in results:
    print("\n---")
    print(doc.page_content)
    print(doc.metadata)
    print(f"Score: {score}")