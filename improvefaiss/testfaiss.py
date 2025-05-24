import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def load_faiss_index(index_path="faiss_local.idx", meta_path="faiss_local_meta.json"):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def search_faiss_local(index, metadata, query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = metadata[idx]
        results.append({
            "distance": float(dist),
            "source_file": meta["source_file"],
            "chunk_index": meta["chunk_index"],
            "text": meta["text"]
        })
    return results

if __name__ == "__main__":
    index_path = r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-2\improvefaiss\faiss_local.idx"
    meta_path = r"C:\Users\ethan\OneDrive\Desktop\ICP_Verztec_Chatbot-2\improvefaiss\faiss_local_meta.json"
    index, metadata = load_faiss_index(index_path, meta_path)

    query = "What to do after I get let go?"
    results = search_faiss_local(index, metadata, query)

    print("\nTop search results:")
    for res in results:
        print(f"Distance: {res['distance']:.4f} | File: {res['source_file']} | Chunk: {res['chunk_index']}")
        print(res['text'])
        print("-" * 60)
