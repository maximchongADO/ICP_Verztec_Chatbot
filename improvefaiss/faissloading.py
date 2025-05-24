import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model_name = 'BAAI/bge-large-en-v1.5'
model = SentenceTransformer(model_name)

def load_chunk_texts(folder):
    chunks = []
    for fname in os.listdir(folder):
        if not fname.endswith("_chunks.json"):
            continue
        with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
            for c in data['chunks']:
                chunks.append({
                    "text": c['chunk_text'],
                    "source_file": c['source_file'],
                    "chunk_index": c['chunk_index']
                })
    return chunks

def build_faiss_index_local(chunks, index_path="faiss_local.idx", meta_path="faiss_local_meta.json"):
    texts = [c['text'] for c in chunks]
    print(f"Encoding {len(texts)} chunks with model '{model_name}'...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    metadata = [{"source_file": c["source_file"], "chunk_index": c["chunk_index"], "text": c["text"]} for c in chunks]

    faiss.write_index(index, index_path)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved FAISS index to {index_path} and metadata to {meta_path}")
    return index, metadata

if __name__ == "__main__":
    chunk_folder = "chunked_output"
    chunks = load_chunk_texts(chunk_folder)
    build_faiss_index_local(chunks)
