import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── 1️⃣  Load the store ──────────────────────────────────────────
store_dir = Path("chatbot/src/backend/python/faiss_GK_index")
embed_fn  = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

vecstore = FAISS.load_local(str(store_dir), embed_fn, allow_dangerous_deserialization=True)

print("🔢 Vectors in store:", vecstore.index.ntotal)

# ── 2️⃣  Ask a question ─────────────────────────────────────────
question = "what industries does Verztec operate in?"
print(f"❓ Question: {question}")

docs = vecstore.similarity_search(question, k=10)   # returns LangChain Documents

# ── 3️⃣  Show results ───────────────────────────────────────────
for i, d in enumerate(docs, 1):
    meta = d.metadata
    print(f"\n#{i}")
    print("Tag  :", meta.get('tag'))
    print("URL  :", meta.get('url'))
    print("Text :", d.page_content)
