import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re
from numpy import dot
from numpy.linalg import norm

# Initialize embedding model
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Initialize Groq LLM client
api_key = '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'
model_name = "compound-beta"
deepseek = ChatGroq(api_key=api_key, model_name=model_name)

def load_faiss_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata

def refine_prompt(user_query: str) -> str:
    prompt = (
        f"Your only job is to correct grammar and spelling mistakes and rephrase the prompt clearly for an internal helpdesk chatbot for a company:\n\n{user_query}"
    )
    messages = [HumanMessage(content=prompt)]
    response = deepseek.generate([messages])
    refined_query = response.generations[0][0].text.strip()
    return refined_query

def search_faiss_and_get_context(refined_query, index, metadata, top_k=4):
    query_emb = embedding_model.encode([refined_query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved_chunks = [metadata[idx]['text'] for idx in indices[0]]
    return retrieved_chunks, distances[0]

def generate_final_answer(original_query: str, context_chunks: list[str]) -> str:
    context_text = "\n---\n".join(context_chunks)
    prompt = (
        "You are an internal company helpdesk chatbot for employees that answers questions "
        "based on the given context (Company Information) and is not allowed to access information outside of the given context. Only answer prompts that are within the company context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {original_query}\n\n"
        "Please provide a clear and concise answer."
    )
    messages = [HumanMessage(content=prompt)]
    response = deepseek.generate([messages])
    answer = response.generations[0][0].text.strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    return answer

def is_related_to_previous(current_query: str, previous_queries: list[str], threshold: float = 0.6) -> bool:
    if not previous_queries:
        return False
    current_vec = embedding_model.encode(current_query)
    similarities = [dot(current_vec, embedding_model.encode(p)) / (norm(current_vec) * norm(embedding_model.encode(p))) for p in previous_queries]
    max_similarity = max(similarities)
    print(f"[Debug] Max similarity to previous queries: {max_similarity:.4f}")
    return max_similarity >= threshold

def interactive_chat(index, metadata):
    chat_history = []
    print("Welcome to the Verztec Helpdesk Bot! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        prev_queries = [q for q, _ in chat_history[-2:]]
        is_related = is_related_to_previous(user_query, prev_queries)
        context_query = " ".join(prev_queries + [user_query]) if is_related else user_query
        clean_query = refine_prompt(context_query)

        print(f"\n[Step 1] Original Query: {user_query}")
        print(f"[Step 1] Cleaned Query: {clean_query}")

        context_chunks, scores = search_faiss_and_get_context(clean_query, index, metadata, top_k=4)
        avg_score = float(np.mean(scores)) if len(scores) > 0 else 1.0
        print("\n[Step 2] Retrieved Chunks:")
        for i, chunk in enumerate(context_chunks, 1):
            snippet = chunk[:300].replace('\n', ' ') + ('...' if len(chunk) > 300 else '')
            print(f"  Chunk {i}: {snippet}")
        print(f"\n[Step 2] FAISS Avg Similarity Score: {avg_score:.4f}")

        if avg_score > 2.5:
            fallback = (
                f"The user's question is likely unrelated to internal company topics.\n"
                f"Query: {clean_query}\n"
                f"Please advise the user that this query is out of scope."
            )
            messages = [HumanMessage(content=fallback)]
            response = deepseek.generate([messages])
            answer = response.generations[0][0].text.strip()
        else:
            recent_history_text = ""
            if is_related:
                for q, a in chat_history[-2:]:
                    recent_history_text += f"User: {q}\nBot: {a}\n"
            recent_history_text += f"User: {user_query}\n"
            combined_context = "\n---\n".join(context_chunks)
            final_prompt = (
                "You are an internal helpdesk chatbot answering questions ONLY based on the given company context.\n\n"
                f"Conversation history:\n{recent_history_text}\n"
                f"Company documents:\n{combined_context}\n\n"
                f"Question: {clean_query}\n"
                "Please provide a clear and concise answer."
            )
            messages = [HumanMessage(content=final_prompt)]
            response = deepseek.generate([messages])
            answer = response.generations[0][0].text.strip()

        print("\n[Step 3] Final Answer:\n", answer)
        chat_history.append((user_query, answer))

        # Save to chat_log.txt
        with open("chat_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_query}\nBot: {answer}\n{'-'*40}\n")

        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    faiss_index_path = "faiss_local_BAAI.idx"
    faiss_meta_path = "faiss_local_BAAI_meta.json"
    index, metadata = load_faiss_and_metadata(faiss_index_path, faiss_meta_path)
    interactive_chat(index, metadata)
