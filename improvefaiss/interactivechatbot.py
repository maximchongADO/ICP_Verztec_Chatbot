import os
from dotenv import load_dotenv
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import re
from numpy import dot
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from numpy.linalg import norm
import sys
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MySQLDatabase.Inserting_data import store_chat_log
import uuid 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.schema import Document
from typing import Any, Dict, List
from langchain.callbacks.manager import CallbackManagerForChainRun
# Initialize embedding model
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')


# Initialize Groq LLM client
api_key = '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'
model_name = "compound-beta"
model = "deepseek-r1-distill-llama-70b" 
deepseek = ChatGroq(api_key=api_key, model_name=model)
compound =ChatGroq(api_key=api_key, model_name=model_name)
## setting up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",        # Input key in your chain
    output_key="answer",         # Explicit output key
    return_messages=True
)

parser = StrOutputParser()

# Chain the Groq model with the parser
deepseek_chain = deepseek | parser

def load_faiss_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return index, metadata
def refine_prompt(user_query: str) -> str:
    prompt = (
        f"You are an assistant that improves grammar and spelling for an internal helpdesk chatbot. "
        f"If the input is offensive, unclear, or meaningless, return the phrase: [UNCLEAR].\n\n"
        f"Input:\n{user_query}"
    )
    messages = [HumanMessage(content=prompt)]
    response = compound.generate([messages])
    refined_query = response.generations[0][0].text.strip()

    # Safety net
    if "[UNCLEAR]" in refined_query.upper() or len(refined_query) < 5:
        return "[UNCLEAR]"
    if "[UNCLEAR]" in refined_query.upper():
        refined_query= 'The user has sent something unintellible, please clarify    '
    return refined_query
def refine_prompt2(user_query: str) -> str:
    prompt = (
        f"Your only job is to correct grammar and spelling mistakes and rephrase the prompt clearly for an internal helpdesk chatbot for a company, You are to ONLY return the cleaned query. No explanation. IF there is no change to be made, return the original query:\n\n{user_query}"
    )
    messages = [HumanMessage(content=prompt)]
    response = compound.generate([messages])
    refined_query = response.generations[0][0].text.strip()
    return refined_query

def search_faiss_and_get_context(refined_query, index, metadata, top_k=4):
    query_emb = embedding_model.encode([refined_query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved_chunks = [metadata[idx]['text'] for idx in indices[0]]
    return retrieved_chunks, distances[0]


# This isint even used idiot vibe coder
def wertyugenerate_final_answer(original_query: str, context_chunks: list[str]) -> str:
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


def is_related_to_previous2(current_query: str, previous_queries: list[str], threshold: float = 0.9) -> bool:
    if not previous_queries:
        return False
    
    current_vec = embedding_model2.embed_query(current_query) 
    
    similarities = [dot(current_vec, embedding_model2.embed_query(current_query)) / (norm(current_vec) * norm(embedding_model2.embed_query(current_query))) for p in previous_queries]
    max_similarity = max(similarities)
    print(f"[Debug] Max similarity to previous queries: {max_similarity:.4f}")
    return max_similarity >= threshold

def interactive_chat(index, metadata):
    chat_history = []
    session_id = str(uuid.uuid4())  # Unique session ID for this chat session
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
            print(['[DEBUG]'])
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

       # Store chat log in MySQL DB instead of file
        store_chat_log(user_message=user_query, bot_response=answer, session_id=session_id)

        print("\n" + "-" * 80 + "\n")

def interactive_chat2(index):
    chat_history = []
    print("Welcome to the Verztec Helpdesk Bot! Type 'exit' to quit.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Step 1: Context linking
        prev_queries = [q for q, _ in chat_history[-2:]]
        is_related = is_related_to_previous2(user_query, prev_queries)
        if is_related:
            print('[LOG] query might not be related to previous query')
        context_query = " ".join(prev_queries + [user_query]) if is_related else user_query
        clean_query = refine_prompt(context_query)

        print(f"\n[Step 1] Original Query: {user_query}")
        print(f"[Step 1] Cleaned Query: {clean_query}")

        # Step 2: Search top documents
        results = index.similarity_search_with_score(clean_query, k=10)
        context_chunks = [doc.page_content for doc, _ in results]
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0

        print("\n[Step 2] Retrieved Chunks:")
        top_3_img=[]
        for i, (doc, score) in enumerate(results, 1):
            #snippet = doc.page_content[:300].replace('\n', ' ') + ('...' if len(doc.page_content) > 300 else '')
            print(f"  Chunk {i} (Score: {score:.4f}): {doc.page_content}")
            print(doc.metadata['images'])
            if i < 3:
                top_3_img.append(doc.metadata['images'])
            print('\n-------------------------------------------------')
        print(f"\n[Step 2] FAISS Avg Similarity Score: {avg_score:.4f}")

        # Step 3: Generate Answer
        if avg_score > 2.5:
            print("[DEBUG] Low relevance. Generating fallback.")
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
                "Please provide a clear and concise answer, include refrences to the images."
            )
            messages = [HumanMessage(content=final_prompt)]
            response = deepseek.generate([messages])
            answer = response.generations[0][0].text.strip()

        print("\n[Step 3] Final Answer:\n", answer)
        for i in top_3_img:
            print (f'Associated document: {i}')
        chat_history.append((user_query, answer))

        # Log conversation
        with open("chat_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_query}\nBot: {answer}\n{'-'*40}\n")

        print("\n" + "-" * 80 + "\n")


def interactive_chat3(index):
    chat_history=[]
    
    retriever = index.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",        # Input key in your chain
        output_key="answer",         # Explicit output key
        return_messages=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=deepseek_chain,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
       
    )

    

    print("Welcome to the Verztec Helpdesk Bot! Type 'exit' to quit.")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Step 1: Context linking
        prev_queries = [q for q, _ in chat_history[-2:]]
        is_related = is_related_to_previous2(user_query, prev_queries)
        if is_related:
            print('[LOG] query might not be related to previous query')
        context_query = " ".join(prev_queries + [user_query]) if is_related else user_query
        
        #clean_query = refine_prompt(context_query)
        clean_query = refine_prompt(user_query)

        print(f"\n[Step 1] Original Query: {user_query}")
        print(f"[Step 1] Cleaned Query: {clean_query}")

        # Step 2: Search top documents
        results = index.similarity_search_with_score(clean_query, k=10)
        context_chunks = [doc.page_content for doc, _ in results]
        scores = [score for _, score in results]
        avg_score = float(np.mean(scores)) if scores else 1.0

        print("\n[Step 2] Retrieved Chunks:")
        top_3_img=[]
        for i, (doc, score) in enumerate(results, 1):
            #snippet = doc.page_content[:300].replace('\n', ' ') + ('...' if len(doc.page_content) > 300 else '')
            #print(f"  Chunk {i} (Score: {score:.4f}): {doc.page_content}")
            #print(doc.metadata['images'])
            if i < 3:
                top_3_img.append(doc.metadata['images'])
            print('\n-------------------------------------------------')
        print(f"\n[Step 2] FAISS Avg Similarity Score: {avg_score:.4f}")

        # Step 3: Generate Answer
        if avg_score > 2.5:
            print("[DEBUG] Low relevance. Generating fallback.")
            fallback = (
                f"The user's question is likely unrelated to internal company topics.\n"
                f"Query: {clean_query}\n"
                f"Please advise the user that this query is out of scope."
            )
            messages = [HumanMessage(content=fallback)]
            response = deepseek.generate([messages])
            answer = response.generations[0][0].text.strip()
        else:
            response = qa_chain.invoke({"question": clean_query})
            answer = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()
            

        print("\n[Step 3] Final Answer:\n", answer)
        if avg_score>1:
            print(f'[LOG] Potentially non useful docs with score of {avg_score}')
        for i in top_3_img:
            print (f'Associated document: {i}')
        chat_history.append((user_query, answer))
        print('------------------------')
        docs = retriever.get_relevant_documents(clean_query)
        for i, doc in enumerate(docs, 1):
            
            print(doc.metadata['chunk_id'])


        # Log conversation
        with open("chat_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_query}\nBot: {answer}\n{'-'*40}\n")

        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    faiss_index_path = "faiss_local_BAAI.idx"
    faiss_meta_path = "faiss_local_BAAI_meta.json"
    #index, metadata = load_faiss_and_metadata(faiss_index_path, faiss_meta_path)
    embedding_model2 = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
        )
    global_db = FAISS.load_local(
        "faiss_index3",
        embedding_model2,
        allow_dangerous_deserialization=True
    )
    
    interactive_chat3(global_db)
    
