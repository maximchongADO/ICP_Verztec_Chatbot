import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pathlib import Path
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

api_key = "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"
model = "deepseek-r1-distill-llama-70b"  # Update model as required
deepseek = ChatGroq(api_key=api_key, model_name=model)

# Define the output parser for handling results
parser = StrOutputParser()

# Chain the Groq model with the parser
deepseek_chain = deepseek | parser

# Print the result from the chainD
#print(deepseek_chain.invoke("hello"))

# Load embedding model
embedding_model2 = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",

    encode_kwargs={'normalize_embeddings': True}  
)
# --- Step 1: Load FAISS Vector Store ---

root_dir = Path(__file__).parent
data_dir = root_dir / "data"
cleaned_dir = data_dir / "cleaned"
chat_history = []  # Store (user, bot) message pairs
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",        # Input key in your chain
    output_key="answer",         # Explicit output key
    return_messages=True
)


# Query function to handle the data retrieval and LLM invocation
def query_faiss_llm(query, chat_history):
    # Load FAISS vector store
    vector_store = FAISS.load_local("faiss_index3", embedding_model2, allow_dangerous_deserialization=True)

    # Retrieve documents with similarity scores
    docs_with_scores = vector_store.similarity_search_with_score(query, k=5)

    # Compute average score
    ttl_score = sum(score for _, score in docs_with_scores)
    avgscore = ttl_score / len(docs_with_scores) if docs_with_scores else 1  # Default to 1 if no docs

    print(f"Average score: {avgscore:.4f}")

    # Check if query is relevant based on score
    if avgscore < 0.93:
        # Proceed with building the retriever and QA chain
        retriever = vector_store.as_retriever(search_type="similarity", k=3)
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=deepseek_chain,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )


        # Extract document content
        docs = [doc for doc, score in docs_with_scores]
        context = "\n".join([doc.page_content for doc in docs])

        # Add memory context
        #memory_context = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history[-3:]])
        #full_context = memory_context + "\n" + context if memory_context else context
        #whprint(full_context)

        # Invoke chain
        response = qa_chain.invoke({"question": query})
        #print(response.dtype)


    else:
        fallback_prompt = (
        f"The user's question is likely unrelated to the available documents.\n"
        f"Query: {query}\n"
        f"Respond by saying this is out of scope and no relevant information was found."
        )
         
        response_text = deepseek_chain.invoke(fallback_prompt)
        response = {
            "answer": response_text,
            "source_documents": []
        }
    
    cleaned_response = re.sub(r"<think>.*?</think>", "", response['answer'], flags=re.DOTALL).strip()

    # Use the cleaned response

    # Save to chat history
    chat_history.append((query, cleaned_response))

    # Display response
    print("\nAnswer:",cleaned_response)
    print("\nSources:")
    for doc, score in docs_with_scores:
        print(f"- Source: {doc.metadata['source']} | Score: {score:.4f}")

    return chat_history

# Main interactive loop for user input
def interactive_bot():
    print("Welcome to the Verztec Helpdesk Bot! You can ask questions anytime.")
    print("Type 'exit' to quit.")

    chat_history = []

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        chat_history=query_faiss_llm(query, chat_history)


# Start the interactive bot
if __name__ == "__main__":
    interactive_bot()
