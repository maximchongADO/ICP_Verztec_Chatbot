import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from pathlib import Path

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

root_dir = Path(__file__).parent
data_dir = root_dir / "data"
cleaned_dir = data_dir / "cleaned"

# Set OpenRouter API details
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

#  LLM and Retrieval Setup (Can be run multiple times)
def query_faiss_llm(query):
    # Load FAISS vector store
    vector_store = FAISS.load_local("verztec_vector_store", embedding_model, allow_dangerous_deserialization=True)

    # Create retriever
    retriever = vector_store.as_retriever(search_type="similarity", k=3)

    # Load OpenRouter DeepSeek model into LangChain
    llm = ChatOpenAI(
        model_name="tngtech/deepseek-r1t-chimera:free",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Retrieve documents for context
    docs = retriever.get_relevant_documents(query)

    # Prepare the context from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])  # Concatenate text from all retrieved docs

    # Now, invoke the LLM chain using the query and context
    response = qa_chain.invoke({"query": query, "context": context})

    # Output the response and sources
    print("Answer:", response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        print("-", doc.metadata["source"])

# Main interactive loop for user input
def interactive_bot():
    print("Welcome to the Verztec Helpdesk Bot! You can ask questions anytime.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("Ask a question: ")
        
        # Exit condition
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        # Call the function to handle the query
        query_faiss_llm(query)

# Start the interactive bot
if __name__ == "__main__":
    interactive_bot()