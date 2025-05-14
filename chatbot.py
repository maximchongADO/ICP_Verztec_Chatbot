import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pathlib import Path

# Load environment variables
load_dotenv()
api_key = "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"
model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Update model as required
deepseek = ChatGroq(api_key=api_key, model_name=model)

# Print test message to verify Groq API
print(deepseek.invoke("hello"))

# Define the output parser for handling results
parser = StrOutputParser()

# Chain the Groq model with the parser
deepseek_chain = deepseek | parser

# Print the result from the chain
print(deepseek_chain.invoke("hello"))

# Load embedding model
embedding_model2 = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",

    encode_kwargs={'normalize_embeddings': True}  
)
# --- Step 1: Load FAISS Vector Store ---

root_dir = Path(__file__).parent
data_dir = root_dir / "data"
cleaned_dir = data_dir / "cleaned"

# Query function to handle the data retrieval and LLM invocation
def query_faiss_llm(query):
    # Load FAISS vector store
    vector_store = FAISS.load_local("faiss_index3", embedding_model2, allow_dangerous_deserialization=True)

    # Create retriever for fetching documents
    retriever = vector_store.as_retriever(search_type="similarity", k=3)

    # Now replace OpenAI API usage with the Groq model setup
    # Build the retrieval QA chain using Groq model instead of OpenAI
    qa_chain = RetrievalQA.from_chain_type(
        llm=deepseek_chain,  # Use the Groq model chain here
        retriever=retriever,
        return_source_documents=True
    )

    # Retrieve documents for context
    docs = retriever.get_relevant_documents(query)

    # Prepare the context from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])  # Concatenate text from all retrieved docs

    # Now, invoke the Groq chain using the query and context
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


