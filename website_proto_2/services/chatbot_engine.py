from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables and API key
load_dotenv()
api_key = "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"
model = "deepseek-r1-distill-llama-70b"

# Initialize Groq LLM with LangChain
deepseek = ChatGroq(api_key=api_key, model_name=model)
parser = StrOutputParser()
deepseek_chain = deepseek | parser

# Load HuggingFace Embeddings
embedding_model2 = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)

def get_bot_reply(query):
    # Go two levels up from this file: services/ → website_proto_2/ → project-root/
    root_dir = Path(__file__).resolve().parent.parent.parent

    # Path to the faiss_master_index folder in chatbot/src/backend/python/
    index_dir = root_dir / "chatbot" / "src" / "backend" / "python" / "faiss_master_index"

    # Load FAISS vector store
    vector_store = FAISS.load_local(str(index_dir), embedding_model2, allow_dangerous_deserialization=True)

    # Create retriever and QA chain
    retriever = vector_store.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=deepseek_chain,
        retriever=retriever,
        return_source_documents=True
    )

    # Get relevant documents and context
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Run the chain
    response = qa_chain.invoke({"query": query, "context": context})
    return response["result"]
