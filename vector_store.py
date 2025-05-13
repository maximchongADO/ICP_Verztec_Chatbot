import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document as langDocument
from langchain.chains import RetrievalQA
from pathlib import Path
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def get_all_file_paths(directory, extensions=[".pdf", ".doc", ".docx",".txt"]):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Filter files by the given extensions (e.g., PDF, DOC, DOCX files)
            if any(file.lower().endswith(ext) for ext in extensions):
                absolute_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(absolute_path)
    return file_paths



root_dir = Path(os.getcwd())
data_dir = root_dir / "data" 
cleaned_dir=data_dir/ "cleaned"
cleaned_files = get_all_file_paths(cleaned_dir)

all_chunks=[]

fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    separators=["\n\n","!", "?"]
)


def spacy_fallback_splitter(text, chunk_size=300, overlap=50):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap (optional)
    if overlap > 0:
        overlapped_chunks = []
        for i in range(len(chunks)):
            start = max(0, i - 1)
            merged = " ".join(chunks[start:i+1])
            overlapped_chunks.append(merged.strip())
        return overlapped_chunks

    return chunks




def split_by_bullets(text):
    # Combine bullet marker + its content into one match using lookahead
    pattern = r'(?m)(?:^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-•*])\s+)(.*?)(?=^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-•*])\s+|$)'

    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE))
    chunks = []

    for match in matches:
        full_chunk = match.group(0).strip()

        # Skip invalid or trivial chunks
        if len(full_chunk) < 5 or full_chunk.strip().isdigit():
            continue

        # Long bullet content gets sub-chunked
        if len(full_chunk) > 600:
            chunks.extend(fallback_splitter.split_text(full_chunk))
        else:
            chunks.append(full_chunk)
        
    return chunks


bullet_pattern = r'(?m)(?:^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-\u2022\*])\s+)(.*?)(?=^\s*(\d+[\.\)]|[a-zA-Z][\.\)]|[-\u2022\*])\s+|$)'
fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50, separators=["\n\n", "!", "?"])


def split_text(text):
    matches = list(re.finditer(bullet_pattern, text, flags=re.DOTALL | re.MULTILINE))
    chunks = [m.group(0).strip() for m in matches if len(m.group(0).strip()) > 5 and not m.group(0).strip().isdigit()]
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(fallback_splitter.split_text(chunk) if len(chunk) > 600 else [chunk])
    return final_chunks or spacy_fallback_splitter(text)



# maybe adding descriptors of the file to the chunk 

llm = ChatOpenAI(
    model_name="tngtech/deepseek-r1t-chimera:free",
    temperature=0.1,
    openai_api_key="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",
    openai_api_base="https://openrouter.ai/api/v1"
)

#chunking
for file_path in cleaned_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    query = 'The following is a internal company document, for verztec, a consulting company. you are to provide a description of the document, for use in a RAG system. ensure to mention the contents, uses and potential queries the document could asnwer: '+text
    response = llm.invoke(query)
    print(response)

    smart_chunks = split_text(text)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for i, chunk in enumerate(smart_chunks):
        enriched = f"[Description: {response}] [Document: {base_name}] {chunk}"
        all_chunks.append(
            langDocument(
                page_content=enriched,
                metadata={
                    "source": base_name,
                    "doc_type": "text",
                    "chunk_id": f"{base_name}_{i}",
                    "clean_chunk": chunk
                }
            )
        )


# embedding. 
embedding_model1 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
     encode_kwargs={'normalize_embeddings': True}
)
embedding_model2 = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",

    encode_kwargs={'normalize_embeddings': True}  # Very important for bge!
)

#for chunk in all_chunks:
 #   print("-----------------------------------------")
  #  print(chunk.page_content)
   # print(chunk.metadata['source'])

vector_db = FAISS.from_documents(all_chunks, embedding_model2)
vector_db.save_local("faiss_index3")

# test embedding 

query="pantry rules"

results = vector_db.similarity_search_with_score(query, k=10)
print ('DIFF PART --------------------------------------------------------------------------------------')
for doc, score in results:
    print("-----")
    print(doc.page_content)
    print(doc.metadata)
    print(f"Score: {score}")