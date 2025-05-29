import os
import time
import faiss
import pickle
import sys
from langchain_community.docstore.in_memory import InMemoryDocstore

def check_faiss_file_updated(file_path, last_checked_time):
    # Get the last modified time of the FAISS index file
    file_mod_time = os.path.getmtime(file_path)
    
    # Compare with the last checked time
    if file_mod_time > last_checked_time:
        print(f"The FAISS file '{file_path}' has been updated.")
        return True
    else:
        print(f"The FAISS file '{file_path}' has not been updated.")
        return False

def inspect_faiss_index(folder_path):
    """
    Display the contents of a FAISS index and its associated metadata from a folder
    """
    try:
        # Check if folder exists and construct full paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_dir, folder_path)
        index_path = os.path.join(folder_path, "index.faiss")
        metadata_path = os.path.join(folder_path, "index.pkl")

        if not os.path.exists(folder_path):
            print(f"Error: FAISS folder not found at: {folder_path}")
            return
        if not os.path.exists(index_path):
            print(f"Error: index.faiss not found in folder")
            return
        if not os.path.exists(metadata_path):
            print(f"Error: index.pkl not found in folder")
            return

        # Load the FAISS index
        index = faiss.read_index(index_path)
        
        # Load the metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Display information
        print(f"\nFAISS Index Information:")
        print(f"Index file: {index_path}")
        print(f"Metadata file: {metadata_path}")
        print(f"Total vectors: {index.ntotal}")
        print(f"Dimension: {index.d}\n")
        
        print("Metadata Content:")
        print(f"Type of metadata: {type(metadata)}")
        
        # Handle tuple format (InMemoryDocstore, id_mapping)
        if isinstance(metadata, tuple) and len(metadata) == 2:
            docstore, id_mapping = metadata
            if isinstance(docstore, InMemoryDocstore):
                print("\nDocument Contents:")
                for index, doc_id in id_mapping.items():
                    doc = docstore._dict.get(doc_id)
                    if doc:
                        print(f"\nChunk {index}:")
                        print(f"ID: {doc_id}")
                        print(f"Content: {str(doc.page_content)}")
                        if hasattr(doc, 'metadata'):
                            print(f"Metadata: {doc.metadata}")
                        print("-" * 80)
            else:
                print("\nUnexpected docstore format")
        else:
            print(f"\nRaw metadata content: {str(metadata)[:500]}...")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Full error details: {sys.exc_info()}")

# Example usage
if __name__ == "__main__":
    # Use relative path from repository root
    faiss_folder = "chatbot/src/backend/python/faiss_index2"
    inspect_faiss_index(faiss_folder)