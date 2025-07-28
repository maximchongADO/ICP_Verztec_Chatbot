#!/usr/bin/env python3
"""
Script to create and populate the admin master FAISS index with all data from country/department indices
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add the current directory to path to import the extractor
sys.path.append(os.path.dirname(__file__))

try:
    from faiss_extractor_optimized import OptimizedFAISSExtractor
    import faiss
    import pickle
    
    def merge_faiss_indices():
        """Merge all country/department FAISS indices into admin master index"""
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_indices_dir = os.path.join(script_dir, "faiss_indices")
        admin_master_dir = os.path.join(faiss_indices_dir, "admin_master", "faiss_index")
        
        print(f"Creating admin master index at: {admin_master_dir}")
        
        # Ensure admin master directory exists
        os.makedirs(admin_master_dir, exist_ok=True)
        
        all_documents = []
        all_vectors = []
        vector_dimension = None
        
        # Iterate through all country/department combinations
        for country_dir in os.listdir(faiss_indices_dir):
            country_path = os.path.join(faiss_indices_dir, country_dir)
            
            # Skip admin_master directory and non-directories
            if not os.path.isdir(country_path) or country_dir == "admin_master":
                continue
                
            print(f"Processing country: {country_dir}")
            
            for dept_dir in os.listdir(country_path):
                dept_path = os.path.join(country_path, dept_dir)
                if not os.path.isdir(dept_path):
                    continue
                    
                index_path = os.path.join(dept_path, "faiss_index")
                if not os.path.exists(index_path):
                    continue
                    
                print(f"  Processing department: {dept_dir}")
                
                try:
                    # Load existing index
                    extractor = OptimizedFAISSExtractor(index_path)
                    documents = extractor.get_all_documents()
                    
                    if documents and len(documents) > 0:
                        print(f"    Found {len(documents)} documents")
                        all_documents.extend(documents)
                        
                        # Load vectors from the FAISS index
                        faiss_index_file = os.path.join(index_path, "index.faiss")
                        if os.path.exists(faiss_index_file):
                            index = faiss.read_index(faiss_index_file)
                            
                            # Get all vectors from the index
                            if index.ntotal > 0:
                                vectors = index.reconstruct_n(0, index.ntotal)
                                all_vectors.extend(vectors)
                                
                                if vector_dimension is None:
                                    vector_dimension = index.d
                                    print(f"    Vector dimension: {vector_dimension}")
                    else:
                        print(f"    No documents found in {index_path}")
                        
                except Exception as e:
                    print(f"    Error processing {index_path}: {e}")
                    continue
        
        if len(all_documents) == 0:
            print("No documents found to merge!")
            return False
            
        print(f"\nTotal documents to merge: {len(all_documents)}")
        print(f"Total vectors to merge: {len(all_vectors)}")
        
        # Create new admin master index
        try:
            import numpy as np
            from langchain_community.vectorstores import FAISS
            from langchain.docstore.document import Document
            
            if len(all_documents) > 0 and len(all_vectors) > 0 and vector_dimension:
                print(f"Creating FAISS vectorstore with {len(all_documents)} documents and {len(all_vectors)} vectors")
                
                # Convert documents to LangChain Document format
                langchain_docs = []
                for doc in all_documents:
                    # Create Document with content and metadata
                    langchain_doc = Document(
                        page_content=doc['content'],
                        metadata=doc['metadata']
                    )
                    langchain_docs.append(langchain_doc)
                
                print(f"Converted {len(langchain_docs)} documents to LangChain format")
                
                # Get embedding model
                from faiss_extractor_optimized import get_embedding_model
                embedding_model = get_embedding_model()
                
                # Create FAISS index from documents
                print("Creating FAISS vectorstore...")
                vectorstore = FAISS.from_documents(langchain_docs, embedding_model)
                
                # Save the vectorstore
                print(f"Saving vectorstore to: {admin_master_dir}")
                vectorstore.save_local(admin_master_dir)
                
                print(f"\nAdmin master index created successfully!")
                print(f"Total files: {len(set(doc['metadata']['source'] for doc in all_documents))}")
                print(f"Total chunks: {len(all_documents)}")
                
                return True
            else:
                print("Error: No documents or vectors found to create index")
                return False
                
        except Exception as e:
            print(f"Error creating admin master index: {e}")
            import traceback
            traceback.print_exc()
            return False

    if __name__ == "__main__":
        success = merge_faiss_indices()
        if success:
            print("\n✅ Admin master index creation completed successfully!")
        else:
            print("\n❌ Admin master index creation failed!")
            
except Exception as e:
    print(f"Error importing required modules: {e}")
    import traceback
    traceback.print_exc()
