#!/usr/bin/env python3
"""
Simple test script to verify knowledge base loading
"""

import os
import sys
import json
import time
from pathlib import Path

def test_faiss_loading():
    """Test FAISS loading with detailed error reporting"""
    print("=== FAISS Knowledge Base Test ===", file=sys.stderr)
    
    try:
        # Test 1: Check if FAISS index exists
        script_dir = Path(__file__).parent
        faiss_paths = [
            script_dir / "chatbot" / "src" / "backend" / "python" / "faiss_master_index",
            script_dir / "faiss_master_index", 
            script_dir / "verztec_vector_store"
        ]
        
        print(f"Script directory: {script_dir}", file=sys.stderr)
        
        found_index = None
        for faiss_path in faiss_paths:
            print(f"Checking: {faiss_path}", file=sys.stderr)
            if faiss_path.exists():
                index_file = faiss_path / "index.faiss"
                pkl_file = faiss_path / "index.pkl"
                if index_file.exists() and pkl_file.exists():
                    found_index = faiss_path
                    print(f"✓ Found FAISS index at: {faiss_path}", file=sys.stderr)
                    break
                else:
                    print(f"✗ Directory exists but missing files: {faiss_path}", file=sys.stderr)
            else:
                print(f"✗ Directory not found: {faiss_path}", file=sys.stderr)
        
        if not found_index:
            return {
                "status": "error",
                "error": "No valid FAISS index found in expected locations",
                "searched_paths": [str(p) for p in faiss_paths]
            }
        
        # Test 2: Try importing required libraries
        print("Testing library imports...", file=sys.stderr)
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("✓ Successfully imported LangChain libraries", file=sys.stderr)
        except ImportError as e:
            return {
                "status": "error", 
                "error": f"Failed to import required libraries: {str(e)}",
                "suggestion": "Run: pip install langchain langchain-community"
            }
        
        # Test 3: Try loading embedding model
        print("Loading embedding model...", file=sys.stderr)
        start_time = time.time()
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                encode_kwargs={'normalize_embeddings': True},
                model_kwargs={'device': 'cpu'}
            )
            embed_time = time.time() - start_time
            print(f"✓ Embedding model loaded in {embed_time:.2f}s", file=sys.stderr)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load embedding model: {str(e)}",
                "suggestion": "Check if sentence-transformers is installed: pip install sentence-transformers"
            }
        
        # Test 4: Try loading FAISS index
        print("Loading FAISS index...", file=sys.stderr)
        start_time = time.time()
        try:
            vectorstore = FAISS.load_local(
                str(found_index),
                embedding_model,
                allow_dangerous_deserialization=True
            )
            load_time = time.time() - start_time
            print(f"✓ FAISS index loaded in {load_time:.2f}s", file=sys.stderr)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load FAISS index: {str(e)}",
                "index_path": str(found_index)
            }
        
        # Test 5: Get basic statistics
        print("Getting index statistics...", file=sys.stderr)
        try:
            total_vectors = vectorstore.index.ntotal
            dimension = vectorstore.index.d
            
            # Count documents
            doc_count = 0
            if hasattr(vectorstore, 'docstore') and hasattr(vectorstore.docstore, '_dict'):
                doc_count = len(vectorstore.docstore._dict)
            
            print(f"✓ Index contains {total_vectors} vectors, {doc_count} documents", file=sys.stderr)
            
            return {
                "status": "success",
                "index_path": str(found_index),
                "total_vectors": total_vectors,
                "vector_dimension": dimension,
                "document_count": doc_count,
                "embedding_load_time": embed_time,
                "index_load_time": load_time,
                "message": "Knowledge base is working correctly!"
            }
            
        except Exception as e:
            return {
                "status": "partial_success",
                "error": f"Index loaded but failed to get statistics: {str(e)}",
                "index_path": str(found_index)
            }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "type": type(e).__name__
        }

if __name__ == "__main__":
    result = test_faiss_loading()
    print(json.dumps(result, indent=2))
