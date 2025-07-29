#!/usr/bin/env python3
"""
Quick test script to check what's in the China HR FAISS index
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to path to import the extractor
sys.path.append(os.path.dirname(__file__))

try:
    from faiss_extractor_optimized import OptimizedFAISSExtractor
    
    # Test the China HR index
    index_path = "faiss_indices/china/hr/faiss_index"
    print(f"Testing index at: {index_path}")
    print(f"Index exists: {os.path.exists(index_path)}")
    
    if os.path.exists(index_path):
        print("Initializing extractor...")
        extractor = OptimizedFAISSExtractor(index_path)
        
        print("Getting statistics...")
        stats = extractor.get_index_statistics()
        print("Statistics:", json.dumps(stats, indent=2))
        
        print("Getting all documents...")
        documents = extractor.get_all_documents()
        if documents:
            print(f"Found {len(documents)} documents")
            grouped = extractor.group_documents_by_source(documents)
            print("Grouped data:", json.dumps(grouped, indent=2))
        else:
            print("No documents found")
    else:
        print("Index path does not exist!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
