#!/usr/bin/env python3
"""
Get available FAISS indices organized by country and department
Returns a dictionary of available indices and their statistics
"""

import sys
import json
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vector_store import list_available_indices
    from faiss_manager import get_all_index_info
except ImportError:
    # Fallback if modules are not available
    print(json.dumps({
        "success": True,
        "available_indices": {},
        "total_indices": 0,
        "error": "FAISS manager modules not available"
    }))
    sys.exit(0)

def main():
    try:
        # Get basic available indices
        indices = list_available_indices()
        
        # Get detailed index information
        try:
            detailed_info = get_all_index_info()
            response = {
                "success": True,
                "available_indices": indices,
                "detailed_info": detailed_info,
                "total_indices": sum(len(departments) for departments in indices.values()) if indices else 0
            }
        except Exception as detail_error:
            # Fall back to basic info if detailed info fails
            response = {
                "success": True,
                "available_indices": indices,
                "total_indices": sum(len(departments) for departments in indices.values()) if indices else 0,
                "detail_error": str(detail_error)
            }
        
        print(json.dumps(response))
        
    except Exception as e:
        # Return empty result on error
        print(json.dumps({
            "success": True,
            "available_indices": {},
            "total_indices": 0,
            "error": str(e)
        }))

if __name__ == "__main__":
    main()
