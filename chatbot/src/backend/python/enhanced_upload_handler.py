#!/usr/bin/env python3
"""
Enhanced Upload Handler for Country/Department-specific FAISS uploads
This script is called by the Node.js internal API to process file uploads
"""

import sys
import json
import os
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fileUpload import process_upload
    from upload_config import validate_country_department
    from vector_store import unified_document_pipeline, embedding_model, images_dir, cleaned_dir, vertztec_collection
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"Failed to import required modules: {str(e)}"
    }))
    sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print(json.dumps({
            "success": False,
            "error": "Usage: enhanced_upload_handler.py <file_path> <country> <department>"
        }))
        sys.exit(1)
    
    file_path = sys.argv[1]
    country = sys.argv[2]
    department = sys.argv[3]
    
    try:
        # Validate country and department
        country, department = validate_country_department(country, department)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Process the upload using the unified pipeline
        result = unified_document_pipeline(
            file_path=file_path,
            images_dir=images_dir,
            cleaned_dir=cleaned_dir,
            vertztec_collection=vertztec_collection,
            faiss_index_path=None,  # Will be auto-generated based on country/department
            embedding_model=embedding_model,
            country=country,
            department=department
        )
        
        if result["success"]:
            # Read the cleaned content
            cleaned_content = ""
            if result.get("cleaned_text_path") and os.path.exists(result["cleaned_text_path"]):
                with open(result["cleaned_text_path"], 'r', encoding='utf-8') as f:
                    cleaned_content = f.read()
            
            response = {
                "success": True,
                "message": f"File processed successfully and added to {country.upper()}/{department.upper()} index",
                "filename": os.path.basename(file_path),
                "country": country,
                "department": department,
                "faiss_index_path": result.get("faiss_index_path"),
                "cleaned_content": cleaned_content,
                "cleaned_text_path": result.get("cleaned_text_path")
            }
        else:
            response = {
                "success": False,
                "error": result.get("error", "Unknown error occurred during processing")
            }
        
        print(json.dumps(response))
        
    except ValueError as e:
        print(json.dumps({
            "success": False,
            "error": f"Validation error: {str(e)}"
        }))
        sys.exit(1)
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Processing error: {str(e)}"
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
