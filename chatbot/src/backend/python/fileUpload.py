from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
from datetime import datetime
import logging
from typing import List
import mimetypes

from vector_store import (
    unified_document_pipeline,
    embedding_model,
    images_dir,
    cleaned_dir,
    vertztec_collection
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define supported file types
SUPPORTED_TYPES = {
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.txt': 'text/plain'
}

async def save_upload_file(upload_file: UploadFile, dest_folder: Path) -> Path:
    """Save uploaded file to destination folder"""
    try:
        os.makedirs(dest_folder, exist_ok=True)
        file_path = dest_folder / upload_file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file {upload_file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

async def process_upload(file: UploadFile = File(...)):
    """Process uploaded file and add to FAISS index"""
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_TYPES.keys())}"
            )

        # Save uploaded file temporarily
        temp_folder = Path("temp_uploads")
        temp_file_path = await save_upload_file(file, temp_folder)
        
        # Get the correct FAISS index path relative to this file
        current_dir = Path(__file__).parent
        faiss_index_path = current_dir / "faiss_master_index2"

        
        # Process document and update FAISS indexs
        result = unified_document_pipeline(
            file_path=temp_file_path,
            images_dir=images_dir,
            cleaned_dir=cleaned_dir,
            vertztec_collection=vertztec_collection,
            faiss_index_path=str(faiss_index_path),  # Convert Path to string
            embedding_model=embedding_model
        )

        # Clean up temp file
        os.remove(temp_file_path)
        if len(os.listdir(temp_folder)) == 0:
            os.rmdir(temp_folder)

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process document: {result['error']}"
            )

        # Read the cleaned text content
        with open(result["cleaned_text_path"], 'r', encoding='utf-8') as f:
            cleaned_content = f.read()

        return JSONResponse(
            content={
                "message": "File processed successfully",
                "filename": file.filename,
                "cleaned_content": cleaned_content,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
