from fastapi import UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
from datetime import datetime
import logging
from typing import List, Optional
import mimetypes

from vector_store import (
    unified_document_pipeline,
    embedding_model,
    images_dir,
    cleaned_dir,
    vertztec_collection,
    list_available_indices,
    get_faiss_index_path
)

from upload_config import (
    SUPPORTED_COUNTRIES,
    SUPPORTED_DEPARTMENTS, 
    SUPPORTED_FILE_TYPES,
    validate_country_department,
    get_config_info
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_country_department_upload(country: str, department: str):
    """Validate that country and department are supported for uploads"""
    try:
        return validate_country_department(country, department)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

async def process_upload(
    file: UploadFile = File(...),
    country: str = Form(...),
    department: str = Form(...)
):
    """Process uploaded file and add to country/department-specific FAISS index"""
    try:
        # Debug logging
        logger.info(f"Received upload request - Country: {country} (type: {type(country)}), Department: {department} (type: {type(department)})")
        
        # Ensure country and department are strings
        if hasattr(country, 'value'):
            country = country.value
        if hasattr(department, 'value'):
            department = department.value
            
        country = str(country).strip()
        department = str(department).strip()
        
        logger.info(f"Processed parameters - Country: '{country}', Department: '{department}'")
        
        # Validate country and department
        country, department = validate_country_department_upload(country, department)
        
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_FILE_TYPES.keys())}"
            )

        # Save uploaded file temporarily
        temp_folder = Path("temp_uploads")
        temp_file_path = await save_upload_file(file, temp_folder)
        
        # Get the country/department-specific FAISS index path (will be auto-created)
        faiss_index_path = get_faiss_index_path(country, department)
        print(f"[INFO] Processing file for {country.upper()}/{department.upper()} index")

        # Process document and update FAISS index
        result = unified_document_pipeline(
            file_path=temp_file_path,
            images_dir=images_dir,
            cleaned_dir=cleaned_dir,
            vertztec_collection=vertztec_collection,
            faiss_index_path=None,  # Let the function auto-generate based on country/department
            embedding_model=embedding_model,
            country=country,
            department=department
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
                "message": f"File processed successfully and added to {country.upper()}/{department.upper()} index",
                "filename": file.filename,
                "country": country,
                "department": department,
                "faiss_index_path": result["faiss_index_path"],
                "cleaned_content": cleaned_content,
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_available_indices():
    """Get list of all available FAISS indices organized by country and department"""
    try:
        indices = list_available_indices()
        config_info = get_config_info()
        
        return JSONResponse(
            content={
                "available_indices": indices,
                "supported_countries": config_info["supported_countries"],
                "supported_departments": config_info["supported_departments"],
                "supported_file_types": list(config_info["supported_file_types"]),
                "total_combinations": config_info["total_combinations"],
                "success": True
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error getting available indices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_system_config():
    """Get complete system configuration information"""
    try:
        config_info = get_config_info()
        return JSONResponse(
            content={
                "config": config_info,
                "success": True
            },
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error getting system config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
