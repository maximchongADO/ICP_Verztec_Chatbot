# upload_routes.py
from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from vector_store import unified_document_pipeline  # ⬅️ Your core function
from chatbot import embedding_model  # ⬅️ Uses the same model loaded in chatbot.py

upload_router = APIRouter()

# Create necessary directories
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)
images_dir = Path("data/images")
cleaned_dir = Path("data/cleaned")
vertztec_collection = Path("data/verztec_logo")
faiss_index_path = "faiss_index3"

for d in [images_dir, cleaned_dir, vertztec_collection]:
    d.mkdir(parents=True, exist_ok=True)

@upload_router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run unified pipeline
        result = unified_document_pipeline(
            file_path=file_path,
            images_dir=images_dir,
            cleaned_dir=cleaned_dir,
            vertztec_collection=vertztec_collection,
            faiss_index_path=faiss_index_path,
            embedding_model=embedding_model
        )

        if result["success"]:
            return {"message": f"✅ File '{file.filename}' uploaded and processed."}
        else:
            return {"error": result["error"]}

    except Exception as e:
        return {"error": f"❌ Upload failed: {str(e)}"}
