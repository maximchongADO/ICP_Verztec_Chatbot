import pymupdf
import re
import ftfy
import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as langDocument
from pathlib import Path
from docx import Document
import magic
import win32com.client
import mimetypes

def get_all_file_paths(directory, extensions=[".pdf", ".doc", ".docx",".txt"]):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Filter files by the given extensions (e.g., PDF, DOC, DOCX files)
            if any(file.lower().endswith(ext) for ext in extensions):
                absolute_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(absolute_path)
    return file_paths

def clean_filename_for_embedding(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    filename = re.sub(r'^\d+[_\s-]*', '', filename)  # Remove leading numbers
    filename = re.sub(r'[_-]\d{6,8}$', '', filename)  # Remove trailing dates
    filename = re.sub(r'[ \-]+', '_', filename)  # Replace spaces/hyphens
    filename = re.sub(r'[^\w_]', '', filename)  # Remove non-word chars
    return filename.lower()

def process_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_doc_using_word(file_path):
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(file_path)
    text = doc.Content.Text
    doc.Close(False)
    word.Quit()
    return text
    

def process_file(file_path):
    # Check file type based on extension and process accordingly
    if file_path.lower().endswith(".pdf"):
        return process_pdf(file_path)
    elif file_path.lower().endswith(".doc"):
        return extract_text_from_doc_using_word(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

root_dir = Path(os.getcwd())  # Get the current working directory (the root of your repository)
data_dir = root_dir / "data"  # Gets the "data" folder relative to the root directory
pdf_dir = data_dir / "pdf"
docx_dir = data_dir / "word"
cleaned_dir=data_dir/ "cleaned"

#PDF file cleaning 
pdf_files = get_all_file_paths(pdf_dir)
word_files= get_all_file_paths(docx_dir)
# Counters
success_count = 0
fail_count = 0

for file_path in pdf_files:
   
    
    # Clean filename for embedding
    cleaned_filename = clean_filename_for_embedding(file_path)

    
    # Process the file (either PDF or DOCX)
    text = process_file(file_path)
    
    # Apply text cleaning
    text = ftfy.fix_text(text)  # Fix broken encoding from PDFs or DOCX
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra line breaks and normalize spaces
    text = re.sub(r'\s*(\d+\.)', r'\n\1', text)  # Clean up bullet numbers

    # Output file path for saving cleaned content
    file_address = cleaned_dir / (cleaned_filename + ".txt")
   

    # Write the extracted text into the cleaned text file
    with open(file_address, "w", encoding="utf-8") as out:
        out.write(text)
    
    print(f"-Finished processing and saved to: {file_address}")





for file_path in word_files:
    mime_type, _ = mimetypes.guess_type(file_path)
    ext = os.path.splitext(file_path)[-1].lower()
    #print(file_path)
    base_filename = os.path.basename(file_path)
    #print(base_filename)
    filename_without_ext = os.path.splitext(base_filename)[0]
    #print(filename_without_ext)

    # Now fully clean the filename
    modified_filename = clean_filename_for_embedding(filename_without_ext)
    #print(modified_filename)
    file_address = cleaned_dir / (modified_filename+ ".txt")
    #print (file_address)
   
    try:
        if ext == '.docx':
            text = extract_text_from_docx(file_path)
        elif ext == '.doc':
            text = extract_text_from_doc_using_word(file_path)
        else:
            print(f"Skipped non-Word file: {file_path} (type: {mime_type})")
            fail_count += 1
            continue

        #print(f"--- Content from {os.path.basename(file_path)} ---")
        
        out = open(file_address, "wb") # create a text output
        
        out.write(text.encode("utf8")) # write text of page encoded in utf-8
        
        out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
        out.close()
    
        
      
        success_count += 1
        print(f"-Finished processing and saved to: {file_address}")

    except Exception as e:
        print(f"Failed to extract from {file_path}: {e}")
        fail_count += 1


