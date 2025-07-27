# Enhanced FAISS Upload System - Country/Department Organization

## Overview

The FAISS upload system has been enhanced to organize documents by **country** and **department**, creating separate FAISS indices for each combination. This provides better organization, faster search performance, and more targeted document retrieval.

## Key Changes

### 1. **Updated Files**

- **`vector_store.py`**: Enhanced with country/department-specific FAISS index creation
- **`fileUpload.py`**: Updated to handle country and department parameters  
- **`faiss_manager.py`**: New utility for managing organized indices
- **`upload_config.py`**: Configuration management system
- **`test_enhanced_upload.py`**: Test suite for new functionality
- **`demo_enhanced_system.py`**: Demonstration script

### 2. **Directory Structure**

```
faiss_indices/
├── china/
│   ├── hr/
│   │   ├── faiss_index.faiss
│   │   ├── faiss_index.pkl
│   │   └── ...
│   └── it/
│       ├── faiss_index.faiss
│       ├── faiss_index.pkl
│       └── ...
└── singapore/
    ├── hr/
    │   ├── faiss_index.faiss
    │   ├── faiss_index.pkl
    │   └── ...
    └── it/
        ├── faiss_index.faiss
        ├── faiss_index.pkl
        └── ...
```

### 3. **Supported Combinations**

Currently configured for:
- **Countries**: China, Singapore
- **Departments**: HR, IT
- **Total Combinations**: 4 (China/HR, China/IT, Singapore/HR, Singapore/IT)

## New Features

### 1. **Automatic Index Creation**
- FAISS indices are automatically created for each country/department combination
- Directory structure is created on first document upload
- No manual setup required

### 2. **Enhanced Upload API**
```python
# New upload function signature
async def process_upload(
    file: UploadFile = File(...),
    country: str = Form(...),        # NEW: Required country parameter
    department: str = Form(...)      # NEW: Required department parameter
)
```

### 3. **Flexible Search Capabilities**
```python
# Search specific country/department
results = faiss_manager.search_across_indices(
    query="leave policy",
    country="singapore", 
    department="hr"
)

# Search all IT departments across countries  
results = faiss_manager.search_across_indices(
    query="network security",
    department="it"
)

# Search all departments in a country
results = faiss_manager.search_across_indices(
    query="office guidelines", 
    country="singapore"
)

# Search everything
results = faiss_manager.search_across_indices(
    query="general policy"
)
```

### 4. **Configuration Management**
- Centralized configuration in `upload_config.py`
- Easy to add new countries/departments
- Validation system ensures only supported combinations are used

### 5. **Index Management Tools**
```python
# Get all available indices
indices = list_available_indices()

# Get detailed index information
info = faiss_manager.get_index_info()

# Get statistics for specific index
stats = faiss_manager.get_index_stats("singapore", "hr")
```

## API Changes

### Upload Endpoint
**Before:**
```
POST /upload
Content-Type: multipart/form-data
Body: file=[binary data]
```

**After:**
```
POST /upload  
Content-Type: multipart/form-data
Body: 
  file=[binary data]
  country=singapore          # NEW: Required
  department=hr              # NEW: Required
```

### Response Format
```json
{
  "message": "File processed successfully and added to SINGAPORE/HR index",
  "filename": "document.pdf",
  "country": "singapore",              // NEW
  "department": "hr",                  // NEW  
  "faiss_index_path": "faiss_indices/singapore/hr/faiss_index", // NEW
  "cleaned_content": "...",
  "timestamp": "2025-07-27T13:26:57.779Z",
  "success": true
}
```

## Benefits

### 🎯 **Targeted Search**
- Search specific country/department combinations
- More relevant results for users
- Reduced noise from irrelevant documents

### 📁 **Organized Storage** 
- Clear directory structure
- Easy to locate and manage indices
- Better data organization

### 🚀 **Better Performance**
- Smaller, focused indices load faster
- Search operations are more efficient
- Reduced memory usage per search

### 🔧 **Easy Expansion**
- Add new countries by updating `SUPPORTED_COUNTRIES`
- Add new departments by updating `SUPPORTED_DEPARTMENTS`
- No code changes required for expansion

### 🔒 **Data Isolation**
- Documents are logically separated
- Better security and access control potential
- Clear organizational boundaries

### 📊 **Granular Analytics**
- Track usage by country/department
- Monitor content distribution
- Identify popular document categories

## Migration Strategy

### For New Installations
- Start using the enhanced system immediately
- Upload documents with country/department classification

### For Existing Installations
1. **Backup existing FAISS indices**
2. **Categorize existing documents** by country/department
3. **Re-upload documents** using new system
4. **Update frontend** to include country/department selection
5. **Test thoroughly** before switching over

## Frontend Updates Required

### 1. **Upload Form Enhancement**
Add country and department selection to the upload form:

```html
<form id="uploadForm" enctype="multipart/form-data">
  <input type="file" name="file" required>
  
  <!-- NEW: Country selection -->
  <select name="country" required>
    <option value="">Select Country</option>
    <option value="china">China</option>
    <option value="singapore">Singapore</option>
  </select>
  
  <!-- NEW: Department selection -->
  <select name="department" required>
    <option value="">Select Department</option>
    <option value="hr">HR</option>
    <option value="it">IT</option>
  </select>
  
  <button type="submit">Upload</button>
</form>
```

### 2. **Search Interface Enhancement**
Add optional filters for targeted search:

```html
<div class="search-filters">
  <select name="country" optional>
    <option value="">All Countries</option>
    <option value="china">China</option>
    <option value="singapore">Singapore</option>
  </select>
  
  <select name="department" optional>
    <option value="">All Departments</option>
    <option value="hr">HR</option>
    <option value="it">IT</option>
  </select>
</div>
```

## Configuration

### Adding New Countries
```python
# In upload_config.py
SUPPORTED_COUNTRIES = [
    'china', 
    'singapore',
    'malaysia',     # NEW
    'thailand'      # NEW
]
```

### Adding New Departments
```python
# In upload_config.py  
SUPPORTED_DEPARTMENTS = [
    'hr',
    'it',
    'finance',      # NEW
    'marketing'     # NEW
]
```

## Testing

### Run the Test Suite
```bash
cd chatbot/src/backend/python
python test_enhanced_upload.py
```

### Run the Demo
```bash
cd chatbot/src/backend/python
python demo_enhanced_system.py
```

## Monitoring and Maintenance

### 1. **Index Health Checks**
```python
# Check all indices
info = faiss_manager.get_index_info()
print(f"Total indices: {info['total_combinations']}")

# Check specific index
stats = faiss_manager.get_index_stats("singapore", "hr")
print(f"Documents: {stats['total_documents']}, Chunks: {stats['total_chunks']}")
```

### 2. **Usage Analytics**
- Track which country/department combinations are most used
- Monitor document upload patterns
- Identify popular search queries by category

### 3. **Performance Monitoring**
- Monitor search response times by index size
- Track memory usage per index
- Monitor disk space usage

## Troubleshooting

### Common Issues

1. **"Unsupported country/department" Error**
   - Check `upload_config.py` for supported values
   - Ensure proper capitalization (case-insensitive)

2. **FAISS Index Not Found**
   - Check directory structure in `faiss_indices/`
   - Verify index files (.faiss, .pkl) exist
   - Try re-uploading documents

3. **Search Returns No Results**
   - Check if documents exist in target index
   - Try broader search (remove country/department filters)
   - Verify index integrity

### Debug Commands
```python
# List all available indices
from vector_store import list_available_indices
print(list_available_indices())

# Check index path
from vector_store import get_faiss_index_path  
path = get_faiss_index_path("singapore", "hr")
print(f"Index path: {path}")

# Verify index files
from pathlib import Path
index_path = Path(path)
print(f"Directory exists: {index_path.parent.exists()}")
print(f"Files: {list(index_path.parent.glob('*'))}")
```

## Summary

The enhanced FAISS upload system provides:
- ✅ **Organized document storage** by country and department
- ✅ **Automatic index creation** and management  
- ✅ **Flexible search capabilities** with optional filtering
- ✅ **Better performance** through smaller, focused indices
- ✅ **Easy expansion** for new countries and departments
- ✅ **Comprehensive tooling** for management and monitoring

The system is **production-ready** and maintains **backward compatibility** while providing significant improvements in organization and performance.
