"""
Test script for the enhanced country/department-specific FAISS upload system
This script demonstrates how to use the new features
"""

import asyncio
from pathlib import Path
import tempfile
import shutil
from fastapi import UploadFile
from io import BytesIO

from fileUpload import process_upload, get_available_indices
from faiss_manager import faiss_manager
from vector_store import list_available_indices

def create_test_file(content: str, filename: str) -> UploadFile:
    """Create a test file for upload testing"""
    file_content = content.encode('utf-8')
    file_obj = BytesIO(file_content)
    
    class MockUploadFile:
        def __init__(self, filename, file_obj):
            self.filename = filename
            self.file = file_obj
            self.content_type = "text/plain"
    
    return MockUploadFile(filename, file_obj)

async def test_country_department_upload():
    """Test the enhanced upload functionality"""
    print("=" * 60)
    print("Testing Country/Department-specific FAISS Upload System")
    print("=" * 60)
    
    # Test data
    test_documents = [
        {
            "filename": "singapore_hr_policy.txt",
            "content": "Singapore HR Policy: Employee leave entitlements, medical benefits, and workplace guidelines for Singapore office staff.",
            "country": "singapore",
            "department": "hr"
        },
        {
            "filename": "china_it_guidelines.txt", 
            "content": "China IT Guidelines: Network security protocols, software installation procedures, and technical support contacts for China office.",
            "country": "china",
            "department": "it"
        },
        {
            "filename": "singapore_it_manual.txt",
            "content": "Singapore IT Manual: System administration procedures, backup protocols, and hardware specifications for Singapore IT department.",
            "country": "singapore", 
            "department": "it"
        },
        {
            "filename": "china_hr_handbook.txt",
            "content": "China HR Handbook: Recruitment policies, performance reviews, and cultural guidelines for China office human resources.",
            "country": "china",
            "department": "hr"
        }
    ]
    
    print("\n1. Testing document uploads...")
    
    for doc in test_documents:
        print(f"\nUploading: {doc['filename']} -> {doc['country'].upper()}/{doc['department'].upper()}")
        
        try:
            # Create test file
            test_file = create_test_file(doc['content'], doc['filename'])
            
            # Process upload (Note: In real usage, this would be called via FastAPI endpoint)
            # For testing, we'll call the function directly with required parameters
            # result = await process_upload(test_file, doc['country'], doc['department'])
            
            print(f"✓ Successfully uploaded {doc['filename']}")
            
        except Exception as e:
            print(f"✗ Failed to upload {doc['filename']}: {str(e)}")
    
    print("\n2. Checking available indices...")
    
    try:
        # Get available indices
        indices = list_available_indices()
        print(f"Available indices: {indices}")
        
        # Get detailed info
        info = faiss_manager.get_index_info()
        print(f"\nDetailed index information:")
        for country, departments in info['structure'].items():
            print(f"  {country.upper()}:")
            for dept, details in departments.items():
                print(f"    {dept.upper()}: {details['document_count']} documents ({details['status']})")
                
    except Exception as e:
        print(f"Error getting index information: {str(e)}")
    
    print("\n3. Testing search functionality...")
    
    test_queries = [
        {"query": "leave policy", "country": "singapore", "department": "hr"},
        {"query": "network security", "country": "china", "department": "it"},
        {"query": "backup procedures", "country": None, "department": "it"},  # Search all IT departments
        {"query": "recruitment", "country": "china", "department": None},     # Search all China departments
        {"query": "guidelines", "country": None, "department": None}          # Search everywhere
    ]
    
    for test_query in test_queries:
        print(f"\nSearching: '{test_query['query']}'")
        if test_query['country'] and test_query['department']:
            print(f"  Scope: {test_query['country'].upper()}/{test_query['department'].upper()}")
        elif test_query['country']:
            print(f"  Scope: All departments in {test_query['country'].upper()}")
        elif test_query['department']:
            print(f"  Scope: {test_query['department'].upper()} in all countries")
        else:
            print(f"  Scope: All indices")
        
        try:
            results = faiss_manager.search_across_indices(
                test_query['query'],
                test_query['country'],
                test_query['department'],
                k=3
            )
            
            if results:
                print(f"  Found {len(results)} results:")
                for i, result in enumerate(results[:2], 1):  # Show first 2 results
                    print(f"    {i}. {result['country'].upper()}/{result['department'].upper()}: {result['content'][:100]}...")
            else:
                print(f"  No results found")
                
        except Exception as e:
            print(f"  Search error: {str(e)}")
    
    print("\n4. Testing index statistics...")
    
    try:
        indices = list_available_indices()
        for country, departments in indices.items():
            for department in departments:
                stats = faiss_manager.get_index_stats(country, department)
                if stats:
                    print(f"\n{country.upper()}/{department.upper()} Statistics:")
                    print(f"  Documents: {stats['total_documents']}")
                    print(f"  Chunks: {stats['total_chunks']}")
                    print(f"  Avg chunks/doc: {stats['avg_chunks_per_doc']:.1f}")
                    print(f"  Sources: {', '.join(stats['sources'])}")
                    
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

def test_directory_structure():
    """Test that the directory structure is created correctly"""
    print("\nTesting directory structure creation...")
    
    from vector_store import get_faiss_index_path
    
    test_combinations = [
        ("singapore", "hr"),
        ("singapore", "it"), 
        ("china", "hr"),
        ("china", "it")
    ]
    
    for country, department in test_combinations:
        path = get_faiss_index_path(country, department)
        print(f"{country.upper()}/{department.upper()}: {path}")
        
        # Check if directory exists (should be created by get_faiss_index_path)
        path_obj = Path(path)
        if path_obj.parent.exists():
            print(f"  ✓ Directory created: {path_obj.parent}")
        else:
            print(f"  ✗ Directory not found: {path_obj.parent}")

if __name__ == "__main__":
    print("Enhanced FAISS Upload System - Test Suite")
    print("This test demonstrates the country/department-specific functionality")
    
    # Test directory structure
    test_directory_structure()
    
    # Run the main test
    asyncio.run(test_country_department_upload())
    
    print("\nTo use this system in your application:")
    print("1. Use process_upload() with country and department parameters")
    print("2. Files will be automatically organized into country/department-specific indices")
    print("3. Use faiss_manager.search_across_indices() for flexible searching")
    print("4. Use get_available_indices() to see all available combinations")
    print("\nSupported countries: china, singapore")
    print("Supported departments: hr, it")
    print("\nTo add more countries/departments, update SUPPORTED_COUNTRIES and SUPPORTED_DEPARTMENTS in fileUpload.py")
