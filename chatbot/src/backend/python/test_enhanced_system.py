#!/usr/bin/env python3
"""
Test the enhanced upload system end-to-end
This script tests the new country/department-specific file upload functionality
"""

import json
import os
import sys
from pathlib import Path

# Add the backend python directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    try:
        from upload_config import get_config_info, validate_country_department
        
        config = get_config_info()
        print(f"✓ Config loaded: {config['total_combinations']} combinations")
        print(f"  Countries: {', '.join(config['supported_countries'])}")
        print(f"  Departments: {', '.join(config['supported_departments'])}")
        
        # Test validation
        try:
            country, dept = validate_country_department("singapore", "hr")
            print(f"✓ Validation works: {country}/{dept}")
        except ValueError as e:
            print(f"✗ Validation error: {e}")
            return False
            
        # Test invalid validation
        try:
            validate_country_department("invalid", "hr")
            print("✗ Validation should have failed for invalid country")
            return False
        except ValueError:
            print("✓ Validation correctly rejects invalid input")
            
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_directory_creation():
    """Test FAISS directory creation"""
    print("\n📁 Testing directory creation...")
    try:
        from vector_store import get_faiss_index_path
        
        test_combinations = [
            ("singapore", "hr"),
            ("singapore", "it"),
            ("china", "hr"),
            ("china", "it")
        ]
        
        for country, dept in test_combinations:
            path = get_faiss_index_path(country, dept)
            print(f"✓ Path created: {country}/{dept} -> {path}")
            
            # Check if directory exists
            path_obj = Path(path)
            if path_obj.parent.exists():
                print(f"  ✓ Directory exists: {path_obj.parent}")
            else:
                print(f"  ✗ Directory not found: {path_obj.parent}")
                
        return True
        
    except Exception as e:
        print(f"✗ Directory test failed: {e}")
        return False

def test_faiss_manager():
    """Test FAISS manager functionality"""
    print("\n🔍 Testing FAISS manager...")
    try:
        from faiss_manager import faiss_manager
        
        # Test index info
        info = faiss_manager.get_index_info()
        print(f"✓ Index info retrieved: {info['total_combinations']} combinations")
        
        # Test search (this will work even with empty indices)
        results = faiss_manager.search_across_indices("test query", k=1)
        print(f"✓ Search functionality works (returned {len(results)} results)")
        
        return True
        
    except Exception as e:
        print(f"✗ FAISS manager test failed: {e}")
        return False

def test_helper_scripts():
    """Test the helper scripts that Node.js will call"""
    print("\n🐍 Testing helper scripts...")
    
    # Test get_config.py
    try:
        import subprocess
        
        result = subprocess.run([
            sys.executable, "get_config.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            config = json.loads(result.stdout)
            if config.get("success"):
                print("✓ get_config.py works")
            else:
                print(f"✗ get_config.py returned error: {config}")
                return False
        else:
            print(f"✗ get_config.py failed: {result.stderr}")
            return False
            
        # Test get_indices.py
        result = subprocess.run([
            sys.executable, "get_indices.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            indices = json.loads(result.stdout)
            if indices.get("success"):
                print("✓ get_indices.py works")
            else:
                print(f"✗ get_indices.py returned error: {indices}")
                return False
        else:
            print(f"✗ get_indices.py failed: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Helper scripts test failed: {e}")
        return False

def create_test_file():
    """Create a test file for upload testing"""
    test_content = """
    Test Document for Enhanced Upload System
    
    This is a test document to verify the country/department-specific
    FAISS upload functionality. 
    
    Content includes:
    - Country: Singapore
    - Department: HR
    - Document Type: Policy
    - Test Date: 2025-07-27
    
    This document should be processed and stored in the Singapore/HR
    FAISS index when uploaded through the enhanced system.
    """
    
    test_file = Path("test_upload_document.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    return test_file

def test_upload_handler():
    """Test the enhanced upload handler"""
    print("\n📤 Testing upload handler...")
    
    try:
        # Create test file
        test_file = create_test_file()
        
        # Test the upload handler
        import subprocess
        
        result = subprocess.run([
            sys.executable, "enhanced_upload_handler.py",
            str(test_file), "singapore", "hr"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if response.get("success"):
                print("✓ Upload handler works")
                print(f"  File processed for: {response.get('country')}/{response.get('department')}")
                print(f"  FAISS index: {response.get('faiss_index_path')}")
                return True
            else:
                print(f"✗ Upload handler returned error: {response.get('error')}")
                return False
        else:
            print(f"✗ Upload handler failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Upload handler test failed: {e}")
        # Clean up test file on error
        test_file = Path("test_upload_document.txt")
        if test_file.exists():
            test_file.unlink()
        return False

def main():
    """Run all tests"""
    print("🚀 Enhanced Upload System - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Directory Creation", test_directory_creation),
        ("FAISS Manager", test_faiss_manager),
        ("Helper Scripts", test_helper_scripts),
        ("Upload Handler", test_upload_handler)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced upload system is ready.")
        print("\nNext steps:")
        print("1. Update your database schema with the migration script")
        print("2. Restart your Node.js server")
        print("3. Test the frontend interface")
        print("4. Upload some documents to verify end-to-end functionality")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("The system may not work correctly until all tests pass.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
