"""
Demo script for the enhanced country/department-specific FAISS upload system
This script shows how to use the new functionality
"""

import os
from pathlib import Path
from vector_store import get_faiss_index_path, list_available_indices
from upload_config import get_config_info, validate_country_department
from faiss_manager import get_all_index_info

def demo_directory_structure():
    """Demonstrate how the directory structure is organized"""
    print("=" * 60)
    print("ENHANCED FAISS UPLOAD SYSTEM - DIRECTORY STRUCTURE DEMO")
    print("=" * 60)
    
    print("\n1. Directory Structure Overview:")
    print("   The system organizes FAISS indices by country and department:")
    print("   faiss_indices/")
    print("   ├── china/")
    print("   │   ├── hr/")
    print("   │   │   ├── faiss_index.faiss")
    print("   │   │   ├── faiss_index.pkl")
    print("   │   │   └── ...")
    print("   │   └── it/")
    print("   │       ├── faiss_index.faiss")
    print("   │       ├── faiss_index.pkl")
    print("   │       └── ...")
    print("   └── singapore/")
    print("       ├── hr/")
    print("       │   ├── faiss_index.faiss")
    print("       │   ├── faiss_index.pkl")
    print("       │   └── ...")
    print("       └── it/")
    print("           ├── faiss_index.faiss")
    print("           ├── faiss_index.pkl")
    print("           └── ...")

def demo_configuration():
    """Demonstrate the configuration system"""
    print("\n2. Configuration System:")
    
    config = get_config_info()
    
    print(f"   Supported Countries: {', '.join(config['supported_countries'])}")
    print(f"   Supported Departments: {', '.join(config['supported_departments'])}")
    print(f"   Supported File Types: {', '.join(config['supported_file_types'])}")
    print(f"   Total Valid Combinations: {config['total_combinations']}")
    
    print("\n   Valid Country/Department Combinations:")
    countries = config['supported_countries']
    departments = config['supported_departments']
    
    for i, country in enumerate(countries, 1):
        for j, dept in enumerate(departments, 1):
            combo_num = (i-1) * len(departments) + j
            print(f"     {combo_num}. {country.upper()}/{dept.upper()}")

def demo_path_generation():
    """Demonstrate how paths are generated"""
    print("\n3. Path Generation Examples:")
    
    examples = [
        ("singapore", "hr"),
        ("singapore", "it"),
        ("china", "hr"), 
        ("china", "it")
    ]
    
    for country, department in examples:
        path = get_faiss_index_path(country, department)
        print(f"   {country.upper()}/{department.upper()}: {path}")
        
        # Check if directory exists
        path_obj = Path(path).parent
        if path_obj.exists():
            print(f"     ✓ Directory exists")
        else:
            print(f"     ○ Directory will be created on first upload")

def demo_validation():
    """Demonstrate the validation system"""
    print("\n4. Validation Examples:")
    
    test_cases = [
        ("singapore", "hr", True),
        ("CHINA", "IT", True),  # Case insensitive
        ("Singapore", "Hr", True),  # Mixed case
        ("malaysia", "hr", False),  # Unsupported country
        ("singapore", "finance", False),  # Unsupported department
        ("invalid", "invalid", False)  # Both invalid
    ]
    
    for country, department, should_pass in test_cases:
        try:
            validated = validate_country_department(country, department)
            if should_pass:
                print(f"   ✓ '{country}'/'{department}' → {validated[0]}/{validated[1]}")
            else:
                print(f"   ⚠ '{country}'/'{department}' → Unexpectedly passed: {validated}")
        except ValueError as e:
            if not should_pass:
                print(f"   ✓ '{country}'/'{department}' → Correctly rejected: {str(e)}")
            else:
                print(f"   ✗ '{country}'/'{department}' → Unexpectedly failed: {str(e)}")

def demo_usage_examples():
    """Show practical usage examples"""
    print("\n5. Usage Examples:")
    
    print("\n   A. File Upload API Call:")
    print("   POST /upload")
    print("   Content-Type: multipart/form-data")
    print("   Body:")
    print("     file: [binary file data]")
    print("     country: 'singapore'")
    print("     department: 'hr'")
    
    print("\n   B. Python Code Example:")
    print("   ```python")
    print("   from fileUpload import process_upload")
    print("   ")
    print("   # Upload file to Singapore HR index")
    print("   result = await process_upload(")
    print("       file=uploaded_file,")
    print("       country='singapore',")
    print("       department='hr'")
    print("   )")
    print("   ```")
    
    print("\n   C. Search Examples:")
    print("   ```python")
    print("   from faiss_manager import faiss_manager")
    print("   ")
    print("   # Search only Singapore HR documents")
    print("   results = faiss_manager.search_across_indices(")
    print("       query='leave policy',")
    print("       country='singapore',")
    print("       department='hr'")
    print("   )")
    print("   ")
    print("   # Search all IT departments")
    print("   results = faiss_manager.search_across_indices(")
    print("       query='network security',")
    print("       department='it'  # All countries")
    print("   )")
    print("   ")
    print("   # Search everything")
    print("   results = faiss_manager.search_across_indices(")
    print("       query='general guidelines'")
    print("   )")
    print("   ```")

def demo_benefits():
    """Explain the benefits of the new system"""
    print("\n6. System Benefits:")
    
    benefits = [
        "🎯 Targeted Search: Search specific country/department combinations",
        "📁 Organized Storage: Clear directory structure by country and department",
        "🔧 Easy Expansion: Add new countries/departments by updating configuration",
        "🚀 Better Performance: Smaller, focused indices load and search faster",
        "🔒 Data Isolation: Documents are logically separated by organization structure",
        "📊 Granular Analytics: Track usage and content by specific divisions",
        "🛠 Flexible Querying: Search across all, specific countries, or specific departments",
        "📈 Scalable Architecture: System grows with organizational structure"
    ]
    
    for benefit in benefits:
        print(f"   {benefit}")

def demo_next_steps():
    """Show next steps for implementation"""
    print("\n7. Next Steps for Implementation:")
    
    steps = [
        "1. Test the system with sample documents",
        "2. Update frontend to include country/department selection",
        "3. Migrate existing documents to new structure if needed",
        "4. Add new countries/departments as required",
        "5. Implement search filtering in the chatbot interface",
        "6. Set up monitoring and analytics for usage tracking",
        "7. Create admin interface for index management"
    ]
    
    for step in steps:
        print(f"   {step}")

if __name__ == "__main__":
    print("Enhanced FAISS Upload System - Complete Demo")
    
    # Run all demo sections
    demo_directory_structure()
    demo_configuration()
    demo_path_generation()
    demo_validation()
    demo_usage_examples()
    demo_benefits()
    demo_next_steps()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    
    print("\nThe enhanced system is ready to use!")
    print("Key files updated:")
    print("  - vector_store.py: Enhanced with country/department support")
    print("  - fileUpload.py: Updated to handle country/department parameters")
    print("  - faiss_manager.py: New utility for managing organized indices")
    print("  - upload_config.py: Configuration management")
    print("  - test_enhanced_upload.py: Test suite")
    print("  - demo_enhanced_system.py: This demo file")
    
    print("\nTo start using:")
    print("  1. Run the test script to verify functionality")
    print("  2. Update your frontend to include country/department selection")
    print("  3. Begin uploading documents to organized indices")
    
    print("\nSupported combinations:")
    config = get_config_info()
    for country in config['supported_countries']:
        for dept in config['supported_departments']:
            print(f"  - {country.upper()}/{dept.upper()}")
    
    print(f"\nTotal: {config['total_combinations']} valid combinations")
