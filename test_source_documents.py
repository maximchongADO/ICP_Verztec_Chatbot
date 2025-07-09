#!/usr/bin/env python3
"""
Test script to debug the source document structure and file serving.
"""
import sys
import os
import json
import requests
import time

def test_file_serving():
    """Test the file serving endpoint."""
    print("Testing File Serving Endpoint...")
    print("-" * 40)
    
    try:
        # Test the debug endpoint first
        response = requests.get('http://localhost:3000/test-documents')
        if response.status_code == 200:
            data = response.json()
            print("File Serving Test Results:")
            print(json.dumps(data, indent=2))
            
            # Try to access a specific file if available
            for dir_info in data['directories']:
                if 'files' in dir_info and dir_info['files']:
                    first_file = dir_info['files'][0]
                    file_url = f"http://localhost:3000{first_file['link']}"
                    print(f"\nTesting file access: {file_url}")
                    
                    file_response = requests.get(file_url)
                    print(f"File response status: {file_response.status_code}")
                    if file_response.status_code == 200:
                        print(f"File size: {len(file_response.content)} bytes")
                        print(f"Content type: {file_response.headers.get('Content-Type', 'N/A')}")
                        print("✅ File serving working correctly!")
                    else:
                        print(f"❌ Error accessing file: {file_response.text}")
                    break
            else:
                print("No files found to test")
        else:
            print(f"❌ File serving test failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing file serving: {e}")

def test_backend_response():
    """Test the backend response structure."""
    print("\nTesting Backend Response Structure...")
    print("-" * 40)
    
    try:
        # Test different types of queries
        test_queries = [
            "What is the meeting etiquette?",
            "How do I use the pantry?",
            "What is the laptop policy?",
            "Hello how are you?"  # This should be handled as casual
        ]
        
        for query in test_queries:
            print(f"\nTesting query: '{query}'")
            
            response = requests.post('http://localhost:5000/chatbot', 
                                   json={
                                       'message': query,
                                       'user_id': 'test_user',
                                       'chat_id': 'test_chat'
                                   })
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend responded successfully")
                print(f"Response keys: {list(data.keys())}")
                
                # Check if sources are present
                if 'sources' in data:
                    sources = data['sources']
                    print(f"Sources found: {len(sources)}")
                    for i, source in enumerate(sources[:2]):  # Show first 2 sources
                        print(f"  Source {i+1}: {source}")
                        
                        # Test if the source files can be accessed
                        if source.get('is_clickable') and source.get('file_path'):
                            filename = os.path.basename(source['file_path'])
                            file_url = f"http://localhost:3000/documents/{filename}"
                            try:
                                file_response = requests.head(file_url, timeout=5)
                                if file_response.status_code == 200:
                                    print(f"    ✅ File accessible: {filename}")
                                else:
                                    print(f"    ❌ File not accessible: {filename} (status: {file_response.status_code})")
                            except Exception as e:
                                print(f"    ❌ Error checking file: {e}")
                else:
                    print("No sources found in response")
                    
            else:
                print(f"❌ Backend request failed with status: {response.status_code}")
                print(f"Response: {response.text}")
            
            time.sleep(1)  # Small delay between requests
            
    except Exception as e:
        print(f"❌ Error testing backend: {e}")

def test_ui_elements():
    """Test UI element generation."""
    print("\nTesting UI Elements...")
    print("-" * 40)
    
    # Sample source data that would come from backend
    sample_sources = [
        {
            "name": "Basic Meeting Etiquette for Professionals",
            "file_path": "c:\\path\\to\\11A_Basic Meeting Etiquette for Professionals.pdf",
            "is_clickable": True
        },
        {
            "name": "Verztec Pantry Rules",
            "file_path": "c:\\path\\to\\18_Verztec_Pantry Rules_060715.pdf",
            "is_clickable": True
        },
        {
            "name": "Policy Document",
            "file_path": None,
            "is_clickable": False
        }
    ]
    
    print("Sample sources structure:")
    for i, source in enumerate(sample_sources):
        print(f"  {i+1}. {source['name']}")
        print(f"     Clickable: {source['is_clickable']}")
        print(f"     File path: {source['file_path']}")

def check_server_status():
    """Check if servers are running."""
    print("Checking Server Status...")
    print("-" * 40)
    
    servers = [
        {"name": "Backend (Python)", "url": "http://localhost:5000/health", "port": 5000},
        {"name": "Frontend (Node.js)", "url": "http://localhost:3000", "port": 3000}
    ]
    
    all_running = True
    for server in servers:
        try:
            response = requests.get(server["url"], timeout=5)
            if response.status_code == 200:
                print(f"✅ {server['name']} is running on port {server['port']}")
            else:
                print(f"❌ {server['name']} responded with status {response.status_code}")
                all_running = False
        except Exception as e:
            print(f"❌ {server['name']} is not accessible: {e}")
            all_running = False
    
    return all_running

if __name__ == '__main__':
    print("🔍 Testing Enhanced Source Document Implementation")
    print("=" * 60)
    
    # Check if servers are running
    if check_server_status():
        print("\n" + "=" * 60)
        test_file_serving()
        test_backend_response()
        test_ui_elements()
    else:
        print("\n⚠️  Some servers are not running. Please start them first:")
        print("   - Backend: Run the Python FastAPI server")
        print("   - Frontend: Run the Node.js Express server")
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
