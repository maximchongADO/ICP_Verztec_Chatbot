"""
Fast FAISS Inspector - No Embedding Model Required
Based on check.py but optimized for web API usage
"""

import os
import sys
import json
import argparse
import pickle
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_community.docstore.in_memory import InMemoryDocstore

class FastFAISSInspector:
    def __init__(self, base_path: str = "faiss_indices"):
        """
        Initialize the fast FAISS inspector
        
        Args:
            base_path: Base path to the FAISS indices directory
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_path = os.path.join(self.script_dir, base_path)
    
    def get_index_path(self, country: str = None, department: str = None, admin_master: bool = False) -> str:
        """Get the correct index path based on filters"""
        if admin_master:
            # Admin master index path
            return os.path.join(self.base_path, "admin_master", "faiss_index")
        elif country and department:
            # Specific country/department path - correct structure is country/department/faiss_index
            return os.path.join(self.base_path, country, department, "faiss_index")
        else:
            # Default to admin master if no specific filters
            return os.path.join(self.base_path, "admin_master", "faiss_index")
    
    def inspect_index(self, country: str = None, department: str = None, admin_master: bool = False) -> Dict[str, Any]:
        """
        Fast inspection of FAISS index without loading embedding model
        
        Args:
            country: Country filter
            department: Department filter
            admin_master: Whether to load admin master index
            
        Returns:
            Dictionary containing index information and document contents
        """
        try:
            index_path = self.get_index_path(country, department, admin_master)
            faiss_file = os.path.join(index_path, "index.faiss")
            metadata_file = os.path.join(index_path, "index.pkl")
            
            print(f"Inspecting index at: {index_path}", file=sys.stderr)
            
            # Check if files exist
            if not os.path.exists(index_path):
                return {
                    "error": f"FAISS index directory not found: {index_path}",
                    "status": "not_found"
                }
            
            if not os.path.exists(faiss_file):
                return {
                    "error": f"FAISS index file not found: {faiss_file}",
                    "status": "not_found"
                }
            
            if not os.path.exists(metadata_file):
                return {
                    "error": f"FAISS metadata file not found: {metadata_file}",
                    "status": "not_found"
                }
            
            # Load FAISS index directly (fast)
            index = faiss.read_index(faiss_file)
            
            # Load metadata directly (fast)
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Extract basic statistics
            total_vectors = index.ntotal
            dimension = index.d
            
            # Get file modification time
            modification_time = os.path.getmtime(faiss_file)
            
            # Process metadata to extract document information
            files_data = []
            total_content_length = 0
            unique_sources = set()
            
            if isinstance(metadata, tuple) and len(metadata) == 2:
                docstore, id_mapping = metadata
                
                if isinstance(docstore, InMemoryDocstore):
                    print(f"Processing {len(id_mapping)} documents...", file=sys.stderr)
                    
                    # Group documents by source file
                    source_files = {}
                    
                    for index_id, doc_id in id_mapping.items():
                        doc = docstore._dict.get(doc_id)
                        if doc:
                            # Extract source information from metadata
                            source = doc.metadata.get('source', 'Unknown')
                            unique_sources.add(source)
                            
                            if source not in source_files:
                                source_files[source] = {
                                    'filename': os.path.basename(source),
                                    'full_path': source,
                                    'chunks': [],
                                    'total_content_length': 0,
                                    'metadata': doc.metadata
                                }
                            
                            # Add chunk information
                            chunk_content = str(doc.page_content)
                            source_files[source]['chunks'].append({
                                'index': index_id,
                                'content': chunk_content,
                                'length': len(chunk_content)
                            })
                            source_files[source]['total_content_length'] += len(chunk_content)
                            total_content_length += len(chunk_content)
                    
                    # Convert to list format
                    files_data = []
                    for source, file_info in source_files.items():
                        files_data.append({
                            'filename': file_info['filename'],
                            'full_path': file_info['full_path'],
                            'chunks_count': len(file_info['chunks']),
                            'content_length': file_info['total_content_length'],
                            'metadata': file_info['metadata'],
                            'chunks': file_info['chunks']  # Include actual chunk content for viewing
                        })
                    
                    # Sort by filename for consistent display
                    files_data.sort(key=lambda x: x['filename'])
            
            # Prepare summary
            summary = {
                'total_files': len(unique_sources),
                'total_chunks': total_vectors,
                'total_content_length': total_content_length,
                'index_dimension': dimension,
                'extraction_time': datetime.fromtimestamp(modification_time).isoformat(),
                'index_path': index_path
            }
            
            return {
                'status': 'success',
                'summary': summary,
                'files': files_data
            }
            
        except Exception as e:
            error_msg = f"Error inspecting FAISS index: {str(e)}"
            print(error_msg, file=sys.stderr)
            return {
                'error': error_msg,
                'status': 'error'
            }
    
    def list_available_indices(self) -> Dict[str, Any]:
        """List all available FAISS indices"""
        try:
            available_indices = []
            
            if os.path.exists(self.base_path):
                for item in os.listdir(self.base_path):
                    item_path = os.path.join(self.base_path, item)
                    if os.path.isdir(item_path):
                        if item == "admin_master":
                            # Check admin master index
                            faiss_index_path = os.path.join(item_path, "faiss_index")
                            if os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
                                available_indices.append({
                                    'type': 'admin_master',
                                    'country': 'admin',
                                    'department': 'master',
                                    'path': faiss_index_path
                                })
                        else:
                            # Check for country directories
                            for dept in os.listdir(item_path):
                                dept_path = os.path.join(item_path, dept)
                                if os.path.isdir(dept_path):
                                    faiss_index_path = os.path.join(dept_path, "faiss_index")
                                    if os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
                                        available_indices.append({
                                            'type': 'specific',
                                            'country': item,
                                            'department': dept,
                                            'path': faiss_index_path
                                        })
            
            return {
                'status': 'success',
                'indices': available_indices
            }
            
        except Exception as e:
            return {
                'error': f"Error listing indices: {str(e)}",
                'status': 'error'
            }

def main():
    parser = argparse.ArgumentParser(description='Fast FAISS Inspector')
    parser.add_argument('command', choices=['list', 'inspect'], help='Command to execute')
    parser.add_argument('--country', help='Country filter')
    parser.add_argument('--department', help='Department filter')
    parser.add_argument('--admin-master', action='store_true', help='Use admin master index')
    parser.add_argument('--user-role', help='User role (for access control)')
    
    args = parser.parse_args()
    
    inspector = FastFAISSInspector()
    
    if args.command == 'list':
        result = inspector.list_available_indices()
    elif args.command == 'inspect':
        result = inspector.inspect_index(
            country=args.country,
            department=args.department,
            admin_master=args.admin_master
        )
    
    # Output JSON result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
