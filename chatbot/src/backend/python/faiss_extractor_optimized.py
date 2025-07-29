"""
Optimized FAISS Vector Store Data Extractor with Model Caching
This script addresses the slow embedding model loading issue by implementing caching.
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Global cache for the embedding model
_embedding_model_cache = None
_model_load_time = None

def get_embedding_model():
    """Get cached embedding model or load it if not cached"""
    global _embedding_model_cache, _model_load_time
    
    if _embedding_model_cache is not None:
        print(f"Using cached embedding model (loaded in {_model_load_time:.2f}s)", file=sys.stderr)
        return _embedding_model_cache
    
    print("Loading embedding model for the first time...", file=sys.stderr)
    import time
    start_time = time.time()
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        _embedding_model_cache = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={'normalize_embeddings': True},
            model_kwargs={'device': 'cpu'}
        )
        
        _model_load_time = time.time() - start_time
        print(f"Embedding model loaded successfully in {_model_load_time:.2f}s", file=sys.stderr)
        return _embedding_model_cache
        
    except Exception as e:
        print(f"Failed to load embedding model: {e}", file=sys.stderr)
        raise

class OptimizedFAISSExtractor:
    def __init__(self, index_path: str = "faiss_master_index3"):
        """
        Initialize the FAISS extractor with cached embedding model
        
        Args:
            index_path: Path to the FAISS index directory
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(self.script_dir, index_path)
        
        # Use cached embedding model
        self.embedding_model = get_embedding_model()
        self.vectorstore = None
        self.load_index()
    
    def load_index(self):
        """Load the existing FAISS index"""
        try:
            if os.path.exists(self.index_path):
                print(f"Loading FAISS index from {self.index_path}", file=sys.stderr)
                
                from langchain_community.vectorstores import FAISS
                
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"FAISS index loaded successfully", file=sys.stderr)
            else:
                error_msg = f"FAISS index not found at {self.index_path}"
                print(error_msg, file=sys.stderr)
                print(json.dumps({"error": error_msg}))
                sys.exit(1)
        except Exception as e:
            error_msg = f"Failed to load FAISS index: {e}"
            print(error_msg, file=sys.stderr)
            print(json.dumps({"error": error_msg}))
            sys.exit(1)

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get general statistics about the FAISS index (fast operation)"""
        if not self.vectorstore:
            return {"status": "no_index", "error": "FAISS index not loaded"}
        
        try:
            total_vectors = self.vectorstore.index.ntotal
            dimension = self.vectorstore.index.d
            
            # Quick document count
            doc_count = 0
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                doc_count = len(self.vectorstore.docstore._dict)
            
            return {
                "status": "loaded",
                "index_info": {
                    "total_vectors": total_vectors,
                    "vector_dimension": dimension,
                    "index_path": self.index_path,
                    "document_count": doc_count
                },
                "content_stats": {
                    "total_files": "calculated_on_demand",  # Don't calculate this in stats call
                    "total_chunks": doc_count,
                    "total_content_length": "calculated_on_demand",
                    "extraction_time": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Extract all documents from the FAISS index"""
        if not self.vectorstore:
            print(json.dumps({"error": "FAISS vectorstore not initialized"}))
            return []
        
        try:
            documents = []
            
            if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                for doc_id, doc in self.vectorstore.docstore._dict.items():
                    doc_info = {
                        "id": doc_id,
                        "content": doc.page_content,
                        "content_length": len(doc.page_content),
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    documents.append(doc_info)
            
            return documents
            
        except Exception as e:
            print(f"Error extracting documents: {e}", file=sys.stderr)
            print(json.dumps({"error": f"Error extracting documents: {str(e)}"}))
            return []

    def group_documents_by_source(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group documents by their source file"""
        files_dict = {}
        
        for doc in documents:
            source = doc["metadata"].get("source", "unknown")
            filename = os.path.basename(source) if source != "unknown" else f"doc_{doc['id']}"
            
            if filename not in files_dict:
                files_dict[filename] = {
                    "filename": filename,
                    "source_path": source,
                    "chunks": [],
                    "total_content_length": 0,
                    "created_at": None,
                    "file_type": self._get_file_type(filename)
                }
            
            files_dict[filename]["chunks"].append({
                "id": doc["id"],
                "content": doc["content"],
                "content_length": doc["content_length"],
                "content_preview": doc["content_preview"],
                "metadata": doc["metadata"]
            })
            
            files_dict[filename]["total_content_length"] += doc["content_length"]
            
            if not files_dict[filename]["created_at"]:
                created_at = doc["metadata"].get("created_at") or doc["metadata"].get("timestamp")
                if created_at:
                    files_dict[filename]["created_at"] = created_at

        # Convert to list and add statistics
        files_list = []
        for filename, file_data in files_dict.items():
            file_data["chunk_count"] = len(file_data["chunks"])
            file_data["avg_chunk_size"] = (
                file_data["total_content_length"] // file_data["chunk_count"] 
                if file_data["chunk_count"] > 0 else 0
            )
            
            file_data["chunks"].sort(key=lambda x: x["content_length"], reverse=True)
            files_list.append(file_data)
        
        files_list.sort(key=lambda x: x["chunk_count"], reverse=True)
        
        return {
            "files": files_list,
            "summary": {
                "total_files": len(files_list),
                "total_chunks": sum(f["chunk_count"] for f in files_list),
                "total_content_length": sum(f["total_content_length"] for f in files_list),
                "extraction_time": datetime.now().isoformat()
            }
        }

    def _get_file_type(self, filename: str) -> str:
        """Get file type from filename extension"""
        try:
            extension = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
            type_mapping = {
                'pdf': 'PDF Document',
                'doc': 'Word Document',
                'docx': 'Word Document', 
                'txt': 'Text File',
                'pptx': 'PowerPoint Presentation',
                'ppt': 'PowerPoint Presentation'
            }
            return type_mapping.get(extension, f'{extension.upper()} File')
        except:
            return 'Unknown File Type'

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "source": doc.metadata.get("source", "unknown")
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching documents: {e}", file=sys.stderr)
            return []

def warmup_model():
    """Warm up the embedding model to cache it"""
    try:
        print("Warming up embedding model...", file=sys.stderr)
        model = get_embedding_model()
        # Test embedding to ensure model is fully loaded
        test_text = "test"
        _ = model.embed_query(test_text)
        print("Model warmed up successfully", file=sys.stderr)
        return {"status": "success", "message": "Embedding model warmed up"}
    except Exception as e:
        error_msg = f"Failed to warm up model: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {"status": "error", "error": error_msg}

def main():
    parser = argparse.ArgumentParser(description='Optimized FAISS vector store extractor')
    parser.add_argument('command', choices=['list', 'stats', 'search', 'warmup'], 
                       help='Command to execute')
    parser.add_argument('--query', '-q', type=str, 
                       help='Search query (for search command)')
    parser.add_argument('--limit', '-l', type=int, default=5,
                       help='Number of search results (default: 5)')
    parser.add_argument('--index-path', '-p', type=str, default='faiss_master_index3',
                       help='Path to FAISS index directory')
    parser.add_argument('--country', '-c', type=str, 
                       help='Filter by country')
    parser.add_argument('--department', '-d', type=str, 
                       help='Filter by department')
    parser.add_argument('--admin-master', action='store_true',
                       help='Use admin master index (contains all data)')
    parser.add_argument('--user-role', type=str, choices=['admin', 'manager', 'user'],
                       help='User role for access control')
    
    args = parser.parse_args()
    
    print(f"Python Script - Received args: {vars(args)}", file=sys.stderr)
    
    try:
        if args.command == 'warmup':
            result = warmup_model()
            print(json.dumps(result, indent=2))
            return
        
        # Determine index path based on user role and filters
        index_path = args.index_path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Admin users have special access
        if args.user_role == 'admin':
            if args.admin_master or (not args.country and not args.department):
                # Admin master index contains all data
                admin_index_path = os.path.join(script_dir, "faiss_indices/admin_master/faiss_index")
                if os.path.exists(admin_index_path):
                    index_path = admin_index_path
                    print(f"Using admin master index: {index_path}", file=sys.stderr)
                else:
                    print(f"Warning: Admin master index not found at {admin_index_path}, using default master index", file=sys.stderr)
            elif args.country and args.department:
                # Admin can access specific country/department index
                specific_index_path = os.path.join(script_dir, f"faiss_indices/{args.country.lower()}/{args.department.lower()}/faiss_index")
                if os.path.exists(specific_index_path):
                    index_path = specific_index_path
                    print(f"Admin accessing specific index: {index_path}", file=sys.stderr)
                else:
                    print(f"Warning: Specific index not found at {specific_index_path}, using master index", file=sys.stderr)
        
        # Manager users can only access their specific country/department
        elif args.user_role == 'manager':
            if args.country and args.department:
                specific_index_path = os.path.join(script_dir, f"faiss_indices/{args.country.lower()}/{args.department.lower()}/faiss_index")
                
                print(f"DEBUG: Manager looking for specific index at: {specific_index_path}", file=sys.stderr)
                print(f"DEBUG: Path exists: {os.path.exists(specific_index_path)}", file=sys.stderr)
                
                if os.path.exists(specific_index_path):
                    index_path = specific_index_path
                    print(f"Using manager filtered index: {index_path}", file=sys.stderr)
                else:
                    print(f"Warning: Manager specific index not found at {specific_index_path}, access denied", file=sys.stderr)
                    print(json.dumps({"error": "Access denied: Your specific index is not available"}))
                    return
            else:
                print(f"Warning: Manager must specify both country and department", file=sys.stderr)
                print(json.dumps({"error": "Access denied: Managers must specify country and department"}))
                return
        
        # Legacy handling for when role is not specified (backward compatibility)
        elif args.country and args.department:
            specific_index_path = os.path.join(script_dir, f"faiss_indices/{args.country.lower()}/{args.department.lower()}/faiss_index")
            
            print(f"DEBUG: Looking for specific index at: {specific_index_path}", file=sys.stderr)
            print(f"DEBUG: Path exists: {os.path.exists(specific_index_path)}", file=sys.stderr)
            
            if os.path.exists(specific_index_path):
                index_path = specific_index_path
                print(f"Using filtered index: {index_path}", file=sys.stderr)
            else:
                print(f"Warning: Specific index not found at {specific_index_path}, using master index", file=sys.stderr)
                # Also try to list what's available in the faiss_indices directory
                faiss_indices_dir = os.path.join(script_dir, "faiss_indices")
                if os.path.exists(faiss_indices_dir):
                    print(f"Available in faiss_indices: {os.listdir(faiss_indices_dir)}", file=sys.stderr)
        
        print(f"DEBUG: Final index path being used: {index_path}", file=sys.stderr)
        
        # Initialize extractor (will use cached model if available)
        extractor = OptimizedFAISSExtractor(index_path)
        
        if args.command == 'stats':
            # Fast statistics operation
            stats = extractor.get_index_statistics()
            print(json.dumps(stats, indent=2))
            
        elif args.command == 'list':
            # Full document listing (slower operation)
            documents = extractor.get_all_documents()
            if documents is None or len(documents) == 0:
                print(json.dumps({"error": "No documents found or failed to extract documents"}))
                return
            grouped_data = extractor.group_documents_by_source(documents)
            print(json.dumps(grouped_data, indent=2))
            
        elif args.command == 'search':
            if not args.query:
                print(json.dumps({"error": "Query required for search command"}))
                return
            
            results = extractor.search_documents(args.query, args.limit)
            output = {
                "query": args.query,
                "results": results,
                "result_count": len(results)
            }
            print(json.dumps(output, indent=2))
            
    except Exception as e:
        error_msg = f"Unexpected error in main: {str(e)}"
        print(f"Error: {error_msg}", file=sys.stderr)
        print(json.dumps({"error": error_msg}))

if __name__ == "__main__":
    main()
