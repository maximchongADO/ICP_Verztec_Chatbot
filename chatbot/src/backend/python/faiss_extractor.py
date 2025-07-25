"""
FAISS Vector Store Data Extractor
This script extracts file information from the FAISS master index
to display documents in the knowledge base systematically.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    print(json.dumps({"error": f"Required packages not installed: {e}"}))
    sys.exit(1)

class FAISSExtractor:
    def __init__(self, index_path: str = "faiss_master_index"):
        """
        Initialize the FAISS extractor
        
        Args:
            index_path: Path to the FAISS index directory
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(self.script_dir, index_path)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={'normalize_embeddings': True},
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = None
        self.load_index()
    
    def load_index(self):
        """Load the existing FAISS index"""
        try:
            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"FAISS index loaded successfully from {self.index_path}", file=sys.stderr)
            else:
                print(f"FAISS index not found at {self.index_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}", file=sys.stderr)
            self.vectorstore = None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Extract all documents from the FAISS index"""
        if not self.vectorstore:
            return []
        
        try:
            # Access the docstore to get all documents
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
            return []

    def group_documents_by_source(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group documents by their source file"""
        files_dict = {}
        
        for doc in documents:
            # Get source from metadata
            source = doc["metadata"].get("source", "unknown")
            filename = os.path.basename(source) if source != "unknown" else f"doc_{doc['id']}"
            
            # Initialize file entry if not exists
            if filename not in files_dict:
                files_dict[filename] = {
                    "filename": filename,
                    "source_path": source,
                    "chunks": [],
                    "total_content_length": 0,
                    "created_at": None,
                    "file_type": self._get_file_type(filename)
                }
            
            # Add chunk to file
            files_dict[filename]["chunks"].append({
                "id": doc["id"],
                "content": doc["content"],
                "content_length": doc["content_length"],
                "content_preview": doc["content_preview"],
                "metadata": doc["metadata"]
            })
            
            files_dict[filename]["total_content_length"] += doc["content_length"]
            
            # Try to get creation date from metadata
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
            
            # Sort chunks by length (largest first)
            file_data["chunks"].sort(key=lambda x: x["content_length"], reverse=True)
            files_list.append(file_data)
        
        # Sort files by chunk count (most chunks first)
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

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get general statistics about the FAISS index"""
        if not self.vectorstore:
            return {"status": "no_index", "error": "FAISS index not loaded"}
        
        try:
            total_vectors = self.vectorstore.index.ntotal
            dimension = self.vectorstore.index.d
            
            # Get documents for detailed stats
            documents = self.get_all_documents()
            grouped_data = self.group_documents_by_source(documents)
            
            return {
                "status": "loaded",
                "index_info": {
                    "total_vectors": total_vectors,
                    "vector_dimension": dimension,
                    "index_path": self.index_path
                },
                "content_stats": grouped_data["summary"]
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        if not self.vectorstore:
            return []
        
        try:
            # Perform similarity search
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

    def delete_file_chunks(self, filename: str) -> Dict[str, Any]:
        """Delete all chunks belonging to a specific file from the FAISS index"""
        if not self.vectorstore:
            return {"success": False, "error": "FAISS index not loaded"}
        
        try:
            # Get all documents to find chunks belonging to the file
            documents = self.get_all_documents()
            chunks_to_delete = []
            
            for doc in documents:
                source = doc["metadata"].get("source", "unknown")
                doc_filename = os.path.basename(source) if source != "unknown" else f"doc_{doc['id']}"
                
                if doc_filename == filename:
                    chunks_to_delete.append(doc["id"])
            
            if not chunks_to_delete:
                return {"success": False, "error": f"No chunks found for file: {filename}"}
            
            # Delete chunks from the vectorstore
            deleted_count = 0
            for chunk_id in chunks_to_delete:
                try:
                    # Remove from docstore
                    if hasattr(self.vectorstore, 'docstore') and hasattr(self.vectorstore.docstore, '_dict'):
                        if chunk_id in self.vectorstore.docstore._dict:
                            del self.vectorstore.docstore._dict[chunk_id]
                            deleted_count += 1
                except Exception as e:
                    print(f"Error deleting chunk {chunk_id}: {e}", file=sys.stderr)
            
            # Save the updated index
            if deleted_count > 0:
                self.vectorstore.save_local(self.index_path)
                print(f"Deleted {deleted_count} chunks for file {filename}", file=sys.stderr)
                
                return {
                    "success": True,
                    "deleted_chunks": deleted_count,
                    "filename": filename,
                    "message": f"Successfully deleted {deleted_count} chunks from {filename}"
                }
            else:
                return {"success": False, "error": "No chunks were deleted"}
            
        except Exception as e:
            print(f"Error deleting file chunks: {e}", file=sys.stderr)
            return {"success": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Extract data from FAISS vector store')
    parser.add_argument('command', choices=['list', 'stats', 'search', 'delete'], 
                       help='Command to execute')
    parser.add_argument('--query', '-q', type=str, 
                       help='Search query (for search command)')
    parser.add_argument('--filename', '-f', type=str,
                       help='Filename to delete (for delete command)')
    parser.add_argument('--limit', '-l', type=int, default=5,
                       help='Number of search results (default: 5)')
    parser.add_argument('--index-path', '-p', type=str, default='faiss_master_index',
                       help='Path to FAISS index directory')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = FAISSExtractor(args.index_path)
    
    try:
        if args.command == 'list':
            # Get all documents grouped by source file
            documents = extractor.get_all_documents()
            grouped_data = extractor.group_documents_by_source(documents)
            print(json.dumps(grouped_data, indent=2))
            
        elif args.command == 'stats':
            # Get index statistics
            stats = extractor.get_index_statistics()
            print(json.dumps(stats, indent=2))
            
        elif args.command == 'search':
            if not args.query:
                print(json.dumps({"error": "Query required for search command"}))
                return
            
            # Search documents
            results = extractor.search_documents(args.query, args.limit)
            output = {
                "query": args.query,
                "results": results,
                "result_count": len(results)
            }
            print(json.dumps(output, indent=2))
            
        elif args.command == 'delete':
            if not args.filename:
                print(json.dumps({"error": "Filename required for delete command"}))
                return
            
            # Delete file chunks
            result = extractor.delete_file_chunks(args.filename)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
