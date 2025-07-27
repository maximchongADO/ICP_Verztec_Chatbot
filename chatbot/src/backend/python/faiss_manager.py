"""
FAISS Index Manager for Country/Department-specific document storage
Handles creation, querying, and management of organized FAISS indices
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

from vector_store import (
    get_faiss_index_path, 
    list_available_indices, 
    get_faiss_db_for_query,
    embedding_model
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FAISSManager:
    """
    Manager class for handling country/department-specific FAISS indices
    """
    
    def __init__(self, embedding_model=embedding_model):
        self.embedding_model = embedding_model
        self.base_dir = Path("faiss_indices")
        self.base_dir.mkdir(exist_ok=True)
    
    def get_index_info(self) -> Dict:
        """Get comprehensive information about all available indices"""
        indices = list_available_indices()
        info = {
            "total_countries": len(indices),
            "total_combinations": sum(len(departments) for departments in indices.values()),
            "indices": indices,
            "structure": {}
        }
        
        for country, departments in indices.items():
            info["structure"][country] = {}
            for department in departments:
                index_path = get_faiss_index_path(country, department)
                try:
                    faiss_db = FAISS.load_local(
                        index_path, 
                        embeddings=self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    doc_count = len(faiss_db.docstore._dict)
                    info["structure"][country][department] = {
                        "path": index_path,
                        "document_count": doc_count,
                        "status": "loaded"
                    }
                except Exception as e:
                    info["structure"][country][department] = {
                        "path": index_path,
                        "document_count": 0,
                        "status": f"error: {str(e)}"
                    }
        
        return info
    
    def search_across_indices(self, query: str, country: Optional[str] = None, 
                            department: Optional[str] = None, k: int = 5) -> List[Dict]:
        """
        Search across FAISS indices with optional country/department filtering
        
        Args:
            query: Search query
            country: Optional country filter
            department: Optional department filter
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        results = []
        
        if country and department:
            # Search specific index
            faiss_db = get_faiss_db_for_query(country, department, self.embedding_model)
            if faiss_db:
                docs = faiss_db.similarity_search(query, k=k)
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "country": country,
                        "department": department
                    })
        elif country:
            # Search all departments in a country
            indices = list_available_indices()
            if country in indices:
                for dept in indices[country]:
                    faiss_db = get_faiss_db_for_query(country, dept, self.embedding_model)
                    if faiss_db:
                        docs = faiss_db.similarity_search(query, k=k//len(indices[country]))
                        for doc in docs:
                            results.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "country": country,
                                "department": dept
                            })
        elif department:
            # Search specific department across all countries
            indices = list_available_indices()
            for country_name, departments in indices.items():
                if department in departments:
                    faiss_db = get_faiss_db_for_query(country_name, department, self.embedding_model)
                    if faiss_db:
                        docs = faiss_db.similarity_search(query, k=k//len([c for c in indices if department in indices[c]]))
                        for doc in docs:
                            results.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "country": country_name,
                                "department": department
                            })
        else:
            # Search all indices
            indices = list_available_indices()
            total_combinations = sum(len(departments) for departments in indices.values())
            per_index_k = max(1, k // total_combinations) if total_combinations > 0 else k
            
            for country_name, departments in indices.items():
                for dept in departments:
                    faiss_db = get_faiss_db_for_query(country_name, dept, self.embedding_model)
                    if faiss_db:
                        docs = faiss_db.similarity_search(query, k=per_index_k)
                        for doc in docs:
                            results.append({
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                                "country": country_name,
                                "department": dept
                            })
        
        # Sort results by relevance (you might want to implement a more sophisticated scoring)
        return results[:k]
    
    def delete_index(self, country: str, department: str) -> bool:
        """
        Delete a specific FAISS index
        
        Args:
            country: Country name
            department: Department name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_path = Path(get_faiss_index_path(country, department))
            if index_path.exists():
                # Remove all FAISS files
                for file_path in index_path.parent.glob("faiss_index.*"):
                    file_path.unlink()
                
                # Remove directory if empty
                if not any(index_path.parent.iterdir()):
                    index_path.parent.rmdir()
                    
                    # Remove country directory if empty
                    country_dir = index_path.parent.parent
                    if not any(country_dir.iterdir()):
                        country_dir.rmdir()
                
                logger.info(f"Deleted FAISS index for {country}/{department}")
                return True
            else:
                logger.warning(f"FAISS index not found for {country}/{department}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting FAISS index for {country}/{department}: {str(e)}")
            return False
    
    def get_index_stats(self, country: str, department: str) -> Optional[Dict]:
        """
        Get statistics for a specific index
        
        Args:
            country: Country name
            department: Department name
            
        Returns:
            Dictionary with index statistics or None if not found
        """
        try:
            faiss_db = get_faiss_db_for_query(country, department, self.embedding_model)
            if not faiss_db:
                return None
            
            docs = faiss_db.docstore._dict
            sources = set()
            total_chunks = len(docs)
            
            for doc in docs.values():
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            return {
                "country": country,
                "department": department,
                "total_documents": len(sources),
                "total_chunks": total_chunks,
                "avg_chunks_per_doc": total_chunks / len(sources) if sources else 0,
                "sources": list(sources)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for {country}/{department}: {str(e)}")
            return None

# Global instance
faiss_manager = FAISSManager()

# Utility functions for use in other modules
def search_documents(query: str, country: Optional[str] = None, 
                    department: Optional[str] = None, k: int = 5) -> List[Dict]:
    """Convenience function for document search"""
    return faiss_manager.search_across_indices(query, country, department, k)

def get_all_index_info() -> Dict:
    """Convenience function to get all index information"""
    return faiss_manager.get_index_info()

def get_stats_for_index(country: str, department: str) -> Optional[Dict]:
    """Convenience function to get index statistics"""
    return faiss_manager.get_index_stats(country, department)
