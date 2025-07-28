"""
Configuration file for the enhanced FAISS upload system
Manages supported countries, departments, and system settings
"""

from typing import Dict, List
from pathlib import Path

# Supported Countries
SUPPORTED_COUNTRIES = [
    'china', 
    'singapore'
]

# Supported Departments  
SUPPORTED_DEPARTMENTS = [
    'hr',      # Human Resources
    'it'       # Information Technology
]

# Future expansion - additional departments that could be added
FUTURE_DEPARTMENTS = [
    'finance',
    'marketing', 
    'sales',
    'operations',
    'legal',
    'admin'
]

# Future expansion - additional countries that could be added
FUTURE_COUNTRIES = [
    'malaysia',
    'thailand', 
    'vietnam',
    'indonesia',
    'philippines'
]

# Directory Configuration
FAISS_BASE_DIR = "faiss_indices"
TEMP_UPLOAD_DIR = "temp_uploads"
CLEANED_DIR = "data/cleaned"
IMAGES_DIR = "data/images"
VERZTEC_COLLECTION_DIR = "data/verztec_logo"

# File Type Configuration
SUPPORTED_FILE_TYPES = {
    '.pdf': 'application/pdf',
    '.doc': 'application/msword', 
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.txt': 'text/plain'
}

# Search Configuration
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 20

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_MODEL_KWARGS = {'normalize_embeddings': True}

# Text Splitting Configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
FALLBACK_CHUNK_SIZE = 250
FALLBACK_CHUNK_OVERLAP = 50

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

class ConfigManager:
    """
    Manager class for handling configuration settings
    """
    
    @staticmethod
    def get_supported_countries() -> List[str]:
        """Get list of supported countries"""
        return SUPPORTED_COUNTRIES.copy()
    
    @staticmethod
    def get_supported_departments() -> List[str]:
        """Get list of supported departments"""
        return SUPPORTED_DEPARTMENTS.copy()
    
    @staticmethod
    def is_valid_country(country: str) -> bool:
        """Check if country is supported"""
        return country.lower() in SUPPORTED_COUNTRIES
    
    @staticmethod
    def is_valid_department(department: str) -> bool:
        """Check if department is supported"""
        return department.lower() in SUPPORTED_DEPARTMENTS
    
    @staticmethod
    def get_all_combinations() -> List[tuple]:
        """Get all valid country/department combinations"""
        combinations = []
        for country in SUPPORTED_COUNTRIES:
            for department in SUPPORTED_DEPARTMENTS:
                combinations.append((country, department))
        return combinations
    
    @staticmethod
    def add_country(country: str) -> bool:
        """
        Add a new country to supported list
        Note: This modifies the runtime configuration only
        """
        country = country.lower().strip()
        if country and country not in SUPPORTED_COUNTRIES:
            SUPPORTED_COUNTRIES.append(country)
            return True
        return False
    
    @staticmethod
    def add_department(department: str) -> bool:
        """
        Add a new department to supported list
        Note: This modifies the runtime configuration only
        """
        department = department.lower().strip()
        if department and department not in SUPPORTED_DEPARTMENTS:
            SUPPORTED_DEPARTMENTS.append(department)
            return True
        return False
    
    @staticmethod
    def get_faiss_base_path() -> Path:
        """Get the base path for FAISS indices"""
        return Path(FAISS_BASE_DIR)
    
    @staticmethod
    def get_config_summary() -> Dict:
        """Get a summary of current configuration"""
        return {
            "supported_countries": SUPPORTED_COUNTRIES,
            "supported_departments": SUPPORTED_DEPARTMENTS,
            "total_combinations": len(SUPPORTED_COUNTRIES) * len(SUPPORTED_DEPARTMENTS),
            "supported_file_types": list(SUPPORTED_FILE_TYPES.keys()),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "directories": {
                "faiss_indices": FAISS_BASE_DIR,
                "temp_uploads": TEMP_UPLOAD_DIR,
                "cleaned_files": CLEANED_DIR,
                "images": IMAGES_DIR,
                "verztec_collection": VERZTEC_COLLECTION_DIR
            },
            "search_settings": {
                "default_results": DEFAULT_SEARCH_RESULTS,
                "max_results": MAX_SEARCH_RESULTS
            },
            "text_splitting": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP
            }
        }

# Create global config manager instance
config = ConfigManager()

# Utility functions for easy access
def get_countries() -> List[str]:
    """Quick access to supported countries"""
    return config.get_supported_countries()

def get_departments() -> List[str]:
    """Quick access to supported departments"""
    return config.get_supported_departments()

def validate_country_department(country: str, department: str) -> tuple:
    """
    Validate and normalize country/department pair
    
    Returns:
        tuple: (normalized_country, normalized_department)
        
    Raises:
        ValueError: If country or department is not supported
    """
    country = country.lower().strip()
    department = department.lower().strip()
    
    # Special case for admin master index
    if country == 'admin' and department == 'master':
        return country, department
    
    if not config.is_valid_country(country):
        raise ValueError(f"Unsupported country '{country}'. Supported: {', '.join(SUPPORTED_COUNTRIES)} or 'admin'")
    
    if not config.is_valid_department(department):
        raise ValueError(f"Unsupported department '{department}'. Supported: {', '.join(SUPPORTED_DEPARTMENTS)} or 'master'")
    
    return country, department

def get_config_info() -> Dict:
    """Get configuration information"""
    return config.get_config_summary()

if __name__ == "__main__":
    print("FAISS Upload System Configuration")
    print("=" * 50)
    
    config_info = get_config_info()
    
    print(f"Supported Countries: {', '.join(config_info['supported_countries'])}")
    print(f"Supported Departments: {', '.join(config_info['supported_departments'])}")
    print(f"Total Combinations: {config_info['total_combinations']}")
    print(f"File Types: {', '.join(config_info['supported_file_types'])}")
    print(f"Embedding Model: {config_info['embedding_model']}")
    
    print("\nDirectory Structure:")
    for name, path in config_info['directories'].items():
        print(f"  {name}: {path}")
    
    print("\nAll Valid Combinations:")
    combinations = config.get_all_combinations()
    for i, (country, dept) in enumerate(combinations, 1):
        print(f"  {i}. {country.upper()}/{dept.upper()}")
    
    print(f"\nFuture Expansion Options:")
    print(f"  Countries: {', '.join(FUTURE_COUNTRIES)}")
    print(f"  Departments: {', '.join(FUTURE_DEPARTMENTS)}")
