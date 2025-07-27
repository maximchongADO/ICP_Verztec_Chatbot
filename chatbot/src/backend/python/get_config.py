#!/usr/bin/env python3
"""
Get configuration information for the enhanced upload system
Returns supported countries, departments, and file types
"""

import sys
import json
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from upload_config import get_config_info
except ImportError:
    # Fallback to default configuration
    print(json.dumps({
        "success": True,
        "supported_countries": ["china", "singapore"],
        "supported_departments": ["hr", "it"],
        "supported_file_types": [".pdf", ".doc", ".docx", ".txt", ".pptx"],
        "total_combinations": 4
    }))
    sys.exit(0)

def main():
    try:
        config_info = get_config_info()
        response = {
            "success": True,
            **config_info
        }
        print(json.dumps(response))
        
    except Exception as e:
        # Return default configuration on error
        print(json.dumps({
            "success": True,
            "supported_countries": ["china", "singapore"],
            "supported_departments": ["hr", "it"],
            "supported_file_types": [".pdf", ".doc", ".docx", ".txt", ".pptx"],
            "total_combinations": 4,
            "error": str(e)
        }))

if __name__ == "__main__":
    main()
