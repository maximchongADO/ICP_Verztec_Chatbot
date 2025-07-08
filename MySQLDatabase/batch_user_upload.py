#!/usr/bin/env python3
"""
Batch User Upload Module for Verztec Chatbot System
Handles Excel/CSV file uploads for bulk user creation
"""

import pandas as pd
import mysql.connector
import bcrypt
import re
import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class UserRecord:
    username: str
    email: str
    password: str
    user_type: str

@dataclass
class ProcessingResult:
    total_records: int
    successful_uploads: int
    failed_uploads: int
    errors: List[Dict]
    duplicates_handled: int
    processing_time: float

class BatchUserUploader:
    def __init__(self):
        """Initialize the batch uploader with database configuration"""
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',
            'database': 'verztec_chatbot'
        }
        self.conn = None
        self.cursor = None
        self.supported_formats = ['.xlsx', '.xls', '.csv']
        self.required_columns = ['username', 'email', 'password', 'user_type']
        
    def _connect_db(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            return True
        except mysql.connector.Error as e:
            print(f"Database connection error: {e}")
            return False
    
    def _disconnect_db(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def _validate_file_format(self, file_path: str) -> bool:
        """Validate if file format is supported"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_formats
    
    def _read_excel_file(self, file_path: str) -> pd.DataFrame:
        """Read Excel or CSV file into DataFrame"""
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        else:
            return pd.read_excel(file_path)
    
    def _validate_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Validate DataFrame structure and required columns"""
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("The uploaded file is empty")
            return errors
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            errors.append(f"Found {empty_rows} completely empty rows. Please remove them.")
        
        return errors
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        return True, "Password is valid"
    
    def _validate_user_type(self, user_type: str) -> bool:
        """Validate user type"""
        valid_types = ['user', 'admin']
        return user_type.lower() in valid_types
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _user_exists(self, username: str, email: str) -> Tuple[bool, str]:
        """Check if user already exists"""
        try:
            # Check by username
            self.cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if self.cursor.fetchone():
                return True, f"User with username '{username}' already exists"
            
            # Check by email
            self.cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            if self.cursor.fetchone():
                return True, f"User with email '{email}' already exists"
            
            return False, "User does not exist"
        except mysql.connector.Error as e:
            return True, f"Database error while checking user: {e}"
    
    def _ensure_roles_exist(self):
        """Ensure user and admin roles exist in the database"""
        try:
            # Check if roles table exists and has the required roles
            self.cursor.execute("SHOW TABLES LIKE 'roles'")
            if not self.cursor.fetchone():
                # Create roles table if it doesn't exist
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS roles (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(50) UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
            # Insert default roles if they don't exist
            default_roles = ['user', 'admin']
            for role in default_roles:
                self.cursor.execute("INSERT IGNORE INTO roles (name) VALUES (%s)", (role,))
            
            self.conn.commit()
        except mysql.connector.Error as e:
            print(f"Error ensuring roles exist: {e}")
    
    def _create_user(self, user_record: UserRecord) -> Tuple[bool, str]:
        """Create a new user in the database"""
        try:
            hashed_password = self._hash_password(user_record.password)
            
            # Insert user
            insert_query = """
                INSERT INTO users (username, email, password, role) 
                VALUES (%s, %s, %s, %s)
            """
            values = (
                user_record.username,
                user_record.email,
                hashed_password,
                user_record.user_type
            )
            
            self.cursor.execute(insert_query, values)
            return True, "User created successfully"
            
        except mysql.connector.Error as e:
            return False, f"Database error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def _validate_user_record(self, row, row_index: int) -> Tuple[bool, Optional[UserRecord], List[str]]:
        """Validate a single user record from the DataFrame"""
        errors = []
        
        # Check for missing required fields
        for col in self.required_columns:
            if pd.isna(row[col]) or str(row[col]).strip() == '':
                errors.append(f"Row {row_index + 2}: Missing required field '{col}'")
        
        if errors:
            return False, None, errors
        
        # Create user record
        try:
            user_record = UserRecord(
                username=str(row['username']).strip(),
                email=str(row['email']).strip(),
                password=str(row['password']).strip(),
                user_type=str(row['user_type']).strip()
            )
        except Exception as e:
            errors.append(f"Row {row_index + 2}: Error creating user record: {e}")
            return False, None, errors
        
        # Validate email format
        if not self._validate_email(user_record.email):
            errors.append(f"Row {row_index + 2}: Invalid email format: {user_record.email}")
        
        # Validate password strength
        password_valid, password_msg = self._validate_password(user_record.password)
        if not password_valid:
            errors.append(f"Row {row_index + 2}: {password_msg}")
        
        # Validate user type
        if not self._validate_user_type(user_record.user_type):
            errors.append(f"Row {row_index + 2}: Invalid user type: {user_record.user_type}. Must be 'user' or 'admin'")
        
        # Validate username (no spaces, reasonable length)
        if len(user_record.username) < 3:
            errors.append(f"Row {row_index + 2}: Username must be at least 3 characters long")
        
        if ' ' in user_record.username:
            errors.append(f"Row {row_index + 2}: Username cannot contain spaces")
        
        return len(errors) == 0, user_record, errors
    
    def process_excel_file(self, file_path: str) -> ProcessingResult:
        """Process Excel file and create users in batch"""
        start_time = datetime.now()
        
        # Initialize result
        result = ProcessingResult(
            total_records=0,
            successful_uploads=0,
            failed_uploads=0,
            errors=[],
            duplicates_handled=0,
            processing_time=0.0
        )
        
        try:
            # Validate file format
            if not self._validate_file_format(file_path):
                result.errors.append({
                    'row': 'File',
                    'error': f"Unsupported file format. Supported formats: {self.supported_formats}"
                })
                return result
            
            # Connect to database
            if not self._connect_db():
                result.errors.append({
                    'row': 'Database',
                    'error': 'Failed to connect to database'
                })
                return result
            
            # Ensure roles exist
            self._ensure_roles_exist()
            
            # Read Excel file
            try:
                df = self._read_excel_file(file_path)
            except Exception as e:
                result.errors.append({
                    'row': 'File',
                    'error': f"Failed to read file: {e}"
                })
                return result
            
            # Validate DataFrame structure
            validation_errors = self._validate_dataframe(df)
            if validation_errors:
                for error in validation_errors:
                    result.errors.append({
                        'row': 'Structure',
                        'error': error
                    })
                return result
            
            result.total_records = len(df)
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Validate user record
                    valid, user_record, validation_errors = self._validate_user_record(row, index)
                    
                    if not valid:
                        result.failed_uploads += 1
                        for error in validation_errors:
                            result.errors.append({
                                'row': index + 2,
                                'error': error.split(': ', 1)[1] if ': ' in error else error
                            })
                        continue
                    
                    # Check if user exists
                    exists, exist_msg = self._user_exists(user_record.username, user_record.email)
                    
                    if exists:
                        result.duplicates_handled += 1
                        result.errors.append({
                            'row': index + 2,
                            'error': f"Skipped - {exist_msg}"
                        })
                        result.failed_uploads += 1
                        continue
                    
                    # Create user
                    success, create_msg = self._create_user(user_record)
                    
                    if success:
                        result.successful_uploads += 1
                    else:
                        result.failed_uploads += 1
                        result.errors.append({
                            'row': index + 2,
                            'error': create_msg
                        })
                
                except Exception as e:
                    result.failed_uploads += 1
                    result.errors.append({
                        'row': index + 2,
                        'error': f"Unexpected error: {e}"
                    })
            
            # Commit all changes
            self.conn.commit()
            
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            
            result.errors.append({
                'row': 'System',
                'error': f"Critical error: {e}"
            })
        
        finally:
            self._disconnect_db()
            result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def generate_sample_excel(self, file_path: str = 'sample_users.xlsx'):
        """Generate a sample Excel file with proper format"""
        sample_data = {
            'username': ['john_doe', 'jane_smith', 'admin_user'],
            'email': ['john.doe@company.com', 'jane.smith@company.com', 'admin@company.com'],
            'password': ['password123', 'securepass456', 'adminpass789'],
            'user_type': ['user', 'user', 'admin']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_excel(file_path, index=False)
        return file_path

    def generate_sample_csv(self, file_path: str = 'sample_users.csv'):
        """Generate a sample CSV file with proper format"""
        sample_data = {
            'username': ['john_doe', 'jane_smith', 'admin_user'],
            'email': ['john.doe@company.com', 'jane.smith@company.com', 'admin@company.com'],
            'password': ['password123', 'securepass456', 'adminpass789'],
            'user_type': ['user', 'user', 'admin']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(file_path, index=False)
        return file_path

if __name__ == "__main__":
    # Command line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch User Upload Tool')
    parser.add_argument('action', choices=['upload', 'sample'], help='Action to perform')
    parser.add_argument('--file', '-f', help='Excel file to upload')
    parser.add_argument('--output', '-o', default='sample_users.xlsx', help='Output file for sample')
    
    args = parser.parse_args()
    
    uploader = BatchUserUploader()
    
    if args.action == 'sample':
        if args.output.endswith('.csv'):
            uploader.generate_sample_csv(args.output)
        else:
            uploader.generate_sample_excel(args.output)
        print(f"Sample file generated: {args.output}")
    elif args.action == 'upload':
        if not args.file:
            print("Error: --file is required for upload action")
            sys.exit(1)
        
        result = uploader.process_excel_file(args.file)
        
        # Output JSON for Node.js integration
        result_json = {
            "total_records": result.total_records,
            "successful_uploads": result.successful_uploads,
            "failed_uploads": result.failed_uploads,
            "duplicates_handled": result.duplicates_handled,
            "processing_time": result.processing_time,
            "errors": result.errors
        }
        
        print(json.dumps(result_json))
        
        # Also output human-readable summary to stderr for debugging
        print(f"Processing complete:", file=sys.stderr)
        print(f"Total records: {result.total_records}", file=sys.stderr)
        print(f"Successful uploads: {result.successful_uploads}", file=sys.stderr)
        print(f"Failed uploads: {result.failed_uploads}", file=sys.stderr)
        print(f"Duplicates handled: {result.duplicates_handled}", file=sys.stderr)
        print(f"Processing time: {result.processing_time:.2f} seconds", file=sys.stderr)
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):", file=sys.stderr)
            for error in result.errors:
                print(f"  Row {error['row']}: {error['error']}", file=sys.stderr)
