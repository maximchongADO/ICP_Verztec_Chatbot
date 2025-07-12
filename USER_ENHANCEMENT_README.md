# User Management Enhancement: Country and Department Fields

## Overview
This update adds `country` and `department` fields to the user management system, both in the backend database and frontend interface.

## Changes Made

### 1. Database Schema Updates

#### Modified Files:
- `src/backend/database/seedScript.js`

#### Changes:
- Added `country VARCHAR(100)` column to Users table
- Added `department VARCHAR(100)` column to Users table
- Both fields are optional (nullable)

#### New Database Schema:
```sql
CREATE TABLE IF NOT EXISTS Users (
  id INT AUTO_INCREMENT,
  username VARCHAR(40) NOT NULL,
  email VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(100) NOT NULL,
  role ENUM('user', 'admin','manager') NOT NULL,
  country VARCHAR(100),
  department VARCHAR(100),
  PRIMARY KEY (id)
)
```

### 2. Backend Model Updates

#### Modified Files:
- `src/backend/models/user.js`

#### Changes:
- Updated User constructor to include `country` and `department` fields
- Modified `toUserObj` method to handle new fields
- Updated all database queries to include new fields:
  - `getAllUsers()` - now returns country and department
  - `getUserById()` - now returns country and department
  - `getUserByIdFull()` - now returns country and department
  - `createUser()` - now accepts and stores country and department
  - `updateUser()` - now allows updating country and department

### 3. Backend Controller Updates

#### Modified Files:
- `src/backend/controllers/userController.js`

#### Changes:
- Updated `adminCreateUser` function to accept country and department from request body
- Updated `adminUpdateUser` function to handle country and department updates
- Modified response objects to include new fields

### 4. Frontend Interface Updates

#### Modified Files:
- `src/public/admin.html`

#### Changes:
- **User Form Modal:**
  - Added country dropdown field with options: China, Singapore
  - Added department dropdown field with options: IT, HR
  - Both fields are optional
  
- **User Table Display:**
  - Added "Country" column to users table
  - Added "Department" column to users table
  - Shows "-" for empty values
  
- **JavaScript Functions:**
  - Updated form submission to include new fields
  - Updated edit user modal to populate country and department fields
  - Enhanced search functionality to include country and department in search
  
- **Batch Upload Documentation:**
  - Updated documentation to mention country and department columns
  - Country options: China, Singapore (optional)
  - Department options: IT, HR (optional)
  - Updated sample file generation to include new fields

### 5. Database Migration

#### New File:
- `src/backend/database/migration_add_country_department.js`

#### Purpose:
- Provides a safe migration script for existing databases
- Checks if columns already exist before adding them
- Can be run on existing installations to add the new fields

## Usage Instructions

### For New Installations:
1. Run the seed script as usual - the new fields are included in the schema

### For Existing Installations:
1. Run the migration script:
   ```bash
   node src/backend/database/migration_add_country_department.js
   ```
2. Restart the application

### Using the New Fields:

#### In Admin Panel:
1. When creating a new user, you can now optionally specify:
   - Country (dropdown: China or Singapore)
   - Department (dropdown: IT or HR)
2. When editing existing users, you can add or modify these fields
3. The user table now displays country and department columns
4. Search functionality includes these fields

#### API Endpoints:
The existing API endpoints now support the new fields:

**Create User (POST /api/users):**
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "password123",
  "role": "user",
  "country": "Singapore",
  "department": "IT"
}
```

**Update User (PATCH /api/users/:id):**
```json
{
  "country": "China",
  "department": "HR"
}
```

**Response Format:**
All user objects now include the new fields:
```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "role": "user",
  "country": "Singapore",
  "department": "IT"
}
```

#### Batch Upload:
The batch upload functionality now supports country and department fields:

**Required Columns:**
- username
- email  
- password
- user_type (user, admin, manager)

**Optional Columns:**
- country (China, Singapore)
- department (IT, HR)

**Sample CSV format:**
```csv
username,email,password,user_type,country,department
john_doe,john@company.com,password123,user,Singapore,IT
jane_smith,jane@company.com,pass456,user,China,HR
admin_user,admin@company.com,admin789,admin,Singapore,IT
```

## Notes
- Both country and department fields are optional
- Existing users will have `null` values for these fields until updated
- The fields accept any text up to 100 characters
- Search functionality includes both fields for better user discovery
- All existing functionality remains unchanged

## Testing
After implementing these changes:
1. Test user creation with and without country/department
2. Test user editing to modify these fields
3. Test search functionality with the new fields
4. Verify batch upload documentation is accurate
5. Test the migration script on a backup database first
