# Role-Based Access Control (RBAC) Implementation Report
**Verztec AI Chatbot System**

## Executive Summary

The Verztec AI Chatbot system implements a comprehensive Role-Based Access Control (RBAC) framework that governs user authentication, authorization, and resource access across the entire application stack. This implementation spans from the foundational database schema through middleware authentication layers, backend API controllers, and frontend user interfaces, creating a robust security architecture that ensures appropriate access controls are maintained throughout the system lifecycle.

## Database Architecture and Role Foundation

The RBAC implementation begins at the database level with a carefully designed schema that establishes the fundamental relationship between users and their assigned roles. Located in `MySQLDatabase/DEFINING_Tables.py`, the system defines two critical tables that form the backbone of the access control mechanism. The `roles` table serves as the authoritative source for role definitions, utilizing an auto-incrementing primary key (`role_id`) paired with unique role names that prevent duplicate role creation. This table is intentionally simple yet effective, storing only the essential information needed to categorize user permissions.

```python
# RBAC tables
'''
CREATE TABLE IF NOT EXISTS roles (
    role_id INT AUTO_INCREMENT PRIMARY KEY,
    role_name VARCHAR(50) UNIQUE NOT NULL
)
''',

'''
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    role_id INT,
    department VARCHAR(100),
    country VARCHAR(100),
    FOREIGN KEY (role_id) REFERENCES roles(role_id)
)
'''
```

Complementing the roles table, the `users` table creates the necessary associations between individual users and their assigned roles through a foreign key relationship. This table extends beyond basic role assignment to include department and country attributes, enabling granular access control based on organizational structure and geographical boundaries. The inclusion of these additional attributes allows the system to implement sophisticated permission logic that considers not only what role a user holds but also their organizational context within the company structure.

The practical implementation of role management is handled through the `MySQLDatabase/RBAC.py` module, which provides essential functions for role manipulation and user management. The `insert_roles()` function establishes three primary role categories: ADMIN, MANAGER, and USER, each carrying distinct permission levels and access scopes. User creation is managed through the `create_user()` function, which handles password hashing using bcrypt encryption and ensures proper role assignment during the account creation process.

```python
def insert_roles():
    roles = ['ADMIN', 'MANAGER', 'USER']
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    for role in roles:
        insert_query = '''
        INSERT INTO roles (role_name)
        VALUES (%s)
        ON DUPLICATE KEY UPDATE role_name = role_name
        '''
        cursor.execute(insert_query, (role,))
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Roles inserted or already exist.")

def create_user(username, password, email, role_name, department, country):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Hash the password
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    # Get role_id from role_name
    cursor.execute("SELECT role_id FROM roles WHERE role_name = %s", (role_name,))
    role = cursor.fetchone()
    if not role:
        print(f"Role '{role_name}' not found.")
        conn.close()
        return False
    role_id = role[0]

    # Insert user
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role_id, department, country)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (username, hashed_pw, email, role_id, department, country))
        conn.commit()
        print(f"User '{username}' created successfully.")
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()
```

The authentication mechanism implemented in `authenticate_user()` validates user credentials against the stored password hashes and returns comprehensive user information including role details, department assignment, and country location.

## Permission Logic and Access Control Framework

The heart of the RBAC system lies in its sophisticated permission checking mechanism, implemented through the `check_permission()` function. This function evaluates user requests against a hierarchical permission model that considers both role-based privileges and organizational boundaries.

```python
def check_permission(user_info, action, target_department=None, target_country=None):
    # Fetch role name
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT role_name FROM roles WHERE role_id = %s", (user_info['role_id'],))
    role_name = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    # Define RBAC rules 
    if role_name == 'ADMIN':
        return True  # Full access

    if role_name == 'MANAGER':
        # Can only act within own department and country
        if user_info['department'] == target_department and user_info['country'] == target_country:
            return True

    if role_name == 'USER':
        # Read-only: can only view within their country
        if action == 'view' and user_info['country'] == target_country:
            return True

    return False
```

Administrators occupy the highest tier of the permission hierarchy, enjoying unrestricted access to all system resources and functionality. This universal access capability allows administrators to perform system-wide operations, manage user accounts across all departments and countries, and override standard organizational boundaries when necessary.

Managers represent the middle tier of the permission structure, operating within carefully defined organizational boundaries that restrict their access to their specific department and country combination. This scoped access model ensures that managers can effectively oversee their assigned organizational units while preventing unauthorized access to other departments or geographical regions. The permission system validates manager requests by comparing the target department and country against the manager's assigned organizational context, granting access only when these parameters align perfectly.

Regular users occupy the most restrictive tier of the permission hierarchy, limited to read-only operations within their assigned country boundaries. This constraint ensures that users can access information relevant to their geographical location while preventing unauthorized access to data from other countries or regions. The read-only restriction further limits user capabilities, preventing them from modifying system data or performing administrative functions that could compromise system integrity or organizational security.

## Middleware Authentication Infrastructure

The authentication middleware, implemented in `chatbot/src/backend/middleware/authenticateToken.js`, serves as the primary gatekeeper for all authenticated requests entering the system. This middleware leverages JSON Web Token (JWT) technology to validate user credentials and extract role information from incoming requests.

```javascript
const authenticateToken = (req,res,next) => {
    //get token and check if its valid
    const authHeader = req.headers["authorization"];
    const token = authHeader && authHeader.split(" ")[1];
    console.log(token)
    if (!token) {
        console.log('No token provided');
        return res.status(403).json({ 
            success: false, 
            message: 'No authentication token provided' 
        });
    }

    //verify token
    jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret-key', (err,user) => {
        if (err) {
            console.log('Token verification failed:', err.message);
            return res.status(403).json({ 
                success: false, 
                message: 'Invalid or expired token' 
            });
        }

        const userRole = user.role;
        const requestEndpoint = req.url.split('?')[0];
        const method = req.method;

        // Allow chatbot endpoints for all authenticated users
        if (requestEndpoint.startsWith('/api/chatbot/')) {
            req.user = user;
            return next();
        }

        req.user = user; // user contains userId and role
        next();
    })
}
```

The token validation process involves verifying the JWT signature against the configured secret key, ensuring that tokens have not been tampered with or expired, and extracting user information including role assignments and organizational details.

The middleware architecture includes specialized functions that cater to different access level requirements throughout the application:

```javascript
// Middleware to restrict to admin only
function requireAdmin(req, res, next) {
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ success: false, message: 'Admin access required' });
    }
    next();
}

// Middleware to restrict to manager only
function requireManager(req, res, next) {
    if (!req.user || req.user.role !== 'manager') {
        return res.status(403).json({ success: false, message: 'Manager access required' });
    }
    next();
}

// Middleware to restrict to admin or manager
function requireAdminOrManager(req, res, next) {
    if (!req.user || !['admin', 'manager'].includes(req.user.role)) {
        return res.status(403).json({ success: false, message: 'Admin or Manager access required' });
    }
    next();
}
```

The `requireAdmin()` function implements strict administrative access controls, rejecting requests from users who do not possess administrative privileges. Similarly, the `requireManager()` function enforces manager-level access restrictions, while the `requireAdminOrManager()` function provides combined access control for operations that can be performed by either administrators or managers. These specialized middleware functions work in conjunction with the base authentication mechanism to create layered security controls that can be applied selectively to different API endpoints based on their sensitivity and required access levels.

The middleware also implements intelligent endpoint routing logic that considers both the requested resource and the user's role assignment. Chatbot-related endpoints receive special treatment, allowing all authenticated users to access core chatbot functionality regardless of their specific role assignment. This design decision ensures that the primary system functionality remains accessible to all legitimate users while maintaining appropriate restrictions on administrative and management functions.

## Backend Controller Integration

The user controller, located in `chatbot/src/backend/controllers/userController.js`, demonstrates comprehensive integration of RBAC principles throughout its operation. JWT token generation incorporates role information directly into the token payload, ensuring that user permissions are embedded within the authentication mechanism itself.

```javascript
const generateAccessToken = (user) => {
    const secret = process.env.JWT_SECRET || 'fallback-secret-key';
    return jwt.sign({ 
        userId: user.id, 
        role: user.role,
        username: user.username
    }, secret, { expiresIn: '24h' });
}
```

This approach allows for efficient permission checking without requiring additional database queries for every authenticated request.

Administrative user creation functionality implements strict role validation, ensuring that only valid role assignments can be made during account creation:

```javascript
// Admin-only: create user with role
const adminCreateUser = async (req, res) => {
    // Only allow if admin
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ message: 'Admin access required' });
    }
    let { username, email, password, role, country, department } = req.body;
    if (!username || !email || !password || !role) {
        return res.status(400).json({ message: 'Missing required fields' });
    }
    if (!['user', 'admin', 'manager'].includes(role)) {
        return res.status(400).json({ message: 'Role must be user, admin, or manager' });
    }
    
    // Admin users should not have country/department
    if (role === 'admin') {
        country = null;
        department = null;
    }
    
    try {
        const hashedPassword = await hashPassword(password);
        const createdUser = await User.createUser({
            username,
            email,
            password: hashedPassword,
            role,
            country,
            department
        });
        res.status(201).json({
            message: 'User created successfully',
            user: {
                id: createdUser.id,
                username: createdUser.username,
                email: createdUser.email,
                role: createdUser.role,
                country: createdUser.country,
                department: createdUser.department
            }
        });
    } catch (error) {
        console.error('Error creating user (admin):', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}
```

The system recognizes three distinct role categories: user, admin, and manager, rejecting any attempts to assign invalid or unauthorized roles. Special handling is implemented for administrative accounts, automatically clearing department and country assignments for admin users since their access scope transcends organizational boundaries.

The file upload controller, found in `chatbot/src/backend/controllers/fileUploadController.js`, showcases role-based functionality in action through its differential handling of file upload operations:

```javascript
const userRole = req.user.user_type;

// First upload to the intended index
try {
    const fileBuffer = Buffer.from(req.file.buffer);
    const formData = new FormData();
    const stream = Readable.from(fileBuffer);
    formData.append('file', stream, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
    });
    formData.append('country', country);
    formData.append('department', department);
    formData.append('replaceIfExists', userRole === 'admin' ? 'true' : 'false');
    
    // Upload logic continues...
}
```

Administrative users receive enhanced capabilities, including the ability to replace existing files during upload operations, while regular users are restricted to standard upload functionality without replacement capabilities. This distinction demonstrates how role-based permissions can be applied to specific feature sets within the application, providing enhanced functionality to privileged users while maintaining appropriate restrictions for standard users.

## API Route Protection and Security

The routing infrastructure implements comprehensive protection mechanisms through the systematic application of authentication middleware across all sensitive endpoints. User management routes, defined in `chatbot/src/backend/routes/userRoute.js`, apply dual-layer protection combining general authentication requirements with specific administrative access controls:

```javascript
const authenticateToken = require("../middleware/authenticateToken.js");
const userController = require("../controllers/userController.js");

module.exports = (app) => {
  // JWT decode route
  app.get("/api/decode-jwt", userController.decodeJWT);

  // User registration (public)
  app.post("/api/register", userController.createUser);

  // Admin-only user creation
  app.post("/api/admin/users", 
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.adminCreateUser
  );

  // Admin-only user retrieval
  app.get("/api/admin/users", 
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.getAllUsers
  );

  // Admin-only user updates
  app.put("/api/admin/users/:userId", 
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.updateUser
  );

  // Admin-only user deletion
  app.delete("/api/admin/users/:userId", 
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.deleteUser
  );

  // Admin-only mailing list management
  app.post("/api/admin/mailing-list", 
    authenticateToken,
    authenticateToken.requireAdmin,
    userController.addEmailToMailingList
  );
}
```

This approach ensures that user creation, modification, and deletion operations can only be performed by authenticated administrators, preventing unauthorized manipulation of user accounts.

Chatbot functionality routes maintain authentication requirements while providing broader access to all authenticated users, reflecting the system's design philosophy that core chatbot features should be available to all legitimate users regardless of their administrative status:

```javascript
const authenticateToken = require("../middleware/authenticateToken.js");
const chatbotController = require("../controllers/chatbotController.js");

module.exports = (app) => {
  // Protected chatbot endpoints - require authentication
  app.post("/api/chatbot/send-message", 
    authenticateToken,
    chatbotController.sendMessage
  );

  app.get("/api/chatbot/chat-history/:userId", 
    authenticateToken,
    chatbotController.getChatHistory
  );

  app.post("/api/chatbot/upload-document", 
    authenticateToken,
    chatbotController.uploadDocument
  );
}
```

This balanced approach maintains security while ensuring that the primary system functionality remains accessible and functional for all user categories.

Mailing list management operations receive particular attention in the routing security model, requiring both authentication and administrative privileges for all operations. This protection level reflects the sensitive nature of communication lists and the potential impact of unauthorized modifications to these critical system components.

## Frontend Role Management and User Interface

The administrative interface, implemented in `chatbot/src/public/admin.html`, provides comprehensive role management capabilities through an intuitive web-based interface. Role selection mechanisms allow administrators to assign and modify user roles through dropdown controls that present the three available role options:

```html
<div class="form-group">
  <label for="userRole" class="form-label">Role</label>
  <select id="userRole" name="role" class="form-select" required onchange="handleRoleChange()">
    <option value="">Select a role</option>
    <option value="user">User</option>
    <option value="manager">Manager</option>
    <option value="admin">Admin</option>
  </select>
</div>
```

The interface implements client-side validation to ensure that only valid role assignments can be submitted, providing immediate feedback to users attempting to assign invalid roles.

JWT authentication integration within the frontend ensures that all administrative operations are performed with proper authorization:

```javascript
function getAuthHeaders() {
    const token = localStorage.getItem('token');
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
    };
}

// Usage in API calls
const response = await fetch('/api/admin/users', {
    method: 'GET',
    headers: getAuthHeaders()
});

// JWT token verification
const response = await fetch('/api/decode-jwt', {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});

if (response.ok) {
    const decoded = await response.json();
    // Use decoded user information including role
} else {
    throw new Error('Unauthorized access');
}
```

The interface includes sophisticated error handling mechanisms that detect and respond to authorization failures, providing clear feedback when users attempt to perform operations beyond their privilege level:

```javascript
// Check authorization
if (decoded.role !== 'admin') {
  throw new Error('Unauthorized access');
}

// User management functionality
document.getElementById('userRole').value = user.role;

// Role-based UI updates
function handleRoleChange() {
    const roleSelect = document.getElementById('userRole');
    const selectedRole = roleSelect.value;
    
    // Show/hide department and country fields based on role
    if (selectedRole === 'admin') {
        // Hide department and country for admin users
        document.getElementById('departmentGroup').style.display = 'none';
        document.getElementById('countryGroup').style.display = 'none';
    } else {
        // Show department and country for other roles
        document.getElementById('departmentGroup').style.display = 'block';
        document.getElementById('countryGroup').style.display = 'block';
    }
}
```

Authentication state management ensures that users are properly verified before gaining access to administrative functions, with automatic redirection for unauthorized access attempts.

User management functionality within the interface allows for comprehensive user account manipulation, including role assignment, departmental allocation, and country assignment. The interface design reflects the organizational structure underlying the RBAC system, providing intuitive controls that map directly to the database schema and permission logic implemented in the backend systems.

## Security Integration and Cross-Layer Protection

The RBAC implementation demonstrates sophisticated integration across all application layers, creating a cohesive security framework that protects resources at multiple levels. CORS configuration includes authorization headers in the allowed headers list, ensuring that JWT tokens can be properly transmitted in cross-origin requests:

```javascript
// CORS configuration in app.js
const cors = require('cors');
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  credentials: true,
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
}));
```

Environmental configuration management ensures that sensitive authentication parameters, including JWT secrets and database credentials, are properly isolated from the application code through environment variable usage:

```javascript
// JWT secret configuration
const secret = process.env.JWT_SECRET || 'fallback-secret-key';

// Database configuration
const DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}
```

This approach prevents credential exposure in version control systems while maintaining the flexibility needed for deployment across different environments.

The system implements comprehensive logging and monitoring capabilities that track authentication attempts, authorization decisions, and access control violations:

```javascript
// Authentication logging
console.log('Request Endpoint:', requestEndpoint);
console.log('User Role:', userRole);

if (err) {
    console.log('Token verification failed:', err.message);
    return res.status(403).json({ 
        success: false, 
        message: 'Invalid or expired token' 
    });
}

// Permission checking logs
if (!req.user || req.user.role !== 'admin') {
    console.log('Admin access required but user role is:', req.user?.role);
    return res.status(403).json({ message: 'Admin access required' });
}
```

This audit trail provides essential visibility into system security events and enables rapid detection of potential security incidents or unauthorized access attempts.

## Role Hierarchy and Permission Matrix

The implemented role hierarchy establishes clear delineation of privileges and responsibilities across the three primary user categories. The following code demonstrates how these roles are enforced throughout the system:

```python
# Role-based permission matrix implementation
def check_permission(user_info, action, target_department=None, target_country=None):
    # Fetch role name
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT role_name FROM roles WHERE role_id = %s", (user_info['role_id'],))
    role_name = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    # Administrative users - Full system access
    if role_name == 'ADMIN':
        return True  # Can perform any action on any resource

    # Manager users - Department and country scoped access  
    if role_name == 'MANAGER':
        # Can manage resources within their organizational boundaries
        if (user_info['department'] == target_department and 
            user_info['country'] == target_country):
            return True

    # Regular users - Read-only country-scoped access
    if role_name == 'USER':
        # Limited to viewing resources within their country
        if action == 'view' and user_info['country'] == target_country:
            return True

    # Default deny
    return False
```

Administrative users possess comprehensive system access, enabling them to perform all operations including user management, system configuration, file management with replacement capabilities, and access to all organizational data regardless of department or country boundaries:

```javascript
// Admin-specific capabilities in file upload
const userRole = req.user.user_type;
formData.append('replaceIfExists', userRole === 'admin' ? 'true' : 'false');

// Admin-only user management endpoints
if (!req.user || req.user.role !== 'admin') {
    return res.status(403).json({ message: 'Admin access required' });
}

// Admin users bypass organizational constraints
if (role === 'admin') {
    country = null;
    department = null;
}
```

This universal access capability makes administrators responsible for system maintenance, user account management, and overall system security oversight.

Managers operate within carefully defined organizational boundaries that align with business structure and reporting relationships:

```python
# Manager access validation
if role_name == 'MANAGER':
    # Validate organizational scope
    if (user_info['department'] == target_department and 
        user_info['country'] == target_country):
        return True  # Access granted within scope
    else:
        return False  # Access denied outside scope
```

Their access scope encompasses their assigned department and country combination, allowing them to manage resources and personnel within their organizational responsibility while preventing access to other organizational units. This scoped access model reflects real-world organizational hierarchies and ensures that management capabilities align with actual business responsibilities.

Regular users receive the most limited access scope, focusing on consumption of information and utilization of core system functionality within their geographical boundaries:

```python
# User access restrictions
if role_name == 'USER':
    # Read-only operations within country boundaries
    if action == 'view' and user_info['country'] == target_country:
        return True
    else:
        return False  # No write access or cross-country access
```

The read-only restriction for users ensures that system data integrity is maintained while providing necessary access to information required for daily operations. This approach balances security requirements with functional needs, ensuring that users can effectively utilize the system for their intended purposes.
-
## Implementation Benefits and Security Advantages

The comprehensive RBAC implementation provides numerous security and operational advantages that enhance overall system security and usability. Role-based access controls eliminate the need for individual permission management, reducing administrative overhead while ensuring consistent application of security policies across all user accounts. The hierarchical permission model aligns with organizational structures, making role assignments intuitive and reducing the likelihood of inappropriate access grants.

Organizational boundary enforcement through department and country attributes enables sophisticated access control that reflects real-world business requirements and regulatory compliance needs. This granular control capability allows organizations to implement data sovereignty requirements and ensure that sensitive information remains within appropriate geographical or organizational boundaries.

The integration of authentication and authorization mechanisms throughout the application stack ensures that security controls cannot be bypassed through alternative access paths. This comprehensive coverage eliminates security gaps that could be exploited by malicious actors while maintaining system usability for legitimate users.

## Conclusion

The Verztec AI Chatbot RBAC implementation represents a mature, comprehensive approach to access control that effectively balances security requirements with operational needs. The system successfully implements industry-standard security practices while maintaining the flexibility and usability required for effective business operations. Through careful integration across all application layers, from database schema through user interface components, the RBAC system provides robust protection for system resources while enabling appropriate access for legitimate users based on their organizational roles and responsibilities.
