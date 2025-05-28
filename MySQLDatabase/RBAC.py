import mysql.connector
import bcrypt
DB_CONFIG = {
    'host': 'localhost',
    'user': 'chatbot_user',
    'password': 'strong_password',
    'database': 'chatbot_db',
    'raise_on_warnings': True
}
#Only when new roles 
def insert_roles():
    roles = ['ADMIN', 'MANAGER', 'USER']
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    for role in roles:
        # INSERT IGNORE equivalent in MySQL Connector is to catch duplicates or use ON DUPLICATE KEY UPDATE
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

def authenticate_user(username, password):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("SELECT password_hash, role_id, department, country FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    if not result:
        print("User not found.")
        conn.close()
        return None

    stored_hash, role_id, department, country = result
    conn.close()

    if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
        return {'username': username, 'role_id': role_id, 'department': department, 'country': country}
    else:
        print("Incorrect password.")
        return None

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

# #pitest scenario
# if __name__ == "__main__":
#     #Only need to do once unless there is a new role
#     insert_roles()
#     #Create a new user
#     create_user('alice', 'securepassword', 'alice@example.com', 'MANAGER', 'Sales', 'Singapore')
#     users_to_create = [
#         # Admin user
#         ('admin_user', 'AdminPass123', 'admin@example.com', 'ADMIN', 'Management', 'Singapore'),

#         # Managers
#         ('manager_sales', 'ManagerPass123', 'salesmgr@example.com', 'MANAGER', 'Sales', 'Singapore'),
#         ('manager_hr', 'ManagerPass123', 'hrmgr@example.com', 'MANAGER', 'HR', 'Singapore'),

#         # Users from different departments
#         ('user_sales1', 'UserPass123', 'salesuser1@example.com', 'USER', 'Sales', 'Singapore'),
#         ('user_hr1', 'UserPass123', 'hruser1@example.com', 'USER', 'HR', 'Singapore'),
#         ('user_it1', 'UserPass123', 'ituser1@example.com', 'USER', 'IT', 'Singapore'),
#     ]

#     for user in users_to_create:
#         create_user(*user)
#     #Authenticate a user
#     user = authenticate_user('alice', 'securepassword')
#     if user:
#         print(f"User authenticated: {user}")

#         # Check permission for some action
#         can_upload = check_permission(user, 'upload', target_department='Sales', target_country='Singapore')
#         print("Permission to upload:", can_upload)