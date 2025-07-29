// Check if user is authenticated
const token = localStorage.getItem("token");
if (!token) {
    window.location.href = "/login.html";
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadList = document.getElementById('uploadList');
const uploadStatus = document.getElementById('uploadStatus');
const countrySelect = document.getElementById('countrySelect');
const departmentSelect = document.getElementById('departmentSelect');
const configInfo = document.getElementById('configInfo');
const configInfoText = document.getElementById('configInfoText');
const uploadWarning = document.getElementById('uploadWarning');

// Initialize country/department selection handlers
document.addEventListener('DOMContentLoaded', function() {
    if (countrySelect && departmentSelect) {
        countrySelect.addEventListener('change', updateConfigInfo);
        departmentSelect.addEventListener('change', updateConfigInfo);
        
        // Load available countries and departments from backend
        loadUploadConfiguration();
    }
});

function updateConfigInfo() {
    // Check if this is admin bulk upload mode
    const adminUploadMode = currentUser && currentUser.role === 'admin' ? getAdminUploadMode() : 'specific';
    
    if (adminUploadMode === 'all') {
        // Show bulk upload info
        configInfo.style.display = 'block';
        configInfoText.innerHTML = `Documents will be stored in: <strong>🌍 ALL KNOWLEDGE BASES</strong> (China/HR, China/IT, Singapore/HR, Singapore/IT + Admin Master)`;
        
        if (currentUser) {
            configInfoText.innerHTML += `<br><small style="color: var(--text-secondary);">Access Level: Administrator Access - Bulk Upload Mode</small>`;
        }
        return;
    }
    
    // Regular specific upload mode
    const country = countrySelect.value;
    const department = departmentSelect.value;
    
    if (country && department) {
        configInfo.style.display = 'block';
        
        const countryFlag = country === 'china' ? '🇨🇳' : country === 'singapore' ? '🇸🇬' : '';
        const departmentIcon = department === 'hr' ? '👥' : department === 'it' ? '💻' : '';
        
        configInfoText.innerHTML = `Documents will be stored in: <strong>${countryFlag} ${country.toUpperCase()}/${departmentIcon} ${department.toUpperCase()}</strong> knowledge base`;
        
        // Show user's access level
        if (currentUser) {
            const accessLevel = currentUser.role === 'admin' ? 'Administrator Access' : 
                               currentUser.role === 'manager' ? `${currentUser.country?.toUpperCase()}/${currentUser.department?.toUpperCase()} Manager Access` :
                               'Limited Access';
            configInfoText.innerHTML += `<br><small style="color: var(--text-secondary);">Access Level: ${accessLevel}</small>`;
        }
    } else {
        configInfo.style.display = 'none';
    }
}

async function loadUploadConfiguration() {
    try {
        const response = await fetch('/api/fileUpload/config', {
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const config = await response.json();
            updateSelectOptions(config);
        }
    } catch (error) {
        console.warn('Could not load upload configuration:', error);
        // Continue with default options
    }
}

function updateSelectOptions(config) {
    if (config.supported_countries && countrySelect) {
        // Clear existing options except the first one
        countrySelect.innerHTML = '<option value="">Select Country</option>';
        
        config.supported_countries.forEach(country => {
            const option = document.createElement('option');
            option.value = country;
            const flag = country === 'china' ? '🇨🇳' : country === 'singapore' ? '🇸🇬' : '';
            option.textContent = `${flag} ${country.charAt(0).toUpperCase() + country.slice(1)}`;
            countrySelect.appendChild(option);
        });
    }
    
    if (config.supported_departments && departmentSelect) {
        // Clear existing options except the first one
        departmentSelect.innerHTML = '<option value="">Select Department</option>';
        
        config.supported_departments.forEach(dept => {
            const option = document.createElement('option');
            option.value = dept;
            const icon = dept === 'hr' ? '👥' : dept === 'it' ? '💻' : '';
            const displayName = dept === 'hr' ? 'Human Resources' : dept === 'it' ? 'Information Technology' : dept;
            option.textContent = `${icon} ${displayName}`;
            departmentSelect.appendChild(option);
        });
    }
}

// Global variable to store current user info
let currentUser = null;

// Helper functions for user authentication (similar to analytics dashboard)
function getToken() { 
    return localStorage.getItem("token"); 
}

function getCurrentUser() { 
    return window.currentUser || null; 
}

// Fetch current user info
async function fetchCurrentUser() {
    const token = getToken();
    console.log('Fetching current user with token:', token ? 'Present' : 'Missing');
    
    if (!token) return null;
    try {
        console.log('Making request to /api/users/me');
        const res = await fetch('/api/users/me', {
            headers: { Authorization: `Bearer ${token}` }
        });
        
        console.log('Response status:', res.status);
        console.log('Response ok:', res.ok);
        
        if (res.ok) {
            const user = await res.json();
            console.log('User data received:', user);
            window.currentUser = user;
            return user;
        } else {
            console.log('Response not ok:', await res.text());
        }
    } catch (error) {
        console.error('Error fetching current user:', error);
    }
    window.currentUser = null;
    return null;
}

// Initialize upload interface based on user role
async function initializeUploadInterface() {
    console.log('Initializing upload interface...');
    const user = await fetchCurrentUser();
    console.log('Fetched user:', user);
    
    if (!user) {
        console.log('No user found, showing access denied');
        showAccessDenied("Unable to verify user authentication");
        return;
    }
    
    console.log('User role:', user.role);
    console.log('User country:', user.country);
    console.log('User department:', user.department);
    
    // Check if user has access to file uploads (only managers and admins)
    if (user.role !== 'admin' && user.role !== 'manager') {
        console.log('User role not admin or manager, denying access');
        showAccessDenied("Access Denied: File upload is restricted to managers and administrators only");
        return;
    }
    
    console.log('User has valid role, setting up interface');
    currentUser = user;
    
    // Filter dropdown options based on user's role and permissions
    filterDropdownOptions(user);
    
    // Show appropriate interface sections
    const uploadInterface = document.getElementById('uploadInterface');
    const accessDenied = document.getElementById('accessDenied');
    
    console.log('uploadInterface element found:', !!uploadInterface);
    console.log('accessDenied element found:', !!accessDenied);
    
    if (uploadInterface) {
        uploadInterface.style.display = 'block';
        uploadInterface.style.visibility = 'visible';
        uploadInterface.style.opacity = '1';
        console.log('Set uploadInterface display to block');
        console.log('uploadInterface computed style:', window.getComputedStyle(uploadInterface).display);
        console.log('uploadInterface offsetHeight:', uploadInterface.offsetHeight);
    }
    if (accessDenied) {
        accessDenied.style.display = 'none';
        console.log('Set accessDenied display to none');
        console.log('accessDenied computed style:', window.getComputedStyle(accessDenied).display);
    }
    
    // Display user access information
    displayUserAccessInfo(user);
    
    // Initialize knowledge base view with role-based filtering
    initializeKnowledgeBaseView(user);
    
    // Final check - log what should be visible
    setTimeout(() => {
        console.log('=== FINAL STATUS CHECK ===');
        console.log('uploadInterface display:', window.getComputedStyle(document.getElementById('uploadInterface')).display);
        console.log('accessDenied display:', window.getComputedStyle(document.getElementById('accessDenied')).display);
        console.log('uploadInterface visible:', document.getElementById('uploadInterface').offsetHeight > 0);
        console.log('accessDenied visible:', document.getElementById('accessDenied').offsetHeight > 0);
    }, 100);
}

// Filter dropdown options based on user role and permissions
function filterDropdownOptions(user) {
    console.log('filterDropdownOptions called for user:', user);
    const countrySelect = document.getElementById('countrySelect');
    const departmentSelect = document.getElementById('departmentSelect');
    const adminUploadOptions = document.getElementById('adminUploadOptions');
    
    console.log('countrySelect found:', !!countrySelect);
    console.log('departmentSelect found:', !!departmentSelect);
    console.log('adminUploadOptions found:', !!adminUploadOptions);
    
    if (!countrySelect || !departmentSelect) {
        console.error('Dropdown elements not found!');
        return;
    }
    
    // Clear existing options
    countrySelect.innerHTML = '<option value="">Select Country</option>';
    departmentSelect.innerHTML = '<option value="">Select Department</option>';
    
    if (user.role === 'admin') {
        console.log('Setting up admin options');
        // Admin can access all countries and departments
        addCountryOption('china', '🇨🇳 China');
        addCountryOption('singapore', '🇸🇬 Singapore');
        addDepartmentOption('hr', '👥 Human Resources');
        addDepartmentOption('it', '💻 Information Technology');
        
        // Show admin upload options
        if (adminUploadOptions) {
            adminUploadOptions.style.display = 'block';
            console.log('Showing admin upload options');
        }
    } else if (user.role === 'manager' && user.country && user.department) {
        console.log('Setting up manager options for:', user.country, user.department);
        // Manager can only access their specific country and department
        const countryFlag = user.country === 'china' ? '🇨🇳' : user.country === 'singapore' ? '🇸🇬' : '';
        const departmentIcon = user.department === 'hr' ? '👥' : user.department === 'it' ? '💻' : '';
        const departmentName = user.department === 'hr' ? 'Human Resources' : user.department === 'it' ? 'Information Technology' : user.department;
        
        addCountryOption(user.country, `${countryFlag} ${user.country.charAt(0).toUpperCase() + user.country.slice(1)}`);
        addDepartmentOption(user.department, `${departmentIcon} ${departmentName}`);
        
        // Auto-select user's country and department
        countrySelect.value = user.country;
        departmentSelect.value = user.department;
        
        // Hide admin upload options for managers
        if (adminUploadOptions) {
            adminUploadOptions.style.display = 'none';
        }
        
        // Update config info immediately
        updateConfigInfo();
    } else {
        showAccessDenied("User profile incomplete - missing country or department information");
        return;
    }
}

// Helper functions to add options to dropdowns
function addCountryOption(value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    document.getElementById('countrySelect').appendChild(option);
}

function addDepartmentOption(value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    document.getElementById('departmentSelect').appendChild(option);
}

// Show access denied message
function showAccessDenied(message) {
    console.log('showAccessDenied called with message:', message);
    const accessDeniedDiv = document.getElementById('accessDenied');
    console.log('accessDeniedDiv found:', !!accessDeniedDiv);
    if (accessDeniedDiv) {
        const messageElement = accessDeniedDiv.querySelector('p');
        console.log('messageElement found:', !!messageElement);
        if (messageElement) {
            messageElement.textContent = message;
        }
        accessDeniedDiv.style.display = 'block';
    }
    const uploadInterface = document.getElementById('uploadInterface');
    console.log('uploadInterface found:', !!uploadInterface);
    if (uploadInterface) {
        uploadInterface.style.display = 'none';
    }
}

// Display user access information
function displayUserAccessInfo(user) {
    const userAccessDisplay = document.getElementById('userAccessDisplay');
    if (userAccessDisplay) {
        const accessLevel = user.role === 'admin' ? 'Administrator Access' : 
                           user.role === 'manager' ? `${user.country?.toUpperCase()}/${user.department?.toUpperCase()} Manager Access` :
                           'Limited Access';
        
        const countryFlag = user.country === 'china' ? '🇨🇳' : user.country === 'singapore' ? '🇸🇬' : '';
        const departmentIcon = user.department === 'hr' ? '👥' : user.department === 'it' ? '💻' : '';
        
        userAccessDisplay.innerHTML = `
            <div class="config-row">
                <div class="config-group">
                    <div class="config-label">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
                            <circle cx="12" cy="7" r="4"/>
                        </svg>
                        USER
                    </div>
                    <div class="access-value">${user.username}</div>
                </div>
                <div class="config-group">
                    <div class="config-label">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M16 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/>
                            <circle cx="8.5" cy="7" r="4"/>
                            <path d="M20 8v6M23 11h-6"/>
                        </svg>
                        ROLE
                    </div>
                    <div class="access-value">${user.role}</div>
                </div>
            </div>
            <div class="config-row">
                ${user.country ? `<div class="config-group">
                    <div class="config-label">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                            <circle cx="12" cy="10" r="3"/>
                        </svg>
                        COUNTRY
                    </div>
                    <div class="access-value">${countryFlag} ${user.country}</div>
                </div>` : ''}
                ${user.department ? `<div class="config-group">
                    <div class="config-label">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/>
                            <circle cx="9" cy="7" r="4"/>
                            <path d="M23 21v-2a4 4 0 00-3-3.87"/>
                            <path d="M16 3.13a4 4 0 010 7.75"/>
                        </svg>
                        DEPARTMENT
                    </div>
                    <div class="access-value">${departmentIcon} ${user.department}</div>
                </div>` : ''}
            </div>
            <div class="config-row">
                <div class="config-group" style="grid-column: 1 / -1;">
                    <div class="config-label">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                        </svg>
                        ACCESS LEVEL
                    </div>
                    <div class="access-value">${accessLevel}</div>
                </div>
            </div>
        `;
        
        const userAccessInfo = document.getElementById('userAccessInfo');
        if (userAccessInfo) {
            userAccessInfo.style.display = 'block';
        }
    }
}

// Initialize knowledge base view dropdowns with role-based filtering
function initializeKnowledgeBaseView(user) {
    console.log('Initializing knowledge base view for user:', user);
    
    const viewCountrySelect = document.getElementById('viewCountrySelect');
    const viewDepartmentSelect = document.getElementById('viewDepartmentSelect');
    
    if (!viewCountrySelect || !viewDepartmentSelect) {
        console.log('Knowledge base view dropdowns not found');
        return;
    }
    
    // Clear existing options
    viewCountrySelect.innerHTML = '<option value="">All Countries</option>';
    viewDepartmentSelect.innerHTML = '<option value="">All Departments</option>';
    
    if (user.role === 'admin') {
        console.log('Setting up admin knowledge base view options');
        // Admin can view all countries and departments
        addViewCountryOption('china', '🇨🇳 China');
        addViewCountryOption('singapore', '🇸🇬 Singapore');
        addViewDepartmentOption('hr', '👥 Human Resources');
        addViewDepartmentOption('it', '💻 Information Technology');
    } else if (user.role === 'manager' && user.country && user.department) {
        console.log('Setting up manager knowledge base view options for:', user.country, user.department);
        // Manager can only view their specific country and department
        const countryFlag = user.country === 'china' ? '🇨🇳' : user.country === 'singapore' ? '🇸🇬' : '';
        const departmentIcon = user.department === 'hr' ? '👥' : user.department === 'it' ? '💻' : '';
        const departmentName = user.department === 'hr' ? 'Human Resources' : user.department === 'it' ? 'Information Technology' : user.department;
        
        addViewCountryOption(user.country, `${countryFlag} ${user.country.charAt(0).toUpperCase() + user.country.slice(1)}`);
        addViewDepartmentOption(user.department, `${departmentIcon} ${departmentName}`);
        
        // Auto-select user's country and department for knowledge base view
        viewCountrySelect.value = user.country;
        viewDepartmentSelect.value = user.department;
    }
}

// Helper functions to add options to knowledge base view dropdowns
function addViewCountryOption(value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    document.getElementById('viewCountrySelect').appendChild(option);
}

function addViewDepartmentOption(value, text) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = text;
    document.getElementById('viewDepartmentSelect').appendChild(option);
}

// Update knowledge base view based on selected filters
function updateKnowledgeBaseView() {
    const viewCountry = document.getElementById('viewCountrySelect').value;
    const viewDepartment = document.getElementById('viewDepartmentSelect').value;
    
    console.log('Knowledge base view filter changed:', viewCountry, viewDepartment);
    
    // Validate user access to selected combination
    if (!validateKnowledgeBaseAccess(viewCountry, viewDepartment)) {
        return;
    }
    
    // Clear previous results when filter changes
    const statsDiv = document.getElementById('faissStats');
    const filesListDiv = document.getElementById('faissFilesList');
    if (statsDiv) statsDiv.style.display = 'none';
    if (filesListDiv) filesListDiv.style.display = 'none';
}

// Validate user access to knowledge base viewing for selected country/department
function validateKnowledgeBaseAccess(country, department) {
    if (!currentUser) {
        console.log('No current user for knowledge base access validation');
        return false;
    }
    
    // Admin can view everything
    if (currentUser.role === 'admin') {
        return true;
    }
    
    // Managers can only view their specific country/department or "all" if it matches their assignment
    if (currentUser.role === 'manager') {
        // If no specific filter selected, allow (they'll see their assigned data)
        if (!country && !department) {
            return true;
        }
        
        // If specific filters selected, must match their assignment
        if (country && country !== currentUser.country) {
            console.log('Manager cannot view knowledge base for different country');
            return false;
        }
        
        if (department && department !== currentUser.department) {
            console.log('Manager cannot view knowledge base for different department');
            return false;
        }
        
        return true;
    }
    
    // Regular users don't have knowledge base view access
    console.log('User role does not have knowledge base view access');
    return false;
}

// Validate user access to selected country/department combination
function validateUserAccess() {
    const country = document.getElementById('countrySelect').value;
    const department = document.getElementById('departmentSelect').value;
    
    if (!currentUser) {
        showUploadWarning("User authentication required");
        return false;
    }
    
    // Admin can access everything
    if (currentUser.role === 'admin') {
        hideUploadWarning();
        return true;
    }
    
    // Managers can only access their specific country/department
    if (currentUser.role === 'manager') {
        if (country && department) {
            if (country !== currentUser.country || department !== currentUser.department) {
                showUploadWarning(`Access Denied: You can only upload to ${currentUser.country?.toUpperCase()}/${currentUser.department?.toUpperCase()}`);
                return false;
            }
        }
        hideUploadWarning();
        return true;
    }
    
    // Regular users don't have file upload access
    showUploadWarning("Access Denied: File upload is restricted to managers and administrators only");
    return false;
}

// Show upload warning
function showUploadWarning(message) {
    const uploadWarning = document.getElementById('uploadWarning');
    if (uploadWarning) {
        uploadWarning.innerHTML = `
            <svg fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
            </svg>
            ${message}
        `;
        uploadWarning.style.display = 'flex';
    }
}

// Hide upload warning
function hideUploadWarning() {
    const uploadWarning = document.getElementById('uploadWarning');
    if (uploadWarning) {
        uploadWarning.style.display = 'none';
    }
}

// Show FAISS loading error
function showFAISSError(message) {
    const knowledgeBaseSection = document.getElementById('knowledgeBaseSection');
    if (knowledgeBaseSection) {
        knowledgeBaseSection.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                ${message}
            </div>
        `;
    }
}

// Admin upload mode handling
function updateAdminUploadMode() {
    const uploadMode = document.querySelector('input[name="uploadMode"]:checked').value;
    const adminModeWarning = document.getElementById('adminModeWarning');
    const countrySelect = document.getElementById('countrySelect');
    const departmentSelect = document.getElementById('departmentSelect');
    
    console.log('Admin upload mode changed to:', uploadMode);
    
    if (uploadMode === 'all') {
        // Show warning for upload to all
        if (adminModeWarning) {
            adminModeWarning.style.display = 'block';
        }
        
        // Disable country/department selection for "upload to all" mode
        if (countrySelect) {
            countrySelect.disabled = true;
            countrySelect.value = '';
        }
        if (departmentSelect) {
            departmentSelect.disabled = true;
            departmentSelect.value = '';
        }
        
        // Update config info
        updateConfigInfo();
    } else {
        // Hide warning for specific upload
        if (adminModeWarning) {
            adminModeWarning.style.display = 'none';
        }
        
        // Enable country/department selection
        if (countrySelect) {
            countrySelect.disabled = false;
        }
        if (departmentSelect) {
            departmentSelect.disabled = false;
        }
        
        // Update config info for specific mode
        updateConfigInfo();
    }
}

// Get current admin upload mode
function getAdminUploadMode() {
    const uploadModeElement = document.querySelector('input[name="uploadMode"]:checked');
    return uploadModeElement ? uploadModeElement.value : 'specific';
}

// Initialize upload interface when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize upload interface with user-based access control
    initializeUploadInterface();
    
    const countrySelect = document.getElementById('countrySelect');
    const departmentSelect = document.getElementById('departmentSelect');
    
    if (countrySelect && departmentSelect) {
        countrySelect.addEventListener('change', function() {
            validateUserAccess();
            updateConfigInfo();
        });
        departmentSelect.addEventListener('change', function() {
            validateUserAccess();
            updateConfigInfo();
        });
    }
});

// Handle drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    if (dropZone) {
        dropZone.addEventListener(eventName, preventDefaults, false);
    }
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    if (dropZone) {
        dropZone.addEventListener(eventName, highlight, false);
    }
});

['dragleave', 'drop'].forEach(eventName => {
    if (dropZone) {
        dropZone.addEventListener(eventName, unhighlight, false);
    }
});

function highlight() {
    if (dropZone) {
        dropZone.classList.add('drag-over');
    }
}

function unhighlight() {
    if (dropZone) {
        dropZone.classList.remove('drag-over');
    }
}

// Handle dropped files
if (dropZone) {
    dropZone.addEventListener('drop', handleDrop, false);
}
if (fileInput) {
    fileInput.addEventListener('change', handleFileSelect, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    // Check if this is admin bulk upload mode
    const adminUploadMode = currentUser && currentUser.role === 'admin' ? getAdminUploadMode() : 'specific';
    
    // For admin bulk uploads, skip country/department validation
    if (adminUploadMode === 'all') {
        if (uploadWarning) {
            uploadWarning.style.display = 'none';
        }
        
        // For bulk uploads, pass dummy values that will be handled by the backend
        [...files].forEach(file => uploadFile(file, 'admin', 'master'));
        return;
    }
    
    // For specific uploads, check if country and department are selected
    const country = countrySelect ? countrySelect.value : '';
    const department = departmentSelect ? departmentSelect.value : '';
    
    if (!country || !department) {
        if (uploadWarning) {
            uploadWarning.style.display = 'flex';
        }
        showStatus('error', 'Please select both country and department before uploading files');
        return;
    }
    
    if (uploadWarning) {
        uploadWarning.style.display = 'none';
    }
    
    [...files].forEach(file => uploadFile(file, country, department));
}

async function uploadFile(file, country, department) {
    // Check user authentication
    if (!currentUser) {
        showStatus('error', 'User authentication required');
        return;
    }
    
    // Check admin upload mode for admins
    const adminUploadMode = currentUser.role === 'admin' ? getAdminUploadMode() : 'specific';
    
    // For admin bulk uploads, skip individual country/department validation
    if (currentUser.role === 'admin' && adminUploadMode === 'all') {
        console.log('Admin bulk upload mode detected');
        // Proceed with bulk upload - no specific country/department restrictions
    } else {
        // Check if user has access to this country/department combination
        if (currentUser.role !== 'admin') {
            if (currentUser.role !== 'manager') {
                showStatus('error', 'Access Denied: File upload is restricted to managers and administrators only');
                return;
            }
            if (country !== currentUser.country || department !== currentUser.department) {
                showStatus('error', `Access Denied: You can only upload to ${currentUser.country?.toUpperCase()}/${currentUser.department?.toUpperCase()}`);
                return;
            }
        }
    }
    
    // Validate file before upload
    if (!validateFile(file)) {
        return;
    }

    const fileItem = createFileItem(file, country, department);
    if (uploadList) {
        uploadList.appendChild(fileItem);
    }

    const formData = new FormData();
    formData.append('file', file);
    
    // For admin bulk uploads, include upload mode
    if (currentUser.role === 'admin' && adminUploadMode === 'all') {
        formData.append('uploadMode', 'all');
        // These are not used in bulk mode but required by validation
        formData.append('country', 'admin');
        formData.append('department', 'master');
    } else {
        formData.append('uploadMode', 'specific');
        formData.append('country', country);
        formData.append('department', department);
    }

    try {
        const response = await fetch('/api/fileUpload/upload', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                // Don't set Content-Type, let browser set it with boundary for FormData
            },
            credentials: 'include',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server response:', errorText);
            throw new Error(errorText || `Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        
        // Handle different response types
        if (result.uploadResults) {
            // Admin bulk upload response
            updateFileStatus(fileItem, 'success', `Bulk upload completed: ${result.successfulUploads}/${result.totalUploads} successful`);
            
            // Show detailed results
            let detailMessage = `Admin Bulk Upload Results:\n`;
            detailMessage += `✅ Successful: ${result.successfulUploads}\n`;
            detailMessage += `❌ Failed: ${result.failedUploads}\n`;
            detailMessage += `📊 Total: ${result.totalUploads}\n\n`;
            
            if (result.uploadResults.length > 0) {
                detailMessage += `Upload Details:\n`;
                result.uploadResults.forEach(upload => {
                    const status = upload.success ? '✅' : '❌';
                    const location = upload.country === 'admin' ? 'ADMIN MASTER' : `${upload.country.toUpperCase()}/${upload.department.toUpperCase()}`;
                    detailMessage += `${status} ${location}: ${upload.message}\n`;
                });
            }
            
            if (result.errors.length > 0) {
                detailMessage += `\nErrors:\n`;
                result.errors.forEach(error => {
                    detailMessage += `❌ ${error}\n`;
                });
            }
            
            showStatus(result.success ? 'success' : 'warning', result.message, detailMessage);
        } else {
            // Regular single upload response
            updateFileStatus(fileItem, 'success', `File processed successfully for ${country.toUpperCase()}/${department.toUpperCase()}`);
            showStatus('success', result.message || `File uploaded successfully to ${country.toUpperCase()}/${department.toUpperCase()} knowledge base`);
        }

    } catch (error) {
        console.error('Upload error:', error);
        updateFileStatus(fileItem, 'error', error.message);
        showStatus('error', `Failed to upload file: ${error.message}`);
    }
}

function validateFile(file) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    ];
    
    if (file.size > maxSize) {
        showStatus('error', 'File too large. Maximum size is 10MB.');
        return false;
    }
    
    if (!allowedTypes.includes(file.type)) {
        showStatus('error', 'Unsupported file type. Please upload PDF, DOC, DOCX, TXT, or PPTX files.');
        return false;
    }
    
    return true;
}

function createFileItem(file, country, department) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    
    const countryFlag = country === 'china' ? '🇨🇳' : country === 'singapore' ? '🇸🇬' : '';
    const departmentIcon = department === 'hr' ? '👥' : department === 'it' ? '💻' : '';
    
    fileItem.innerHTML = `
        <div class="file-info">
            <div class="file-name">${file.name}</div>
            <div class="file-destination">${countryFlag} ${country.toUpperCase()} / ${departmentIcon} ${department.toUpperCase()}</div>
            <div class="file-status status-pending">Uploading...</div>
        </div>
    `;
    return fileItem;
}

function updateFileStatus(fileItem, status, message) {
    const statusDiv = fileItem.querySelector('.file-status');
    if (statusDiv) {
        statusDiv.className = `file-status status-${status}`;
        statusDiv.textContent = message;
    }
}

function showStatus(type, message, detailMessage = null) {
    if (uploadStatus) {
        uploadStatus.className = `upload-status ${type}`;
        uploadStatus.textContent = message;
        uploadStatus.style.display = 'block';
        
        // If detail message provided, show in console and potentially in alert for admin bulk uploads
        if (detailMessage) {
            console.log('Upload Details:', detailMessage);
            
            // For admin bulk uploads, show detailed results in an alert
            if (detailMessage.includes('Admin Bulk Upload Results')) {
                // Create a more user-friendly display
                const formattedDetails = detailMessage.replace(/\n/g, '\n');
                alert(formattedDetails);
            }
        }
        
        setTimeout(() => {
            uploadStatus.style.display = 'none';
        }, 5000);
    }
}

// Navigation functions
function logout() {
    localStorage.removeItem("token");
    localStorage.removeItem("userId");
    window.location.href = "/login.html";
}

function returnToChat() {
    window.location.href = "/chatbot.html";
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// FAISS Knowledge Base Functions
async function loadFAISSData() {
    // Check user access first
    if (!currentUser) {
        console.error('No user authenticated for knowledge base access');
        return;
    }
    
    // Validate user has access to knowledge base viewing
    if (currentUser.role !== 'admin' && currentUser.role !== 'manager') {
        console.error('User does not have access to view knowledge base');
        showFAISSError('Access Denied: Knowledge base viewing is restricted to managers and administrators only');
        return;
    }
    
    // Get selected filters
    const viewCountry = document.getElementById('viewCountrySelect')?.value || '';
    const viewDepartment = document.getElementById('viewDepartmentSelect')?.value || '';
    
    // Validate access to selected filters
    if (!validateKnowledgeBaseAccess(viewCountry, viewDepartment)) {
        showFAISSError('Access Denied: You can only view knowledge base for your assigned country/department');
        return;
    }
    
    const loadingSpinner = document.getElementById('faissLoadingSpinner');
    const statsDiv = document.getElementById('faissStats');
    const filesListDiv = document.getElementById('faissFilesList');
    const errorDiv = document.getElementById('faissError');
    
    // Show loading state with better messaging
    loadingSpinner.style.display = 'block';
    statsDiv.style.display = 'none';
    filesListDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    // Update loading message
    const loadingMessage = document.querySelector('#faissLoadingSpinner p');
    if (loadingMessage) {
        loadingMessage.textContent = 'Loading knowledge base... This may take up to 2 minutes for the first load.';
    }
    
    try {
        // First, try to warm up the model (this helps with subsequent calls)
        try {
            if (loadingMessage) {
                loadingMessage.textContent = 'Warming up AI model... (1/2)';
            }
            
            const warmupResponse = await fetch('/api/faiss/warmup', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (warmupResponse.ok) {
                console.log('Model warmed up successfully');
            }
        } catch (warmupError) {
            console.warn('Warmup failed, continuing with regular load:', warmupError);
        }
        
        // Now load the actual data
        if (loadingMessage) {
            loadingMessage.textContent = 'Loading knowledge base data... (2/2)';
        }
        
        // Prepare request body with filters and user role information
        const requestBody = { 
            command: 'list',
            userRole: currentUser?.role,
            filters: {
                country: viewCountry,
                department: viewDepartment
            }
        };
        
        // For admin users, check if they want to view the master index
        if (currentUser?.role === 'admin') {
            // If no specific filters are selected, use admin master index
            if (!viewCountry && !viewDepartment) {
                requestBody.adminMaster = true;
                console.log('Admin user loading master index (all data)');
            } else {
                console.log('Admin user loading specific filters:', requestBody.filters);
            }
        }
        
        console.log('Loading FAISS data with request:', requestBody);
        
        const response = await fetch('/api/faiss/extract', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display statistics
        displayFAISSStats(data.summary);
        
        // Display files
        displayFAISSFiles(data.files);
        
        loadingSpinner.style.display = 'none';
        statsDiv.style.display = 'block';
        filesListDiv.style.display = 'block';
        
    } catch (error) {
        console.error('Error loading FAISS data:', error);
        loadingSpinner.style.display = 'none';
        errorDiv.style.display = 'block';
        
        let errorMessage = error.message;
        let suggestion = '';
        
        if (error.message.includes('timed out')) {
            suggestion = `
                <p><strong>This usually happens on the first load.</strong></p>
                <p>The AI model needs to be downloaded and loaded, which can take 1-2 minutes.</p>
                <p>Please try again - subsequent loads will be much faster.</p>
            `;
        }
        
        errorDiv.innerHTML = `
            <h3>⚠️ Failed to Load Knowledge Base</h3>
            <p><strong>Error:</strong> ${errorMessage}</p>
            ${suggestion}
            <button class="btn btn-primary" onclick="loadFAISSData()" style="margin-top: 1rem;">
                🔄 Retry
            </button>
        `;
    }
}

function displayFAISSStats(summary) {
    const statsDiv = document.getElementById('faissStats');
    
    statsDiv.innerHTML = `
        <div class="stat-card">
            <span class="stat-number">${summary.total_files || 0}</span>
            <span class="stat-label">Total Files</span>
        </div>
        <div class="stat-card">
            <span class="stat-number">${summary.total_chunks || 0}</span>
            <span class="stat-label">Text Chunks</span>
        </div>
        <div class="stat-card">
            <span class="stat-number">${formatFileSize(summary.total_content_length || 0)}</span>
            <span class="stat-label">Total Content</span>
        </div>
        <div class="stat-card">
            <span class="stat-number">${summary.extraction_time ? new Date(summary.extraction_time).toLocaleDateString() : 'Unknown'}</span>
            <span class="stat-label">Last Updated</span>
        </div>
    `;
}

function displayFAISSFiles(files) {
    const filesListDiv = document.getElementById('faissFilesList');
    
    if (!files || files.length === 0) {
        filesListDiv.innerHTML = `
            <div class="error-message">
                <h3>📭 No Files Found</h3>
                <p>The knowledge base appears to be empty. Upload some documents to get started.</p>
            </div>
        `;
        return;
    }
    
    const filesHTML = files.map(file => `
        <div class="faiss-file-card">
            <div class="file-card-header">
                <div class="file-card-info">
                    <h3>📄 ${escapeHtml(file.filename)}</h3>
                    <div class="file-card-meta">
                        <div class="meta-item">
                            <span class="file-type-badge">${file.file_type}</span>
                        </div>
                        ${file.created_at ? `
                        <div class="meta-item">
                            📅 ${new Date(file.created_at).toLocaleDateString()}
                        </div>
                        ` : ''}
                    </div>
                </div>
            </div>
            
            <div class="chunk-info">
                <div class="chunk-stat">
                    <span class="chunk-stat-number">${file.chunk_count}</span>
                    <span class="chunk-stat-label">Chunks</span>
                </div>
                <div class="chunk-stat">
                    <span class="chunk-stat-number">${formatFileSize(file.total_content_length)}</span>
                    <span class="chunk-stat-label">Total Size</span>
                </div>
                <div class="chunk-stat">
                    <span class="chunk-stat-number">${formatFileSize(file.avg_chunk_size)}</span>
                    <span class="chunk-stat-label">Avg Chunk</span>
                </div>
            </div>
            
            <div class="file-actions-faiss">
                <button class="btn btn-secondary btn-small" onclick="viewFileChunks('${escapeHtml(file.filename)}')">
                    👁️ View Chunks
                </button>
                <button class="btn btn-secondary btn-small" onclick="searchInFile('${escapeHtml(file.filename)}')">
                    🔍 Search Content
                </button>
                <button class="btn btn-danger btn-small" onclick="deleteFile('${escapeHtml(file.filename)}')">
                    🗑️ Delete File
                </button>
            </div>
        </div>
    `).join('');
    
    filesListDiv.innerHTML = filesHTML;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function viewFileChunks(filename) {
    try {
        const response = await fetch('/api/faiss/extract', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ command: 'list' })
        });
        
        const data = await response.json();
        const file = data.files?.find(f => f.filename === filename);
        
        if (!file) {
            alert('File not found in knowledge base');
            return;
        }
        
        // Create modal to show chunks
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>📄 Chunks in ${escapeHtml(filename)}</h3>
                    <button class="modal-close" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="modal-body">
                    ${file.chunks.map((chunk, index) => `
                        <div class="chunk-preview">
                            <div class="chunk-header">
                                <strong>Chunk ${index + 1}</strong>
                                <span class="chunk-size">${formatFileSize(chunk.content_length)}</span>
                            </div>
                            <div class="chunk-content">${escapeHtml(chunk.content_preview)}</div>
                            ${chunk.content_length > 200 ? '<div class="chunk-note">Content truncated...</div>' : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('Error viewing chunks:', error);
        alert('Failed to load file chunks: ' + error.message);
    }
}

async function searchInFile(filename) {
    const query = prompt(`Search in "${filename}":`);
    if (!query) return;
    
    try {
        const response = await fetch('/api/faiss/extract', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                command: 'search', 
                query: query,
                limit: 10
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Filter results for this file
        const fileResults = data.results?.filter(result => 
            result.source.includes(filename) || result.metadata?.source?.includes(filename)
        ) || [];
        
        // Show search results
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>🔍 Search Results in ${escapeHtml(filename)}</h3>
                    <button class="modal-close" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="modal-body">
                    <p><strong>Query:</strong> "${escapeHtml(query)}"</p>
                    <p><strong>Found:</strong> ${fileResults.length} results</p>
                    ${fileResults.length > 0 ? fileResults.map((result, index) => `
                        <div class="search-result">
                            <div class="result-header">
                                <strong>Result ${index + 1}</strong>
                                <span class="similarity-score">Similarity: ${(result.similarity_score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="result-content">${escapeHtml(result.content_preview)}</div>
                        </div>
                    `).join('') : '<p>No results found in this file.</p>'}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
    } catch (error) {
        console.error('Error searching file:', error);
        alert('Search failed: ' + error.message);
    }
}

// Delete file function
async function deleteFile(filename) {
    // Show confirmation dialog
    const confirmed = await showDeleteConfirmation(filename);
    if (!confirmed) {
        return;
    }
    
    // Find and disable the delete button for this file
    const deleteButtons = document.querySelectorAll('button[onclick*="deleteFile"]');
    let targetButton = null;
    deleteButtons.forEach(btn => {
        if (btn.onclick.toString().includes(filename)) {
            targetButton = btn;
            btn.disabled = true;
            btn.innerHTML = '⏳ Deleting...';
            btn.className = 'btn btn-secondary btn-small';
        }
    });
    
    // Create a more detailed loading indicator
    const loadingModal = createLoadingModal('Deleting file...', 
        `Removing "${filename}" from the knowledge base. This may take a few moments.`);
    document.body.appendChild(loadingModal);
    
    try {
        showStatus('info', `🗑️ Deleting ${filename}...`);
        
        const response = await fetch(`/api/faiss/file/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            credentials: 'include'
        });
        
        // Remove loading modal
        loadingModal.remove();
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to delete file');
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Show detailed success message
            const chunksText = result.deletedChunks === 1 ? 'chunk' : 'chunks';
            
            showStatus('success', `✅ File "${filename}" deleted successfully! (${result.deletedChunks} ${chunksText} removed)`);
            
            // Show detailed success modal
            showSuccessModal('File Deleted Successfully', [
                `File "${filename}" has been completely removed from the knowledge base.`,
                `• ${result.deletedChunks} text ${chunksText} deleted`,
                `• Knowledge base automatically updated`,
                `• Changes are effective immediately`
            ]);
            
            // Refresh the FAISS data display
            await loadFAISSData();
        } else {
            throw new Error(result.error || 'Delete operation failed');
        }
        
    } catch (error) {
        // Remove loading modal if still present
        if (document.body.contains(loadingModal)) {
            loadingModal.remove();
        }
        
        // Restore the delete button
        if (targetButton) {
            targetButton.disabled = false;
            targetButton.innerHTML = '🗑️ Delete File';
            targetButton.className = 'btn btn-danger btn-small';
        }
        
        console.error('Error deleting file:', error);
        
        const errorMessage = error.message || 'Unknown error occurred';
        showStatus('error', `❌ Failed to delete ${filename}: ${errorMessage}`);
        
        // Show error modal with retry option
        showErrorModal('Deletion Failed', 
            `Failed to delete "${filename}": ${errorMessage}`,
            () => deleteFile(filename) // Retry function
        );
    }
}

// Show delete confirmation dialog
function showDeleteConfirmation(filename) {
    return new Promise((resolve) => {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>⚠️ Confirm Deletion</h3>
                    <button class="modal-close" onclick="this.parentElement.parentElement.parentElement.remove(); resolve(false);">×</button>
                </div>
                <div class="modal-body">
                    <p><strong>Are you sure you want to delete "${escapeHtml(filename)}"?</strong></p>
                    <p>This will permanently remove:</p>
                    <ul style="margin-left: 1rem; color: var(--text-secondary);">
                        <li>All text chunks from this file in the knowledge base</li>
                        <li>Any search results related to this document</li>
                        <li>The file's contribution to chatbot responses</li>
                    </ul>
                    <p style="color: var(--error); font-weight: 500;"><em>⚠️ This action cannot be undone.</em></p>
                    
                    <div class="modal-actions">
                        <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove(); resolve(false);">
                            Cancel
                        </button>
                        <button class="btn btn-danger" onclick="this.closest('.modal-overlay').remove(); resolve(true);">
                            🗑️ Delete File
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Add resolve function to buttons
        const cancelBtn = modal.querySelector('.btn-secondary');
        const deleteBtn = modal.querySelector('.btn-danger');
        const closeBtn = modal.querySelector('.modal-close');
        
        cancelBtn.onclick = () => {
            modal.remove();
            resolve(false);
        };
        
        deleteBtn.onclick = () => {
            modal.remove();
            resolve(true);
        };
        
        closeBtn.onclick = () => {
            modal.remove();
            resolve(false);
        };
        
        document.body.appendChild(modal);
    });
}

// Create loading modal for delete operations
function createLoadingModal(title, message) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>⏳ ${escapeHtml(title)}</h3>
            </div>
            <div class="modal-body">
                <div class="loading-spinner"></div>
                <p>${escapeHtml(message)}</p>
                <p><em>Please wait...</em></p>
            </div>
        </div>
    `;
    return modal;
}

// Show success modal with detailed information
function showSuccessModal(title, messages) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    
    const messageList = messages.map(msg => `<p>${escapeHtml(msg)}</p>`).join('');
    
    modal.innerHTML = `
        <div class="modal-content success-modal">
            <div class="modal-header">
                <h3>✅ ${escapeHtml(title)}</h3>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove();">×</button>
            </div>
            <div class="modal-body">
                ${messageList}
                <div class="modal-actions">
                    <button class="btn btn-primary" onclick="this.closest('.modal-overlay').remove();">
                        OK
                    </button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Auto-remove after 8 seconds
    setTimeout(() => {
        if (document.body.contains(modal)) {
            modal.remove();
        }
    }, 8000);
}

// Show error modal with retry option
function showErrorModal(title, message, retryFunction) {
    const modal = document.createElement('div');
    modal.className = 'modal-overlay';
    modal.innerHTML = `
        <div class="modal-content error-modal">
            <div class="modal-header">
                <h3>❌ ${escapeHtml(title)}</h3>
                <button class="modal-close" onclick="this.closest('.modal-overlay').remove();">×</button>
            </div>
            <div class="modal-body">
                <p>${escapeHtml(message)}</p>
                <div class="modal-actions">
                    <button class="btn btn-secondary" onclick="this.closest('.modal-overlay').remove();">
                        Cancel
                    </button>
                    <button class="btn btn-primary retry-btn">
                        🔄 Retry
                    </button>
                </div>
            </div>
        </div>
    `;
    
    const retryBtn = modal.querySelector('.retry-btn');
    retryBtn.onclick = () => {
        modal.remove();
        if (retryFunction) {
            retryFunction();
        }
    };
    
    document.body.appendChild(modal);
}
