// Check if user is authenticated
const token = localStorage.getItem("token");
if (!token) {
    window.location.href = "/login.html";
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadList = document.getElementById('uploadList');
const uploadStatus = document.getElementById('uploadStatus');

// Check admin access and disable UI for non-admins
(function() {
    const token = localStorage.getItem("token");
    if (!token) return;
    fetch('/api/users/me', {
        headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => res.ok ? res.json() : null)
    .then(user => {
        if (user && user.role !== 'admin') {
            showAdminPopup();
            // Disable upload UI
            if (document.getElementById('dropZone')) {
                document.getElementById('dropZone').style.pointerEvents = 'none';
                document.getElementById('dropZone').style.opacity = '0.6';
            }
            if (document.getElementById('uploadList')) {
                document.getElementById('uploadList').style.opacity = '0.6';
            }
        }
    })
    .catch(() => {});
})();

function showAdminPopup() {
    let popup = document.createElement('div');
    popup.textContent = "Admin access required to upload files.";
    popup.style.position = "fixed";
    popup.style.top = "30px";
    popup.style.left = "50%";
    popup.style.transform = "translateX(-50%)";
    popup.style.background = "#222";
    popup.style.color = "#FFD700";
    popup.style.padding = "18px 36px";
    popup.style.borderRadius = "18px";
    popup.style.fontSize = "1.1rem";
    popup.style.boxShadow = "0 4px 24px rgba(255,215,0,0.18)";
    popup.style.zIndex = "9999";
    popup.style.opacity = "0.97";
    document.body.appendChild(popup);
    setTimeout(() => {
        popup.remove();
    }, 3500);
}

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
    [...files].forEach(uploadFile);
}

async function uploadFile(file) {
    // Validate file before upload
    if (!validateFile(file)) {
        return;
    }

    const fileItem = createFileItem(file);
    if (uploadList) {
        uploadList.appendChild(fileItem);
    }

    const formData = new FormData();
    formData.append('file', file);

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
        updateFileStatus(fileItem, 'success', 'File processed successfully');
        showStatus('success', result.message || 'File uploaded successfully');

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

function createFileItem(file) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.innerHTML = `
        <div class="file-info">
            <div class="file-name">${file.name}</div>
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

function showStatus(type, message) {
    if (uploadStatus) {
        uploadStatus.className = `upload-status ${type}`;
        uploadStatus.textContent = message;
        uploadStatus.style.display = 'block';
        
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
