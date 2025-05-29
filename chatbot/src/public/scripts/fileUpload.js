// Check if user is authenticated
const token = localStorage.getItem("token");
if (!token) {
    window.location.href = "/login.html";
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadList = document.getElementById('uploadList');
const uploadStatus = document.getElementById('uploadStatus');

// Handle drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    dropZone.classList.add('drag-over');
}

function unhighlight() {
    dropZone.classList.remove('drag-over');
}

// Handle dropped files
dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', handleFileSelect, false);

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
    const fileItem = createFileItem(file);
    uploadList.appendChild(fileItem);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/fileUpload/upload', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                // Remove Content-Type, let browser set it with boundary for FormData
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
    statusDiv.className = `file-status status-${status}`;
    statusDiv.textContent = message;
}

function showStatus(type, message) {
    uploadStatus.className = `upload-status ${type}`;
    uploadStatus.textContent = message;
    uploadStatus.style.display = 'block';
    
    setTimeout(() => {
        uploadStatus.style.display = 'none';
    }, 5000);
}
