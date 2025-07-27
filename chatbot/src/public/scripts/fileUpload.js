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
    const country = countrySelect.value;
    const department = departmentSelect.value;
    
    if (country && department) {
        configInfo.style.display = 'block';
        uploadWarning.style.display = 'none';
        
        const countryFlag = country === 'china' ? '🇨🇳' : country === 'singapore' ? '🇸🇬' : '';
        const departmentIcon = department === 'hr' ? '👥' : department === 'it' ? '💻' : '';
        
        configInfoText.innerHTML = `Documents will be stored in: <strong>${countryFlag} ${country.toUpperCase()}/${departmentIcon} ${department.toUpperCase()}</strong> knowledge base`;
    } else {
        configInfo.style.display = 'none';
        uploadWarning.style.display = 'none';
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
    // Check if country and department are selected
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
    formData.append('country', country);
    formData.append('department', department);

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
        updateFileStatus(fileItem, 'success', `File processed successfully for ${country.toUpperCase()}/${department.toUpperCase()}`);
        showStatus('success', result.message || `File uploaded successfully to ${country.toUpperCase()}/${department.toUpperCase()} knowledge base`);

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

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// FAISS Knowledge Base Functions
async function loadFAISSData() {
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
        
        const response = await fetch('/api/faiss/extract', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ command: 'list' })
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
