// VIEW - UI Management
class ChatView {
    constructor() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Event handlers (to be connected by controller)
        this.onSendMessage = null;
        this.onInputChange = null;
        
        this.setupEventListeners();
    }

    /**
     * Set up DOM event listeners
     */
    setupEventListeners() {
        // Auto-resize textarea
        this.messageInput.addEventListener('input', (e) => {
            this.handleInputResize();
            if (this.onInputChange) {
                this.onInputChange(e.target.value);
            }
        });

        // Handle Enter key
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.triggerSendMessage();
            }
        });

        // Send button click
        this.sendButton.addEventListener('click', () => {
            this.triggerSendMessage();
        });

        // Handle paste events for better UX
        this.messageInput.addEventListener('paste', () => {
            setTimeout(() => this.handleInputResize(), 0);
        });
    }

    /**
     * Handle textarea auto-resize
     */
    handleInputResize() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    /**
     * Trigger send message event
     */
    triggerSendMessage() {
        if (this.onSendMessage && this.getInputValue().trim()) {
            this.onSendMessage();
        }
    }

    /**
     * Display a message in the chat interface
     * @param {Object} message - Message object to display
     */
    displayMessage(message) {
        // Remove welcome message if it exists
        this.removeWelcomeMessage();

        const messageElement = this.createMessageElement(message);
        this.messagesContainer.appendChild(messageElement);
        this.scrollToBottom();
        this.animateMessage(messageElement);
    }

    /**
     * Create DOM element for a message
     * @param {Object} message - Message object
     * @returns {HTMLElement} Message DOM element
     */
    createMessageElement(message) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.sender}`;
        messageElement.setAttribute('data-message-id', message.id);
        
        const avatarContent = message.sender === 'user' 
            ? 'You' 
            : '<i class="fas fa-robot"></i>';
            
        messageElement.innerHTML = `
            <div class="message-avatar">
                ${avatarContent}
            </div>
            <div class="message-content">
                ${this.formatMessage(message.content)}
            </div>
        `;

        return messageElement;
    }

    /**
     * Format message content for display
     * @param {string} content - Raw message content
     * @returns {string} Formatted HTML content
     */
    formatMessage(content) {
        // Convert line breaks to HTML
        let formatted = content.replace(/\n/g, '<br>');
        
        // Basic markdown-like formatting
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
        
        return formatted;
    }

    /**
     * Animate message appearance
     * @param {HTMLElement} messageElement - Message element to animate
     */
    animateMessage(messageElement) {
        messageElement.style.opacity = '0';
        messageElement.style.transform = 'translateY(10px)';
        
        requestAnimationFrame(() => {
            messageElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            messageElement.style.opacity = '1';
            messageElement.style.transform = 'translateY(0)';
        });
    }

    /**
     * Remove welcome message from display
     */
    removeWelcomeMessage() {
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.transition = 'opacity 0.3s ease';
            welcomeMessage.style.opacity = '0';
            setTimeout(() => welcomeMessage.remove(), 300);
        }
    }

    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        this.typingIndicator.style.display = 'flex';
        this.typingIndicator.style.opacity = '0';
        this.scrollToBottom();
        
        requestAnimationFrame(() => {
            this.typingIndicator.style.transition = 'opacity 0.3s ease';
            this.typingIndicator.style.opacity = '1';
        });
    }

    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        this.typingIndicator.style.transition = 'opacity 0.3s ease';
        this.typingIndicator.style.opacity = '0';
        
        setTimeout(() => {
            this.typingIndicator.style.display = 'none';
        }, 300);
    }

    /**
     * Clear input field
     */
    clearInput() {
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
    }

    /**
     * Get current input value
     * @returns {string} Current input value
     */
    getInputValue() {
        return this.messageInput.value;
    }

    /**
     * Set input field value
     * @param {string} value - Value to set
     */
    setInputValue(value) {
        this.messageInput.value = value;
        this.handleInputResize();
    }

    /**
     * Enable or disable input controls
     * @param {boolean} disabled - Whether to disable controls
     */
    setInputDisabled(disabled) {
        this.messageInput.disabled = disabled;
        this.sendButton.disabled = disabled;
        
        if (disabled) {
            this.messageInput.style.opacity = '0.6';
            this.sendButton.style.opacity = '0.6';
        } else {
            this.messageInput.style.opacity = '1';
            this.sendButton.style.opacity = '1';
        }
    }

    /**
     * Scroll messages container to bottom
     */
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        });
    }

    /**
     * Focus on input field
     */
    focusInput() {
        setTimeout(() => {
            this.messageInput.focus();
        }, 0);
    }

    /**
     * Clear all messages from display
     */
    clearMessages() {
        // Remove all message elements except welcome message
        const messages = this.messagesContainer.querySelectorAll('.message');
        messages.forEach(message => message.remove());
        
        // Show welcome message again
        this.showWelcomeMessage();
    }

    /**
     * Show welcome message
     */
    showWelcomeMessage() {
        if (!this.messagesContainer.querySelector('.welcome-message')) {
            const welcomeElement = document.createElement('div');
            welcomeElement.className = 'welcome-message';
            welcomeElement.innerHTML = `
                <h2>Hello! I'm Claude</h2>
                <p>I'm an AI assistant created by Anthropic. I can help you with a wide variety of tasks. How can I assist you today?</p>
            `;
            this.messagesContainer.appendChild(welcomeElement);
        }
    }

    /**
     * Show error message
     * @param {string} errorMessage - Error message to display
     */
    showError(errorMessage) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.style.cssText = `
            background: #fee2e2;
            color: #dc2626;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid #fecaca;
        `;
        errorElement.textContent = errorMessage;
        
        this.messagesContainer.appendChild(errorElement);
        this.scrollToBottom();
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorElement.parentNode) {
                errorElement.remove();
            }
        }, 5000);
    }

    /**
     * Update send button state based on input
     * @param {string} inputValue - Current input value
     */
    updateSendButtonState(inputValue) {
        const hasContent = inputValue && inputValue.trim().length > 0;
        this.sendButton.disabled = !hasContent;
        this.sendButton.style.opacity = hasContent ? '1' : '0.5';
    }

    /**
     * Get message element by ID
     * @param {number} messageId - Message ID
     * @returns {HTMLElement|null} Message element or null
     */
    getMessageElement(messageId) {
        return this.messagesContainer.querySelector(`[data-message-id="${messageId}"]`);
    }

    /**
     * Add loading state to a message
     * @param {HTMLElement} messageElement - Message element
     */
    addMessageLoading(messageElement) {
        const content = messageElement.querySelector('.message-content');
        if (content) {
            content.classList.add('loading');
            content.style.opacity = '0.6';
        }
    }

    /**
     * Remove loading state from a message
     * @param {HTMLElement} messageElement - Message element
     */
    removeMessageLoading(messageElement) {
        const content = messageElement.querySelector('.message-content');
        if (content) {
            content.classList.remove('loading');
            content.style.opacity = '1';
        }
    }
}