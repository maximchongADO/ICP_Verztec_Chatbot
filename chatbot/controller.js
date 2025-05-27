// CONTROLLER - Business Logic
class ChatController {
    constructor() {
        this.model = new ChatModel();
        this.view = new ChatView();
        
        this.isProcessing = false;
        
        this.initializeEventHandlers();
        this.initializeApp();
    }

    /**
     * Initialize event handlers connecting view to controller
     */
    initializeEventHandlers() {
        // Connect view events to controller methods
        this.view.onSendMessage = () => this.handleSendMessage();
        this.view.onInputChange = (value) => this.handleInputChange(value);
        
        // Set up keyboard shortcuts
        this.setupKeyboardShortcuts();
    }

    /**
     * Initialize the application
     */
    initializeApp() {
        // Focus input on load
        this.view.focusInput();
        
        // Update initial send button state
        this.view.updateSendButtonState('');
        
        console.log('Claude Chatbot initialized successfully');
    }

    /**
     * Set up keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to focus input
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.view.focusInput();
            }
            
            // Ctrl/Cmd + L to clear conversation
            if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
                e.preventDefault();
                this.clearConversation();
            }
            
            // Escape to cancel current processing
            if (e.key === 'Escape' && this.isProcessing) {
                this.cancelCurrentMessage();
            }
        });
    }

    /**
     * Handle input change events
     * @param {string} value - Current input value
     */
    handleInputChange(value) {
        this.view.updateSendButtonState(value);
    }

    /**
     * Handle sending a message
     */
    async handleSendMessage() {
        if (this.isProcessing) {
            return;
        }

        const messageContent = this.view.getInputValue().trim();
        
        if (!messageContent) {
            return;
        }

        try {
            await this.processMessage(messageContent);
        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Process a user message and generate AI response
     * @param {string} messageContent - User's message content
     */
    async processMessage(messageContent) {
        this.isProcessing = true;
        
        // Disable input while processing
        this.view.setInputDisabled(true);
        this.view.clearInput();

        try {
            // Add user message to model and display
            const userMessage = this.model.addMessage(messageContent, 'user');
            this.view.displayMessage(userMessage);

            // Show typing indicator
            this.model.setTyping(true);
            this.view.showTypingIndicator();

            // Generate AI response
            const response = await this.model.generateResponse(messageContent);
            
            // Hide typing indicator
            this.model.setTyping(false);
            this.view.hideTypingIndicator();
            
            // Add AI response to model and display
            const aiMessage = this.model.addMessage(response, 'assistant');
            this.view.displayMessage(aiMessage);
            
            // Log conversation stats (optional)
            this.logConversationStats();
            
        } catch (error) {
            // Hide typing indicator on error
            this.model.setTyping(false);
            this.view.hideTypingIndicator();
            
            throw error;
        } finally {
            this.isProcessing = false;
            
            // Re-enable input
            this.view.setInputDisabled(false);
            this.view.focusInput();
        }
    }

    /**
     * Handle errors during message processing
     * @param {Error} error - Error object
     */
    handleError(error) {
        console.error('Error processing message:', error);
        
        // Show error message to user
        const errorMessage = this.model.addMessage(
            "I apologize, but I encountered an error while processing your message. Please try again.",
            'assistant'
        );
        this.view.displayMessage(errorMessage);
        
        // Optionally show error details in development
        if (process?.env?.NODE_ENV === 'development') {
            this.view.showError(`Development Error: ${error.message}`);
        }
    }

    /**
     * Cancel current message processing
     */
    cancelCurrentMessage() {
        if (this.isProcessing) {
            this.isProcessing = false;
            this.model.setTyping(false);
            this.view.hideTypingIndicator();
            this.view.setInputDisabled(false);
            this.view.focusInput();
            
            console.log('Message processing cancelled by user');
        }
    }

    /**
     * Add custom response for specific triggers
     * @param {string} trigger - Trigger word or phrase
     * @param {string} response - Response to return
     */
    addCustomResponse(trigger, response) {
        this.model.addCustomResponse(trigger, response);
    }

    /**
     * Get conversation history
     * @returns {Array} Array of message objects
     */
    getConversationHistory() {
        return this.model.getMessages();
    }

    /**
     * Clear the entire conversation
     */
    clearConversation() {
        if (this.isProcessing) {
            this.cancelCurrentMessage();
        }
        
        this.model.clearMessages();
        this.view.clearMessages();
        this.view.focusInput();
        
        console.log('Conversation cleared');
    }

    /**
     * Export conversation data
     * @returns {Object} Exportable conversation data
     */
    exportConversation() {
        return this.model.exportConversation();
    }

    /**
     * Import conversation data
     * @param {Object} conversationData - Previously exported conversation data
     */
    importConversation(conversationData) {
        try {
            if (conversationData && conversationData.messages) {
                this.clearConversation();
                
                // Import messages
                conversationData.messages.forEach(message => {
                    this.model.addMessage(message.content, message.sender);
                    this.view.displayMessage(message);
                });
                
                console.log('Conversation imported successfully');
            }
        } catch (error) {
            console.error('Error importing conversation:', error);
            this.view.showError('Failed to import conversation data');
        }
    }

    /**
     * Get conversation statistics
     * @returns {Object} Conversation statistics
     */
    getConversationStats() {
        return this.model.getConversationStats();
    }

    /**
     * Log conversation statistics to console
     */
    logConversationStats() {
        const stats = this.getConversationStats();
        console.log('Conversation Stats:', stats);
    }

    /**
     * Set up auto-save functionality
     * @param {number} intervalMs - Auto-save interval in milliseconds
     */
    setupAutoSave(intervalMs = 30000) {
        setInterval(() => {
            if (this.model.getMessages().length > 0) {
                const conversationData = this.exportConversation();
                // In a real app, you would save to localStorage or send to server
                console.log('Auto-saving conversation...', conversationData);
            }
        }, intervalMs);
    }

    /**
     * Handle window beforeunload event
     */
    setupBeforeUnloadHandler() {
        window.addEventListener('beforeunload', (e) => {
            if (this.isProcessing) {
                e.preventDefault();
                e.returnValue = 'A message is currently being processed. Are you sure you want to leave?';
                return e.returnValue;
            }
        });
    }

    /**
     * Set up the application with additional features
     * @param {Object} options - Configuration options
     */
    configure(options = {}) {
        if (options.autoSave) {
            this.setupAutoSave(options.autoSaveInterval);
        }
        
        if (options.beforeUnloadWarning) {
            this.setupBeforeUnloadHandler();
        }
        
        if (options.customResponses) {
            Object.entries(options.customResponses).forEach(([trigger, response]) => {
                this.addCustomResponse(trigger, response);
            });
        }
    }

    /**
     * Simulate a message from the assistant
     * @param {string} message - Message content
     */
    simulateAssistantMessage(message) {
        const aiMessage = this.model.addMessage(message, 'assistant');
        this.view.displayMessage(aiMessage);
    }

    /**
     * Get current processing status
     * @returns {boolean} Whether a message is currently being processed
     */
    isCurrentlyProcessing() {
        return this.isProcessing;
    }

    /**
     * Restart the conversation with a new welcome message
     */
    restartConversation() {
        this.clearConversation();
        this.simulateAssistantMessage("Hello! I'm Claude, your AI assistant. How can I help you today?");
    }

    /**
     * Search through conversation history
     * @param {string} query - Search query
     * @returns {Array} Matching messages
     */
    searchConversation(query) {
        const messages = this.model.getMessages();
        const searchTerm = query.toLowerCase();
        
        return messages.filter(message => 
            message.content.toLowerCase().includes(searchTerm)
        );
    }

    /**
     * Get the last N messages
     * @param {number} count - Number of messages to retrieve
     * @returns {Array} Last N messages
     */
    getRecentMessages(count = 10) {
        const messages = this.model.getMessages();
        return messages.slice(-count);
    }
    
}

async function sendToBackend(query, history=[]) {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: query, history: history })
  });
  const data = await response.json();
  return data.answer;
}
