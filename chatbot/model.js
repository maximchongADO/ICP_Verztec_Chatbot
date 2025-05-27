// MODEL - Data Management
class ChatModel {
    constructor() {
        this.messages = [];
        this.isTyping = false;
        this.customResponses = {};
    }

    /**
     * Add a new message to the conversation
     * @param {string} content - Message content
     * @param {string} sender - Message sender ('user' or 'assistant')
     * @returns {Object} The created message object
     */
    addMessage(content, sender) {
        const message = {
            id: Date.now(),
            content: content.trim(),
            sender: sender,
            timestamp: new Date()
        };
        this.messages.push(message);
        return message;
    }

    /**
     * Get all messages in the conversation
     * @returns {Array} Array of message objects
     */
    getMessages() {
        return [...this.messages];
    }

    /**
     * Set typing status
     * @param {boolean} status - Whether AI is typing
     */
    setTyping(status) {
        this.isTyping = status;
    }

    /**
     * Get current typing status
     * @returns {boolean} Current typing status
     */
    getTypingStatus() {
        return this.isTyping;
    }

    /**
     * Clear all messages from conversation
     */
    clearMessages() {
        this.messages = [];
    }

    /**
     * Add custom response for specific triggers
     * @param {string} trigger - Trigger word or phrase
     * @param {string} response - Response to return
     */
    addCustomResponse(trigger, response) {
        this.customResponses[trigger.toLowerCase()] = response;
    }

    /**
     * Simulate AI response generation
     * @param {string} userMessage - User's message
     * @returns {Promise<string>} Generated response
     */
    async generateResponse(userMessage) {
        // Check for custom responses first
        const messageKey = userMessage.toLowerCase();
        for (const [trigger, response] of Object.entries(this.customResponses)) {
            if (messageKey.includes(trigger)) {
                await this.simulateThinkingTime();
                return response;
            }
        }

        // Default response generation
        const responses = [
            "I understand your question. Let me help you with that.",
            "That's an interesting point. Here's what I think about it:",
            "I'd be happy to assist you with this topic.",
            "Thank you for asking. Based on what you've shared:",
            "That's a great question! Let me provide some insights:",
            "I can help you understand this better. Here's my perspective:",
            "I appreciate you bringing this up. From my understanding:"
        ];

        // Simulate thinking time
        await this.simulateThinkingTime();
        
        const randomResponse = responses[Math.floor(Math.random() * responses.length)];
        const contextualResponse = this.generateContextualResponse(userMessage);
        
        return `${randomResponse} ${contextualResponse}`;
    }

    /**
     * Generate contextual responses based on message content
     * @param {string} userMessage - User's message
     * @returns {string} Contextual response
     */
    generateContextualResponse(userMessage) {
        const message = userMessage.toLowerCase();
        
        if (message.includes('hello') || message.includes('hi') || message.includes('hey')) {
            return "Hello! I'm Claude, an AI assistant created by Anthropic. I'm here to help you with any questions or tasks you might have. What would you like to discuss today?";
        }
        
        if (message.includes('help') || message.includes('assist')) {
            return "I'm here to help! I can assist with a wide range of topics including answering questions, helping with analysis, creative writing, coding, math problems, and much more. What specific area would you like help with?";
        }
        
        if (message.includes('code') || message.includes('program')) {
            return "I'd be happy to help with coding! I can assist with various programming languages, debug code, explain concepts, or help you build applications. What programming challenge are you working on?";
        }
        
        if (message.includes('write') || message.includes('essay') || message.includes('story')) {
            return "I love helping with writing projects! Whether you need help with essays, creative stories, business writing, or any other type of content, I can provide guidance, suggestions, and even help with brainstorming ideas.";
        }

        if (message.includes('math') || message.includes('calculate')) {
            return "I can help with mathematical problems and calculations! From basic arithmetic to complex equations, algebra, calculus, statistics, and more. What math problem would you like help with?";
        }

        if (message.includes('explain') || message.includes('what is')) {
            return "I'd be happy to explain that concept to you! I can break down complex topics into understandable parts and provide examples to make things clearer. What would you like me to explain in more detail?";
        }
        
        return "I'd be happy to explore this topic with you further. Could you provide a bit more context or let me know what specific aspect you'd like to focus on?";
    }

    /**
     * Simulate AI thinking time
     * @returns {Promise} Promise that resolves after random delay
     */
    async simulateThinkingTime() {
        const delay = 1000 + Math.random() * 2000; // 1-3 seconds
        return new Promise(resolve => setTimeout(resolve, delay));
    }

    /**
     * Get conversation statistics
     * @returns {Object} Statistics about the conversation
     */
    getConversationStats() {
        const userMessages = this.messages.filter(msg => msg.sender === 'user');
        const assistantMessages = this.messages.filter(msg => msg.sender === 'assistant');
        
        return {
            totalMessages: this.messages.length,
            userMessages: userMessages.length,
            assistantMessages: assistantMessages.length,
            conversationStarted: this.messages.length > 0 ? this.messages[0].timestamp : null,
            lastMessage: this.messages.length > 0 ? this.messages[this.messages.length - 1].timestamp : null
        };
    }

    /**
     * Export conversation data
     * @returns {Object} Exportable conversation data
     */
    exportConversation() {
        return {
            messages: this.getMessages(),
            stats: this.getConversationStats(),
            exportedAt: new Date()
        };
    }
}