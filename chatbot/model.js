// MODEL - Data Management
class ChatModel {
    constructor() {
        this.messages = [];
        this.isTyping = false;
        this.customResponses = {};
    }

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

    getMessages() {
        return [...this.messages];
    }

    setTyping(status) {
        this.isTyping = status;
    }

    getTypingStatus() {
        return this.isTyping;
    }

    clearMessages() {
        this.messages = [];
    }

    addCustomResponse(trigger, response) {
        this.customResponses[trigger.toLowerCase()] = response;
    }

    async generateResponse(userMessage) {
        // Check for hardcoded triggers
        const messageKey = userMessage.toLowerCase();
        for (const [trigger, response] of Object.entries(this.customResponses)) {
            if (messageKey.includes(trigger)) {
                await this.simulateThinkingTime();
                return response;
            }
        }

        // Send to backend if no trigger matched
        await this.simulateThinkingTime(); // Optional: fake delay before real call
        const history = this.messages
            .filter(m => m.sender === 'user')
            .map(m => m.content);

        try {
            const response = await sendToBackend(userMessage, history);
            return response;
        } catch (err) {
            console.error("Error from backend:", err);
            return "Sorry, I couldn't process your request due to a system error.";
        }
    }

    async simulateThinkingTime() {
        const delay = 1000 + Math.random() * 1500;
        return new Promise(resolve => setTimeout(resolve, delay));
    }

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

    exportConversation() {
        return {
            messages: this.getMessages(),
            stats: this.getConversationStats(),
            exportedAt: new Date()
        };
    }
}

// Make sure this is included in the same file or globally accessible
async function sendToBackend(query, history = []) {
    const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, history: history })
    });
    const data = await response.json();
    return data.answer;
}
