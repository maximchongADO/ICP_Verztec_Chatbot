const fetch = require('node-fetch');

const PYTHON_CHATBOT_URL = process.env.PYTHON_CHATBOT_URL || 'http://localhost:8000';

const callPythonChatbot = async (message, userId = null, chatHistory = []) => {
    try {
        const response = await fetch(`${PYTHON_CHATBOT_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: userId,
                chat_history: chatHistory
            }),
            timeout: 30000 // 30 second timeout
        });

        if (!response.ok) {
            console.error(`Python API error: ${response.status} ${response.statusText}`);
            return {
                success: false,
                error: `API request failed with status ${response.status}`,
                message: 'I\'m sorry, I couldn\'t process your request at the moment.'
            };
        }

        const result = await response.json();
        return result;

    } catch (error) {
        console.error('Python API call error:', error);
        
        if (error.code === 'ECONNREFUSED') {
            return {
                success: false,
                error: 'Python chatbot service is not running',
                message: 'I\'m sorry, the chatbot service is currently unavailable. Please ensure the Python API is running on port 8000.'
            };
        }

        return {
            success: false,
            error: error.message,
            message: 'I\'m sorry, I couldn\'t connect to the chatbot service.'
        };
    }
};

const processMessage = async (req, res) => {
    try {
        const { message } = req.body;
        const userId = req.user?.userId; // From JWT token
        const chatHistory = req.session.chatHistory || [];

        // Validate input
        if (!message || message.trim() === '') {
            return res.status(400).json({
                success: false,
                error: 'Message is required'
            });
        }

        // Call Python script
        const response = await callPythonChatbot(message, userId, chatHistory);
        
        if (response.success !== false) {
            // Update chat history in session
            if (!req.session.chatHistory) {
                req.session.chatHistory = [];
            }
            req.session.chatHistory.push(message);
            
            // Keep only last 10 messages for context
            if (req.session.chatHistory.length > 10) {
                req.session.chatHistory = req.session.chatHistory.slice(-10);
            }
        }

        res.json({
            success: response.success !== false,
            message: response.message,
            timestamp: new Date().toISOString(),
            ...response
        });
    } catch (error) {
        console.error('Chatbot controller error:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error',
            message: 'I\'m sorry, I encountered an error while processing your request.'
        });
    }
};

// Get chat history for a user
const getChatHistory = (req, res) => {
    const userChatHistory = req.session.chatHistory || [];
    res.json({
        success: true,
        chatHistory: userChatHistory
    });
};

// Clear chat history
const clearChatHistory = (req, res) => {
    req.session.chatHistory = [];
    res.json({
        success: true,
        message: 'Chat history cleared'
    });
};

module.exports = {
    processMessage,
    getChatHistory,
    clearChatHistory
};
