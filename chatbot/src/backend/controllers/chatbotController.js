const fetch = require('node-fetch');

const PYTHON_CHATBOT_URL = process.env.PYTHON_CHATBOT_URL || 'http://localhost:3000';

const callPythonChatbot = async (message, userId = null, chatHistory = []) => {
    try {
        console.log('Calling Python chatbot with:', { message, userId, chatHistory });
        
        const response = await fetch(`${PYTHON_CHATBOT_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                user_id: userId,
                chat_history: chatHistory || []
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Python API error:', errorText);
            throw new Error(`API request failed with status ${response.status}`);
        }

        const result = await response.json();
        console.log('Python API response:', result);
        return result;

    } catch (error) {
        console.error('Python API call error:', error);
        throw error;
    }
};

const processMessage = async (req, res) => {
    try {
        const { message } = req.body;
        if (!message?.trim()) {
            return res.status(400).json({
                success: false,
                message: 'Message is required'
            });
        }

        const userId = req.user?.userId;
        const chatHistory = req.session?.chatHistory || [];

        const response = await callPythonChatbot(message, userId, chatHistory);
        
        if (!response || !response.message) {
            throw new Error('Invalid response from Python chatbot');
        }

        // Update session chat history
        if (!req.session.chatHistory) {
            req.session.chatHistory = [];
        }
        req.session.chatHistory.push({
            message,
            timestamp: new Date().toISOString()
        });

        res.json({
            success: true,
            message: response.message,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error processing message:', error);
        res.status(500).json({
            success: false,
            message: 'Sorry, I encountered an error processing your request.',
            error: error.message
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
