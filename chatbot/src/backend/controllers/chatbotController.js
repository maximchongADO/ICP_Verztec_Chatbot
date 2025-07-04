const fetch = require('node-fetch');
const mysql = require('mysql2/promise');
const dbConfig = require('../database/dbConfig');
const crypto = require('crypto');

const PYTHON_CHATBOT_URL = process.env.PYTHON_CHATBOT_URL || 'http://localhost:3000';

const callPythonChatbot = async (message, userId = "YABBABAABBBABBABAAB", chatHistory = []) => {
    try {
        console.log('Calling Python chatbot with:', { message, userId, chatHistory });
        const fullMessage = `${message} YABABDODD`;   // ← append here
        const response = await fetch(`${PYTHON_CHATBOT_URL}/chatbot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                message: fullMessage,
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
        const { message, chat_id } = req.body;
        if (!message?.trim()) {
            return res.status(400).json({
                success: false,
                message: 'Message is required'
            });
        }

        const userId = req.user?.userId;
        const chatId = chat_id || req.session.chat_id;
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
        req.session.chat_id = chatId; // Store chat_id in session for later use

        // Store chat in database using user_id and chat_id
        const connection = await mysql.createConnection(dbConfig);
        await connection.execute(
            'INSERT INTO chat_logs (timestamp, user_message, bot_response, feedback, query_score, relevance_score, user_id, chat_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',

            [new Date(), message, response.message, null, response.query_score || null, response.relevance_score || null, userId, chatId]
        );
        await connection.end();

        res.json({
            success: true,
            message: response.message,
            images: response.images,
            timestamp: new Date().toISOString()
        });
        console.log('Response sent to client:', response.images);

    } catch (error) {
        console.error('Error processing message:', error);
        res.status(500).json({
            success: false,
            message: 'Error processing request',
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
const clearChatHistory = async (req, res) => {
    req.session.chatHistory = [];
    try {
        const userId = req.user?.userId;
        const chatId = req.body.chat_id || req.session.chat_id;
        console.log('ClearChatHistory called with:', { userId, chatId });
        if (!userId || !chatId) {
            console.error('Missing userId or chatId:', { userId, chatId });
            return res.status(400).json({
                success: false,
                message: 'userId and chatId are required to clear chat history',
                userId,
                chatId
            });
        }
        const connection = await mysql.createConnection(dbConfig);
        const [result] = await connection.execute(
            'DELETE FROM chat_logs WHERE user_id = ? AND chat_id = ?',
            [userId, chatId]
        );
        console.log('Rows affected by delete:', result.affectedRows);
        await connection.end();
        res.json({
            success: true,
            message: 'Chat history cleared from session and database',
            deleted: result.affectedRows
        });
    } catch (error) {
        console.error('Error clearing chat history:', error);
        res.status(500).json({
            success: false,
            message: 'Error clearing chat history',
            error: error.message
        });
    }
};

// Handle feedback from user
const handleFeedback = async (req, res) => {
    try {
        const { feedback } = req.body;

        if (!feedback) {
            return res.status(400).json({
                success: false,
                message: 'feedback is required'
            });
        }

        const connection = await mysql.createConnection(dbConfig);
        await connection.execute(
            `UPDATE chat_logs
             SET feedback = ?
             ORDER BY timestamp DESC
             LIMIT 1`,
            [feedback]
        );
        await connection.end();

        res.json({
            success: true,
            message: 'Feedback recorded successfully'
        });

    } catch (error) {
        console.error('Error handling feedback:', error);
        res.status(500).json({
            success: false,
            message: 'Error recording feedback',
            error: error.message
        });
    }
};

// Create a new chat and return a new chat_id
const newChat = async (req, res) => {
    try {
        // Log the incoming request for debugging
        console.log('newChat endpoint called. req.user:', req.user, 'req.body:', req.body);
        const userId = req.user?.userId || req.user?.id || req.user?.sub || req.body.user_id || "defaultUser";
        // Generate a random chat_id (UUID v4 style)
        let chatId;
        if (crypto.randomUUID) {
            chatId = crypto.randomUUID();
        } else {
            chatId = crypto.randomBytes(16).toString('hex');
        }
        // Log the generated chatId and userId
        console.log('Generated new chat_id:', chatId, 'for userId:', userId);
        // Always return a JSON object with chat_id and success
        res.status(200).json({ success: true, chat_id: chatId });
    } catch (error) {
        console.error('Error in newChat endpoint:', error);
        res.status(500).json({ success: false, message: 'Failed to create new chat', error: error.message });
    }
};

module.exports = {
    processMessage,
    getChatHistory,
    clearChatHistory,
    handleFeedback,
    newChat
};