const fetch = require('node-fetch');
const mysql = require('mysql2/promise');
const dbConfig = require('../database/dbConfig');
const crypto = require('crypto');
const textToSpeech = require('@google-cloud/text-to-speech');
const path = require('path');
const fs = require('fs');
const { lipSyncMessage, audioFileToBase64, readJsonTranscript } = require('./rhubarbController');

const PYTHON_CHATBOT_URL = process.env.PYTHON_CHATBOT_URL || 'http://localhost:3000';

// Initialize the Google Cloud TTS client with proper authentication
const ttsClient = new textToSpeech.TextToSpeechClient({
  keyFilename: path.resolve(__dirname, 'service-account-key.json'), // Updated to correct path
  projectId: 'golden-frame-461314-e6'
});

const audioDir = path.resolve(__dirname, '../../public/audio');
if (!fs.existsSync(audioDir)) {
  fs.mkdirSync(audioDir, { recursive: true });
}

const callPythonChatbot = async (message, userId = "YABBABAABBBABBABAAB", chatHistory = []) => {
    try {
        console.log('Calling Python chatbot with:', { message, userId, chatHistory });
        const response = await fetch(`${PYTHON_CHATBOT_URL}/chatbot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                message: message, // Send original message without appending
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

// Avatar chatbot endpoint with TTS and lip sync
const processAvatarMessage = async (req, res) => {
    try {
        const { message } = req.body;
        
        if (!message?.trim()) {
            return res.status(400).json({
                success: false,
                message: 'Message is required'
            });
        }

        console.log('Processing avatar message:', message);

        // Get response from Python chatbot
        const chatbotResponse = await callPythonChatbot(message);
        
        if (!chatbotResponse || !chatbotResponse.message) {
            throw new Error('Invalid response from Python chatbot');
        }

        const textInput = chatbotResponse.message;
        console.log('Chatbot response text:', textInput);

        // Generate unique filename
        const timestamp = Date.now();
        const fileName = `message_${timestamp}.mp3`;
        const audioFilePath = path.join(audioDir, fileName);

        // Generate TTS audio
        console.log('Generating TTS audio...');
        const request = {
            input: { text: textInput },
            voice: {
                languageCode: 'en-GB',
                name: 'en-GB-Standard-A',
                ssmlGender: 'FEMALE'
            },
            audioConfig: {
                audioEncoding: 'MP3',
                speakingRate: 1.25,
                pitch: 0.0,
                volumeGainDb: 0.0
            }
        };

        const [ttsResponse] = await ttsClient.synthesizeSpeech(request);
        fs.writeFileSync(audioFilePath, ttsResponse.audioContent);
        console.log('TTS audio saved:', fileName);

        // Generate lip sync data
        console.log('Generating lip sync...');
        await lipSyncMessage(timestamp, true);
        
        // Convert audio to base64
        const audioBase64 = await audioFileToBase64(audioFilePath);
        
        // Read lip sync data
        const lipSyncPath = path.join(audioDir, `message_${timestamp}.json`);
        const lipsync = await readJsonTranscript(lipSyncPath);
        
        console.log('Lip sync data loaded:', lipsync.mouthCues?.length, 'mouth cues');

        // Create response message
        const responseMessage = {
            type: 'bot',
            text: textInput,
            audio: audioBase64,
            lipsync: lipsync
        };

        console.log('Sending response with audio and lipsync data');
        res.json({ messages: [responseMessage] });

    } catch (error) {
        console.error('Avatar message processing error:', error);
        res.status(500).json({
            success: false,
            message: 'Error processing avatar message',
            error: error.message
        });
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
            sources: response.sources || [],  // Pass through sources data
            timestamp: new Date().toISOString()
        });
        console.log('Response sent to client:', {
            message: response.message ? response.message.substring(0, 100) + '...' : 'No message',
            images: response.images,
            sources: response.sources || []
        });

    } catch (error) {
        console.error('Error processing message:', error);
        res.status(500).json({
            success: false,
            message: 'Error processing request',
            error: error.message
        });
    }
};

// Get chat history for a user (grouped by chat_id)
const getChatHistory = async (req, res) => {
    try {
        const userId = req.user?.userId || req.user?.id || req.user?.sub || req.query.user_id || req.body.user_id || "defaultUser";
        let chatId;
        if (req && req.params && typeof req.params === 'object' && Object.prototype.hasOwnProperty.call(req.params, 'chat_id')) {
            chatId = req.params.chat_id;
        } else if (req && req.query && typeof req.query === 'object' && Object.prototype.hasOwnProperty.call(req.query, 'chat_id')) {
            chatId = req.query.chat_id;
        } else if (req && req.body && typeof req.body === 'object' && Object.prototype.hasOwnProperty.call(req.body, 'chat_id')) {
            chatId = req.body.chat_id;
        } else {
            chatId = undefined;
        }
        if (!userId) {
            return res.status(400).json({ success: false, message: 'userId is required' });
        }
        const connection = await mysql.createConnection(dbConfig);
        if (chatId) {
            // Return all messages for this chat_id
            const [rows] = await connection.execute(
                'SELECT chat_id, user_id, user_message, bot_response, timestamp FROM chat_logs WHERE user_id = ? AND chat_id = ? ORDER BY timestamp ASC',
                [userId, chatId]
            );
            await connection.end();
            // Return messages in the format expected by frontend
            const messages = rows.map(row => ({
                chat_id: row.chat_id,
                user_id: row.user_id,
                message: row.user_message,
                sender: row.user_id === userId ? "user" : "bot", // for user messages
                bot_response: row.bot_response,
                timestamp: row.timestamp
            }));
            // Interleave user and bot messages for chat display
            let chatMessages = [];
            rows.forEach(row => {
                chatMessages.push({
                    message: row.user_message,
                    sender: "user",
                    timestamp: row.timestamp
                });
                if (row.bot_response) {
                    chatMessages.push({
                        message: row.bot_response,
                        sender: "bot",
                        timestamp: row.timestamp
                    });
                }
            });
            return res.json(chatMessages);
        } else {
            // Return all chat sessions for this user
            const [rows] = await connection.execute(
                'SELECT chat_id, MIN(timestamp) as created_at, MAX(timestamp) as last_message, COUNT(*) as message_count FROM chat_logs WHERE user_id = ? GROUP BY chat_id ORDER BY last_message DESC',
                [userId]
            );
            await connection.end();
            const chats = rows.map(row => ({
                chat_id: row.chat_id,
                date: row.created_at,
                last_message: row.last_message,
                message_count: row.message_count,
                title: `Chat (${row.created_at ? new Date(row.created_at).toLocaleString() : row.chat_id})`
            }));
            return res.json(chats);
        }
    } catch (error) {
        console.error('Error fetching chat history:', error);
        res.status(500).json({ success: false, message: 'Error fetching chat history', error: error.message });
    }
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
    newChat,
    processAvatarMessage
};