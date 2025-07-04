const User = require('../models/user.js');
const bcrypt = require('bcrypt');
const fs = require('fs');
const jwt = require('jsonwebtoken');
const mysql = require('mysql2/promise');
const dbConfig = require('../database/dbConfig.js');
require('dotenv').config();

const hashPassword = async (password) => {
    const saltRounds = 10;
    return await bcrypt.hash(password, saltRounds);
}

const generateAccessToken = (user) => {
    const secret = process.env.JWT_SECRET || process.env.ACCESS_TOKEN_SECRET || 'fallback-secret-key';
    return jwt.sign({ userId: user.id, role: user.role }, secret, { expiresIn: '1h' });
}

const getAllUsers = async (req, res) => {
    try {
        const users = await User.getAllUsers();
        res.status(200).json(users);
    } catch (error) {
        console.error('Error fetching users:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}

const getUserById = async (req, res) => {
    const userId = req.params.id;
    try {
        const user = await User.getUserById(userId);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.status(200).json(user);
    } catch (error) {
        console.error('Error fetching user:', error);
        res.status(500).json({ message: 'User Internal server error' });
    }
}

const loginUser = async (req, res) => {
    const { username, email, password } = req.body;
    
    try {
        let user;
        
        // Try to find user by email first, then by username
        if (email) {
            user = await User.getUserByEmail(email);
        } else if (username) {
            // We need to add a method to find by username
            user = await User.getUserByUsername(username);
        }
        
        if (!user) {
            return res.status(401).json({ 
                success: false, 
                message: 'Invalid credentials' 
            });
        }
        
        const isPasswordValid = await bcrypt.compare(password, user.password);
        if (!isPasswordValid) {
            return res.status(401).json({ 
                success: false, 
                message: 'Invalid credentials' 
            });
        }
        
        const token = generateAccessToken(user);
        res.status(200).json({ 
            success: true, 
            token, 
            userId: user.id,
            message: 'Login successful'
        });
    } catch (error) {
        console.error('Error logging in:', error);
        res.status(500).json({ 
            success: false, 
            message: 'Test Internal server error' 
        });
    }
}

const createUser = async (req, res, next, _generateAccessToken = generateAccessToken) => {
    const newUser = req.body;
    try {
        newUser.password = await hashPassword(newUser.password);
        const createdUser = await User.createUser(newUser);
        const token = _generateAccessToken(createdUser);
        res.status(201).json({ message: 'User created successfully', token, userId: createdUser.id });
    } catch (error) {
        console.error('Error creating user:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
}

// Admin-only: create user with role
const adminCreateUser = async (req, res) => {
    // Only allow if admin
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ message: 'Admin access required' });
    }
    const { username, email, password, role } = req.body;
    if (!username || !email || !password || !role) {
        return res.status(400).json({ message: 'Missing required fields' });
    }
    if (!['user', 'admin'].includes(role)) {
        return res.status(400).json({ message: 'Role must be user or admin' });
    }
    try {
        const hashedPassword = await hashPassword(password);
        const createdUser = await User.createUser({
            username,
            email,
            password: hashedPassword,
            role
        });
        res.status(201).json({
            message: 'User created successfully',
            user: {
                id: createdUser.id,
                username: createdUser.username,
                email: createdUser.email,
                role: createdUser.role
            }
        });
    } catch (error) {
        console.error('Error creating user (admin):', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};

// Admin-only: update user profile
const adminUpdateUser = async (req, res) => {
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ message: 'Admin access required' });
    }
    const userId = req.params.id;
    const { username, email, role, password } = req.body;
    if (!username && !email && !role && !password) {
        return res.status(400).json({ message: 'No fields to update' });
    }
    if (role && !['user', 'admin'].includes(role)) {
        return res.status(400).json({ message: 'Role must be user or admin' });
    }
    try {
        let updateFields = {};
        if (username) updateFields.username = username;
        if (email) updateFields.email = email;
        if (role) updateFields.role = role;
        if (password) updateFields.password = await hashPassword(password);

        const updatedUser = await User.updateUser(userId, updateFields);
        if (!updatedUser) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.json({
            message: 'User updated successfully',
            user: updatedUser
        });
    } catch (error) {
        console.error('Error updating user (admin):', error);
        res.status(500).json({ message: 'Internal server error' });
    }
};

const decodeJWT = async (req, res) => {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
        return res.status(401).json({ message: 'No token provided' });
    }
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        res.status(200).json(decoded);
    } catch (error) {
        console.error('Error decoding JWT:', error);
        res.status(401).json({ message: 'Invalid token' });
    }
}

const getCurrentUser = async (req, res) => {
    try {
        const userId = req.user?.userId;
        if (!userId) return res.status(401).json({ message: "Not authenticated" });
        const user = await User.getUserByIdFull(userId);
        if (!user) return res.status(404).json({ message: "User not found" });
        res.json(user);
    } catch (error) {
        res.status(500).json({ message: "Internal server error" });
    }
};

// User analytics dashboard (per user)
const getUserAnalytics = async (req, res) => {
    try {
        // Accept userId from query or JWT
        const userId = req.query.userId || req.user?.userId;
        if (!userId) return res.status(400).json({ message: "Missing userId" });

        const connection = await mysql.createConnection(dbConfig);

        // Number of unique chats
        const [chats] = await connection.execute(
            'SELECT COUNT(DISTINCT chat_id) AS chatCount FROM chat_logs WHERE user_id = ?',
            [userId]
        );
        // Number of queries/messages sent
        const [queries] = await connection.execute(
            'SELECT COUNT(*) AS queryCount FROM chat_logs WHERE user_id = ?',
            [userId]
        );
        // Last interaction timestamp
        const [last] = await connection.execute(
            'SELECT MAX(timestamp) AS lastInteraction FROM chat_logs WHERE user_id = ?',
            [userId]
        );
        // Number of feedbacks given
        const [feedbacks] = await connection.execute(
            'SELECT COUNT(*) AS feedbackCount FROM chat_logs WHERE user_id = ? AND feedback IS NOT NULL AND feedback != ""',
            [userId]
        );
        // --- Satisfaction Calculation ---
        // Get number of "not helpful" feedbacks
        const [notHelpfulRows] = await connection.execute(
            'SELECT COUNT(*) AS notHelpfulCount FROM chat_logs WHERE user_id = ? AND feedback = "not helpful"',
            [userId]
        );
        // Get total number of chats (or feedbacks, depending on your logic)
        const [totalChatsRows] = await connection.execute(
            'SELECT COUNT(DISTINCT chat_id) AS totalChats FROM chat_logs WHERE user_id = ?',
            [userId]
        );
        const notHelpfulCount = notHelpfulRows[0]?.notHelpfulCount || 0;
        const totalChats = totalChatsRows[0]?.totalChats || 0;
        // Satisfaction = 100% - (notHelpfulCount / totalChats * 100)
        let satisfaction = null;
        if (totalChats > 0) {
            satisfaction = Math.round(100 - (notHelpfulCount / totalChats) * 100);
        }

        // --- Topic Analytics Logic ---
        // Define your topics and keywords here
        const TOPIC_KEYWORDS = [
            { label: "Pantry Rules", keywords: ["pantry rules", "pantry", "food", "snacks"] },
            { label: "Leave Policy", keywords: ["leave policy", "leave", "annual leave", "medical leave", "mc", "vacation"] },
            { label: "Offboarding Process", keywords: ["offboarding", "resign", "exit process", "last day", "clearance"] },
            { label: "Onboarding Process", keywords: ["onboarding", "new joiner", "welcome", "orientation"] },
            { label: "E-invoices", keywords: ["e-invoice", "einvoice", "invoice", "upload invoice"] },
            { label: "IT Support", keywords: ["it support", "helpdesk", "computer", "laptop", "reset password"] },
            { label: "Company Policies", keywords: ["company policy", "policies", "regulations"] },
            // Add more topics as needed
        ];

        // Fetch all user messages for this user
        const [userMessages] = await connection.execute(
            'SELECT user_message FROM chat_logs WHERE user_id = ?',
            [userId]
        );

        // Count topic occurrences
        const topicCounts = TOPIC_KEYWORDS.map(topic => {
            let count = 0;
            for (const row of userMessages) {
                const msg = (row.user_message || "").toLowerCase();
                if (topic.keywords.some(kw => msg.includes(kw))) {
                    count++;
                }
            }
            return { label: topic.label, count };
        }).filter(t => t.count > 0); // Only show topics with at least 1 interaction

        await connection.end();

        res.json({
            chatCount: chats[0]?.chatCount || 0,
            queryCount: queries[0]?.queryCount || 0,
            lastInteraction: last[0]?.lastInteraction || null,
            feedbackCount: feedbacks[0]?.feedbackCount || 0,
            topicAnalytics: topicCounts,
            satisfaction,
            notHelpfulCount,
            totalChats
        });
    } catch (error) {
        console.error('Error fetching user analytics:', error);
        res.status(500).json({ message: "Internal server error" });
    }
};

// Get recent chats for analytics
const getUserChats = async (req, res) => {
    try {
        const userId = req.query.userId || req.user?.userId;
        if (!userId) return res.status(400).json({ message: "Missing userId" });

        const connection = await mysql.createConnection(dbConfig);
        // Get recent chats: group by chat_id, get first timestamp/message, count, feedback count
        const [rows] = await connection.execute(
            `SELECT 
                chat_id,
                MIN(timestamp) AS firstTimestamp,
                SUBSTRING_INDEX(GROUP_CONCAT(user_message ORDER BY timestamp ASC), ',', 1) AS firstMessage,
                COUNT(*) AS messageCount,
                SUM(CASE WHEN feedback IS NOT NULL AND feedback != '' THEN 1 ELSE 0 END) AS feedbackCount
            FROM chat_logs
            WHERE user_id = ?
            GROUP BY chat_id
            ORDER BY firstTimestamp DESC
            LIMIT 10`,
            [userId]
        );
        await connection.end();
        res.json(rows);
    } catch (error) {
        console.error('Error fetching user chats:', error);
        res.status(500).json({ message: "Internal server error" });
    }
};

// Get recent feedback for analytics
const getUserFeedback = async (req, res) => {
    try {
        const userId = req.query.userId || req.user?.userId;
        if (!userId) return res.status(400).json({ message: "Missing userId" });

        const connection = await mysql.createConnection(dbConfig);
        // Get recent feedbacks (latest 20)
        const [rows] = await connection.execute(
            `SELECT 
                timestamp,
                user_message,
                bot_response,
                feedback
            FROM chat_logs
            WHERE user_id = ? AND feedback IS NOT NULL AND feedback != ''
            ORDER BY timestamp DESC
            LIMIT 20`,
            [userId]
        );
        await connection.end();
        res.json(rows);
    } catch (error) {
        console.error('Error fetching user feedback:', error);
        res.status(500).json({ message: "Internal server error" });
    }
};

// Company-wide analytics (admin only)
const getCompanyAnalytics = async (req, res) => {
    try {
        if (!req.user || req.user.role !== 'admin') {
            return res.status(403).json({ message: "Admin access required" });
        }
        const connection = await mysql.createConnection(dbConfig);

        // Total unique users
        const [users] = await connection.execute(
            'SELECT COUNT(DISTINCT user_id) AS userCount FROM chat_logs'
        );
        // Total unique chats
        const [chats] = await connection.execute(
            'SELECT COUNT(DISTINCT chat_id) AS chatCount FROM chat_logs'
        );
        // Total queries/messages
        const [queries] = await connection.execute(
            'SELECT COUNT(*) AS queryCount FROM chat_logs'
        );
        // Last interaction timestamp
        const [last] = await connection.execute(
            'SELECT MAX(timestamp) AS lastInteraction FROM chat_logs'
        );
        // Total feedbacks
        const [feedbacks] = await connection.execute(
            'SELECT COUNT(*) AS feedbackCount FROM chat_logs WHERE feedback IS NOT NULL AND feedback != ""'
        );
        // --- Satisfaction Calculation ---
        const [helpfulRows] = await connection.execute(
            'SELECT COUNT(*) AS helpfulCount FROM chat_logs WHERE feedback = "helpful"'
        );
        const [notHelpfulRows] = await connection.execute(
            'SELECT COUNT(*) AS notHelpfulCount FROM chat_logs WHERE feedback = "not helpful"'
        );
        const helpfulCount = helpfulRows[0]?.helpfulCount || 0;
        const notHelpfulCount = notHelpfulRows[0]?.notHelpfulCount || 0;
        const totalFeedback = helpfulCount + notHelpfulCount;
        const satisfaction = totalFeedback > 0 ? Math.round((helpfulCount / totalFeedback) * 100) : null;
        // Topic analytics for all users
        const TOPIC_KEYWORDS = [
            { label: "Pantry Rules", keywords: ["pantry rules", "pantry", "food", "snacks"] },
            { label: "Leave Policy", keywords: ["leave policy", "leave", "annual leave", "medical leave", "mc", "vacation"] },
            { label: "Offboarding Process", keywords: ["offboarding", "resign", "exit process", "last day", "clearance"] },
            { label: "Onboarding Process", keywords: ["onboarding", "new joiner", "welcome", "orientation"] },
            { label: "E-invoices", keywords: ["e-invoice", "einvoice", "invoice", "upload invoice"] },
            { label: "IT Support", keywords: ["it support", "helpdesk", "computer", "laptop", "reset password"] },
            { label: "Company Policies", keywords: ["company policy", "policies", "regulations"] },
            // Add more topics as needed
        ];
        const [allMessages] = await connection.execute(
            'SELECT user_message FROM chat_logs'
        );
        const topicCounts = TOPIC_KEYWORDS.map(topic => {
            let count = 0;
            for (const row of allMessages) {
                const msg = (row.user_message || "").toLowerCase();
                if (topic.keywords.some(kw => msg.includes(kw))) {
                    count++;
                }
            }
            return { label: topic.label, count };
        }).filter(t => t.count > 0);

        await connection.end();

        res.json({
            userCount: users[0]?.userCount || 0,
            chatCount: chats[0]?.chatCount || 0,
            queryCount: queries[0]?.queryCount || 0,
            lastInteraction: last[0]?.lastInteraction || null,
            feedbackCount: feedbacks[0]?.feedbackCount || 0,
            topicAnalytics: topicCounts,
            satisfaction
        });
    } catch (error) {
        console.error('Error fetching company analytics:', error);
        res.status(500).json({ message: "Internal server error" });
    }
};

// Per-user analytics for all users (admin only)
const getAllUsersAnalytics = async (req, res) => {
    try {
        if (!req.user || req.user.role !== 'admin') {
            return res.status(403).json({ message: "Admin access required" });
        }
        const connection = await mysql.createConnection(dbConfig);

        // Get all users
        const [users] = await connection.execute(
            'SELECT DISTINCT user_id FROM chat_logs'
        );
        const userIds = users.map(u => u.user_id);

        // For each user, get analytics (reuse logic from getUserAnalytics)
        const TOPIC_KEYWORDS = [
            { label: "Pantry Rules", keywords: ["pantry rules", "pantry", "food", "snacks"] },
            { label: "Leave Policy", keywords: ["leave policy", "leave", "annual leave", "medical leave", "mc", "vacation"] },
            { label: "Offboarding Process", keywords: ["offboarding", "resign", "exit process", "last day", "clearance"] },
            { label: "Onboarding Process", keywords: ["onboarding", "new joiner", "welcome", "orientation"] },
            { label: "E-invoices", keywords: ["e-invoice", "einvoice", "invoice", "upload invoice"] },
            { label: "IT Support", keywords: ["it support", "helpdesk", "computer", "laptop", "reset password"] },
            { label: "Company Policies", keywords: ["company policy", "policies", "regulations"] },
        ];

        const results = [];
        for (const userId of userIds) {
            // Number of unique chats
            const [[chats]] = await connection.execute(
                'SELECT COUNT(DISTINCT chat_id) AS chatCount FROM chat_logs WHERE user_id = ?',
                [userId]
            );
            // Number of queries/messages sent
            const [[queries]] = await connection.execute(
                'SELECT COUNT(*) AS queryCount FROM chat_logs WHERE user_id = ?',
                [userId]
            );
            // Last interaction timestamp
            const [[last]] = await connection.execute(
                'SELECT MAX(timestamp) AS lastInteraction FROM chat_logs WHERE user_id = ?',
                [userId]
            );
            // Number of feedbacks given
            const [[feedbacks]] = await connection.execute(
                'SELECT COUNT(*) AS feedbackCount FROM chat_logs WHERE user_id = ? AND feedback IS NOT NULL AND feedback != ""',
                [userId]
            );
            // --- Satisfaction Calculation ---
            const [[helpfuls]] = await connection.execute(
                'SELECT COUNT(*) AS helpfulCount FROM chat_logs WHERE user_id = ? AND feedback = "helpful"',
                [userId]
            );
            const [[notHelpfuls]] = await connection.execute(
                'SELECT COUNT(*) AS notHelpfulCount FROM chat_logs WHERE user_id = ? AND feedback = "not helpful"',
                [userId]
            );
            const helpfulCount = helpfuls?.helpfulCount || 0;
            const notHelpfulCount = notHelpfuls?.notHelpfulCount || 0;
            const totalFeedback = helpfulCount + notHelpfulCount;
            const satisfaction = totalFeedback > 0 ? Math.round((helpfulCount / totalFeedback) * 100) : null;
            // Topic analytics
            const [userMessages] = await connection.execute(
                'SELECT user_message FROM chat_logs WHERE user_id = ?',
                [userId]
            );
            const topicCounts = TOPIC_KEYWORDS.map(topic => {
                let count = 0;
                for (const row of userMessages) {
                    const msg = (row.user_message || "").toLowerCase();
                    if (topic.keywords.some(kw => msg.includes(kw))) {
                        count++;
                    }
                }
                return { label: topic.label, count };
            }).filter(t => t.count > 0);

            results.push({
                userId,
                chatCount: chats?.chatCount || 0,
                queryCount: queries?.queryCount || 0,
                lastInteraction: last?.lastInteraction || null,
                feedbackCount: feedbacks?.feedbackCount || 0,
                topicAnalytics: topicCounts,
                satisfaction
            });
        }

        await connection.end();
        res.json(results);
    } catch (error) {
        console.error('Error fetching all users analytics:', error);
        res.status(500).json({ message: "Internal server error" });
    }
};

module.exports = {
    getAllUsers,
    getUserById,
    loginUser,
    createUser,
    decodeJWT,
    hashPassword,
    generateAccessToken,
    getCurrentUser,
    adminCreateUser,
    adminUpdateUser,
    getUserAnalytics,
    getUserChats,
    getUserFeedback,
    getCompanyAnalytics,
    getAllUsersAnalytics
};