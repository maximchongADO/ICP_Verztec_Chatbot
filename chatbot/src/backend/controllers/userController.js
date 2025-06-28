const User = require('../models/user.js');
const bcrypt = require('bcrypt');
const fs = require('fs');
const jwt = require('jsonwebtoken');
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

module.exports = {
    getAllUsers,
    getUserById,
    loginUser,
    createUser,
    decodeJWT,
    hashPassword,
    generateAccessToken,
    getCurrentUser
};