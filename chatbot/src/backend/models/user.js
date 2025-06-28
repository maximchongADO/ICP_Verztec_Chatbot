const mysql = require('mysql2/promise');
const dbConfig = require('../database/dbConfig.js');

class User {
    constructor(id, username, email, password, role) {
        this.id = id;
        this.username = username;
        this.email = email;
        this.password = password;
        this.role = role;
    }

    static toUserObj(row) {
        return row ? new User(
            row.id,
            row.username,
            row.email,
            row.password,
            row.role
        ) : null;
    }

    static async query(sql, params = []) {
        const connection = await mysql.createConnection(dbConfig);
        try {
            const [rows] = await connection.execute(sql, params);
            return rows;
        } finally {
            await connection.end();
        }
    }

    static async getAllUsers() {
        const sql = 'SELECT id, username, email, role FROM Users';
        const rows = await this.query(sql);
        return rows.map(row => this.toUserObj(row));
    }

    static async getUserById(id) {
        const sql = 'SELECT id, username, role FROM Users WHERE id = ?';
        const rows = await this.query(sql, [id]);
        return this.toUserObj(rows[0]);
    }

    static async getUserByIdFull(id) {
        const sql = 'SELECT id, username, email, role FROM Users WHERE id = ?';
        const rows = await this.query(sql, [id]);
        return rows[0];
    }

    static async getUserByEmail(email) {
        const sql = 'SELECT * FROM Users WHERE email = ?';
        const rows = await this.query(sql, [email]);
        return rows[0];
    }

    static async getUserByUsername(username) {
        const sql = 'SELECT * FROM Users WHERE username = ?';
        const rows = await this.query(sql, [username]);
        return rows[0];
    }

    static async createUser(user) {
        const sql = 'INSERT INTO Users (username, email, password, role) VALUES (?, ?, ?, ?)';
        const result = await this.query(sql, [
            user.username,
            user.email,
            user.password,
            user.role
        ]);
        
        return this.getUserById(result.insertId);
    }

    static async updateUser(id, fields) {
        if (!id || !fields || Object.keys(fields).length === 0) return null;
        const allowed = ['username', 'email', 'password', 'role'];
        const set = [];
        const params = [];
        for (const key of allowed) {
            if (fields[key] !== undefined) {
                set.push(`${key} = ?`);
                params.push(fields[key]);
            }
        }
        if (set.length === 0) return null;
        params.push(id);
        const sql = `UPDATE Users SET ${set.join(', ')} WHERE id = ?`;
        await this.query(sql, params);
        return this.getUserByIdFull(id);
    }
}

module.exports = User;
