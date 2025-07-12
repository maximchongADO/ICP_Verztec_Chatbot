const jwt = require("jsonwebtoken");
require("dotenv").config();

const authenticateToken = (req,res,next) => {
    //get token and check if its valid
    const authHeader = req.headers["authorization"];
    const token = authHeader && authHeader.split(" ")[1];
    console.log(token)
    if (!token) {
        console.log('No token provided');
        return res.status(403).json({ 
            success: false, 
            message: 'No authentication token provided' 
        });
    }

    //verify token
    jwt.verify(token, process.env.ACCESS_TOKEN_SECRET || process.env.JWT_SECRET || 'fallback-secret-key', (err,user) => {
        if (err) {
            console.log('Token verification failed:', err.message);
            return res.status(403).json({ 
                success: false, 
                message: 'Invalid or expired token' 
            });
        }


        const authorizedRoles = {
            
        };
        
        const userRole = user.role;
        const requestEndpoint = req.url.split('?')[0]; // Remove query parameters if present
        const method = req.method;

        console.log('Request Endpoint:', requestEndpoint); // debugging log
        console.log('User Role:', userRole); // debugging log

        // Allow chatbot endpoints for all authenticated users
        if (requestEndpoint.startsWith('/api/chatbot/')) {
            req.user = user;
            return next();
        }

        // Iterate over authorized roles and handle dynamic segments
        const authorizedRole = Object.entries(authorizedRoles).find(
            ([endpoint, roles]) => {
                const [endpointMethod, path] = endpoint.split(' ');
                const pathPattern = path.replace(/:\w+/g, '\\w+'); // Replace: param with regex pattern
                const regex = new RegExp(`^${pathPattern}$`);
                return endpointMethod === method && regex.test(requestEndpoint) && roles.includes(userRole);
            }
        );

        console.log(authorizedRole)
        if (authorizedRole === false) {
            console.log('Role not authorized for this endpoint'); 
            return res.status(403).json({ message: 'Forbidden' });
        }

        req.user = user; // user contains userId and role
        next();
    })
}

// Middleware to restrict to admin only
function requireAdmin(req, res, next) {
    if (!req.user || req.user.role !== 'admin') {
        return res.status(403).json({ success: false, message: 'Admin access required' });
    }
    next();
}

// Middleware to restrict to admin or manager
function requireAdminOrManager(req, res, next) {
    if (!req.user || !['admin', 'manager'].includes(req.user.role)) {
        return res.status(403).json({ success: false, message: 'Admin or Manager access required' });
    }
    next();
}

module.exports = authenticateToken;
module.exports.requireAdmin = requireAdmin;
module.exports.requireAdminOrManager = requireAdminOrManager;