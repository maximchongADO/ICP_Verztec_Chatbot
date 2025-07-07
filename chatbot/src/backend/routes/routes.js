const userRoute = require("./userRoute.js");
const chatbotRoute = require("./chatbotRoute.js");
const fileUploadRoute = require("./uploadroute.js");
const ttsRoute = require("./ttsRoute.js");
const userController = require('../controllers/userController');
const authenticateToken = require('../middleware/authenticateToken');

const route = (app, upload) => {
    userRoute(app, upload);
    chatbotRoute(app, upload);
    fileUploadRoute(app, upload);
    ttsRoute(app);
    app.get('/api/users/me', authenticateToken, userController.getCurrentUser);
    // Add more routes here as needed
};

module.exports = route;