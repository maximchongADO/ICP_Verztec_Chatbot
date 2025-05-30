const userRoute = require("./userRoute.js");
const chatbotRoute = require("./chatbotRoute.js");
const fileUploadRoute = require("./uploadroute.js");

const route = (app, upload) => {
    userRoute(app, upload);
    chatbotRoute(app, upload);
    fileUploadRoute(app, upload);
    // Add more routes here as needed
};

module.exports = route;