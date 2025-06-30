import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ChatProvider } from "./hooks/useChat";
import "./index.css";

const canvas = document.querySelector('canvas');
if (canvas) {
  canvas.addEventListener('webglcontextlost', function (event) {
    event.preventDefault();
    alert('WebGL context lost. Please reload the page.');
  }, false);
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ChatProvider>
      <App />
    </ChatProvider>
  </React.StrictMode>
);
