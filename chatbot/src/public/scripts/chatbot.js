// Check if user is authenticated
const token = localStorage.getItem("token");
if (!token) {
  window.location.href = "/login.html";
}

function handleKeyPress(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

function sendMessage() {
  const input = document.getElementById("messageInput");
  const message = input.value.trim();

  if (!message) return;

  // Clear welcome message on first message
  clearWelcomeContent();

  // Disable send button
  const sendButton = document.getElementById("sendButton");
  sendButton.disabled = true;

  // Add user message to chat
  addMessage(message, "user");

  // Clear input and reset height
  input.value = "";
  input.style.height = "auto";

  // Show typing indicator
  showTypingIndicator();

  // Call real chatbot API
  callChatbotAPI(message)
    .then((response) => {
      // Remove typing indicator
      hideTypingIndicator();

      // Add bot response
      if (response.success !== false) {
        addMessage(response.message, "bot");
      } else {
        addMessage(response.message || "Sorry, I encountered an error.", "bot");
      }
    })
    .catch((error) => {
      console.error("Chatbot API error:", error);
      // Remove typing indicator
      hideTypingIndicator();

      // Add error message
      addMessage(
        "Sorry, I'm having trouble responding right now. Please try again.",
        "bot"
      );
    })
    .finally(() => {
      // Re-enable send button
      sendButton.disabled = false;
    });
}

async function callChatbotAPI(message) {
  // Get chat history from session storage
  const chatHistory = JSON.parse(sessionStorage.getItem("chatHistory") || "[]");

  const response = await fetch("/api/chatbot/message", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      message: message,
      chat_history: chatHistory,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const result = await response.json();

  // Update chat history in session storage
  if (result.success) {
    chatHistory.push(message);
    // Keep only last 10 messages for context
    if (chatHistory.length > 10) {
      chatHistory.splice(0, chatHistory.length - 10);
    }
    sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  }

  return result;
}

// Add function to clear chat history
async function clearChatHistory() {
  try {
    const response = await fetch("/api/chatbot/history", {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (response.ok) {
      // Clear session storage
      sessionStorage.removeItem("chatHistory");

      // Clear chat messages on screen and show welcome message
      const chatMessages = document.getElementById("chatMessages");
      chatMessages.innerHTML = `
        <div class="welcome-message">
          <h2>Welcome to AI Assistant</h2>
          <p>
            I'm here to help you with any questions or tasks you might have.
            Feel free to ask me anything!
          </p>
          <div class="suggestions">
            <div
              class="suggestion"
              onclick="sendSuggestion('How can I reset my password?')"
            >
              How can I reset my password?
            </div>
            <div
              class="suggestion"
              onclick="sendSuggestion('What are the office hours?')"
            >
              What are the office hours?
            </div>
            <div
              class="suggestion"
              onclick="sendSuggestion('How do I submit a support ticket?')"
            >
              How do I submit a support ticket?
            </div>
            <div
              class="suggestion"
              onclick="sendSuggestion('Where can I find company policies?')"
            >
              Where can I find company policies?
            </div>
          </div>
        </div>
      `;
    }
  } catch (error) {
    console.error("Error clearing chat history:", error);
  }
}

// Clear welcome message and demo content
function clearWelcomeContent() {
  const welcomeMsg = document.querySelector(".welcome-message");
  if (welcomeMsg) {
    welcomeMsg.remove();
  }
}

// Show typing indicator
function showTypingIndicator() {
  const messagesContainer = document.getElementById("chatMessages");
  const typingDiv = document.createElement("div");
  typingDiv.className = "typing-indicator show";
  typingDiv.id = "typingIndicator";
  typingDiv.innerHTML = `
    <div class="ai-message-avatar">AI</div>
    <div class="typing-dots">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;

  messagesContainer.appendChild(typingDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
  const typingIndicator = document.getElementById("typingIndicator");
  if (typingIndicator) {
    typingIndicator.remove();
  }
}

// Send suggestion message
function sendSuggestion(text) {
  const messageInput = document.getElementById("messageInput");
  if (messageInput) {
    messageInput.value = text;
    sendMessage();
  }
}

function addMessage(text, sender) {
  const chatMessages = document.getElementById("chatMessages");
  const messageDiv = document.createElement("div");

  if (sender === "user") {
    messageDiv.className = "message message-user";
    messageDiv.innerHTML = `
      <div class="message-content user-message">
        ${escapeHtml(text)}
      </div>
    `;
  } else {
    messageDiv.className = "message message-ai";
    messageDiv.innerHTML = `
      <div class="ai-message-avatar">AI</div>
      <div class="message-content ai-message">
        ${escapeHtml(text)}
      </div>
    `;
  }

  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return messageDiv; // Return the element for potential removal
}

// Helper function to escape HTML to prevent XSS
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("userId");
  window.location.href = "/login.html";
}

// Toggle sidebar functionality
function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.querySelector(".sidebar-overlay");

  sidebar.classList.toggle("collapsed");

  if (window.innerWidth <= 768) {
    overlay.classList.toggle("active");
  }
}

// Auto-resize textarea based on content
function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

// Export chat functionality
function exportChat() {
  const messages = document.querySelectorAll(".message");
  let chatText = "AI Assistant Chat Export\n" + "=".repeat(50) + "\n\n";
  chatText += `Exported on: ${new Date().toLocaleString()}\n\n`;

  messages.forEach((message) => {
    const isUser = message.classList.contains("message-user");
    const content = message.querySelector(".message-content");
    if (content) {
      const text = content.textContent || content.innerText;
      chatText += `${isUser ? "You" : "AI Assistant"}: ${text}\n\n`;
    }
  });

  // Create and download file
  const blob = new Blob([chatText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `chat-export-${new Date().toISOString().split("T")[0]}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Handle file upload (placeholder function)
function handleFileUpload(event) {
  const files = event.target.files;
  if (files.length > 0) {
    // Add your file upload logic here
    console.log("Files selected:", files);
    // For now, just show a message
    addMessage(
      `File(s) selected: ${Array.from(files)
        .map((f) => f.name)
        .join(", ")}`,
      "bot"
    );
  }
}

// Initialize sidebar state on page load
document.addEventListener("DOMContentLoaded", function () {
  // Close sidebar on mobile by default
  if (window.innerWidth <= 768) {
    document.getElementById("sidebar").classList.add("collapsed");
  }
});

// Handle window resize
window.addEventListener("resize", function () {
  const sidebar = document.getElementById("sidebar");
  const overlay = document.querySelector(".sidebar-overlay");

  if (window.innerWidth > 768) {
    // Desktop: show sidebar, hide overlay
    sidebar.classList.remove("collapsed");
    overlay.classList.remove("active");
  } else {
    // Mobile: hide sidebar by default
    if (!sidebar.classList.contains("collapsed")) {
      overlay.classList.add("active");
    }
  }
});
