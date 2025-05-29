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


// Add these new variables at the top of the file with other global variables
let currentSpeechText = null;
let isCurrentlySpeaking = false;

function stopAvatarAnimation() {
    const avatar = document.getElementById('chatbotAvatar');
    const avatarOpen = document.getElementById('avatarOpen');
    
    if (currentMouthInterval) {
        clearInterval(currentMouthInterval);
        currentMouthInterval = null;
    }
    
    avatar.classList.remove('speaking');
    avatarOpen.classList.add('avatar-hidden');
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

  // Call chatbot API
  callChatbotAPI(message)
    .then((response) => {
      // Remove typing indicator
      hideTypingIndicator();

      // Add bot response
      if (response) {
        addMessage(response.message, "bot"); 
        if (Array.isArray(response.images) && response.images.length > 1) {
          addMessage(response.images[1], "bot"); // optional: specify sender
          
        }
        else {
          addMessage("No additional images available.", "bot");
        }
        //addMessage(response.images[1])// Pass the whole response object
      } else {
        addMessage("Sorry, I received an invalid response. Please try again.", "bot");
      }
    })
    .catch((error) => {
      console.error("Chatbot API error:", error);
      // Remove typing indicator
      hideTypingIndicator();

      // Add error message
      addMessage(
        error,
        "bot"
      );
    })
    .finally(() => {
      // Re-enable send button
      sendButton.disabled = false;
    });
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

  // Call chatbot API
  callChatbotAPI(message)
    .then((response) => {
      // Remove typing indicator
      hideTypingIndicator();

      // Add bot response
      if (response) {
        addMessage(response.message, "bot");
        if (Array.isArray(response.images) && response.images.length > 0) {
          sendImages(response.images);
        }
      } else {
        addMessage("Sorry, I received an invalid response. Please try again.", "bot");
      }
    })
    .catch((error) => {
      console.error("Chatbot API error:", error);
      // Remove typing indicator
      hideTypingIndicator();

      // Add error message
      addMessage(
        error,
        "bot"
      );
    })
    .finally(() => {
      // Re-enable send button
      sendButton.disabled = false;
    });
}







async function callChatbotAPI(message) {
  const chatHistory = JSON.parse(sessionStorage.getItem("chatHistory") || "[]");

  try {
    const response = await fetch("http://localhost:3000/chatbot", {
      method: "POST",
      credentials: 'include',
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify({
        message: message,
        chat_history: chatHistory,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("API Response:", data);

    if (data && data.message) {
      return {
        success: true,
        message: data.message,
        images: data.images || [], // Ensure images is an array
      };
    } else {
      throw new Error("Invalid response format from chatbot");
    }
  } catch (error) {
    console.error("Chatbot API Error:", error);
    throw error;
  }
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
    <div class="ai-message-avatar"></div>
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

function sendImages(images) {
  if (!Array.isArray(images) || images.length === 0) return;

  // Each image will be passed as a filename (e.g., "example.png")
  // We pass it to `addMessage` which now handles raw image filenames
  addMessage({ message: "", images: images }, "bot");
}


function addMessage(textOrResponse, sender) {
  let text = textOrResponse;
  let images = [];

  // Check if it's an object with message and images
  if (typeof textOrResponse === "object" && textOrResponse !== null && "message" in textOrResponse) {
    text = textOrResponse.message?.trim() || "";
    images = textOrResponse.images || [];
  } 
  
  // NEW: If it's a plain image filename string like "example.png"
  else if (typeof textOrResponse === "string" && /\.(png|jpg|jpeg|gif|bmp)$/i.test(textOrResponse.trim())) {
    images = [textOrResponse.trim()];
    text = "";  // No text message, only image
  }

  // If no message or image, do nothing
  if (!text && images.length === 0) {
    console.error("Empty message and no images received");
    return;
  }

  const chatMessages = document.getElementById("chatMessages");
  const messageDiv = document.createElement("div");

  if (sender === "user") {
    messageDiv.className = "message message-user";
    messageDiv.innerHTML = `
      <div class="message-content user-message">${escapeHtml(text)}</div>`;
  } else {
    messageDiv.className = "message message-ai";
    let imagesHtml = "";

    if (Array.isArray(images) && images.length > 0) {
      imagesHtml = `<div class="ai-message-images">` +
        images.map(src => {
          const filename = escapeHtml(src.split('/').pop() || src);
          return `<img src="/data/images/${filename}" alt="${filename}" class="chat-image" />`;
        }).join("") +
        `</div>`;
    }

    messageDiv.innerHTML = `
      <div class="ai-message-avatar"></div>
      <div class="message-content ai-message">${escapeHtml(text)}${imagesHtml}</div>`;
  }

  if (sender === "bot" && text) {
    setTimeout(() => speakMessage(text), 100);
  }

  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return messageDiv;
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

function handleFileUpload(event) {
  // Prevent default file input behavior
  event.preventDefault();
  
  // Redirect to the file upload page
  window.location.href = "/fileupload.html";
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

// Add mute toggle functionality
let isMuted = false;

function toggleMute() {
    isMuted = !isMuted;
    const toggleButton = document.getElementById('toggleSpeechButton');
    const avatar = document.getElementById('chatbotAvatar');
    
    if (isMuted) {
        responsiveVoice.pause();  // Pause instead of cancel
        avatar.classList.add('muted');
        toggleButton.classList.add('muted');
        toggleButton.innerHTML = '<i class="fas fa-volume-mute"></i> Muted';
    } else {
        responsiveVoice.resume();  // Resume if paused
        avatar.classList.remove('muted');
        toggleButton.classList.remove('muted');
        toggleButton.innerHTML = '<i class="fas fa-volume-up"></i> Unmuted';
    }
}

let currentMouthInterval = null; // Add this at the top level of your file

async function speakMessage(text) {
    if (!text || !text.trim()) return;
    
    const avatar = document.getElementById('chatbotAvatar');
    const avatarOpen = document.getElementById('avatarOpen');
    
    // Store the current text being spoken
    currentSpeechText = text;
    
    try {
        avatar.classList.add('speaking');
        isCurrentlySpeaking = true;
        
        // Clear any existing animation
        if (currentMouthInterval) {
            clearInterval(currentMouthInterval);
        }
        
        // Start new animation
        currentMouthInterval = setInterval(() => {
            if (isCurrentlySpeaking) {
                avatarOpen.classList.toggle('avatar-hidden');
            }
        }, 200);
        
        responsiveVoice.speak(text, "UK English Female", {
            onend: () => {
                currentSpeechText = null;
                isCurrentlySpeaking = false;
                stopAvatarAnimation();
            },
            onstart: () => {
                isCurrentlySpeaking = true;
                avatar.classList.add('speaking');
            },
            volume: isMuted ? 0 : 1  // Set volume based on mute state
        });
        
    } catch (error) {
        console.error('Speech Error:', error);
        stopAvatarAnimation();
    }
}

// Add function to cancel current speech
function cancelSpeech() {
    responsiveVoice.cancel();
    currentSpeechText = null;
    isCurrentlySpeaking = false;
    stopAvatarAnimation();
}