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
    
    if (avatar) avatar.classList.remove('speaking');
    if (avatarOpen) avatarOpen.classList.add('avatar-hidden');
}

function sendMessage() {
  const input = document.getElementById("messageInput");
  const message = input.value.trim();
  const user_id = localStorage.getItem("userId") || "defaultUser";
  // Always use the latest chat_id from localStorage or sessionStorage
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id") || "chat123";

  if (!message) return;

  // Clear welcome message on first message
  clearWelcomeContent();

  // Disable send button
  const sendButton = document.getElementById("sendButton");
  sendButton.disabled = true;
  const fullMessage = `${message} YABABDODD`;

  // Add user message to chat
  addMessage(message, "user");

  // Clear input and reset height
  input.value = "";
  input.style.height = "auto";

  // Show typing indicator with realistic staged status updates
  showTypingIndicator("Retrieving relevant documents...");
  setTimeout(() => updateTypingIndicatorStatus("Analyzing your question..."), 1000);
  setTimeout(() => updateTypingIndicatorStatus("Generating response..."), 2200);
  setTimeout(() => updateTypingIndicatorStatus("Finalizing..."), 3200);

  // Call chatbot API with correct chat_id
  callChatbotAPI(message, user_id, chat_id)
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

// Chat History Sidebar logic
async function getChatHistorySidebar() {
  const userId = localStorage.getItem("userId") || "defaultUser";
  try {
    const response = await fetch(`/api/chatbot/history?user_id=${encodeURIComponent(userId)}`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("token")}`
      }
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const chatLogs = await response.json();
    renderChatHistorySidebar(chatLogs);
  } catch (error) {
    console.error("Error fetching chat history:", error);
    renderChatHistorySidebar([]);
  }
}

function renderChatHistorySidebar(chatLogs) {
  const sidebar = document.getElementById("chatHistorySidebar");
  const list = document.getElementById("chatHistoryList");
  if (!sidebar || !list) return;
  list.innerHTML = "";
  if (!Array.isArray(chatLogs) || chatLogs.length === 0) {
    list.innerHTML = '<div class="chat-history-empty">No chat history found.</div>';
    return;
  }
  chatLogs.forEach(log => {
    const item = document.createElement("div");
    item.className = "chat-history-item";
    item.textContent = log.title || `Chat on ${log.date || log.created_at || "Unknown"}`;
    item.onclick = () => loadChatHistory(log.chat_id);
    list.appendChild(item);
  });
}

function openChatHistorySidebar() {
  const sidebar = document.getElementById("chatHistorySidebar");
  if (!sidebar) return;
  sidebar.classList.add("open");
  getChatHistorySidebar();
}

function closeChatHistorySidebar() {
  const sidebar = document.getElementById("chatHistorySidebar");
  if (!sidebar) return;
  sidebar.classList.remove("open");
}

function loadChatHistory(chatId) {
  // Load the selected chat's messages and display in main chat area
  const userId = localStorage.getItem("userId") || "defaultUser";
  fetch(`/api/chatbot/history/${encodeURIComponent(chatId)}?user_id=${encodeURIComponent(userId)}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("token")}`
    }
  })
    .then(res => res.ok ? res.json() : Promise.reject(res))
    .then(chatLogs => {
      // Replace chat UI with selected chat's messages
      const chatMessages = document.getElementById("chatMessages");
      chatMessages.innerHTML = "";
      if (Array.isArray(chatLogs) && chatLogs.length > 0) {
        chatLogs.forEach(msg => {
          addMessage(msg.message, msg.sender === "user" ? "user" : "bot");
        });
      } else {
        chatMessages.innerHTML = `<div class='welcome-message'><h2>No messages in this chat.</h2></div>`;
      }
      closeChatHistorySidebar();
    })
    .catch(() => {
      alert("Failed to load chat history.");
      closeChatHistorySidebar();
    });
}



async function get_frequentmsg() {
  // Fallback suggestions
  const fallback = [
    "What are the pantry rules?",
    "What is the leave policy?",
    "How do I upload e-invoices?"
  ];

  try {
    const response = await fetch("http://localhost:3000/frequent", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Frequent Messages:", data);

    if (Array.isArray(data) && data.length > 0) {
      updateSuggestions(data);
    } else {
      updateSuggestions(fallback);
    }
  } catch (error) {
    console.error("Error fetching frequent messages:", error);
    // Use fallback if API is unreachable or any error occurs
    updateSuggestions(fallback);
  }
}


async function callChatbotAPI(message,
  User_id,
  Chat_id
) {
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
        user_id :User_id,
        chat_id: Chat_id
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
  const user_id = localStorage.getItem("userId") || "defaultUser";
  // Always use the latest chat_id from localStorage or sessionStorage
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id") || "chat123";
  try {
    const response = await fetch("/api/chatbot/history", {
      method: "POST", // Use POST to allow a body
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ user_id, chat_id })
    });

    let result;
    try {
      result = await response.json();
    } catch (jsonErr) {
      // If not JSON, fallback to text
      result = await response.text();
    }

    if (response.ok) {
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
          <div id="suggestionsContainer" class="suggestions"></div>
        </div>
      `;
    } else {
      let errorMsg = 'Unknown error';
      if (typeof result === 'object' && result !== null && result.message) {
        errorMsg = result.message;
      } else if (typeof result === 'string') {
        errorMsg = result;
      }
      alert('Failed to clear chat: ' + errorMsg);
    }
  } catch (error) {
    console.error("Error clearing chat history:", error);
    alert('Error clearing chat history: ' + error.message);
  }
}

// Clear welcome message and demo content
function clearWelcomeContent() {
  const welcomeMsg = document.querySelector(".welcome-message");
  if (welcomeMsg) {
    welcomeMsg.remove();
  }
}

function updateSuggestions(suggestionsArray) {
  const container = document.getElementById("suggestionsContainer");
  if (!container) return;

  // Fallback suggestions
  const fallback = [
    "What are the pantry rules?",
    "What is the leave policy?",
    "How do I upload e-invoices?"
  ];

  // Use fallback if suggestionsArray is not an array or empty
  const suggestions = Array.isArray(suggestionsArray) && suggestionsArray.length > 0
    ? suggestionsArray
    : fallback;

  container.innerHTML = ""; // Clear old suggestions

  suggestions.forEach(text => {
    const div = document.createElement("div");
    div.className = "suggestion";
    div.textContent = text;
    div.onclick = () => sendSuggestion(text);
    container.appendChild(div);
  });
}

async function fetchSuggestions(query = "") {
  try {
    const res = await fetch(`/api/chatbot/suggestions?query=${encodeURIComponent(query)}`, {
      headers: { Authorization: `Bearer ${token}` }
    });

    const data = await res.json();
    if (data && Array.isArray(data.suggestions)) {
      updateSuggestions(data.suggestions);
    }
  } catch (err) {
    console.error("Failed to fetch suggestions:", err);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  if (window.innerWidth <= 768) {
    document.getElementById("sidebar").classList.add("collapsed");
  }

  // Fetch default welcome suggestions
  get_frequentmsg();  // fixed syntax
});



// Show typing indicator with status message
function showTypingIndicator(status = "Getting documents...") {
  const messagesContainer = document.getElementById("chatMessages");
  const typingDiv = document.createElement("div");
  typingDiv.className = "typing-indicator show";
  typingDiv.id = "typingIndicator";
  typingDiv.innerHTML = `
    <div class="ai-message-avatar"></div>
    <div class="typing-bubble">
      <span class="typing-status" id="typingStatus">${status}</span>
      <span class="typing-dots">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </span>
    </div>
  `;
  messagesContainer.appendChild(typingDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Update typing indicator status message
function updateTypingIndicatorStatus(status) {
  const statusSpan = document.getElementById("typingStatus");
  if (statusSpan) statusSpan.textContent = status;
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
      <div class="message-content user-message">${escapeHtml(text)}</div>
      <div class="user-message-avatar"></div>
    `;
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
      <div class="message-content ai-message">
        ${escapeHtml(text)}${imagesHtml}
        <button class="copy-btn" title="Copy response" onclick="copyMessage(this)">📋</button>
      </div>
      <div class="feedback-buttons">
        <button class="feedback-btn positive" onclick="handleFeedback(this, true)">
          👍 Helpful
        </button>
        <button class="feedback-btn negative" onclick="handleFeedback(this, false)">
          👎 Not Helpful
        </button>
      </div>`;
  }

  if (sender === "bot" && text) {
    setTimeout(() => speakMessage(text), 100);

    // Disable all previous feedback buttons
    const allFeedbackGroups = chatMessages.querySelectorAll('.feedback-buttons');
    allFeedbackGroups.forEach(group => {
      group.querySelectorAll('.feedback-btn').forEach(btn => {
        btn.disabled = true;
      });
    });

    // Enable feedback buttons for the latest bot message
    setTimeout(() => {
      const latestFeedbackGroup = messageDiv.querySelector('.feedback-buttons');
      if (latestFeedbackGroup) {
        latestFeedbackGroup.querySelectorAll('.feedback-btn').forEach(btn => {
          btn.disabled = false;
        });
      }
    }, 0);
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

  // Check admin before redirecting
  fetch('/api/users/me', {
    headers: { Authorization: `Bearer ${token}` }
  })
    .then(res => res.ok ? res.json() : null)
    .then(user => {
      if (user && user.role === 'admin') {
        window.location.href = "/fileupload.html";
      } else {
        showNoAccessPopup();
      }
    })
    .catch(() => {
      showNoAccessPopup();
    });
}

// Show a non-intrusive popup in the middle of the page for no access
function showNoAccessPopup() {
  if (document.getElementById('noAccessPopup')) return;
  const popup = document.createElement('div');
  popup.id = 'noAccessPopup';
  popup.textContent = "You do not have access to the file upload feature.";
  popup.style.position = "fixed";
  popup.style.top = "50%";
  popup.style.left = "50%";
  popup.style.transform = "translate(-50%, -50%)";
  popup.style.background = "#222";
  popup.style.color = "#FFD700";
  popup.style.padding = "22px 44px";
  popup.style.borderRadius = "18px";
  popup.style.fontSize = "1.2rem";
  popup.style.boxShadow = "0 4px 24px rgba(255,215,0,0.18)";
  popup.style.zIndex = "9999";
  popup.style.opacity = "0.97";
  document.body.appendChild(popup);
  setTimeout(() => {
    popup.remove();
  }, 3500);
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
    if (overlay) overlay.classList.remove("active");
  } else {
    // Mobile: hide sidebar by default
    if (!sidebar.classList.contains("collapsed")) {
      if (overlay) overlay.classList.add("active");
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
        window.googleTTS?.pause();
        if (avatar) avatar.classList.add('muted');
        if (toggleButton) {
            toggleButton.classList.add('muted');
            toggleButton.innerHTML = '<i class="fas fa-volume-mute"></i> Muted';
        }
    } else {
        window.googleTTS?.resume();
        if (avatar) avatar.classList.remove('muted');
        if (toggleButton) {
            toggleButton.classList.remove('muted');
            toggleButton.innerHTML = '<i class="fas fa-volume-up"></i> Unmuted';
        }
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
        if (avatar) avatar.classList.add('speaking');
        isCurrentlySpeaking = true;
      
        // Use Google Cloud TTS instead of ResponsiveVoice
        if (window.googleTTS) {
          await window.googleTTS.speak(text, {
            voice: 'en-GB-Standard-A',
            languageCode: 'en-GB',
            volume: isMuted ? 0 : 1,
            onend: () => {
              currentSpeechText = null;
              isCurrentlySpeaking = false;
              stopAvatarAnimation();
            },
            onstart: () => {
              isCurrentlySpeaking = true;
              if (avatar) avatar.classList.add('speaking');
            }
          });
        } else {
          console.warn('Google TTS not loaded');
          stopAvatarAnimation();
        }
        
    } catch (error) {
        console.error('Speech Error:', error);
        stopAvatarAnimation();
    }
}

// Add function to cancel current speech
function cancelSpeech() {
    if (window.googleTTS) {
        window.googleTTS.cancel();
    }
    currentSpeechText = null;
    isCurrentlySpeaking = false;
    stopAvatarAnimation();
}

function handleFeedback(button, isPositive) {
    const messageContainer = button.closest('.message');
    if (!messageContainer) return;

    const feedbackGroup = button.closest('.feedback-buttons');
    if (!feedbackGroup) return;

    feedbackGroup.querySelectorAll('.feedback-btn').forEach(btn => {
        btn.classList.remove('selected');
    });

    button.classList.add('selected');

    // Get bot response text
    const bot_response = messageContainer.querySelector('.message-content').textContent.trim();

    // Get the previous user message (search backwards for .message-user)
    let user_message = '';
    let prev = messageContainer.previousElementSibling;
    while (prev) {
        if (prev.classList.contains('message-user')) {
            user_message = prev.querySelector('.message-content').textContent.trim();
            break;
        }
        prev = prev.previousElementSibling;
    }

    fetch('/api/chatbot/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
            feedback: isPositive ? 'helpful' : 'not helpful'
        })
    }).catch(error => console.error('Error sending feedback:', error));

    feedbackGroup.querySelectorAll('.feedback-btn').forEach(btn => {
        btn.disabled = true;
    });
}

// Render user message
function appendUserMessage(userMessage) {
  const userMessageDiv = document.createElement('div');
  userMessageDiv.className = 'message message-user';
  userMessageDiv.innerHTML = `
    <div class="message-content user-message">${escapeHtml(userMessage)}</div>
    <div class="user-message-avatar"></div>
  `;
  chatMessages.appendChild(userMessageDiv);
}

// Render AI message with copy button
function appendAIMessage(aiMessage) {
  const aiMessageDiv = document.createElement('div');
  aiMessageDiv.className = 'message message-ai';
  aiMessageDiv.innerHTML = `
    <div class="ai-message-avatar"></div>
    <div class="message-content ai-message">
      ${escapeHtml(aiMessage)}
      <button class="copy-btn" title="Copy response" onclick="copyMessage(this)">📋</button>
    </div>
  `;
  chatMessages.appendChild(aiMessageDiv);
}

// Only trigger copy on click, not on keydown/keyup/keypress
window.copyMessage = function(btn) {
  // Prevent multiple triggers
  if (btn.disabled) return;
  btn.disabled = true;

  const content = btn.parentElement.textContent.replace('📋', '').replace('✔', '').trim();
  navigator.clipboard.writeText(content).then(() => {
    btn.classList.add('copied');
    btn.innerHTML = '<span style="font-size:18px;">✔</span> Copied';
    showCopyPopup();
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = '<span style="font-size:18px;">📋</span> Copy';
      btn.disabled = false;
    }, 1200);
  });
};

function showCopyPopup() {
  const popup = document.getElementById('copyPopup');
  if (!popup) return;
  popup.classList.add('show');
  clearTimeout(window._copyPopupTimeout);
  window._copyPopupTimeout = setTimeout(() => {
    popup.classList.remove('show');
  }, 1400);
}

// Helper to get current user info (populated by the HTML script)
function getCurrentUser() {
  return window.currentUser || null;
}

// Helper to check if current user is admin
function isAdmin() {
  return getCurrentUser() && getCurrentUser().role === 'admin';
}

// Profile dropdown logic
function populateProfileSection() {
  const user = getCurrentUser();
  if (!user) return;
  // Sidebar summary
  document.getElementById("profileName").textContent = user.username || "User";
  document.getElementById("profileRole").textContent = user.role || "";
  // Dropdown
  document.getElementById("profileDropdownName").textContent = user.username || "";
  document.getElementById("profileDropdownEmail").textContent = user.email || "";
  document.getElementById("profileDropdownRole").textContent = user.role || "";
}
function toggleProfileDropdown(event) {
  event.stopPropagation();
  const profile = document.getElementById("sidebarProfile");
  profile.classList.toggle("active");
  // Close on outside click
  if (profile.classList.contains("active")) {
    document.addEventListener("click", closeProfileDropdownOnClick);
  }
}
function closeProfileDropdownOnClick(e) {
  const profile = document.getElementById("sidebarProfile");
  if (!profile.contains(e.target)) {
    profile.classList.remove("active");
    document.removeEventListener("click", closeProfileDropdownOnClick);
  }
}

// Wait for user info to be loaded and then populate profile
document.addEventListener("DOMContentLoaded", function () {
  // ...existing code...

// Chat History button logic
window.addEventListener("DOMContentLoaded", function () {
  setTimeout(() => {
    const chatHistoryBtn = document.getElementById("chatHistoryNavBtn");
    const closeBtn = document.getElementById("closeChatHistoryBtn");
    if (chatHistoryBtn) chatHistoryBtn.onclick = openChatHistorySidebar;
    if (closeBtn) closeBtn.onclick = closeChatHistorySidebar;
  }, 200);
});
  // Wait for window.currentUser to be set (from HTML inline script)
  let tries = 0;
  function tryPopulateProfile() {
    if (window.currentUser) {
      populateProfileSection();
    } else if (tries < 20) {
      tries++;
      setTimeout(tryPopulateProfile, 100);
    }
  }
  tryPopulateProfile();
  // ...existing code...
});

// Show the confirmation popup for clearing chat
function showClearChatConfirmPopup() {
  const popup = document.getElementById("clearChatConfirmPopup");
  if (popup) popup.style.display = "flex";
}
// Hide the confirmation popup
function hideClearChatConfirmPopup() {
  const popup = document.getElementById("clearChatConfirmPopup");
  if (popup) popup.style.display = "none";
}

// Attach popup logic on DOMContentLoaded
window.addEventListener("DOMContentLoaded", function () {
  const clearBtn = document.getElementById("confirmClearChatBtn");
  const cancelBtn = document.getElementById("cancelClearChatBtn");
  if (clearBtn) clearBtn.onclick = function() {
    hideClearChatConfirmPopup();
    clearChatHistory();
  };
  if (cancelBtn) cancelBtn.onclick = hideClearChatConfirmPopup;
});

// Replace sidebar clear chat click to show popup
function triggerClearChat() {
  showClearChatConfirmPopup();
}

// Start a new chat: call backend to get a new chat_id, reset UI and session
async function startNewChat() {
  try {
    const token = localStorage.getItem("token");
    const response = await fetch("/api/chatbot/newchat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ user_id: localStorage.getItem("userId") || "defaultUser" })
    });

    // Defensive: check content-type before parsing as JSON
    const contentType = response.headers.get("content-type") || "";
    let data = null;
    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      const text = await response.text();
      console.error("Non-JSON response from /api/chatbot/newchat:", text);
      alert("Failed to start a new chat. Server did not return valid JSON.");
      return;
    }

    // Log for debugging
    console.log("/api/chatbot/newchat response:", data);

    if (response.ok && data && data.chat_id) {
      localStorage.setItem("chat_id", data.chat_id);
      sessionStorage.setItem("chat_id", data.chat_id);
      sessionStorage.removeItem("chatHistory");
      // Reset chat UI
      const chatMessages = document.getElementById("chatMessages");
      chatMessages.innerHTML = `
        <div class="welcome-message">
          <h2>Welcome to AI Assistant</h2>
          <p>
            I'm here to help you with any questions or tasks you might have.
            Feel free to ask me anything!
          </p>
          <div id="suggestionsContainer" class="suggestions"></div>
        </div>
      `;
      get_frequentmsg();
    } else {
      // Log error details for debugging
      console.error("Failed to start new chat. Response:", data, "Status:", response.status);
      alert("Failed to start a new chat. Please try again.\n" + (data && data.message ? data.message : ""));
    }
  } catch (error) {
    console.error("Error in startNewChat:", error);
    alert("Error starting new chat: " + error.message);
  }
}

// On page load, start a new chat if not already present (persist chat_id across reloads)
window.addEventListener("DOMContentLoaded", async function () {
  // Use localStorage for chat_id persistence
  if (!localStorage.getItem("chat_id")) {
    await startNewChat();
  } else {
    // If chat_id exists in localStorage, sync it to sessionStorage for compatibility
    sessionStorage.setItem("chat_id", localStorage.getItem("chat_id"));
  }
  if (window.innerWidth <= 768) {
    document.getElementById("sidebar").classList.add("collapsed");
  }
  // Fetch default welcome suggestions
  get_frequentmsg();
});

// Open profile modal
function openProfileModal(event) {
  event.stopPropagation();
  const modal = document.getElementById("profileModal");
  if (!modal) return;
  // Populate user info
  const user = getCurrentUser();
  document.getElementById("modalProfileName").textContent = user?.username || "";
  document.getElementById("modalProfileId").textContent = user?.id ? `User ID: ${user.id}` : "";
  document.getElementById("modalProfileEmail").textContent = user?.email || "";
  document.getElementById("modalProfileRole").textContent = user?.role || "";
  // Render analytics
  renderProfileAnalytics();
  modal.style.display = "flex";
  // Close on outside click
  setTimeout(() => {
    document.addEventListener("mousedown", closeProfileModalOnOutsideClick);
  }, 0);
}

// Close profile modal
function closeProfileModal() {
  const modal = document.getElementById("profileModal");
  if (modal) modal.style.display = "none";
  document.removeEventListener("mousedown", closeProfileModalOnOutsideClick);
}

// Close modal if click outside dialog
function closeProfileModalOnOutsideClick(e) {
  const modal = document.getElementById("profileModal");
  if (!modal) return;
  const dialog = modal.querySelector(".profile-modal-dialog");
  if (modal.style.display !== "none" && !dialog.contains(e.target)) {
    closeProfileModal();
  }
}

// Render analytics dashboard in modal
function renderProfileAnalytics() {
  const dashboard = document.getElementById("profileAnalyticsDashboard");
  if (!dashboard) return;

  // Show loading state
  dashboard.innerHTML = `<div>Loading analytics...</div>`;

  const user = getCurrentUser();
  if (!user || !user.id) {
    dashboard.innerHTML = `<div>Unable to load analytics.</div>`;
    return;
  }

  fetch(`/api/users/analytics?userId=${encodeURIComponent(user.id)}`, {
    headers: { Authorization: `Bearer ${localStorage.getItem("token")}` }
  })
    .then(res => res.ok ? res.json() : null)
    .then(data => {
      if (!data) {
        dashboard.innerHTML = `<div>Unable to load analytics.</div>`;
        return;
      }
      dashboard.innerHTML = `
        <div><strong>Number of Chats:</strong> ${data.chatCount}</div>
        <div><strong>Queries Sent:</strong> ${data.queryCount}</div>
        <div><strong>Feedback Given:</strong> ${data.feedbackCount}</div>
        <div><strong>Last Interaction:</strong> ${data.lastInteraction ? new Date(data.lastInteraction).toLocaleString() : 'N/A'}</div>
        <div style="margin-top:10px;color:#a08a3c;font-size:0.97em;">(Analytics are based on your chat history)</div>
        <div style="margin-top:8px;color:#888;font-size:0.95em;"><em>Click for more details</em></div>
      `;
    })
    .catch(() => {
      dashboard.innerHTML = `<div>Unable to load analytics.</div>`;
    });
}

