// ==========================================
// CHAT PERSISTENCE SYSTEM
// ==========================================

// Clean up old chat data from localStorage (keep only last 10 chats)
function cleanupOldChatData() {
  const chatKeys = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('chat_messages_')) {
      chatKeys.push({
        key: key,
        timestamp: parseInt(localStorage.getItem(key + '_timestamp')) || 0
      });
    }
  }
  
  // Sort by timestamp (newest first) and keep only the last 10
  chatKeys.sort((a, b) => b.timestamp - a.timestamp);
  if (chatKeys.length > 10) {
    const keysToRemove = chatKeys.slice(10);
    keysToRemove.forEach(item => {
      localStorage.removeItem(item.key);
      localStorage.removeItem(item.key + '_timestamp');
    });
  }
}

// Save current chat messages to localStorage
function saveChatToLocalStorage() {
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id");
  if (!chat_id) return;
  
  const chatMessages = document.getElementById("chatMessages");
  if (!chatMessages) return;
  
  // Clean up old chat data periodically
  cleanupOldChatData();
  
  const messages = [];
  const messageElements = chatMessages.querySelectorAll('.message');
  
  messageElements.forEach(messageEl => {
    if (messageEl.classList.contains('welcome-message') || 
        messageEl.classList.contains('typing-indicator')) {
      return; // Skip welcome messages and typing indicators
    }
    
    const isUser = messageEl.classList.contains('message-user');
    const messageContent = messageEl.querySelector('.message-content');
    const messageImages = messageEl.querySelectorAll('.chat-image, .ai-message-images img');
    
    if (messageContent) {
      // Get the text content, removing copy button text
      let textContent = messageContent.textContent || '';
      // Remove copy button text and other UI elements
      textContent = textContent.replace(/📋.*?Copy/g, '').replace(/✓.*?Helpful/g, '').replace(/👎.*?Not Helpful/g, '').trim();
      
      const messageData = {
        text: textContent,
        sender: isUser ? 'user' : 'bot',
        timestamp: Date.now(),
        images: Array.from(messageImages).map(img => img.src.split('/').pop() || img.src)
      };
      
      // For bot messages, preserve the original HTML content (excluding UI buttons)
      if (!isUser) {
        let htmlContent = messageContent.innerHTML;
        // Remove copy button and feedback buttons
        htmlContent = htmlContent.replace(/<button[^>]*copy-btn[^>]*>.*?<\/button>/g, '');
        htmlContent = htmlContent.replace(/<div[^>]*feedback-buttons[^>]*>.*?<\/div>/g, '');
        messageData.html = htmlContent;
      }
      
      messages.push(messageData);
    }
  });
  
  // Store messages and timestamp for cleanup
  localStorage.setItem(`chat_messages_${chat_id}`, JSON.stringify(messages));
  localStorage.setItem(`chat_messages_${chat_id}_timestamp`, Date.now().toString());
}

// Load chat messages from localStorage
function loadChatFromLocalStorage() {
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id");
  if (!chat_id) {
    console.log("No chat_id found for loading messages");
    return false;
  }
  
  const savedMessages = localStorage.getItem(`chat_messages_${chat_id}`);
  if (!savedMessages) {
    console.log("No saved messages found for chat_id:", chat_id);
    return false;
  }
  
  try {
    const messages = JSON.parse(savedMessages);
    const chatMessages = document.getElementById("chatMessages");
    
    if (!chatMessages) {
      console.error("Chat messages container not found");
      return false;
    }
    
    if (messages.length > 0) {
      console.log(`Loading ${messages.length} saved messages...`);
      
      // Clear any existing content (including welcome message)
      chatMessages.innerHTML = '';
      
      // Restore messages
      messages.forEach((messageData, index) => {
        try {
          addMessageFromStorage(messageData);
        } catch (error) {
          console.error(`Error loading message ${index}:`, error, messageData);
        }
      });
      
      // Scroll to bottom
      setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
      }, 100);
      
      return true;
    } else {
      console.log("No messages in saved data");
    }
  } catch (error) {
    console.error("Error loading chat from localStorage:", error);
    // Remove corrupted data
    localStorage.removeItem(`chat_messages_${chat_id}`);
    localStorage.removeItem(`chat_messages_${chat_id}_timestamp`);
  }
  
  return false;
}

// Add a message from storage (specialized version of addMessage)
function addMessageFromStorage(messageData) {
  const chatMessages = document.getElementById("chatMessages");
  const messageDiv = document.createElement("div");
  
  if (messageData.sender === "user") {
    messageDiv.className = "message message-user";
    messageDiv.innerHTML = `
      <div class="message-content user-message">${escapeHtml(messageData.text)}</div>
      <div class="user-message-avatar"></div>
    `;
  } else {
    messageDiv.className = "message message-ai";
    
    let imagesHtml = "";
    if (messageData.images && messageData.images.length > 0) {
      imagesHtml = `<div class="ai-message-images">` +
        messageData.images.map(filename => {
          // Handle different image sources - if it starts with 'images/', serve from public, otherwise from data/images
          const imageSrc = filename.startsWith('images/') ? `/${filename}` : `/data/images/${filename}`;
          return `<img src="${imageSrc}" alt="${filename}" class="chat-image" />`;
        }).join("") +
        `</div>`;
    }
    
    messageDiv.innerHTML = `
      <div class="ai-message-avatar"></div>
      <div class="message-content ai-message">
        ${messageData.html || escapeHtml(messageData.text)}${imagesHtml}
        <button class="copy-btn" title="Copy response" onclick="copyMessage(this)">📋</button>
      </div>
      <div class="feedback-buttons">
        <button class="feedback-btn positive" onclick="handleFeedback(this, true)">
          👍 Helpful
        </button>
        <button class="feedback-btn negative" onclick="handleFeedback(this, false)">
          👎 Not Helpful
        </button>
      </div>
    `;
  }
  
  chatMessages.appendChild(messageDiv);
}

// Clear chat messages from localStorage for current chat
function clearChatFromLocalStorage() {
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id");
  if (chat_id) {
    localStorage.removeItem(`chat_messages_${chat_id}`);
    localStorage.removeItem(`chat_messages_${chat_id}_timestamp`);
  }
}

// Save chat ID to localStorage with timestamp
function saveChatIdToLocalStorage(chat_id) {
  localStorage.setItem("chat_id", chat_id);
  sessionStorage.setItem("chat_id", chat_id);
  localStorage.setItem("chat_id_timestamp", Date.now().toString());
}

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
  
  // Update button state after any key press
  setTimeout(updateActionButton, 0);
}

// Add these new variables at the top of the file with other global variables
let currentSpeechText = null;
let isCurrentlySpeaking = false;

// Function to check if avatar is active and should handle TTS
function isAvatarActive() {
    // Check if draggable avatar exists and is visible
    if (window.draggableAvatar && window.draggableAvatar.isAvatarVisible()) {
        console.log('🎭 Draggable avatar is active');
        return true;
    }
    
    // Check if avatar iframe exists
    const avatarIframe = document.querySelector('iframe[src*="avatar"]');
    if (avatarIframe) {
        console.log('🎭 Avatar iframe is active');
        return true;
    }
    
    // Check if avatar window is open
    if (window.avatarWindow && !window.avatarWindow.closed) {
        console.log('🎭 Avatar window is active');
        return true;
    }
    
    console.log('🎭 No avatar is active');
    return false;
}

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

  // Disable action button
  const actionButton = document.getElementById("actionButton");
  if (actionButton) actionButton.disabled = true;
  // const fullMessage = `${message} YABABDODD`;

  // Add user message to chat
  addMessage(message, "user");

  // Clear input and reset height
  input.value = "";
  input.style.height = "auto";
  
  // Update action button state after clearing input
  updateActionButton();

  // Show typing indicator with realistic staged status updates
  showTypingIndicator("Retrieving relevant documents...");
  setTimeout(() => updateTypingIndicatorStatus("Analyzing your question..."), 1000);
  setTimeout(() => updateTypingIndicatorStatus("Generating response..."), 2200);
  setTimeout(() => updateTypingIndicatorStatus("Finalizing..."), 3200);

  // Call chatbot API with correct chat_id - send original message
  callChatbotAPI(message, user_id, chat_id) // Don't send fullMessage
    .then((response) => {
      // Remove typing indicator
      hideTypingIndicator();

      // Add bot response
      if (response) {
        // Check if tool_used is true and add confirmation buttons
        const messageData = {
          message: response.message,
          images: response.images || [],
          sources: response.sources || [],
          tool_used: response.tool_used || false,
          tool_identified: response.tool_identified || "none",
          tool_confidence: response.tool_confidence || "",
          original_message: message // Store original message for reprocessing
        };
        
        addMessage(messageData, "bot");
        
        // Log tool information for debugging
        if (response.tool_used) {
          console.log(`Tool detected - Type: ${response.tool_identified}, Confidence: ${response.tool_confidence}`);
        }
        
        // Handle sources if available (only if not tool_used)
        if (!response.tool_used && Array.isArray(response.sources) && response.sources.length > 0) {
          addSourcesToMessage(response.sources);
        }
        
        // REMOVE this block to prevent duplicate image rendering:
        // if (Array.isArray(response.images) && response.images.length > 0) {
        //   sendImages(response.images);
        // }
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
      // Re-enable action button
      const actionButton = document.getElementById("actionButton");
      if (actionButton) actionButton.disabled = false;
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
  const list = document.getElementById("chatHistoryList");
  if (!list) return;
  list.innerHTML = "";
  if (!Array.isArray(chatLogs) || chatLogs.length === 0) {
    list.innerHTML = '<div class="chat-history-empty">No chat history found.</div>';
    return;
  }
  chatLogs.forEach(log => {
    const item = document.createElement("div");
    item.className = "chat-history-item";
    item.setAttribute('data-chat-id', log.chat_id);

    // Create chat history item structure
    const icon = document.createElement("div");
    icon.className = "chat-history-item-icon";
    icon.innerHTML = `<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>`;

    const textDiv = document.createElement("div");
    textDiv.className = "chat-history-item-text";
    // Prefer chat_name, then title, then fallback
    textDiv.textContent =
      log.chat_name && typeof log.chat_name === "string" && log.chat_name.trim()
        ? log.chat_name
        : (log.title || `Chat on ${log.date || log.created_at || "Unknown"}`); // <-- fixed: added missing closing parenthesis

    // Move timestamp under the chat name/title
    let timeDiv = null;
    if (log.date || log.created_at) {
      timeDiv = document.createElement("div");
      timeDiv.className = "chat-history-item-time";
      const date = new Date(log.date || log.created_at);
      timeDiv.textContent = date.toLocaleDateString();
      textDiv.appendChild(document.createElement("br"));
      textDiv.appendChild(timeDiv);
    }

    const actionsDiv = document.createElement("div");
    actionsDiv.className = "chat-history-item-actions";
    
    const deleteBtn = document.createElement("button");
    deleteBtn.innerHTML = "×";
    deleteBtn.title = "Delete this chat";
    deleteBtn.style.cssText = "background: none; border: none; color: #d4b24c; cursor: pointer; padding: 6px 8px; border-radius: 6px; font-size: 12px; font-weight: 500; z-index: 1000; position: relative;";
    
    deleteBtn.onclick = function(event) {
      console.log("Delete button clicked for chat:", log.chat_id);
      event.stopPropagation();
      event.preventDefault();
      deleteChatHistory(log.chat_id);
    };
    
    actionsDiv.appendChild(deleteBtn);

    item.appendChild(icon);
    item.appendChild(textDiv);
    // Remove the old timeDiv append here (if present)
    // item.appendChild(timeDiv);
    item.appendChild(actionsDiv);

    item.onclick = () => loadChatHistory(log.chat_id);
    list.appendChild(item);
  });
}

// Remove separate sidebar functions since we're integrating into main sidebar
function openChatHistorySidebar() {
  // Chat history is now always visible in the main sidebar
  getChatHistorySidebar();
}

function closeChatHistorySidebar() {
  // No longer needed since chat history is integrated
}

function loadChatHistory(chatId) {
  // Update current chat_id to the selected chat
  saveChatIdToLocalStorage(chatId);
  
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
          addMessage(msg.message, msg.sender === "user" ? "user" : "bot", true); // true = isHistorical
        });
        
        // Save the loaded messages to localStorage for persistence
        saveChatToLocalStorage();
      } else {
        chatMessages.innerHTML = `<div class='welcome-message'><h2>No messages in this chat.</h2></div>`;
      }
      
      // Mark the selected chat as active
      const chatItems = document.querySelectorAll('.chat-history-item');
      chatItems.forEach(item => {
        item.classList.remove('active');
        // Check if this item corresponds to the loaded chat
        if (item.getAttribute('data-chat-id') === chatId) {
          item.classList.add('active');
        }
      });
    })
    .catch(() => {
      alert("Failed to load chat history.");
    });
}

function deleteChatHistory(chatId) {
  console.log("deleteChatHistory called with chatId:", chatId);
  
  // Store the chatId for use in the confirmation handlers
  window.pendingDeleteChatId = chatId;
  
  // Show the custom confirmation popup
  const popup = document.getElementById('deleteChatConfirmPopup');
  if (popup) {
    popup.style.display = 'flex';
  } else {
    console.error("Delete confirmation popup not found, falling back to browser confirm");
    // Fallback to browser confirm if popup doesn't exist
    const confirmDelete = window.confirm("Are you sure you want to delete this chat? This action cannot be undone.");
    if (confirmDelete) {
      performDeleteChat(chatId);
    }
  }
}

function performDeleteChat(chatId) {
  console.log("Proceeding with delete for chatId:", chatId);
  
  const userId = localStorage.getItem("userId") || "defaultUser";
  fetch(`/api/chatbot/history/${encodeURIComponent(chatId)}?user_id=${encodeURIComponent(userId)}`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("token")}`
    }
  })
    .then(res => {
      console.log("Delete response status:", res.status);
      if (res.ok) {
        // Show success message (optional)
        console.log("Chat deleted successfully!");
        // Refresh the chat history list
        getChatHistorySidebar();
      } else {
        throw new Error(`Failed to delete chat: ${res.status}`);
      }
    })
    .catch((error) => {
      console.error("Delete error:", error);
      alert("Failed to delete chat history: " + error.message);
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
    // Get user_id for regional filtering
    const user_id = localStorage.getItem("userId") || "defaultUser";
    
    const response = await fetch("http://localhost:3000/frequent", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: user_id
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Frequent Messages for user", user_id, ":", data);

    if (Array.isArray(data) && data.length > 2) {
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
        sources: data.sources || [], // Include sources data
        tool_used: data.tool_used || false, // Include tool_used flag
        tool_identified: data.tool_identified || "none", // Include tool identification
        tool_confidence: data.tool_confidence || "" // Include tool confidence
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
  console.log("clearChatHistory called");
  const user_id = localStorage.getItem("userId") || "defaultUser";
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id") || "chat123";
  
  console.log("Clearing chat - user_id:", user_id, "chat_id:", chat_id);
  
  try {
    // First, try to delete the current chat from the database
    if (chat_id && chat_id !== "chat123") { // Don't try to delete default/fallback chat_id
      try {
        console.log("Deleting current chat from database:", chat_id);
        const response = await fetch(`/api/chatbot/history/${encodeURIComponent(chat_id)}?user_id=${encodeURIComponent(user_id)}`, {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${localStorage.getItem("token")}`
          }
        });
        
        if (response.ok) {
          console.log("Successfully deleted chat from database");
        } else {
          console.warn(`Failed to delete chat from database: ${response.status} (continuing anyway)`);
        }
      } catch (dbError) {
        console.warn("Failed to delete chat from database (continuing anyway):", dbError);
        // Continue with local cleanup even if database deletion fails
      }
    }
    
    // Clear local storage and session data
    sessionStorage.removeItem("chatHistory");
    clearChatFromLocalStorage();
    
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
    
    // Load suggestions
    get_frequentmsg();
    
    // Start a new chat (which will get a new chat_id from the server)
    await startNewChat();
    
    // Refresh the chat history sidebar to reflect any changes
    getChatHistorySidebar();
    
    console.log("Chat cleared successfully and deleted from database");
    
  } catch (error) {
    console.error("Error clearing chat history:", error);
    alert('Error clearing chat history: ' + error.message);
  }
}

// Periodically save chat messages (every 30 seconds)
setInterval(function() {
  if (localStorage.getItem("chat_id")) {
    saveChatToLocalStorage();
  }
}, 30000);

// Save chat when page is about to unload
window.addEventListener("beforeunload", function() {
  saveChatToLocalStorage();
});

// Debug function to log persistence status
function debugPersistence() {
  const chat_id = localStorage.getItem("chat_id");
  const saved_messages = localStorage.getItem(`chat_messages_${chat_id}`);
  console.log("Chat Persistence Debug:", {
    chat_id: chat_id,
    has_saved_messages: !!saved_messages,
    message_count: saved_messages ? JSON.parse(saved_messages).length : 0,
    localStorage_keys: Object.keys(localStorage).filter(k => k.startsWith('chat_'))
  });
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


// Enhanced text formatting function for policy responses
function formatBoldText(text) {
  if (!text) return '';
  
  // Convert **text** to <strong>text</strong> for proper bold formatting
  let formatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Handle single asterisks for emphasis if needed
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  
  // Handle line breaks for better readability
  formatted = formatted.replace(/\n/g, '<br>');
  
  // Format bullet points with proper HTML structure
  formatted = formatted.replace(/^- (.+)$/gm, '<li>$1</li>');
  
  // Wrap consecutive list items in ul tags
  formatted = formatted.replace(/(<li>.*?<\/li>)(\s*<br>\s*<li>.*?<\/li>)*/g, function(match) {
    return '<ul>' + match.replace(/<br>\s*/g, '') + '</ul>';
  });
  
  // Format numbered lists
  formatted = formatted.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  
  // Clean up any double line breaks
  formatted = formatted.replace(/<br>\s*<br>/g, '<br>');
  
  return formatted;
}

function addMessage(textOrResponse, sender, isHistorical = false) {
  let text = textOrResponse;
  let images = [];
  let tool_used = false;
  let tool_identified = "none";
  let tool_confidence = "";
  let original_message = null;

  // Check if it's an object with message and images
  if (typeof textOrResponse === "object" && textOrResponse !== null && "message" in textOrResponse) {
    text = textOrResponse.message?.trim() || "";
    images = textOrResponse.images || [];
    tool_used = textOrResponse.tool_used || false;
    tool_identified = textOrResponse.tool_identified || "none";
    tool_confidence = textOrResponse.tool_confidence || "";
    original_message = textOrResponse.original_message || null;
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

    // Add confirmation buttons if tool_used is true
    let confirmationHtml = "";
    if (tool_used && original_message) {
      // Customize confirmation message based on tool type
      let confirmationText = "Do you want me to proceed with this action?";
      let yesButtonText = "✓ Yes, proceed";
      let noButtonText = "✗ No, cancel";
      let additionalInputs = "";
      

    if (tool_identified === "raise_to_hr") {
      confirmationText = `<div class="tool-confirm-title"><span class="tool-icon hr">👤</span>Escalate to HR</div><div class="tool-confirm-desc">This will escalate your issue to <b>Human Resources</b>. Please provide additional details about the incident:</div>`;
      yesButtonText = "✓ Yes, escalate to HR";
      noButtonText = "✗ No, cancel";
      additionalInputs = `<div class="incident-details-section"><label for="incidentDetails" class="incident-label">Incident Details:</label><textarea id="incidentDetails" class="incident-textarea" placeholder="Describe the incident in detail (when, who, what happened)..." rows="1"></textarea><div class="incident-note"><small>All information will be included in your HR escalation request and handled confidentially.</small></div></div>`;
    } else if (tool_identified === "schedule_meeting") {
      confirmationText = `
        <div class="tool-confirm-title">
          <span class="tool-icon meeting">📅</span>
          Schedule a Meeting
        </div>
        <div class="tool-confirm-desc">
          This will schedule a meeting. Do you want to proceed?
        </div>
      `;
      yesButtonText = "✓ Yes, schedule meeting";
      noButtonText = "✗ No, cancel";
    }

    confirmationHtml = `
      <div class="tool-confirmation modern" data-tool-type="${tool_identified}" data-tool-confidence="${tool_confidence}" data-original-message="${escapeHtml(original_message)}">
        ${confirmationText}
        ${additionalInputs}
        <div class="confirmation-buttons">
          <button class="confirm-btn yes" onclick="handleToolConfirmation(this, true)">
            ${yesButtonText}
          </button>
          <button class="confirm-btn no" onclick="handleToolConfirmation(this, false)">
            ${noButtonText}
          </button>
        </div>
      </div>
    `;
  }

  messageDiv.innerHTML = `
    <div class="ai-message-avatar"></div>
    <div class="message-content ai-message">
      ${formatBoldText(text)}${imagesHtml}
      ${confirmationHtml}
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

  if (sender === "bot" && text && !tool_used && !isHistorical) {
    console.log('🤖 ========== BOT MESSAGE TTS CHECK ==========');
    console.log('🤖 Bot message detected, checking TTS status...');
    console.log('🤖 isMuted status:', typeof isMuted !== 'undefined' ? isMuted : 'undefined');
    console.log('🤖 localStorage tts_muted:', localStorage.getItem('tts_muted'));
    
    // Check if there's a mute state preventing TTS
    if (typeof isMuted !== 'undefined' && isMuted) {
        console.error('🤖 ❌ TTS is muted! Bot message will not generate TTS.');
        console.error('🤖 ❌ This is likely why TTS stopped working after first stop.');
        return; // Don't continue with TTS if muted
    }
    
    // Check if avatar is available and active
    const avatarActive = isAvatarActive();
    console.log('🤖 Avatar active:', avatarActive);
    console.log('🤖 Message text:', text.substring(0, 50) + '...');
    
    if (avatarActive) {
      // Generate TTS first, then send audio to avatar (no duplication)
      console.log('🎭 Generating TTS for avatar (single source)');
      
      // Show stop button when starting TTS generation
      showStopTTSButton();
      isTTSPlaying = true;
      
      try {
        // Generate TTS with lipsync for avatar
        const token = localStorage.getItem("token");
        fetch("/api/tts/synthesize-enhanced", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            text: text,
            voice: 'en-GB-Standard-A',
            languageCode: 'en-GB',
            generateLipSyncData: true,
            facialExpression: 'default',
            animation: 'Talking_1'
          }),
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            console.log('🎭 TTS generated successfully, sending to avatar');
            // Send the generated audio and lipsync to avatar
            if (typeof sendMessageToAvatar === 'function') {
              sendMessageToAvatar({
                type: 'tts_with_lipsync',
                text: text,
                audio: data.audio,
                lipsync: data.lipSyncData
              });
            }
          } else {
            console.error('🎭 TTS generation failed, falling back to local TTS');
            // Fallback to local TTS
            speakMessage(text);
          }
        })
        .catch(error => {
          console.error('🎭 TTS generation error, falling back to local TTS:', error);
          // Fallback to local TTS
          speakMessage(text);
        });
      } catch (error) {
        console.error('🎭 TTS setup error, falling back to local TTS:', error);
        speakMessage(text);
      }
    } else {
      // Fallback to local TTS if no avatar is active
      console.log('🔊 Using local TTS (no avatar active)');
      // Show stop button for local TTS too
      showStopTTSButton();
      isTTSPlaying = true;
      setTimeout(() => speakMessage(text), 100);
    }

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
  
  // Save chat to localStorage after adding message
  saveChatToLocalStorage();
  
  return messageDiv;
}

// Function to add source document links to the latest bot message
function addSourcesToMessage(sources) {
  const chatMessages = document.getElementById("chatMessages");
  const latestBotMessage = chatMessages.querySelector(".message-ai:last-child .message-content");
  
  if (!latestBotMessage || !Array.isArray(sources) || sources.length === 0) {
    return;
  }
  
  // Create sources container
  const sourcesDiv = document.createElement("div");
  sourcesDiv.className = "message-sources";
  
  // Create header
  const headerDiv = document.createElement("div");
  headerDiv.className = "sources-header";
  headerDiv.textContent = "References";
  
  // Create sources list
  const listDiv = document.createElement("div");
  listDiv.className = "sources-list";
  
  sources.forEach(source => {
    const sourceItem = document.createElement("div");
    sourceItem.className = "source-item";
    
    // Determine file type and icon
    const fileType = getFileType(source.file_path);
    const iconData = getFileIcon(fileType);
    
    if (source.is_clickable && source.file_path) {
      sourceItem.classList.add("clickable");
      sourceItem.addEventListener("click", () => openSourceDocument(source.file_path, sourceItem));
    }

    // Create icon
    const iconDiv = document.createElement("div");
    iconDiv.className = "source-icon";
    iconDiv.textContent = iconData.icon;
    
    // Create content container
    const contentDiv = document.createElement("div");
    contentDiv.className = "source-content";
    
    // Create name
    const nameDiv = document.createElement("div");
    nameDiv.className = "source-name";
    nameDiv.textContent = source.name;
    
    // Create type
    const typeDiv = document.createElement("div");
    typeDiv.className = "source-type";
    typeDiv.textContent = iconData.type;
    
    contentDiv.appendChild(nameDiv);
    contentDiv.appendChild(typeDiv);
    
    // Create action (only show for clickable items)
    if (source.is_clickable) {
      const actionDiv = document.createElement("div");
      actionDiv.className = "source-action";
      actionDiv.textContent = "View";
      sourceItem.appendChild(iconDiv);
      sourceItem.appendChild(contentDiv);
      sourceItem.appendChild(actionDiv);
    } else {
      sourceItem.appendChild(iconDiv);
      sourceItem.appendChild(contentDiv);
    }
    
    listDiv.appendChild(sourceItem);
  });
  
  sourcesDiv.appendChild(headerDiv);
  sourcesDiv.appendChild(listDiv);
  latestBotMessage.appendChild(sourcesDiv);
}

// Helper function to determine file type
function getFileType(filePath) {
  if (!filePath) return 'unknown';
  
  const ext = filePath.split('.').pop().toLowerCase();
  switch (ext) {
    case 'pdf':
      return 'pdf';
    case 'doc':
    case 'docx':
      return 'word';
    case 'ppt':
    case 'pptx':
      return 'powerpoint';
    case 'xls':
    case 'xlsx':
      return 'excel';
    case 'txt':
      return 'text';
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'gif':
      return 'image';
    default:
      return 'document';
  }
}

// Helper function to get file icon and type description
function getFileIcon(fileType) {
  const icons = {
    pdf: { icon: '📄', type: 'PDF Document' },
    word: { icon: '📝', type: 'Word Document' },
    powerpoint: { icon: '📊', type: 'PowerPoint Presentation' },
    excel: { icon: '📈', type: 'Excel Spreadsheet' },
    text: { icon: '📄', type: 'Text Document' },
    image: { icon: '🖼️', type: 'Image File' },
    document: { icon: '📄', type: 'Document' },
    unknown: { icon: '📄', type: 'Document' }
  };
  
  return icons[fileType] || icons.unknown;
}

// Enhanced function to open source document
function openSourceDocument(filePath, sourceItem = null) {
  try {
    console.log("Opening source document:", filePath);
    
    // Add loading state
    if (sourceItem) {
      sourceItem.classList.add("loading");
      const actionDiv = sourceItem.querySelector(".source-action");
      if (actionDiv) {
        actionDiv.textContent = "Opening...";
      }
    }
    
    // Extract filename from the full path
    let fileName = '';
    if (filePath.includes('\\')) {
      // Windows path
      fileName = filePath.split('\\').pop();
    } else if (filePath.includes('/')) {
      // Unix path
      fileName = filePath.split('/').pop();
    } else {
      // Just filename
      fileName = filePath;
    }
    
    if (fileName) {
      // Use the new document serving endpoint
      const documentUrl = `/documents/${encodeURIComponent(fileName)}`;
      
      // Test if the file exists first
      fetch(documentUrl, { method: 'HEAD' })
        .then(response => {
          if (response.ok) {
            window.open(documentUrl, '_blank');
            console.log("Successfully opened document URL:", documentUrl);
            
            // Show success feedback
            if (sourceItem) {
              sourceItem.classList.remove("loading");
              const actionDiv = sourceItem.querySelector(".source-action");
              if (actionDiv) {
                actionDiv.textContent = "Opened ✓";
                setTimeout(() => {
                  actionDiv.textContent = "Click to open";
                }, 2000);
              }
            }
          } else {
            throw new Error(`File not found (${response.status})`);
          }
        })
        .catch(error => {
          console.error("Error accessing document:", error);
          handleDocumentError(sourceItem, error.message);
        });
    } else {
      throw new Error("Could not extract filename from path");
    }
  } catch (error) {
    console.error("Error opening source document:", error);
    handleDocumentError(sourceItem, error.message);
  }
}

// Helper function to handle document errors
function handleDocumentError(sourceItem, errorMessage) {
  if (sourceItem) {
    sourceItem.classList.remove("loading");
    sourceItem.classList.add("error");
    
    const actionDiv = sourceItem.querySelector(".source-action");
    if (actionDiv) {
      actionDiv.textContent = "Error opening";
    }
    
    // Show error message
    setTimeout(() => {
      alert(`Unable to open document: ${errorMessage}`);
    }, 100);
    
    // Reset error state after a delay
    setTimeout(() => {
      sourceItem.classList.remove("error");
      const actionDiv = sourceItem.querySelector(".source-action");
      if (actionDiv) {
        actionDiv.textContent = "Click to open";
      }
    }, 3000);
  } else {
    alert(`Unable to open document: ${errorMessage}`);
  }
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

  // Check admin or manager access before redirecting
  fetch('/api/users/me', {
    headers: { 
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  })
    .then(res => {
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      return res.json();
    })
    .then(user => {
      if (user && (user.role === 'admin' || user.role === 'manager')) {
        window.location.href = "/fileupload.html";
      } else {
        showNoAccessPopup();
      }
    })
    .catch((error) => {
      console.error('Error checking admin access:', error);
      showNoAccessPopup();
    });
}

// Show a non-intrusive popup in the middle of the page for no access
function showNoAccessPopup() {
  if (document.getElementById('noAccessPopup')) return;
  const popup = document.createElement('div');
  popup.id = 'noAccessPopup';
  popup.textContent = "Access Denied: File upload is restricted to managers and administrators only.";
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

// Add stop TTS functionality
let currentTTSAudio = null;
let isTTSPlaying = false;

function showStopTTSButton() {
    const stopButton = document.getElementById('stopTtsButton');
    console.log('STOP-TTS: Attempting to show button. Button found:', !!stopButton);
    if (stopButton) {
        stopButton.style.display = 'flex';
        console.log('STOP-TTS: Button display set to flex');
        console.log('STOP-TTS: Button current style:', stopButton.style.display);
        console.log('STOP-TTS: Button visible:', stopButton.offsetWidth > 0 && stopButton.offsetHeight > 0);
    } else {
        console.error('STOP-TTS: Button element not found!');
    }
}

function hideStopTTSButton() {
    const stopButton = document.getElementById('stopTtsButton');
    console.log('STOP-TTS: Attempting to hide button. Button found:', !!stopButton);
    if (stopButton) {
        stopButton.style.display = 'none';
        console.log('STOP-TTS: Button hidden');
    }
}

function stopCurrentTTS() {
    console.log('STOP-TTS: Stopping current TTS...');
    console.log('STOP-TTS: isMuted before stop:', typeof isMuted !== 'undefined' ? isMuted : 'undefined');
    
    // Stop Google TTS
    if (window.googleTTS) {
        window.googleTTS.pause();
        console.log('STOP-TTS: Stopped Google TTS');
    }
    
    // Stop HTML5 audio elements
    const audioElements = document.querySelectorAll('audio');
    audioElements.forEach(audio => {
        if (!audio.paused) {
            audio.pause();
            audio.currentTime = 0;
        }
    });
    console.log('STOP-TTS: Stopped HTML5 audio elements');
    
    // Send stop command to avatar (stops current audio without persistent mute)
    if (typeof sendMessageToAvatar === 'function') {
        sendMessageToAvatar({ type: 'mute_immediate' });
        console.log('STOP-TTS: Sent stop command to avatar');
    }
    
    // Reset TTS state
    isTTSPlaying = false;
    currentTTSAudio = null;
    
    // Hide the stop button
    hideStopTTSButton();
    
    // Make sure isMuted is not accidentally set
    if (typeof isMuted !== 'undefined') {
        console.log('STOP-TTS: isMuted after stop:', isMuted);
        if (isMuted) {
            console.warn('STOP-TTS: ⚠️ isMuted is true - this might prevent future TTS!');
        }
    }
    
    console.log('STOP-TTS: Current TTS stopped successfully');
}

function toggleMute() {
    console.log('=== TOGGLE MUTE CALLED ===');
    console.log('Before toggle - isMuted:', isMuted);
    
    // Toggle the state
    isMuted = !isMuted;
    
    console.log('After toggle - isMuted:', isMuted);
    
    // Get UI elements
    const muteButton = document.getElementById('muteButton');
    const muteIcon = document.getElementById('muteIcon');
    const mutedIcon = document.getElementById('mutedIcon');
    
    if (!muteButton || !muteIcon || !mutedIcon) {
        console.error('MISSING UI ELEMENTS:', {
            muteButton: !!muteButton,
            muteIcon: !!muteIcon, 
            mutedIcon: !!mutedIcon
        });
        return;
    }
    
    if (isMuted) {
        console.log('APPLYING MUTE STATE');
        
        // Stop any current audio
        if (window.googleTTS) {
            window.googleTTS.pause();
        }
        
        const audioElements = document.querySelectorAll('audio');
        audioElements.forEach(audio => {
            if (!audio.paused) {
                audio.pause();
                audio.currentTime = 0;
            }
        });
        
        // Send mute command to avatar
        sendMessageToAvatar({
            type: 'mute_immediate'
        });
        
        // Update UI for muted state
        muteButton.classList.add('muted');
        muteIcon.style.display = 'none';
        mutedIcon.style.display = 'block';
        muteButton.title = 'Unmute TTS';
        
        console.log('MUTE STATE APPLIED');
        
    } else {
        // UNMUTE - Allow new audio to play
        console.log('🔊 Applying unmute...');
        
        // Resume Google TTS
        if (window.googleTTS) {
            window.googleTTS.resume();
            console.log('  ✓ Google TTS resumed');
        }
        
        // Send unmute command to avatar to allow audio
        sendMessageToAvatar({
            type: 'unmute'
        });
        console.log('  ✓ Unmute command sent to avatar');
        
        // Update UI for unmuted state
        muteButton.classList.remove('muted');
        muteIcon.style.display = 'block';
        mutedIcon.style.display = 'none';
        muteButton.title = 'Mute TTS';
        
        console.log('UNMUTE STATE APPLIED');
    }
    
    // Save state to localStorage
    localStorage.setItem('tts_muted', isMuted.toString());
    console.log('Saved to localStorage:', localStorage.getItem('tts_muted'));
    console.log('=== TOGGLE COMPLETE ===');
    console.log('� Mute state saved to localStorage:', isMuted.toString());
    
    // Verify the save worked
    const verification = localStorage.getItem('tts_muted');
    console.log('✓ Verification - localStorage now contains:', verification);
}

// Initialize stop TTS button as hidden
document.addEventListener('DOMContentLoaded', function() {
    hideStopTTSButton();
    console.log('STOP-TTS: Initialized with button hidden');
    
    // Listen for avatar TTS end events
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'avatar_tts_ended') {
            console.log('STOP-TTS: ✅ Received avatar TTS ended event');
            console.log('STOP-TTS: Resetting TTS state and hiding button');
            isTTSPlaying = false;
            hideStopTTSButton();
        }
    });
    
    // Test function to manually show button (for debugging)
    window.testShowStopButton = function() {
        console.log('TEST: Manually showing stop button');
        showStopTTSButton();
    };
    
    window.testHideStopButton = function() {
        console.log('TEST: Manually hiding stop button');
        hideStopTTSButton();
    };
    
    console.log('STOP-TTS: Test functions available - testShowStopButton() and testHideStopButton()');
});

let currentMouthInterval = null; // Add this at the top level of your file

async function speakMessage(text) {
    console.log('TTS: ========== speakMessage CALLED ==========');
    console.log('TTS: Text to speak:', text ? text.substring(0, 50) + '...' : 'null');
    console.log('TTS: isMuted status:', typeof isMuted !== 'undefined' ? isMuted : 'undefined');
    console.log('TTS: isTTSPlaying status:', isTTSPlaying);
    
    if (!text || !text.trim()) {
        console.log('TTS: No text provided, returning');
        return;
    }
    
    // Check if there's a mute state that's preventing TTS
    if (typeof isMuted !== 'undefined' && isMuted) {
        console.error('TTS: ❌ TTS is muted! This is why audio is not playing.');
        console.error('TTS: ❌ isMuted =', isMuted, 'localStorage tts_muted =', localStorage.getItem('tts_muted'));
        return;
    }
    
    console.log('TTS: About to check if should show stop button');
    
    // Check if avatar is available and active
    const avatarActive = isAvatarActive();
    console.log('TTS: Avatar active:', avatarActive);
    
    if (avatarActive) {
        console.log('TTS: Avatar is active, sending message with TTS and lip sync');
        // Always generate TTS for avatar so lipsync works
        try {
            // Show stop button when generating TTS for avatar
            showStopTTSButton();
            isTTSPlaying = true;
            
            const token = localStorage.getItem("token");
            const response = await fetch("/api/tts/synthesize-enhanced", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                    text: text,
                    voice: 'en-GB-Standard-A',
                    languageCode: 'en-GB',
                    generateLipSyncData: true,
                    facialExpression: 'default',
                    animation: 'Talking_1'
                }),
            });

            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    console.log('TTS: TTS generated, sending to avatar');
                    sendMessageToAvatar({
                        type: 'tts_with_lipsync',
                        text: text,
                        audio: data.audio,
                        lipsync: data.lipSyncData
                    });
                    return; // Don't play locally when avatar is active
                }
            }
        } catch (error) {
            console.error('TTS: Failed to generate TTS for avatar:', error);
            // Reset state on error
            isTTSPlaying = false;
            hideStopTTSButton();
        }
    }
    
    // Fallback to local TTS if avatar is not available or TTS failed
    console.log('TTS: Playing TTS locally');
    
    const avatar = document.getElementById('chatbotAvatar');
    
    try {
        if (avatar) avatar.classList.add('speaking');
        isCurrentlySpeaking = true;
        isTTSPlaying = true;
        
        // Show stop button when TTS starts
        showStopTTSButton();
        
        if (window.googleTTS) {
          await window.googleTTS.speak(text, {
            voice: 'en-GB-Standard-A',
            languageCode: 'en-GB',
            volume: 1,
            generateLipSync: false, // Don't need lip sync for local playback
            onend: () => {
              currentSpeechText = null;
              isCurrentlySpeaking = false;
              isTTSPlaying = false;
              hideStopTTSButton(); // Hide stop button when TTS ends
              stopAvatarAnimation();
              console.log('TTS: Local TTS ended');
            },
            onstart: () => {
              isCurrentlySpeaking = true;
              isTTSPlaying = true;
              if (avatar) avatar.classList.add('speaking');
              console.log('TTS: Local TTS started');
            }
          });
        } else {
          console.warn('Google TTS not loaded');
          isTTSPlaying = false;
          hideStopTTSButton();
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
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message message-user';
    userMessageDiv.innerHTML = `
        <div class="user-message-avatar"></div>
        <div class="message-content user-message">${escapeHtml(userMessage)}</div>
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

// Helper to check if current user is admin or manager
function isAdminOrManager() {
  const user = getCurrentUser();
  return user && ['admin', 'manager'].includes(user.role);
}

// Profile dropdown logic
function populateProfileSection() {
  const user = getCurrentUser();
  if (!user) return;
  
  // Sidebar summary - check if elements exist before accessing them
  const profileName = document.getElementById("profileName");
  const profileRole = document.getElementById("profileRole");
  const profileDropdownName = document.getElementById("profileDropdownName");
  const profileDropdownEmail = document.getElementById("profileDropdownEmail");
  const profileDropdownRole = document.getElementById("profileDropdownRole");
  
  if (profileName) profileName.textContent = user.username || "User";
  if (profileRole) profileRole.textContent = user.role || "";
  if (profileDropdownName) profileDropdownName.textContent = user.username || "";
  if (profileDropdownEmail) profileDropdownEmail.textContent = user.email || "";
  if (profileDropdownRole) profileDropdownRole.textContent = user.role || "";
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
  // Initialize chat history on page load
  setTimeout(() => {
    getChatHistorySidebar();
  }, 200);
  
  // Wait for window.currentUser to be set (from HTML inline script)
  let tries = 0;
  function tryPopulateProfile() {
    if (window.currentUser) {
      populateProfileSection();
      
      // Show admin button for admin and manager roles
      const adminBtn = document.getElementById("adminAddUserBtn");
      if (adminBtn && isAdminOrManager()) {
        adminBtn.style.display = "flex";
        // Update button text for managers
        if (window.currentUser.role === 'manager') {
          const btnSpan = adminBtn.querySelector('span');
          if (btnSpan) btnSpan.textContent = 'Manage Users';
        }
      }
    } else if (tries < 20) {
      tries++;
      setTimeout(tryPopulateProfile, 100);
    }
  }
  tryPopulateProfile();
});

// Show the confirmation popup for clearing chat
function showClearChatConfirmPopup() {
  console.log("showClearChatConfirmPopup called");
  const popup = document.getElementById("clearChatConfirmPopup");
  console.log("Popup element found:", popup);
  console.log("Popup display style before:", popup ? popup.style.display : "element not found");
  if (popup) {
    popup.style.display = "flex";
    console.log("Popup display style after:", popup.style.display);
    console.log("Popup should now be visible");
  } else {
    console.error("ERROR: clearChatConfirmPopup element not found in DOM!");
    // Let's check what elements with 'popup' in their ID exist
    const allElements = document.querySelectorAll('[id*="popup"], [id*="Popup"]');
    console.log("Elements with 'popup' in ID:", Array.from(allElements).map(el => el.id));
  }
}
// Hide the confirmation popup
function hideClearChatConfirmPopup() {
  console.log("hideClearChatConfirmPopup called");
  const popup = document.getElementById("clearChatConfirmPopup");
  if (popup) popup.style.display = "none";
}

function hideDeleteChatConfirmPopup() {
  console.log("hideDeleteChatConfirmPopup called");
  const popup = document.getElementById("deleteChatConfirmPopup");
  if (popup) popup.style.display = "none";
}

// Attach popup logic on DOMContentLoaded
window.addEventListener("DOMContentLoaded", function () {
  console.log("DOMContentLoaded - setting up clear chat popup handlers");
  const clearBtn = document.getElementById("confirmClearChatBtn");
  const cancelBtn = document.getElementById("cancelClearChatBtn");
  console.log("Clear button found:", clearBtn);
  console.log("Cancel button found:", cancelBtn);
  if (clearBtn) clearBtn.onclick = function() {
    console.log("Clear chat confirmed by user");
    hideClearChatConfirmPopup();
    clearChatHistory();
  };
  if (cancelBtn) cancelBtn.onclick = hideClearChatConfirmPopup;

  // Set up delete chat popup handlers
  console.log("Setting up delete chat popup handlers");
  const deleteBtn = document.getElementById("confirmDeleteChatBtn");
  const cancelDeleteBtn = document.getElementById("cancelDeleteChatBtn");
  console.log("Delete button found:", deleteBtn);
  console.log("Cancel delete button found:", cancelDeleteBtn);
  if (deleteBtn) deleteBtn.onclick = function() {
    console.log("Delete chat confirmed by user");
    hideDeleteChatConfirmPopup();
    if (window.pendingDeleteChatId) {
      performDeleteChat(window.pendingDeleteChatId);
      window.pendingDeleteChatId = null;
    }
  };
  if (cancelDeleteBtn) cancelDeleteBtn.onclick = function() {
    console.log("Delete chat cancelled by user");
    hideDeleteChatConfirmPopup();
    window.pendingDeleteChatId = null;
  };
});

// Replace sidebar clear chat click to show popup
function triggerClearChat() {
  console.log("triggerClearChat called");
  showClearChatConfirmPopup();
}

// Start a new chat: call backend to get a new chat_id, reset UI and session
async function startNewChat() {
  console.log("startNewChat called");
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
      saveChatIdToLocalStorage(data.chat_id);
      sessionStorage.removeItem("chatHistory");
      // Clear previous chat messages from localStorage
      clearChatFromLocalStorage();
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
      
      // Refresh chat history to show the new chat
      getChatHistorySidebar();
      
      // Remove active state from all chat history items
      const chatItems = document.querySelectorAll('.chat-history-item');
      chatItems.forEach(item => item.classList.remove('active'));
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
  console.log("Page loaded, initializing chat persistence...");
  
  // Debug current state
  debugPersistence();
  
  // Use localStorage for chat_id persistence
  if (!localStorage.getItem("chat_id")) {
    console.log("No chat_id found, starting new chat...");
    await startNewChat();
  } else {
    const chat_id = localStorage.getItem("chat_id");
    console.log("Found existing chat_id:", chat_id);
    
    // If chat_id exists in localStorage, sync it to sessionStorage for compatibility
    sessionStorage.setItem("chat_id", chat_id);
    
    // Try to load saved chat messages
    const messagesLoaded = loadChatFromLocalStorage();
    if (messagesLoaded) {
      console.log("Successfully loaded saved messages from localStorage");
    } else {
      console.log("No saved messages found, showing welcome message");
      // If no saved messages, show welcome message and fetch suggestions
      get_frequentmsg();
    }
  }
  
  if (window.innerWidth <= 768) {
    document.getElementById("sidebar").classList.add("collapsed");
  }
  
  // Debug final state
  setTimeout(() => {
    debugPersistence();
  }, 1000);
});

// Open profile modal
function openProfileModal(event) {
  // Prevent default and stop propagation to ensure click is captured
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  
  console.log("Profile modal opened"); // Debug log
  
  const modal = document.getElementById("profileModal");
  if (!modal) {
    console.error("Profile modal not found");
    return;
  }
  
  // Populate user info
  const user = getCurrentUser();
  const modalProfileName = document.getElementById("modalProfileName");
  const modalProfileId = document.getElementById("modalProfileId");
  const modalProfileEmail = document.getElementById("modalProfileEmail");
  const modalProfileRole = document.getElementById("modalProfileRole");
  
  if (modalProfileName) modalProfileName.textContent = user?.username || "";
  if (modalProfileId) modalProfileId.textContent = user?.id ? `User ID: ${user.id}` : "";
  if (modalProfileEmail) modalProfileEmail.textContent = user?.email || "";
  if (modalProfileRole) modalProfileRole.textContent = user?.role || "";
  
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

// Add direct event listener to the sidebar profile
document.addEventListener('DOMContentLoaded', function() {
  const sidebarProfile = document.getElementById('sidebarProfile');
  if (sidebarProfile) {
    sidebarProfile.addEventListener('click', function(e) {
      openProfileModal(e);
    });
    
    console.log("Added event listener to sidebar profile");
  }
});

// Handle tool confirmation button clicks
function handleToolConfirmation(button, confirmed) {
  const messageDiv = button.closest('.message');
  const confirmationDiv = button.closest('.tool-confirmation');
  
  if (!messageDiv || !confirmationDiv) return;
  
  // Get tool information from data attributes
  const toolIdentified = confirmationDiv.getAttribute('data-tool-type');
  const toolConfidence = confirmationDiv.getAttribute('data-tool-confidence');
  const originalMessage = confirmationDiv.getAttribute('data-original-message');
  
  // Debug logging to verify we're getting the tool information
  console.log('Tool confirmation - toolIdentified:', toolIdentified, 'toolConfidence:', toolConfidence);
  
  // Collect incident details if it's an HR escalation and user confirmed
  let incidentDetails = null;
  if (confirmed && toolIdentified === "raise_to_hr") {
    const incidentTextarea = confirmationDiv.querySelector('#incidentDetails');
    if (incidentTextarea) {
      incidentDetails = incidentTextarea.value.trim();
      // Validate that incident details are provided
      if (!incidentDetails) {
        // Highlight the textarea and show validation message
        incidentTextarea.style.borderColor = '#ff4444';
        incidentTextarea.style.boxShadow = '0 0 5px rgba(255, 68, 68, 0.3)';
        
        // Show validation message
        let validationMsg = confirmationDiv.querySelector('.validation-message');
        if (!validationMsg) {
          validationMsg = document.createElement('div');
          validationMsg.className = 'validation-message';
          validationMsg.style.color = '#ff4444';
          validationMsg.style.fontSize = '0.9em';
          validationMsg.style.marginTop = '5px';
          incidentTextarea.parentNode.appendChild(validationMsg);
        }
        validationMsg.textContent = 'Please provide incident details before proceeding.';
        
        // Focus on the textarea
        incidentTextarea.focus();
        return; // Don't proceed without incident details
      }
    }
  }
  
  // Disable all confirmation buttons
  const allButtons = confirmationDiv.querySelectorAll('.confirm-btn');
  allButtons.forEach(btn => {
    btn.disabled = true;
    btn.style.opacity = '0.6';
  });
  
  if (confirmed) {
    // User confirmed - proceed with tool execution
    confirmationDiv.innerHTML = `
      <div class="confirmation-result confirmed">
        <span class="confirmation-icon">✓</span>
        <span class="confirmation-message">Processing your request...</span>
      </div>
    `;
    
    // Call the API with the cached tool identification and incident details
    executeConfirmedTool(originalMessage, toolIdentified, toolConfidence, incidentDetails);
  } else {
    // User cancelled
    confirmationDiv.innerHTML = `
      <div class="confirmation-result cancelled">
        <span class="confirmation-icon">✗</span>
        <span class="confirmation-message">Action cancelled</span>
      </div>
    `;
  }
}

// Execute the tool after user confirmation
async function executeConfirmedTool(originalMessage, toolIdentified, toolConfidence, incidentDetails = null) {
  const user_id = localStorage.getItem("userId") || "defaultUser";
  const chat_id = localStorage.getItem("chat_id") || sessionStorage.getItem("chat_id") || "chat123";
  
  try {
    // Show typing indicator
    showTypingIndicator("Executing action...");
    
    // Prepare request body
    const requestBody = {
      message: originalMessage,
      user_id: user_id,
      chat_id: chat_id,
      tool_identified: toolIdentified,
      tool_confidence: toolConfidence
    };
    
    // Add incident details if provided
    if (incidentDetails && toolIdentified === "raise_to_hr") {
      requestBody.user_description = incidentDetails;
    }
    
    // Call tool confirmation API endpoint with cached tool identification
    const response = await fetch("http://localhost:3000/tool_confirmation", {
      method: "POST",
      credentials: 'include',
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    // Remove typing indicator
    hideTypingIndicator();
    
    if (data && data.message) {
      // Add the final response
      addMessage(data.message, "bot");
      
      // Log the tool execution result
      console.log(`Tool executed - Type: ${data.tool_identified || toolIdentified}, Success: ${data.success}`);
      
      // Handle sources if available
      if (Array.isArray(data.sources) && data.sources.length > 0) {
        addSourcesToMessage(data.sources);
      }
      
      if (Array.isArray(data.images) && data.images.length > 0) {
        sendImages(data.images);
      }
    } else {
      throw new Error("Invalid response from tool confirmation API");
    }
  } catch (error) {
    console.error("Tool execution error:", error);
    hideTypingIndicator();
    addMessage("Sorry, there was an error executing the action. Please try again.", "bot");
  }
}

// ==========================================
// THEME CUSTOMIZATION FUNCTIONALITY
// ==========================================

// Theme management
let isThemePanelOpen = false;

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', function() {
  initializeTheme();
  updateThemeUI();
});

function initializeTheme() {
  const savedTheme = localStorage.getItem('selectedTheme') || 'light';
  const customColors = localStorage.getItem('customColors');
  
  if (customColors) {
    const colors = JSON.parse(customColors);
    applyCustomColors(colors);
  } else {
    setTheme(savedTheme);
  }
}

function toggleThemePanel() {
  const panel = document.getElementById('themePanel');
  const isVisible = panel.style.display === 'block';
  
  if (isVisible) {
    // Animate out
    panel.style.maxHeight = '0px';
    setTimeout(() => {
      panel.style.display = 'none';
    }, 300);
    isThemePanelOpen = false;
  } else {
    // Animate in
    panel.style.display = 'block';
    panel.style.maxHeight = '500px';
    isThemePanelOpen = true;
    updateThemeUI();
  }
}

function setTheme(themeName) {
  // Use the universal theme manager to set theme across all pages
  if (window.ThemeManager && window.ThemeManager.setTheme) {
    window.ThemeManager.setTheme(themeName);
  } else {
    // Fallback for direct theme setting
    document.documentElement.removeAttribute('data-theme');
    localStorage.removeItem('customColors');
    
    if (themeName !== 'light') {
      document.documentElement.setAttribute('data-theme', themeName);
    }
    
    localStorage.setItem('selectedTheme', themeName);
  }
  
  // Update UI and show feedback
  updateThemeUI();
  showThemeChangeNotification(`${capitalizeFirstLetter(themeName)} theme applied universally to all pages!`);
  
  // Trigger storage event so other pages update their themes
  window.dispatchEvent(new StorageEvent('storage', {
    key: 'selectedTheme',
    newValue: themeName,
    url: window.location.href
  }));
}

function updateThemeUI() {
  const currentTheme = getCurrentTheme();
  
  // Update theme option active states
  document.querySelectorAll('.theme-option').forEach(option => {
    const themeData = option.getAttribute('data-theme');
    if (themeData === currentTheme) {
      option.classList.add('active');
    } else {
      option.classList.remove('active');
    }
  });
  
  // Update custom color inputs
  updateColorInputs();
}

function getCurrentTheme() {
  const dataTheme = document.documentElement.getAttribute('data-theme');
  return dataTheme || 'light';
}

function updateCustomColor(colorType, colorValue) {
  const customColors = JSON.parse(localStorage.getItem('customColors') || '{}');
  customColors[colorType] = colorValue;
  
  // Also update background mode if not set
  if (!customColors.backgroundMode) {
    customColors.backgroundMode = document.getElementById('backgroundMode').value;
  }
  
  localStorage.setItem('customColors', JSON.stringify(customColors));
  
  // Use universal theme manager for custom colors
  if (window.ThemeManager && window.ThemeManager.applyCustomColors) {
    window.ThemeManager.applyCustomColors(customColors);
  } else {
    applyCustomColors(customColors);
  }
  
  // Clear theme selection
  document.querySelectorAll('.theme-option').forEach(option => {
    option.classList.remove('active');
  });
  
  localStorage.setItem('selectedTheme', 'custom');
  
  // Show feedback
  showThemeChangeNotification(`Custom ${colorType} color applied universally to all pages!`);
  
  // Trigger storage event so other pages update
  window.dispatchEvent(new StorageEvent('storage', {
    key: 'customColors',
    newValue: JSON.stringify(customColors),
    url: window.location.href
  }));
}

function updateBackgroundMode(mode) {
  const customColors = JSON.parse(localStorage.getItem('customColors') || '{}');
  customColors.backgroundMode = mode;
  localStorage.setItem('customColors', JSON.stringify(customColors));
  
  // Use universal theme manager for custom colors
  if (window.ThemeManager && window.ThemeManager.applyCustomColors) {
    window.ThemeManager.applyCustomColors(customColors);
  } else {
    applyCustomColors(customColors);
  }
  
  // Show feedback
  showThemeChangeNotification(`${mode === 'dark' ? 'Dark' : 'Light'} background applied universally to all pages!`);
  
  // Trigger storage event so other pages update
  window.dispatchEvent(new StorageEvent('storage', {
    key: 'customColors',
    newValue: JSON.stringify(customColors),
    url: window.location.href
  }));
}

function applyCustomColors(colors) {
  const root = document.documentElement;
  
  // Clear existing theme
  root.removeAttribute('data-theme');
  
  // Apply custom colors
  if (colors.primary) {
    root.style.setProperty('--primary-color', colors.primary);
    root.style.setProperty('--text-primary', colors.primary);
  }
  
  if (colors.accent) {
    root.style.setProperty('--primary-accent', colors.accent);
    root.style.setProperty('--text-accent', colors.accent);
    // Create a darker version for hover state
    const darkerAccent = adjustBrightness(colors.accent, -20);
    root.style.setProperty('--primary-accent-hover', darkerAccent);
  }
  
  // Apply background mode
  if (colors.backgroundMode === 'dark') {
    root.style.setProperty('--background-main', 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)');
    root.style.setProperty('--background-card', '#1a1a1a');
    root.style.setProperty('--background-header', '#1a1a1a');
    root.style.setProperty('--background-footer', '#0a0a0a');
    root.style.setProperty('--background-btn', '#2c2c2c');
    root.style.setProperty('--background-btn-hover', '#333');
    root.style.setProperty('--border-color', '#333');
    root.style.setProperty('--text-secondary', '#aaa');
    root.style.setProperty('--text-muted', '#666');
    root.style.setProperty('--secondary-color', '#1a1a1a');
  } else {
    root.style.setProperty('--background-main', 'linear-gradient(135deg, #f6f6f4 0%, #ecebe7 100%)');
    root.style.setProperty('--background-card', '#f7f6f2');
    root.style.setProperty('--background-header', '#f7f6f2');
    root.style.setProperty('--background-footer', '#ecebe7');
    root.style.setProperty('--background-btn', '#fff');
    root.style.setProperty('--background-btn-hover', '#f7f6f2');
    root.style.setProperty('--border-color', '#e5e3dc');
    root.style.setProperty('--text-secondary', '#666');
    root.style.setProperty('--text-muted', '#999');
    root.style.setProperty('--secondary-color', '#ecebe7');
  }
}

function updateColorInputs() {
  const customColors = JSON.parse(localStorage.getItem('customColors') || '{}');
  
  if (customColors.primary) {
    document.getElementById('primaryColor').value = customColors.primary;
  }
  if (customColors.accent) {
    document.getElementById('accentColor').value = customColors.accent;
  }
  if (customColors.backgroundMode) {
    document.getElementById('backgroundMode').value = customColors.backgroundMode;
  }
}

function resetToDefault() {
  // Clear all custom settings
  localStorage.removeItem('customColors');
  localStorage.removeItem('selectedTheme');
  
  // Reset to default light theme using universal manager
  if (window.ThemeManager && window.ThemeManager.setTheme) {
    window.ThemeManager.setTheme('light');
  } else {
    setTheme('light');
  }
  
  // Reset input values
  document.getElementById('primaryColor').value = '#232323';
  document.getElementById('accentColor').value = '#d4b24c';
  document.getElementById('backgroundMode').value = 'light';
  
  // Clear custom CSS properties
  const root = document.documentElement;
  root.style.removeProperty('--primary-color');
  root.style.removeProperty('--primary-accent');
  root.style.removeProperty('--primary-accent-hover');
  root.style.removeProperty('--text-primary');
  root.style.removeProperty('--text-accent');
  root.style.removeProperty('--background-main');
  root.style.removeProperty('--background-card');
  root.style.removeProperty('--background-header');
  root.style.removeProperty('--background-footer');
  root.style.removeProperty('--background-btn');
  root.style.removeProperty('--background-btn-hover');
  root.style.removeProperty('--border-color');
  root.style.removeProperty('--text-secondary');
  root.style.removeProperty('--text-muted');
  root.style.removeProperty('--secondary-color');
  
  showThemeChangeNotification('Theme reset to default!');
}

function showThemeChangeNotification(message) {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = 'theme-notification';
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--primary-accent);
    color: white;
    padding: 12px 20px;
    border-radius: 6px;
    font-size: 0.9em;
    font-weight: 500;
    box-shadow: var(--shadow);
    z-index: 10000;
    transform: translateX(300px);
       transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  `;
  
  document.body.appendChild(notification);
  
  // Animate in
  setTimeout(() => {
    notification.style.transform = 'translateX(0)';
  }, 100);
  
  // Remove after delay
  setTimeout(() => {
    notification.style.transform = 'translateX(300px)';
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }, 3000);
}

function adjustBrightness(hexColor, amount) {
  // Convert hex to RGB
  const hex = hexColor.replace('#', '');
  const r = parseInt(hex.substr(0, 2), 16);
  const g = parseInt(hex.substr(2, 2), 16);
  const b = parseInt(hex.substr(4, 2), 16);
  
  // Adjust brightness
  const newR = Math.max(0, Math.min(255, r + amount));
  const newG = Math.max(0, Math.min(255, g + amount));
  const newB = Math.max(0, Math.min(255, b + amount));
  
  // Convert back to hex
  return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`;
}

function capitalizeFirstLetter(string) {
   return string.charAt(0).toUpperCase() + string.slice(1);
}

// Close theme panel when clicking outside
document.addEventListener('click', function(event) {
  const themePanel = document.getElementById('themePanel');
  const themesSection = document.querySelector('.sidebar-themes');
  
  if (isThemePanelOpen && themePanel && themesSection) {
    if (!themesSection.contains(event.target)) {
      themePanel.style.display = 'none';
      isThemePanelOpen = false;
    }
  }
});

// ==========================================
// END THEME CUSTOMIZATION FUNCTIONALITY
// ==========================================

// ==========================================
// SPEECH-TO-TEXT FUNCTIONALITY
// ==========================================

let speechToText = null;
let isRecording = false;

// Initialize speech recognition when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Initialize SpeechToText (preferring browser API for faster response)
  if (typeof SpeechToText !== 'undefined') {
    speechToText = new SpeechToText(false); // false = prefer browser API over Google Cloud
    
    // Set up event listeners
    speechToText.onResult = function(transcript, isFinal) {
      console.log('Speech recognized:', transcript, 'Final:', isFinal);
      
      // Insert the transcript into the input field
      const messageInput = document.getElementById('messageInput');
      if (messageInput && isFinal) {
        const currentValue = messageInput.value.trim();
        const newValue = currentValue ? currentValue + ' ' + transcript : transcript;
        messageInput.value = newValue;
        
        // Trigger input event to resize textarea if needed
        messageInput.dispatchEvent(new Event('input'));
        
        // Update button state after adding text
        updateActionButton();
        
        // Focus the input field
        messageInput.focus();
        
        // Auto-stop recording after getting final result
        stopSpeechRecognition();
      }
    };
    
    speechToText.onEnd = function() {
      console.log('Speech recognition ended');
      stopSpeechRecognition();
    };
    
    speechToText.onError = function(error) {
      console.error('Speech recognition error:', error);
      showSpeechError(error);
      stopSpeechRecognition();
    };
    
    speechToText.onStart = function() {
      console.log('Speech recognition started');
      showSpeechStatus('Listening... Speak now');
    };
    
    console.log('Speech-to-Text initialized with service:', speechToText.getServiceType());
  } else {
    console.warn('SpeechToText class not available');
  }

  // Initialize button state
  updateActionButton();
});

// Handle the unified action button (mic/send)
function handleActionButton() {
  const messageInput = document.getElementById('messageInput');
  const inputText = messageInput ? messageInput.value.trim() : '';
  
  console.log('Action button clicked:', { inputText, isRecording });
  
  if (isRecording) {
    // If recording, stop it
    console.log('Stopping speech recognition');
    stopSpeechRecognition();
  } else if (inputText) {
    // If there's text, send message
    console.log('Sending message:', inputText);
    sendMessage();
  } else {
    // If no text, start speech recognition
    console.log('Starting speech recognition');
    startSpeechRecognition();
  }
}

// Update action button based on input state
function updateActionButton() {
  const messageInput = document.getElementById('messageInput');
  const actionButton = document.getElementById('actionButton');
  const micIcon = document.getElementById('micIcon');
  const recordingIcon = document.getElementById('recordingIcon');
  const sendIcon = document.getElementById('sendIcon');
  
  if (!messageInput || !actionButton || !micIcon || !recordingIcon || !sendIcon) return;
  
  const inputText = messageInput.value.trim();
  
  if (isRecording) {
    // Recording state
    actionButton.classList.add('recording');
    actionButton.classList.remove('send-mode');
    actionButton.title = 'Click to stop recording';
    micIcon.style.display = 'none';
    recordingIcon.style.display = 'block';
    sendIcon.style.display = 'none';
  } else if (inputText) {
    // Send mode
    actionButton.classList.remove('recording');
    actionButton.classList.add('send-mode');
    actionButton.title = 'Send message';
    micIcon.style.display = 'none';
    recordingIcon.style.display = 'none';
    sendIcon.style.display = 'block';
  } else {
    // Microphone mode
    actionButton.classList.remove('recording', 'send-mode');
    actionButton.title = 'Click to speak';
    micIcon.style.display = 'block';
    recordingIcon.style.display = 'none';
    sendIcon.style.display = 'none';
  }
}

// Toggle speech recognition (legacy function for keyboard shortcuts)
function toggleSpeechRecognition() {
  if (!speechToText || !speechToText.isSupported()) {
    showSpeechError('Speech recognition not available in this browser');
    return;
  }
  
  if (isRecording) {
    stopSpeechRecognition();
  } else {
    startSpeechRecognition();
  }
}

// Start speech recognition
function startSpeechRecognition() {
  if (!speechToText || isRecording) return;
  
  try {
    speechToText.startListening('en-GB');
    isRecording = true;
    updateActionButton();
    showSpeechStatus('Listening... Speak now');
    console.log('Started speech recognition');
  } catch (error) {
    console.error('Failed to start speech recognition:', error);
    showSpeechError('Failed to start speech recognition: ' + error.message);
  }
}

// Stop speech recognition
function stopSpeechRecognition() {
  if (!speechToText || !isRecording) return;
  
  try {
    speechToText.stopListening();
    isRecording = false;
    updateActionButton();
    hideSpeechStatus();
    console.log('Stopped speech recognition');
  } catch (error) {
    console.error('Failed to stop speech recognition:', error);
  }
}

// Update microphone button appearance (legacy function - now handled by updateActionButton)
function updateMicrophoneButton(recording) {
  updateActionButton();
}

// Show speech recognition status
function showSpeechStatus(message) {
  const speechStatus = document.getElementById('speechStatus');
  const statusText = document.getElementById('statusText');
  
  if (speechStatus && statusText) {
    statusText.textContent = message;
    speechStatus.style.display = 'flex';
  }
}

// Hide speech recognition status
function hideSpeechStatus() {
  const speechStatus = document.getElementById('speechStatus');
  if (speechStatus) {
    speechStatus.style.display = 'none';
  }
}

// Show speech recognition error
function showSpeechError(error) {
  let errorMessage = 'Speech recognition error';
  
  if (typeof error === 'string') {
    errorMessage = error;
  } else if (error && error.error) {
    switch (error.error) {
      case 'no-speech':
        errorMessage = 'No speech detected. Please try again.';
        break;
      case 'audio-capture':
        errorMessage = 'Microphone not accessible. Please check permissions.';
        break;
      case 'not-allowed':
        errorMessage = 'Microphone access denied. Please allow microphone access.';
        break;
      case 'network':
        errorMessage = 'Network error. Please check your connection.';
        break;
      case 'service-not-allowed':
        errorMessage = 'Speech recognition service not allowed.';
        break;
      default:
        errorMessage = `Speech recognition error: ${error.error}`;
    }
  }
  
  console.error('Speech recognition error:', errorMessage);
  
  // Show error message temporarily
  showSpeechStatus(errorMessage);
  setTimeout(() => {
    hideSpeechStatus();
  }, 3000);
}

// Handle keyboard shortcuts for speech recognition
document.addEventListener('keydown', function(event) {
  // Ctrl + Shift + M or Cmd + Shift + M to toggle speech recognition
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'M') {
    event.preventDefault();
    const messageInput = document.getElementById('messageInput');
    const inputText = messageInput ? messageInput.value.trim() : '';
    
    // Only start speech recognition if there's no text
    if (!inputText) {
      toggleSpeechRecognition();
    }
  }
  
  // Escape key to stop speech recognition
  if (event.key === 'Escape' && isRecording) {
    event.preventDefault();
    stopSpeechRecognition();
  }
});

// ==========================================
// END SPEECH-TO-TEXT FUNCTIONALITY
// ==========================================

