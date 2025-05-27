// APPLICATION - Main Initialization and Configuration

/**
 * Application configuration
 */
const AppConfig = {
    autoSave: true,
    autoSaveInterval: 30000, // 30 seconds
    beforeUnloadWarning: true,
    customResponses: {
        'weather': "I don't have access to real-time weather data, but I can help you find weather resources or discuss weather-related topics!",
        'time': "I don't have access to real-time data, but you can check your system clock for the current time!",
        'joke': "Why don't scientists trust atoms? Because they make up everything! 😄",
        'thanks': "You're very welcome! I'm happy to help. Is there anything else you'd like to discuss?",
        'goodbye': "Goodbye! It was great chatting with you. Feel free to come back anytime if you need help!"
    }
};

/**
 * Initialize the chatbot application
 */
function initializeApp() {
    try {
        // Create main controller instance
        window.chatApp = new ChatController();
        
        // Configure the application
        window.chatApp.configure(AppConfig);
        
        // Set up development tools (only in development)
        if (isDevelopmentMode()) {
            setupDevelopmentTools();
        }
        
        // Set up application event listeners
        setupApplicationEvents();
        
        console.log('✅ Claude Chatbot Application initialized successfully');
        
        // Optional: Show a welcome notification
        showWelcomeNotification();
        
    } catch (error) {
        console.error('❌ Failed to initialize application:', error);
        showErrorNotification('Failed to initialize the chatbot. Please refresh the page.');
    }
}

/**
 * Check if running in development mode
 * @returns {boolean} Whether in development mode
 */
function isDevelopmentMode() {
    return window.location.hostname === 'localhost' || 
           window.location.hostname === '127.0.0.1' ||
           window.location.protocol === 'file:';
}

/**
 * Set up development tools and debugging utilities
 */
function setupDevelopmentTools() {
    // Expose useful methods to global scope for debugging
    window.devTools = {
        clearChat: () => window.chatApp.clearConversation(),
        exportChat: () => window.chatApp.exportConversation(),
        getStats: () => window.chatApp.getConversationStats(),
        simulateMessage: (msg) => window.chatApp.simulateAssistantMessage(msg),
        searchChat: (query) => window.chatApp.searchConversation(query),
        addResponse: (trigger, response) => window.chatApp.addCustomResponse(trigger, response)
    };
    
    console.log('🔧 Development tools available via window.devTools');
    console.log('Available commands: clearChat(), exportChat(), getStats(), simulateMessage(msg), searchChat(query), addResponse(trigger, response)');
}

/**
 * Set up application-level event listeners
 */
function setupApplicationEvents() {
    // Handle visibility change (tab switching)
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && window.chatApp) {
            // Refocus input when tab becomes visible
            setTimeout(() => {
                window.chatApp.view.focusInput();
            }, 100);
        }
    });
    
    // Handle online/offline status
    window.addEventListener('online', () => {
        console.log('🌐 Connection restored');
        showNotification('Connection restored', 'success');
    });
    
    window.addEventListener('offline', () => {
        console.log('📡 Connection lost');
        showNotification('Connection lost - chatbot will work offline', 'warning');
    });
    
    // Handle errors globally
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
        if (window.chatApp) {
            showNotification('An error occurred. Please try refreshing the page.', 'error');
        }
    });
}

/**
 * Show welcome notification
 */
function showWelcomeNotification() {
    setTimeout(() => {
        showNotification('Welcome to Claude Chatbot! Press Ctrl+K to focus input, Ctrl+L to clear chat.', 'info', 5000);
    }, 1000);
}

/**
 * Show notification to user
 * @param {string} message - Notification message
 * @param {string} type - Notification type ('info', 'success', 'warning', 'error')
 * @param {number} duration - Duration in milliseconds
 */
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Notification styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        max-width: 300px;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    
    // Type-specific colors
    const colors = {
        info: '#3b82f6',
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444'
    };
    
    notification.style.background = colors[type] || colors.info;
    
    document.body.appendChild(notification);
    
    // Animate in
    requestAnimationFrame(() => {
        notification.style.transform = 'translateX(0)';
    });
    
    // Auto remove
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, duration);
}

/**
 * Show error notification
 * @param {string} message - Error message
 */
function showErrorNotification(message) {
    showNotification(message, 'error', 5000);
}

/**
 * Performance monitoring
 */
function setupPerformanceMonitoring() {
    // Monitor page load performance
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            console.log('📊 Page Load Performance:', {
                loadTime: Math.round(perfData.loadEventEnd - perfData.loadEventStart),
                domContentLoaded: Math.round(perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart),
                totalTime: Math.round(perfData.loadEventEnd - perfData.fetchStart)
            });
        }, 0);
    });
}

/**
 * Application utility functions
 */
const AppUtils = {
    /**
     * Format timestamp for display
     * @param {Date} timestamp - Timestamp to format
     * @returns {string} Formatted timestamp
     */
    formatTimestamp(timestamp) {
        return new Intl.DateTimeFormat('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).format(timestamp);
    },
    
    /**
     * Copy text to clipboard
     * @param {string} text - Text to copy
     * @returns {Promise<boolean>} Success status
     */
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            showNotification('Copied to clipboard!', 'success');
            return true;
        } catch (error) {
            console.error('Failed to copy to clipboard:', error);
            showNotification('Failed to copy to clipboard', 'error');
            return false;
        }
    },
    
    /**
     * Download conversation as JSON
     */
    downloadConversation() {
        if (!window.chatApp) return;
        
        const data = window.chatApp.exportConversation();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `claude-conversation-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showNotification('Conversation downloaded!', 'success');
    }
};

// Expose utilities globally
window.AppUtils = AppUtils;

// Initialize performance monitoring
setupPerformanceMonitoring();

// Initialize the application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Handle module loading errors
window.addEventListener('error', (event) => {
    if (event.filename && event.filename.includes('.js')) {
        console.error('Script loading error:', event.filename);
        showErrorNotification('Failed to load application resources. Please refresh the page.');
    }
});

console.log('🚀 Claude Chatbot App.js loaded successfully');