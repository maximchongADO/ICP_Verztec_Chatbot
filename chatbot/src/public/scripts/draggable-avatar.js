class DraggableAvatar {
  constructor() {
    this.isVisible = false;
    this.isDragging = false;
    this.startX = 0;
    this.startY = 0;
    this.currentX = 0;
    this.currentY = 0;
    this.initialTransform = null;
    this.avatarContainer = null;
    this.avatarIframe = null;
    this.toggleButton = null;
    this.dragHandle = null;
    
    this.init();
  }

  init() {
    this.createAvatarSidebar();
    this.setupEventListeners();
    this.loadAvatarState();
  }

  createAvatarSidebar() {
    // Create main avatar container
    this.avatarContainer = document.createElement('div');
    this.avatarContainer.id = 'draggable-avatar-container';
    this.avatarContainer.className = 'draggable-avatar-container';
    
    // Create drag handle
    this.dragHandle = document.createElement('div');
    this.dragHandle.className = 'avatar-drag-handle';
    this.dragHandle.innerHTML = `
      <div class="drag-icon">⋮⋮</div>
      <span class="drag-text">Avatar Assistant</span>
    `;
    
    // Create close button
    const closeButton = document.createElement('button');
    closeButton.className = 'avatar-close-btn';
    closeButton.innerHTML = '×';
    closeButton.onclick = () => this.hideAvatar();
    
    // Create iframe for avatar app
    this.avatarIframe = document.createElement('iframe');
    this.avatarIframe.src = '/avatar/index.html';
    this.avatarIframe.className = 'avatar-iframe';
    this.avatarIframe.frameBorder = '0';
    this.avatarIframe.allow = 'microphone; camera; autoplay';
    
    // Assemble container
    this.avatarContainer.appendChild(this.dragHandle);
    this.avatarContainer.appendChild(closeButton);
    this.avatarContainer.appendChild(this.avatarIframe);
    
    // Create toggle button
    this.toggleButton = document.createElement('button');
    this.toggleButton.id = 'avatar-toggle-btn';
    this.toggleButton.className = 'avatar-toggle-btn';
    this.toggleButton.innerHTML = `
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
      </svg>
      Avatar
    `;
    this.toggleButton.onclick = () => this.toggleAvatar();
    
    // Add to DOM
    document.body.appendChild(this.avatarContainer);
    document.body.appendChild(this.toggleButton);
  }

  setupEventListeners() {
    // Drag functionality
    this.dragHandle.addEventListener('mousedown', (e) => this.startDrag(e));
    this.dragHandle.addEventListener('touchstart', (e) => this.startDrag(e));
    
    document.addEventListener('mousemove', (e) => this.drag(e));
    document.addEventListener('touchmove', (e) => this.drag(e));
    
    document.addEventListener('mouseup', () => this.stopDrag());
    document.addEventListener('touchend', () => this.stopDrag());
    
    // Window resize handler
    window.addEventListener('resize', () => this.handleResize());
    
    // Message communication with avatar iframe
    window.addEventListener('message', (event) => this.handleAvatarMessage(event));
  }

  startDrag(e) {
    if (!this.isVisible) return;
    
    this.isDragging = true;
    this.dragHandle.classList.add('dragging');
    
    const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
    const clientY = e.type === 'touchstart' ? e.touches[0].clientY : e.clientY;
    
    this.startX = clientX - this.currentX;
    this.startY = clientY - this.currentY;
    
    e.preventDefault();
  }

  drag(e) {
    if (!this.isDragging) return;
    
    const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
    const clientY = e.type === 'touchmove' ? e.touches[0].clientY : e.clientY;
    
    this.currentX = clientX - this.startX;
    this.currentY = clientY - this.startY;
    
    // Constrain to viewport
    const maxX = window.innerWidth - this.avatarContainer.offsetWidth;
    const maxY = window.innerHeight - this.avatarContainer.offsetHeight;
    
    this.currentX = Math.max(0, Math.min(this.currentX, maxX));
    this.currentY = Math.max(0, Math.min(this.currentY, maxY));
    
    this.avatarContainer.style.transform = `translate(${this.currentX}px, ${this.currentY}px)`;
    
    e.preventDefault();
  }

  stopDrag() {
    if (!this.isDragging) return;
    
    this.isDragging = false;
    this.dragHandle.classList.remove('dragging');
    this.saveAvatarState();
  }

  toggleAvatar() {
    if (this.isVisible) {
      this.hideAvatar();
    } else {
      this.showAvatar();
    }
  }

  showAvatar() {
    // Cancel main chatbot TTS when showing avatar
    if (window.googleTTS) {
      window.googleTTS.cancel();
      console.log('Cancelled main chatbot TTS when showing avatar');
    }
    
    this.isVisible = true;
    this.avatarContainer.classList.add('visible');
    this.toggleButton.classList.add('active');
    
    // Send message to main chatbot that avatar is visible
    this.notifyChatbot('avatar_visible', true);
    this.saveAvatarState();
  }

  hideAvatar() {
    this.isVisible = false;
    this.avatarContainer.classList.remove('visible');
    this.toggleButton.classList.remove('active');
    
    // Send message to main chatbot that avatar is hidden
    this.notifyChatbot('avatar_visible', false);
    this.saveAvatarState();
  }

  handleResize() {
    if (!this.isVisible) return;
    
    // Ensure avatar stays within viewport after resize
    const maxX = window.innerWidth - this.avatarContainer.offsetWidth;
    const maxY = window.innerHeight - this.avatarContainer.offsetHeight;
    
    this.currentX = Math.max(0, Math.min(this.currentX, maxX));
    this.currentY = Math.max(0, Math.min(this.currentY, maxY));
    
    this.avatarContainer.style.transform = `translate(${this.currentX}px, ${this.currentY}px)`;
  }

  handleAvatarMessage(event) {
    // Handle messages from avatar iframe
    if (event.origin !== window.location.origin) return;
    
    const data = event.data;
    
    switch (data.type) {
      case 'avatar_ready':
        console.log('Avatar app is ready');
        break;
      case 'avatar_speaking':
        this.notifyChatbot('avatar_speaking', data.payload);
        break;
      case 'avatar_response':
        this.notifyChatbot('avatar_response', data.payload);
        break;
    }
  }

  notifyChatbot(type, payload) {
    // Send messages to main chatbot
    const event = new CustomEvent('avatarEvent', {
      detail: { type, payload }
    });
    document.dispatchEvent(event);
  }

  sendMessageToAvatar(message) {
    if (this.avatarIframe && this.avatarIframe.contentWindow) {
      this.avatarIframe.contentWindow.postMessage({
        type: 'chat_message',
        payload: message
      }, window.location.origin);
    }
  }

  saveAvatarState() {
    const state = {
      isVisible: this.isVisible,
      position: {
        x: this.currentX,
        y: this.currentY
      }
    };
    localStorage.setItem('draggable_avatar_state', JSON.stringify(state));
  }

  loadAvatarState() {
    const savedState = localStorage.getItem('draggable_avatar_state');
    if (savedState) {
      const state = JSON.parse(savedState);
      
      this.currentX = state.position?.x || 0;
      this.currentY = state.position?.y || 0;
      
      // Apply position
      this.avatarContainer.style.transform = `translate(${this.currentX}px, ${this.currentY}px)`;
      
      // Restore visibility state
      if (state.isVisible) {
        this.showAvatar();
      }
    }
  }
  
  // Method to check if avatar is visible (used by main chatbot)
  isAvatarVisible() {
    return this.isVisible;
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.draggableAvatar = new DraggableAvatar();
});
