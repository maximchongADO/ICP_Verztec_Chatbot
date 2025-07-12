import { createContext, useContext, useEffect, useState } from "react";

const ChatContext = createContext();

const backendUrl = "http://localhost:8000";

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState();
  const [loading, setLoading] = useState(false);
  const [cameraZoomed, setCameraZoomed] = useState(true);
  const [processedMessageIds, setProcessedMessageIds] = useState(new Set());

  const token = localStorage.getItem("token");

  const chat = async (message) => {
    setLoading(true);
    console.log('Chat request starting for:', message);
    
    try {
      const response = await fetch(`${backendUrl}/chatbot_avatar_test`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({
          message: message,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Chat response received:', data);

      if (data.messages && data.messages.length > 0) {
        // Add unique IDs to messages to prevent duplicates
        const newMessages = data.messages.map((msg, index) => {
          const uniqueId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}_${index}`;
          console.log('Creating message with ID:', uniqueId);
          return {
            ...msg,
            id: uniqueId
          };
        });
        
        console.log('Adding messages to queue:', newMessages.map(m => ({ id: m.id, hasAudio: !!m.audio })));
        setMessages((messages) => [...messages, ...newMessages]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      // Add error message
      const errorMessage = {
        id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: "bot",
        text: "Sorry, I encountered an error. Please try again."
      };
      setMessages((messages) => [...messages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const onMessagePlayed = () => {
    const currentMessage = messages[0];
    if (currentMessage) {
      console.log('Message played, marking as processed:', currentMessage.id);
      setProcessedMessageIds(prev => new Set([...prev, currentMessage.id]));
    }
    setMessages((messages) => messages.slice(1));
  };

  useEffect(() => {
    if (messages.length > 0) {
      const nextMessage = messages[0];
      // Only set message if it hasn't been processed yet
      if (!processedMessageIds.has(nextMessage.id)) {
        console.log('Setting next message for processing:', nextMessage.id);
        setMessage(nextMessage);
      } else {
        console.log('Message already processed, skipping:', nextMessage.id);
        setMessage(null);
      }
    } else {
      setMessage(null);
    }
  }, [messages, processedMessageIds]);

  return (
    <ChatContext.Provider
      value={{
        chat,
        message,
        onMessagePlayed,
        loading,
        cameraZoomed,
        setCameraZoomed,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};