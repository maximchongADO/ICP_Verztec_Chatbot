import { createContext, useContext, useEffect, useState } from "react";

const ChatContext = createContext(); // <-- Add this line

const backendUrl = "http://localhost:8000";
export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState();
  const [loading, setLoading] = useState(false);
  const [cameraZoomed, setCameraZoomed] = useState(true);

  const token = localStorage.getItem("token");

  const chat = async (message) => {
    setLoading(true);
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
      console.log('Received data:', data);

      if (data.messages && data.messages.length > 0) {
        setMessages((messages) => [...messages, ...data.messages]);
      }
    } catch (error) {
      console.error("Chat error:", error);
      // Add error message
      const errorMessage = {
        type: "bot",
        text: "Sorry, I encountered an error. Please try again."
      };
      setMessages((messages) => [...messages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const onMessagePlayed = () => {
    setMessages((messages) => messages.slice(1));
  };

  useEffect(() => {
    if (messages.length > 0) {
      setMessage(messages[0]);
    } else {
      setMessage(null);
    }
  }, [messages]);

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