import { createContext, useContext, useEffect, useState } from "react";

const ChatContext = createContext(); // <-- Add this line

const backendUrl = "http://localhost:3000";
export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState();
  const [loading, setLoading] = useState(false);
  const [cameraZoomed, setCameraZoomed] = useState(true);

  // Get token from localStorage (or however you store it)
  const token = localStorage.getItem("token"); // <-- Add this line

  const chat = async (message) => {
    setLoading(true);
    try {
      const response = await fetch(`${backendUrl}/chatbot_avatar`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: message,
        }),
      });

      const data = await response.json();

      const resp = [{ type: "bot", text: data.message }];
      setMessages((messages) => [...messages, ...resp]);
    } catch (error) {
      console.error("Chat error:", error);
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