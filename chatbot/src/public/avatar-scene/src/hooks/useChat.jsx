import { createContext, useContext, useEffect, useState, useCallback } from "react";

const ChatContext = createContext();

const backendUrl = "http://localhost:8000"; // Node.js server for TTS

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState();
  const [loading, setLoading] = useState(false);
  const [cameraZoomed, setCameraZoomed] = useState(true);
  const [processingTexts, setProcessingTexts] = useState(new Set()); // Track texts being processed

  // Function to generate TTS with lipsync from text
  const generateTTSWithLipsync = async (text) => {
    // Prevent duplicate TTS generation for the same text
    if (processingTexts.has(text)) {
      console.log('🚫 Already processing TTS for this text, skipping:', text.substring(0, 50) + '...');
      return;
    }

    setProcessingTexts(prev => new Set([...prev, text]));

    try {
      console.log('🎤 Generating TTS for text:', text.substring(0, 50) + '...');
      setLoading(true);

      const response = await fetch(`${backendUrl}/api/tts/synthesize-enhanced`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${localStorage.getItem("token")}`,
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

      if (!response.ok) {
        throw new Error(`TTS API error: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ TTS generated successfully', data);

      if (!data.success) {
        throw new Error(`TTS generation failed: ${data.error || 'Unknown error'}`);
      }

      const newMessage = {
        id: `tts_generated_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: "bot",
        text,
        audio: data.audio,
        lipsync: data.lipSyncData || data.lipsync // Handle both possible response formats
      };

      console.log('✅ Adding TTS message to queue:', newMessage.id);
      setMessages((messages) => [...messages, newMessage]);

    } catch (error) {
      console.error('❌ Error generating TTS:', error);
    } finally {
      setLoading(false);
      // Remove from processing set after a delay to prevent immediate duplicates
      setTimeout(() => {
        setProcessingTexts(prev => {
          const newSet = new Set(prev);
          newSet.delete(text);
          return newSet;
        });
      }, 2000);
    }
  };

  const handleMessage = useCallback(async (event) => {
    // Check origin for security
    if (event.origin !== window.location.origin) {
      console.log('🚫 Message from invalid origin:', event.origin);
      return;
    }

    console.log('🎭 Avatar received message:', event.data);
    console.log('🎭 Message type:', event.data?.type);

    // Handle pre-generated TTS with lipsync data
    if (event.data?.type === 'tts_with_lipsync') {
      const { text, audio, lipsync } = event.data;
      console.log('✅ Avatar received pre-generated TTS:', text?.substring(0, 50) + '...');
      
      const newMessage = {
        id: `tts_ready_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: "bot",
        text,
        audio,
        lipsync
      };
      console.log('✅ Adding ready TTS message to queue:', newMessage.id);
      setMessages((messages) => [...messages, newMessage]);
      return;
    }

    // Handle bot response that needs TTS generation
    if (event.data?.type === 'bot_response_for_tts') {
      const { text, timestamp, isNewMessage } = event.data.payload || event.data;
      console.log('🎤 Avatar received bot response for TTS:', text?.substring(0, 50) + '...');
      
      if (!text) {
        console.error('❌ No text provided in bot_response_for_tts message');
        return;
      }

      // Only generate TTS for new messages, not historical ones
      const now = Date.now();
      const messageTime = timestamp || now;
      const timeDiff = now - messageTime;
      
      // If message is older than 5 seconds or explicitly marked as old, skip TTS
      if (timeDiff > 5000 || isNewMessage === false) {
        console.log('🚫 Skipping TTS for old message (age:', timeDiff, 'ms)');
        return;
      }

      console.log('✅ Processing fresh message for TTS (age:', timeDiff, 'ms)');
      // Generate TTS + lipsync for the provided text
      await generateTTSWithLipsync(text);
      return;
    }

    // Log unhandled message types for debugging
    if (event.data?.type) {
      console.log('⚠️ Unhandled message type:', event.data.type, event.data);
    }
  }, []);

  useEffect(() => {
    console.log('🎭 Setting up message listener for avatar');
    window.addEventListener('message', handleMessage);
    return () => {
      console.log('🎭 Cleaning up message listener for avatar');
      window.removeEventListener('message', handleMessage);
    };
  }, [handleMessage]);

  const onMessagePlayed = () => {
    const currentMessage = messages[0];
    if (currentMessage) {
      console.log('🎵 Message played, removing from queue:', currentMessage.id);
    }
    setMessages((messages) => messages.slice(1));
  };

  useEffect(() => {
    if (messages.length > 0) {
      const nextMessage = messages[0];
      console.log('🎵 Setting next message for processing:', nextMessage.id);
      setMessage(nextMessage);
    } else {
      console.log('🎵 No messages in queue, clearing current message');
      setMessage(null);
    }
  }, [messages]);

  return (
    <ChatContext.Provider
      value={{
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