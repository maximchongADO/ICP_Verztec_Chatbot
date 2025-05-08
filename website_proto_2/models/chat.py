import uuid
from datetime import datetime
from typing import Dict, List, Optional

class Message:
    """Class representing a chat message"""
    def __init__(self, role: str, text: str):
        self.role = role
        self.text = text
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            "role": self.role,
            "text": self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create message from dictionary"""
        return cls(data["role"], data["text"])

class Chat:
    """Class representing a chat session"""
    def __init__(self, 
                 title: str = None, 
                 user: str = None, 
                 chat_id: str = None,
                 messages: List[Message] = None):
        self.id = chat_id or str(uuid.uuid4())
        self.title = title or f"Chat {datetime.now().strftime('%H:%M:%S')}"
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.user = user
        self.messages = messages or []
    
    def add_message(self, role: str, text: str) -> None:
        """Add a new message to the chat"""
        self.messages.append(Message(role, text))
    
    def rename(self, new_title: str) -> None:
        """Rename the chat"""
        self.title = new_title
    
    def to_dict(self) -> Dict:
        """Convert chat to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "timestamp": self.timestamp,
            "user": self.user,
            "messages": [msg.to_dict() for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Chat':
        """Create chat from dictionary"""
        chat = cls(
            title=data["title"],
            user=data["user"],
            chat_id=data["id"],
        )
        chat.timestamp = data["timestamp"]
        chat.messages = [Message.from_dict(msg) for msg in data["messages"]]
        return chat