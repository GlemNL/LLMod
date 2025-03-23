"""
Conversation context manager for tracking message history in channels
"""
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Deque, Optional

logger = logging.getLogger(__name__)

@dataclass
class MessageContext:
    """
    Data class for storing message context information
    """
    message_id: int
    author_id: int
    author_name: str
    content: str
    timestamp: datetime
    
    def to_dict(self):
        """Convert to dictionary for LLM processing"""
        return {
            "author_name": self.author_name,
            "author_id": str(self.author_id),
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

class ConversationContextManager:
    """
    Manages conversation context by keeping track of recent messages per channel
    """
    def __init__(self, max_messages_per_channel=25, max_age_minutes=60):
        """
        Initialize the conversation context manager
        
        Args:
            max_messages_per_channel (int): Maximum number of messages to store per channel
            max_age_minutes (int): Maximum age of messages to keep in minutes
        """
        self.max_messages = max_messages_per_channel
        self.max_age = timedelta(minutes=max_age_minutes)
        # Dictionary of channel_id -> deque of MessageContext objects
        self.channel_messages: Dict[int, Deque[MessageContext]] = defaultdict(
            lambda: deque(maxlen=max_messages_per_channel)
        )
        self.logger = logging.getLogger(__name__)
    
    def add_message(self, message):
        """
        Add a message to the context
        
        Args:
            message: Discord message object
        """
        # Create a message context object
        # Ensure timestamp is timezone-aware
        timestamp = message.created_at
        if timestamp.tzinfo is None:
            # If the timestamp is naive, make it aware using UTC timezone
            timestamp = timestamp.replace(tzinfo=timezone.utc)
            
        ctx = MessageContext(
            message_id=message.id,
            author_id=message.author.id,
            author_name=message.author.name,
            content=message.content,
            timestamp=timestamp
        )
        
        # Add to the appropriate channel's deque
        channel_id = message.channel.id
        self.channel_messages[channel_id].append(ctx)
        
        # Log debug information
        self.logger.debug(
            f"Added message to context for channel {channel_id}, "
            f"now tracking {len(self.channel_messages[channel_id])} messages"
        )
    
    def get_conversation_context(self, channel_id, num_messages=None) -> List[MessageContext]:
        """
        Get the conversation context for a channel
        
        Args:
            channel_id: Discord channel ID
            num_messages (int, optional): Number of messages to return. 
                                         If None, returns all messages up to max_messages.
        
        Returns:
            list: List of MessageContext objects, ordered from oldest to newest
        """
        # Clean up old messages first
        self._clean_old_messages(channel_id)
        
        # Get the channel's message queue
        messages = list(self.channel_messages.get(channel_id, []))
        
        # Limit to the requested number of messages
        if num_messages is not None and num_messages < len(messages):
            messages = messages[-num_messages:]
        
        return messages
    
    def get_formatted_context(self, channel_id, num_messages=None) -> List[Dict]:
        """
        Get the conversation context in a format suitable for LLM processing
        
        Args:
            channel_id: Discord channel ID
            num_messages (int, optional): Number of messages to include
            
        Returns:
            list: List of message dictionaries with author and content
        """
        messages = self.get_conversation_context(channel_id, num_messages)
        return [msg.to_dict() for msg in messages]
    
    def _clean_old_messages(self, channel_id):
        """
        Remove messages older than max_age
        
        Args:
            channel_id: Discord channel ID
        """
        if channel_id not in self.channel_messages or not self.channel_messages[channel_id]:
            return
            
        # Get current time in UTC with timezone awareness
        now = datetime.now(timezone.utc)
        
        while (self.channel_messages[channel_id] and 
               now - self.channel_messages[channel_id][0].timestamp > self.max_age):
            old_msg = self.channel_messages[channel_id].popleft()
            self.logger.debug(f"Removed old message from {old_msg.author_name} (age: {now - old_msg.timestamp})")