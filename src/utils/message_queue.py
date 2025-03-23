"""
Queue for handling Discord messages for moderation
"""
import logging
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

class MessageQueue:
    """
    A queue for processing Discord messages
    
    This queue allows the bot to process messages asynchronously,
    preventing the bot from getting overwhelmed during high traffic.
    """
    def __init__(self, max_size=100):
        """
        Initialize the message queue
        
        Args:
            max_size (int): Maximum number of messages to keep in the queue
        """
        self.queue = deque(maxlen=max_size)
        self.max_size = max_size
        self._lock = asyncio.Lock()  # Add lock to prevent race conditions
    
    async def put(self, message):
        """
        Add a message to the queue (async version)
        
        Args:
            message: The Discord message to add
        """
        async with self._lock:
            # If the queue is full, log a warning
            if len(self.queue) >= self.max_size:
                logger.warning("Message queue is full, dropping oldest message")
            
            # Add the message to the queue
            self.queue.append(message)
            logger.debug(f"Added message to queue, current size: {len(self.queue)}")
    
    def put_sync(self, message):
        """
        Add a message to the queue (sync version for legacy compatibility)
        
        Args:
            message: The Discord message to add
        """
        # If the queue is full, log a warning
        if len(self.queue) >= self.max_size:
            logger.warning("Message queue is full, dropping oldest message")
        
        # Add the message to the queue
        self.queue.append(message)
        logger.debug(f"Added message to queue, current size: {len(self.queue)}")
    
    async def get(self):
        """
        Get the next message from the queue
        
        Returns:
            The next Discord message, or None if the queue is empty
        """
        async with self._lock:
            if self.is_empty():
                return None
            
            message = self.queue.popleft()
            logger.debug(f"Retrieved message from queue, remaining: {len(self.queue)}")
            return message
    
    def is_empty(self):
        """
        Check if the queue is empty
        
        Returns:
            bool: True if the queue is empty, False otherwise
        """
        return len(self.queue) == 0
    
    def size(self):
        """
        Get the current size of the queue
        
        Returns:
            int: The number of messages in the queue
        """
        return len(self.queue)