"""
Logging utilities for DiscordLLModerator
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

def setup_logging(log_level=None):
    """
    Set up logging configuration for the application
    
    Args:
        log_level (str, optional): Log level to use. Defaults to None.
            If None, the LOG_LEVEL environment variable is used, or INFO if not set.
    """
    # Determine log level
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    # Create file handler for JSON logs
    json_handler = RotatingFileHandler(
        os.path.join(log_dir, "discord_llmoderator.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    json_handler.setLevel(log_level)
    json_formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)s"
    )
    json_handler.setFormatter(json_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(json_handler)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level {log_level}")
    
    return root_logger

def get_logger(name):
    """
    Get a logger with the given name
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: The logger instance
    """
    return logging.getLogger(name)

class LoggerMixin:
    """
    Mixin to add logging capabilities to a class
    """
    @property
    def logger(self):
        """Get a logger for this class"""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger