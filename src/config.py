"""
Enhanced configuration loading and management for DiscordLLModerator with conversation context support
"""

import logging
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for DiscordLLModerator
    Handles loading configuration from YAML file
    """

    def __init__(self, config_file: str = "config/config.yaml"):
        """
        Initialize configuration from file

        Args:
            config_file (str): Path to the configuration file
        """
        self.config_file = config_file
        self.data = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Returns:
            dict: Configuration data
        """
        try:
            with open(self.config_file, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(
                f"Configuration file {self.config_file} not found, using empty configuration"
            )
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file

        Returns:
            dict: Updated configuration data
        """
        self.data = self._load_config()
        return self.data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value

        Args:
            key (str): The configuration key
            default: Default value if key not found

        Returns:
            The configuration value or default
        """
        return self.data.get(key, default)

    @property
    def bot_token(self) -> str:
        """Get the Discord bot token"""
        return self.data.get("bot_token", "")

    @property
    def status_message(self) -> str:
        """Get the bot status message"""
        return self.data.get("status_message", "")

    @property
    def log_level(self) -> str:
        """Get the logging level"""
        return self.data.get("log_level", "INFO")

    @property
    def max_messages(self) -> int:
        """Get the maximum number of messages to store for context"""
        return self.data.get("max_messages", 25)

    @property
    def conversation_interval(self) -> int:
        """Get the interval in seconds between conversation analyses"""
        return self.data.get("conversation_interval", 300)  # Default: 5 minutes

    @property
    def conversation_max_age(self) -> int:
        """Get the maximum age in minutes of messages to keep for context"""
        return self.data.get("conversation_max_age", 60)  # Default: 60 minutes

    @property
    def moderation_channel_id(self) -> Optional[str]:
        """Get the moderation channel ID"""
        return self.data.get("moderation_channel_id")

    @property
    def moderation_threshold(self) -> float:
        """Get the moderation confidence threshold"""
        return float(self.data.get("moderation_threshold", 0.7))

    @property
    def providers(self) -> Dict[str, Dict[str, str]]:
        """Get the configured LLM providers"""
        return self.data.get("providers", {})

    @property
    def model(self) -> str:
        """Get the configured model"""
        return self.data.get("model", "")

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return self.data.get("system_prompt", "")

    @property
    def extra_api_parameters(self) -> Dict[str, Any]:
        """Get extra API parameters for LLM requests"""
        return self.data.get("extra_api_parameters", {})


def load_config(config_file: str = "config/config.yaml") -> Config:
    """
    Load the configuration

    Args:
        config_file (str): Path to the configuration file

    Returns:
        Config: Configuration object
    """
    config = Config(config_file)

    # Check for missing required values
    missing = []
    if not config.bot_token:
        missing.append("bot_token")

    if not config.model:
        missing.append("model")

    if not config.providers:
        missing.append("providers")

    if missing:
        logger.warning(f"Missing required configuration values: {', '.join(missing)}")

    return config
