"""
Discord client setup for DiscordLLModerator
"""
import logging
import discord

logger = logging.getLogger(__name__)

def setup_discord_client(intents):
    """
    Set up and configure the Discord client
    
    Args:
        intents (discord.Intents): The intents to use for the client
        
    Returns:
        discord.Client: The configured Discord client
    """
    # Create the Discord client
    client = discord.Client(intents=intents)
    
    logger.info("Discord client initialized with required intents")
    
    return client