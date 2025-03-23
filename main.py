#!/usr/bin/env python3
"""
DiscordLLModerator - Main entry point
A Discord bot that uses LLM to moderate conversations
"""
import asyncio
import logging
import signal
import sys
from src.bot import DiscordLLModerator
from src.config import load_config
from src.utils.logging import setup_logging

async def main():
    """Main entry point for the DiscordLLModerator bot"""
    # Load configuration first
    try:
        config = load_config("config/config.yaml")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Set up logging based on config
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    
    # Configuration is already loaded
    
    # Create and run the bot
    bot = DiscordLLModerator(config)
    
    # Set up graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(bot.shutdown())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        logger.info("Starting DiscordLLModerator bot")
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
    finally:
        # Close any remaining resources
        if hasattr(bot, 'cleanup'):
            await bot.cleanup()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)