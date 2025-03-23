"""
Enhanced event handlers for Discord events with conversation context support and fixes
"""
import logging
from src.discord.responses import ResponseTemplates

logger = logging.getLogger(__name__)

def register_events(client, bot):
    """Register event handlers for the Discord client"""
    
    @client.event
    async def on_ready():
        """Called when the bot is ready and connected to Discord"""
        logger.info(f"Logged in as {client.user.name} (ID: {client.user.id})")
        logger.info(f"Connected to {len(client.guilds)} guilds")
        
        # Log provider and model
        provider, model = bot.config.model.split("/", 1) if "/" in bot.config.model else ("default", bot.config.model)
        logger.info(f"Using LLM provider: {provider}, model: {model}")
        
        # Log conversation context settings
        logger.info(f"Tracking up to {bot.config.max_messages} messages per channel for context")
        
        for guild in client.guilds:
            logger.info(f"Connected to guild: {guild.name} (ID: {guild.id})")
            
        # Set up slash commands if commands module is available
        try:
            from src.discord.commands import CommandHandler
            bot.command_handler = CommandHandler(client, bot.config)
            await bot.command_handler.sync_commands()
        except ImportError:
            logger.info("Command handler not available, skipping slash command registration")
    
    @client.event
    async def on_message(message):
        """Called when a message is received"""
        try:
            # Log the message received with more detail
            logger.debug(
                f"Received message ID {message.id} from {message.author.name} "
                f"{message.content}..."
            )
            
            # Skip messages from bots including our own
            if message.author.bot:
                logger.debug(f"Skipping message from bot: {message.author.name}")
                return
                
            # Add message to the context manager for conversation tracking
            # This happens before queueing to ensure context is up to date even if processing is delayed
            if hasattr(bot, 'context_manager'):
                bot.context_manager.add_message(message)
                logger.debug(f"Added message to context manager for channel {message.channel.id}")
            
            # Add the message to the processing queue
            # The MessageQueue class should have a put_sync method for synchronous calls
            # from event handlers
            if hasattr(bot.message_queue, 'put_sync'):
                bot.message_queue.put_sync(message)
            else:
                # Fallback to the original put method if put_sync is not available
                bot.message_queue.put(message)
                
            logger.debug(f"Added message from {message.author.name} to queue for processing")
        except Exception as e:
            logger.error(f"Error in on_message handler: {e}", exc_info=True)
    
    @client.event
    async def on_message_delete(message):
        """Called when a message is deleted"""
        try:
            # We don't remove deleted messages from the context
            # This is intentional - deleted messages can still be part of problematic patterns
            logger.debug(f"Message deleted in {message.channel.name}, but kept in context history")
        except Exception as e:
            logger.error(f"Error in on_message_delete handler: {e}", exc_info=True)
    
    @client.event
    async def on_message_edit(before, after):
        """Called when a message is edited"""
        try:
            # If the message content changed, update it in our context
            if before.content != after.content and hasattr(bot, 'context_manager'):
                # For now, we just add the edited message as a new message
                # A more sophisticated approach would be to update the existing message
                logger.debug(f"Message edited in {after.channel.name}, adding updated version to context")
                bot.context_manager.add_message(after)
        except Exception as e:
            logger.error(f"Error in on_message_edit handler: {e}", exc_info=True)
    
    @client.event
    async def on_guild_join(guild):
        """Called when the bot joins a new guild"""
        try:
            logger.info(f"Joined new guild: {guild.name} (ID: {guild.id})")
            
            # Find a suitable channel to send an introduction message
            system_channel = guild.system_channel
            if system_channel and system_channel.permissions_for(guild.me).send_messages:
                welcome_message = ResponseTemplates.get_server_join_message()
                await system_channel.send(welcome_message)
        except Exception as e:
            logger.error(f"Error in on_guild_join handler: {e}", exc_info=True)
    
    @client.event
    async def on_error(event, *args, **kwargs):
        """Called when an error occurs in an event handler"""
        logger.error(f"Error in event {event}", exc_info=True)