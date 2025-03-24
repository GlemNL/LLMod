"""
Enhanced main bot class for DiscordLLModerator with conversation context analysis and fixed error handling
"""

import asyncio
import logging
import time
from typing import Any, Dict

from discord import Intents
from src.discord.client import setup_discord_client
from src.discord.events import register_events
from src.discord.responses import ResponseTemplates
from src.llm.client import LLMClient
from src.llm.moderation import ModerationService
from src.utils.conversation_context import ConversationContextManager
from src.utils.logging import setup_logging
from src.utils.message_queue import MessageQueue


class DiscordLLModerator:
    """
    Main bot class that manages the Discord client and LLM moderation
    Now with support for conversation context analysis and improved error handling
    """

    def __init__(self, config):
        """Initialize the bot with provided configuration"""
        # Set up logging
        setup_logging(config.log_level)
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.running = False

        # Set up Discord client with required intents
        intents = Intents.default()
        intents.message_content = True  # Required to read message content
        intents.guild_messages = True  # Required to read guild messages

        # Initialize components
        self.message_queue = MessageQueue(max_size=100)

        # Initialize conversation context manager
        self.context_manager = ConversationContextManager(
            max_messages_per_channel=config.max_messages,
            max_age_minutes=config.conversation_max_age,
        )

        self.llm_client = LLMClient(config)

        # Pass the context manager to the moderation service
        self.moderation_service = ModerationService(
            self.llm_client, self.context_manager
        )

        self.discord_client = setup_discord_client(intents)

        # Register event handlers
        register_events(self.discord_client, self)

        # Keep track of channels and when they were last analyzed
        self.channel_last_analyzed = {}
        self.channel_analysis_interval = (
            config.conversation_interval
        )  # Interval between conversation analyses

    async def process_message_queue(self):
        """Process messages in the queue and check for moderation issues"""
        self.logger.info("Message queue processor started")

        # Track processing stats
        stats: Dict[str, Any] = {
            "messages_processed": 0,
            "messages_moderated": 0,
            "conversations_analyzed": 0,
            "last_report_time": time.time(),
        }

        while self.running:  # Changed from "while True" for clarity
            try:
                # Process messages in the queue
                if not self.message_queue.is_empty():
                    message = await self.message_queue.get()  # Use async get

                    if message is None:
                        await asyncio.sleep(0.1)  # Short sleep if no message
                        continue

                    # Skip messages from bots including our own
                    if message.author.bot:
                        continue

                    self.logger.debug(f"Processing message: {message.content[:200]}...")
                    stats["messages_processed"] += 1

                    try:
                        # Get moderation result for individual message from LLM
                        needs_moderation, reason = (
                            await self.moderation_service.needs_moderation(
                                message.content
                            )
                        )

                        self.logger.debug(
                            f"Moderation result for message: {needs_moderation}, reason: {reason[:200] if reason else 'None'}..."
                        )

                        if needs_moderation:
                            stats["messages_moderated"] += 1
                            self.logger.info(
                                f"Moderating message from {message.author.name}: {reason}"
                            )

                            # Use response template for warning message
                            warning_message = ResponseTemplates.get_disrespect_warning(
                                message.author.mention, reason
                            )
                            await message.channel.send(warning_message)

                            # Log to moderation channel if configured
                            await self._log_to_moderation_channel(
                                message.author, message.channel, message.content, reason
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Error processing individual message: {e}", exc_info=True
                        )

                    try:
                        # Check if we should perform conversation analysis
                        channel_id = message.channel.id
                        current_time = time.time()

                        # Check if it's time to analyze this channel's conversation
                        if channel_id not in self.channel_last_analyzed or (
                            current_time - self.channel_last_analyzed[channel_id]
                            >= self.channel_analysis_interval
                        ):

                            # Update the last analyzed time
                            self.channel_last_analyzed[channel_id] = current_time

                            # Analyze the conversation
                            await self._analyze_conversation(message.channel)
                            stats["conversations_analyzed"] += 1
                    except Exception as e:
                        self.logger.error(
                            f"Error during conversation analysis: {e}", exc_info=True
                        )

                # Report stats periodically (every 10 minutes)
                current_time = time.time()
                if current_time - stats["last_report_time"] > 600:  # 10 minutes
                    if stats["messages_processed"] > 0:
                        moderation_rate = (
                            stats["messages_moderated"] / stats["messages_processed"]
                        ) * 100
                        self.logger.info(
                            f"Stats: Processed {stats['messages_processed']} messages, "
                            f"moderated {stats['messages_moderated']} "
                            f"({moderation_rate:.1f}%), "
                            f"analyzed {stats['conversations_analyzed']} conversations"
                        )
                    stats["last_report_time"] = current_time

                # Sleep to prevent hammering the CPU
                await asyncio.sleep(
                    0.1
                )  # Reduced sleep time for more responsive processing
            except Exception as e:
                self.logger.error(
                    f"Unhandled error in message processor: {e}", exc_info=True
                )
                # Continue running even after errors - don't break the loop
                await asyncio.sleep(1)  # Sleep a bit longer after an error

    async def _analyze_conversation(self, channel):
        """
        Analyze conversation context for a channel

        Args:
            channel: Discord channel object
        """
        try:
            self.logger.debug(f"Analyzing conversation in channel {channel.name}")

            analysis = await self.moderation_service.analyze_conversation(
                channel.id, message_limit=self.config.max_messages
            )

            if analysis["needs_moderation"]:
                self.logger.info(
                    f"Conversation moderation needed in {channel.name}: "
                    f"{len(analysis['violators'])} violators found"
                )

                # Handle each violator
                for violator in analysis["violators"]:
                    # Use response template for warning message
                    user_id = violator["user_id"]
                    user_name = violator["user_name"]
                    reason = violator["reason"]

                    # Format the mention
                    user_mention = f"<@{user_id}>"

                    # Send contextual warning
                    warning_message = ResponseTemplates.get_contextual_warning(
                        user_mention, reason
                    )
                    await channel.send(warning_message)

                    # Log to moderation channel
                    await self._log_to_moderation_channel(
                        {
                            "id": user_id,
                            "name": user_name,
                        },  # Mock user object with necessary fields
                        channel,
                        "Multiple messages in conversation",
                        reason,
                        is_conversation=True,
                    )
            else:
                self.logger.debug(
                    f"No conversation moderation needed in {channel.name}"
                )
        except Exception as e:
            self.logger.error(f"Error in _analyze_conversation: {e}", exc_info=True)

    async def _log_to_moderation_channel(
        self, user, channel, content, reason, is_conversation=False
    ):
        """
        Log moderation action to the moderation channel

        Args:
            user: Discord user object or dict with id and name
            channel: Discord channel object
            content: Message content
            reason: Moderation reason
            is_conversation: Whether this is from conversation analysis
        """
        if not self.config.moderation_channel_id:
            return

        try:
            mod_channel = await self.discord_client.fetch_channel(
                int(self.config.moderation_channel_id)
            )

            mod_log = ResponseTemplates.format_mod_log(
                user, channel, content, reason, is_conversation
            )

            await mod_channel.send(mod_log)
        except Exception as e:
            self.logger.error(f"Failed to log to moderation channel: {e}")

    async def run(self):
        """Run the Discord bot and message processor"""
        # Set the running flag
        self.running = True

        try:
            # Start the message processing task
            self.processor_task = asyncio.create_task(self.process_message_queue())

            # Log startup information with provider and model
            provider, model = (
                self.config.model.split("/", 1)
                if "/" in self.config.model
                else ("default", self.config.model)
            )
            self.logger.info(
                f"Starting bot with LLM provider: {provider}, model: {model}"
            )
            self.logger.info(
                f"Conversation analysis will run every {self.channel_analysis_interval} seconds"
            )
            self.logger.info(
                f"Keeping message context for up to {self.config.conversation_max_age} minutes"
            )

            # Start the Discord client
            await self.discord_client.start(self.config.bot_token)
        except Exception as e:
            self.running = False
            self.logger.error(f"Error starting bot: {e}", exc_info=True)
            raise

    async def shutdown(self):
        """Handle graceful shutdown of the bot"""
        self.logger.info("Shutting down bot")
        self.running = False

        # Close the Discord client
        if self.discord_client:
            await self.discord_client.close()

        # Cancel the message processor task
        if hasattr(self, "processor_task") and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                self.logger.info("Message processor task cancelled")
            except Exception as e:
                self.logger.error(
                    f"Error cancelling processor task: {e}", exc_info=True
                )

    async def cleanup(self):
        """Clean up resources on shutdown"""
        # Close the LLM client if it has a close method
        if hasattr(self.llm_client, "close"):
            try:
                await self.llm_client.close()
            except Exception as e:
                self.logger.error(f"Error closing LLM client: {e}", exc_info=True)

        self.logger.info("Resource cleanup complete")
