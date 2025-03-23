"""
Enhanced response templates for the Discord bot with conversation context support
"""
import random
import logging

logger = logging.getLogger(__name__)

class ResponseTemplates:
    """
    Templates for bot responses
    """
    
    # Warning messages for disrespectful content
    DISRESPECT_WARNINGS = [
        "Hey {user}, please be respectful in this channel. {reason}",
        "{user}, I've noticed something disrespectful in your message. {reason} Please be more mindful of how your words might affect others.",
        "Reminder {user}: We want to maintain a respectful environment here. {reason}",
        "{user}, your message was flagged as potentially disrespectful. {reason} Let's keep conversations constructive.",
        "Please watch your tone, {user}. {reason} Remember that text can sometimes come across differently than intended."
    ]
    
    # Warning messages for contextual analysis
    CONTEXTUAL_WARNINGS = [
        "Hey {user}, I've been monitoring the conversation and noticed some concerning patterns. {reason} Let's keep our discussions respectful.",
        "{user}, looking at the recent conversation history, I've identified some potentially disrespectful content. {reason}",
        "Based on the conversation context, {user}, I need to issue a warning. {reason} Please be mindful of how your messages contribute to the overall tone.",
        "After analyzing the conversation, {user}, I've detected content that may be problematic. {reason} Let's maintain a supportive atmosphere.",
        "{user}, when looking at your recent messages together, a pattern emerges that needs addressing. {reason}"
    ]
    
    # Greeting messages when bot joins a server
    SERVER_JOIN_MESSAGES = [
        "Hello! I'm DiscordLLModerator, a bot that helps keep conversations respectful. I'll be monitoring chats and may send reminders if disrespectful language is detected.",
        "Greetings! I'm DiscordLLModerator, here to help maintain a positive community atmosphere. I'll be monitoring for disrespectful language.",
        "Hi everyone! I'm DiscordLLModerator, your new moderation assistant. I use AI to help keep conversations respectful and constructive."
    ]
    
    @staticmethod
    def get_disrespect_warning(user_mention, reason):
        """
        Get a randomly selected disrespect warning message
        
        Args:
            user_mention (str): The user mention string (e.g., "<@123456789>")
            reason (str): The reason for the warning
            
        Returns:
            str: Formatted warning message
        """
        template = random.choice(ResponseTemplates.DISRESPECT_WARNINGS)
        return template.format(user=user_mention, reason=reason)
    
    @staticmethod
    def get_contextual_warning(user_mention, reason):
        """
        Get a randomly selected contextual warning message
        
        Args:
            user_mention (str): The user mention string (e.g., "<@123456789>")
            reason (str): The reason for the warning
            
        Returns:
            str: Formatted warning message
        """
        template = random.choice(ResponseTemplates.CONTEXTUAL_WARNINGS)
        return template.format(user=user_mention, reason=reason)
    
    @staticmethod
    def get_server_join_message():
        """
        Get a randomly selected server join message
        
        Returns:
            str: Server join message
        """
        return random.choice(ResponseTemplates.SERVER_JOIN_MESSAGES)

    @staticmethod
    def get_help_message():
        """
        Get the help message for the bot
        
        Returns:
            str: Help message
        """
        return (
            "**DiscordLLModerator Help**\n\n"
            "I'm a moderation bot that uses AI to detect and respond to disrespectful messages.\n\n"
            "**What I do:**\n"
            "• Monitor channels for disrespectful content\n"
            "• Analyze conversation context to detect patterns of disrespect\n"
            "• Send warnings when disrespectful messages or patterns are detected\n"
            "• Help maintain a positive community atmosphere\n\n"
            "I don't have commands yet, but I'm always watching to help keep conversations respectful."
        )

    @staticmethod
    def format_mod_log(user, channel, message_content, reason, is_conversation=False):
        """
        Format a moderation log entry
        
        Args:
            user (discord.User or dict): The user who sent the message
            channel (discord.TextChannel): The channel where the message was sent
            message_content (str): The content of the message
            reason (str): The reason for moderation
            is_conversation (bool): Whether this is from conversation analysis
            
        Returns:
            str: Formatted moderation log message
        """
        # Handle different user object types
        user_name = user.name if hasattr(user, 'name') else user.get('name', 'Unknown')
        user_id = user.id if hasattr(user, 'id') else user.get('id', 'Unknown')
        
        action_type = "**Conversation Moderation**" if is_conversation else "**Single Message Moderation**"
        
        return (
            f"{action_type}\n"
            f"**User:** {user_name} ({user_id})\n"
            f"**Channel:** {channel.name} ({channel.id})\n"
            f"**Content:** {message_content[:200]}{'...' if len(message_content) > 200 else ''}\n"
            f"**Reason:** {reason}"
        )