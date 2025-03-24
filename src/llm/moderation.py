"""
Enhanced moderation service that uses LLM to detect disrespectful messages with conversation context
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.llm.prompts import CONVERSATION_MODERATION_PROMPT, MODERATION_PROMPT

logger = logging.getLogger(__name__)


class ModerationService:
    """
    Service that analyzes messages using LLM to detect disrespectful content
    Now with support for contextual conversation analysis
    """

    def __init__(self, llm_client, context_manager=None):
        """
        Initialize the moderation service with an LLM client

        Args:
            llm_client: The LLM client to use for moderation
            context_manager: ConversationContextManager instance (optional)
        """
        self.llm_client = llm_client
        self.context_manager = context_manager
        self._lock = asyncio.Lock()  # Add a lock to prevent concurrent API calls

    async def needs_moderation(self, message_content):
        """
        Check if a message should be moderated based on content

        Args:
            message_content (str): The content of the message to check

        Returns:
            tuple: (needs_moderation, reason)
                - needs_moderation (bool): Whether the message should be moderated
                - reason (str): The reason for moderation, if applicable
        """
        if not message_content.strip():
            return False, ""

        # Prepare the prompt with the message content
        prompt = MODERATION_PROMPT.format(message=message_content)

        # Use a lock to prevent too many concurrent API calls
        try:
            async with self._lock:
                try:
                    # Get response from LLM
                    response = await self.llm_client.get_completion(prompt)

                    # Try to parse the JSON response, handling markdown formatting
                    result = extract_json_from_llm_response(response)

                    if result:
                        needs_moderation = result.get("needs_moderation", False)
                        reason = result.get("reason", "")

                        return needs_moderation, reason

                    # Fallback if JSON extraction fails
                    logger.warning(
                        f"Failed to extract JSON from LLM response: {response}"
                    )

                    # Simple heuristic fallback
                    if "yes" in response.lower() and "because" in response.lower():
                        return (
                            True,
                            "Your message was flagged as potentially disrespectful.",
                        )
                    return False, ""

                except Exception as e:
                    logger.error(f"Error getting LLM completion: {e}")
                    return False, ""
        except Exception as e:
            logger.error(f"Lock handling error in moderation service: {e}")
            return False, ""  # Ensure we return even if lock fails

    async def analyze_conversation(self, channel_id, message_limit=None):
        """
        Analyze a conversation for disrespectful content by looking at recent messages

        Args:
            channel_id: The Discord channel ID
            message_limit (int, optional): Number of messages to analyze

        Returns:
            dict: Analysis result containing:
                - needs_moderation (bool): Whether moderation is needed
                - violators (list): List of dict with users who violated rules
                    Each dict contains user_id, user_name, and reason
                - summary (str): Summary of moderation decision
        """
        if not self.context_manager:
            logger.warning(
                "ConversationContextManager not provided, cannot analyze conversation"
            )
            return {
                "needs_moderation": False,
                "violators": [],
                "summary": "Context manager not available",
            }

        # Get conversation context
        conversation = self.context_manager.get_formatted_context(
            channel_id, message_limit
        )

        if not conversation:
            return {
                "needs_moderation": False,
                "violators": [],
                "summary": "No conversation to analyze",
            }

        # Prepare the prompt for conversation analysis
        prompt = CONVERSATION_MODERATION_PROMPT.format(
            conversation=json.dumps(conversation, indent=2)
        )

        # Use a lock to prevent too many concurrent API calls
        try:
            async with self._lock:
                try:
                    # Get response from LLM
                    response = await self.llm_client.get_completion(prompt)

                    # Try to parse the JSON response, handling markdown formatting
                    result = extract_json_from_llm_response(response)

                    if result:
                        return {
                            "needs_moderation": result.get("needs_moderation", False),
                            "violators": result.get("violators", []),
                            "summary": result.get("summary", ""),
                        }

                    # Fallback if JSON extraction fails
                    logger.warning(
                        f"Failed to extract JSON from LLM conversation analysis: {response}"
                    )
                    return {
                        "needs_moderation": False,
                        "violators": [],
                        "summary": "Error parsing LLM response",
                    }

                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")
                    return {
                        "needs_moderation": False,
                        "violators": [],
                        "summary": f"Error: {str(e)}",
                    }
        except Exception as e:
            logger.error(f"Lock handling error in conversation analysis: {e}")
            return {
                "needs_moderation": False,
                "violators": [],
                "summary": f"Error: {str(e)}",
            }


def extract_json_from_llm_response(response_text):
    """
    Extract JSON from LLM response text, handling markdown formatting

    Args:
        response_text (str): The raw response text from the LLM

    Returns:
        dict: The parsed JSON object, or empty dict if parsing fails
    """
    import json
    import re

    # First try direct JSON parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, response_text)
    if match:
        try:
            json_str = match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to find anything that looks like JSON with curly braces
    # This is a fallback for malformed responses
    json_pattern = r"\{[\s\S]*?\}"
    match = re.search(json_pattern, response_text)
    if match:
        try:
            json_str = match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Return empty dict if all parsing attempts fail
    return {}
