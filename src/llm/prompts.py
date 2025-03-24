"""
Optimized prompts for Discord moderation with scam/spam detection
"""

MODERATION_PROMPT = """
Message: "{message}"

MODERATE IF:
- Severely toxic (not banter)
- Harassment/threats
- Scams/phishing/suspicious links
- Requests for personal info
- Spam/unsolicited ads

DON'T MODERATE:
- Teasing, jokes, sarcasm
- Swearing, disagreements
- Dark humor, memes
- Casual link sharing

Respond with JSON:
{{
    "needs_moderation": true/false,
    "reason": "Brief explanation if applicable"
}}
"""

CONVERSATION_MODERATION_PROMPT = """
Conversation:
{conversation}

MODERATE IF:
- Severely toxic (not banter)
- Harassment/threats
- Scams/phishing/suspicious links
- Requests for personal info
- Spam/unsolicited ads

DON'T MODERATE:
- Teasing, jokes, sarcasm
- Swearing, disagreements
- Dark humor, memes
- Casual link sharing

Respond with JSON:
{{
    "needs_moderation": true/false,
    "violators": [
        {{
            "user_id": "user_id_string",
            "user_name": "username",
            "reason": "Brief explanation"
        }}
    ],
    "summary": "Brief decision summary"
}}
"""
