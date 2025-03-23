"""
Optimized prompts for Discord moderation with scam/spam detection
"""

MODERATION_PROMPT = """
Determine if this message needs moderation:

Message: "{message}"

Flag patterns of:
- Severe toxicity beyond banter
- Harassment, threats
- Scams, phishing, suspicious links
- Message spam or flooding
- Unsolicited advertising

Respond with JSON:
{{
    "needs_moderation": true/false,
    "reason": "Brief explanation if applicable"
}}

DON'T flag:
- Teasing, jokes, sarcasm
- Swearing
- Wild Disagreements or dark humor
- Memes or occasional content sharing via harmless links

Only moderate genuinely harmful behavior. Watch for suspicious links, unrealistic promises, personal info requests, or promotional patterns.

Respond ONLY with the JSON object.
"""

CONVERSATION_MODERATION_PROMPT = """
You're a Discord mod for a friendly community. Analyze this conversation for moderation needs:

Conversation:
{conversation}

Flag patterns of:
- Severe toxicity beyond banter
- Harassment, threats
- Scams, phishing, suspicious links
- Message spam or flooding
- Unsolicited advertising

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

DON'T flag:
- Teasing, jokes, sarcasm
- Swearing
- Wild Disagreements or dark humor
- Memes or occasional content sharing via harmless links

Only moderate genuinely harmful behavior. Watch for suspicious links, unrealistic promises, personal info requests, or promotional patterns.

Respond ONLY with the JSON object.
"""