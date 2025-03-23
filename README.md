# LLMod

A Discord moderation bot powered by Large Language Models that helps maintain respectful conversations in your community.
Developed with the help of Claude 3.5 Sonnet.

## Overview

LLMod monitors Discord channels for disrespectful content and helps maintain a positive community atmosphere. It uses Large Language Models to analyze messages and conversation patterns, providing intelligent moderation capabilities.

## Work in Progress

⚠️ Note: This project is under active development

LLMod is currently in early development, and many improvements are needed:

- The current moderation action is limited to posting warning messages to users who violate community standards

- Future updates will include:
  - Configurable moderation actions (message deletion, timeouts, bans, etc.)
  - Better conversation tracking and context awareness
  - User reputation system
  - ...


If you're interested in contributing to any of these features, please check the open issues or submit a pull request!
## Features

- **Smart Content Moderation**: Detects disrespectful messages using powerful LLMs
- **Conversation Context Analysis**: Analyzes conversation patterns to detect problematic behaviors
- **Multi-Provider Support**: Works with various LLM providers including OpenAI, Anthropic, Mistral, Groq, and more
- **Self-hosted**: Run it on your own hardware for full control over your data
- **Customizable**: Configure moderation policies and LLM settings to match your community needs

## How It Works

LLMod monitors messages in Discord channels and analyzes them using LLMs to determine if they contain disrespectful content. When problematic content is detected, the bot issues a warning to the user and can log the incident to a moderation channel.

The bot also analyzes conversation patterns over time to detect issues that might not be obvious from single messages, such as subtle forms of harassment or coordinated disrespectful behavior.

## Setup

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- Docker (optional, for containerized deployment)
- A Discord bot token
- API key(s) for your preferred LLM provider(s)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llmod.git
cd llmod
```

2. Install dependencies with Poetry:
```bash
# Install Poetry if you don't have it
# curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --no-root
```

3. Copy the example configuration file:
```bash
cp config/config-example.yaml config/config.yaml
```

4. Edit `config/config.yaml` with your Discord bot token and LLM provider API keys.

### Configuration

The `config.yaml` file contains all the settings for the bot. Key configurations include:

```yaml
# Discord settings
bot_token: YOUR_DISCORD_BOT_TOKEN
status_message: "Monitoring messages for moderation"

# Moderation settings
max_messages: 25
log_level: INFO

# Moderation channel for logging (optional)
moderation_channel_id: YOUR_CHANNEL_ID

# LLM providers configuration
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key: YOUR_OPENAI_API_KEY
  anthropic:
    base_url: https://api.anthropic.com/v1
    api_key: YOUR_ANTHROPIC_API_KEY

# Default model to use (format: provider/model_name)
model: openai/gpt-4o
```

### Running the Bot

#### With Poetry

```bash
poetry run python main.py
```

#### Using Docker

You can also run LLMod in a Docker container:

1. Build the Docker image:
```bash
docker build -t llmod .
```

2. Run the container:
```bash
docker run -v $(pwd)/config:/app/config -v $(pwd)/logs:/app/logs llmod
```

This mounts your local config directory and logs directory to the container, ensuring your configuration is used and logs are persisted.

## Commands

LLMod supports the following Discord slash commands:

- `/ping` - Check if the bot is online
- `/info` - Get information about the bot's configuration
- `/providers` - List available LLM providers

## Advanced Configuration

### Conversation Analysis

Configure how the bot analyzes conversations:

```yaml
# Number of messages to track for context
max_messages: 25

# Maximum age of messages to keep (in minutes)
conversation_max_age: 60

# Interval between conversation analyses (in seconds)
conversation_interval: 300
```

### Moderation Channel

Set up a dedicated channel where moderation actions are logged:

```yaml
moderation_channel_id: YOUR_CHANNEL_ID
```

### Custom System Prompt

Customize the moderation criteria by modifying the system prompt:

```yaml
system_prompt: >
  You are a Discord moderation assistant. Your task is to determine if messages contain
  disrespectful content that should be moderated.
```

## Supported LLM Providers

- OpenAI (gpt-4, gpt-3.5-turbo, etc.)
- Anthropic (claude-3-opus, claude-3-sonnet, etc.)
- Mistral AI
- Groq
- OpenRouter
- Local models via Ollama, LM Studio, or vLLM

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
