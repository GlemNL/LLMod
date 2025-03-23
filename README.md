# LLMod

A Discord bot that uses Large Language Models (LLMs) to moderate conversations and detect disrespectful messages.
Developed with the help of Claude 3.5 Sonnet.

## Features

- Connects to Discord and monitors all channels
- Analyzes messages in real-time using LLM technology
- Automatically identifies and responds to disrespectful messages
- Supports multiple LLM providers (OpenAI, Anthropic, Mistral, etc.)
- Customizable moderation settings and thresholds

## Setup

### Prerequisites

- Python 3.11 or higher
- Discord Bot Token (from [Discord Developer Portal](https://discord.com/developers/applications))
- LLM API Key (from one of the supported providers)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/LLMod.git
   cd LLMod
   ```

2. Install the required dependencies:
   ```
   poetry install --no-root
   ```

3. Create a configuration file:
   ```
   cp config/config-example.yaml config/config.yaml
   ```

4. Edit the `config.yaml` file and add your Discord token and LLM API key(s).

### Discord Bot Setup

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Navigate to the "Bot" tab and click "Add Bot"
4. Under "Privileged Gateway Intents", enable:
   - Message Content Intent
   - Server Members Intent
5. Copy the bot token and add it to your `config.yaml` file
6. Use the following URL to add the bot to your server (replace YOUR_CLIENT_ID):
   ```
   https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=274878221376&scope=bot
   ```

## Running the Bot

Start the bot with:

```
python main.py
```

## LLM Provider Configuration

The bot supports multiple LLM providers. Configure them in the `config.yaml` file:

```yaml
providers:
  openai:
    base_url: https://api.openai.com/v1
    api_key: YOUR_OPENAI_API_KEY
  anthropic:
    base_url: https://api.anthropic.com/v1
    api_key: YOUR_ANTHROPIC_API_KEY
  mistral:
    base_url: https://api.mistral.ai/v1
    api_key: YOUR_MISTRAL_API_KEY
  # Add more providers as needed
```

Specify which model to use with the `model` setting:

```yaml
# Format: provider/model_name
model: openai/gpt-4o
```

## Available Providers

- `openai`: OpenAI API (GPT-4, GPT-3.5, etc.)
- `anthropic`: Anthropic API (Claude models)
- `mistral`: Mistral AI API
- `groq`: Groq API
- `openrouter`: OpenRouter API
- `ollama`: Local Ollama server
- `lmstudio`: Local LM Studio server
- `vllm`: Local vLLM server

## Slash Commands

The bot provides several slash commands:

- `/ping`: Check if the bot is online
- `/info`: Get information about the bot's configuration
- `/providers`: List all configured LLM providers

## How It Works

1. The bot connects to Discord and listens for messages in all channels it has access to
2. When a message is received, it's added to a processing queue
3. The LLM analyzes the message content to determine if it contains disrespectful language
4. If disrespectful content is detected, the bot responds in the same channel with a warning to the user

## Extending the Bot

You can extend the bot's functionality by:

- Adding new moderation types beyond just disrespect
- Implementing more sophisticated response templates
- Adding admin commands for configuration
- Creating a web dashboard for analytics and settings
