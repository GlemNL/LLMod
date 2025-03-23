"""
Enhanced client for interacting with various LLM providers
"""
import logging
import asyncio
import json
from typing import Dict, Any, Tuple, Optional, AsyncGenerator
from datetime import datetime

import aiohttp

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for making requests to different LLM APIs
    Supports multiple providers through a unified interface
    """
    def __init__(self, config):
        """
        Initialize the LLM client with configuration
        
        Args:
            config: Configuration object with providers and settings
        """
        self.config = config
        self.session = None
        
        # Define constants
        self.OPENAI_COMPATIBLE_PROVIDERS = [
            "openai", "groq", "openrouter", "ollama", 
            "lmstudio", "vllm", "oobabooga"
        ]
        self.ANTHROPIC_PROVIDERS = ["anthropic"]
        self.VISION_MODEL_TAGS = [
            "gpt-4", "claude-3", "gemini", "gemma",
            "pixtral", "mistral-small", "llava", "vision", "vl",
        ]
    
    async def ensure_session(self):
        """Ensure an aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the aiohttp session if it exists"""
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None
    
    def get_provider_and_model(self) -> Tuple[str, str]:
        """
        Parse the model string into provider and model name
        
        Returns:
            tuple: (provider, model)
        """
        if "/" not in self.config.model:
            # Default to OpenAI if no provider specified
            return "openai", self.config.model
        
        provider, model = self.config.model.split("/", 1)
        return provider, model
    
    def model_supports_images(self, model: str) -> bool:
        """
        Check if the model supports image inputs
        
        Args:
            model (str): The model name
            
        Returns:
            bool: True if the model supports images
        """
        return any(tag in model.lower() for tag in self.VISION_MODEL_TAGS)
    
    def prepare_system_message(self) -> Dict[str, str]:
        """
        Prepare the system message with appropriate context
        
        Returns:
            dict: The system message in the appropriate format
        """
        if not self.config.system_prompt:
            return {}

        system_prompt_extras = [f"Today's date: {datetime.now().strftime('%B %d %Y')}."]
        full_system_prompt = "\n".join([self.config.system_prompt] + system_prompt_extras)
        
        return {"role": "system", "content": full_system_prompt}
    
    async def get_completion(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        """
        Get a completion from the LLM API
        
        Args:
            prompt (str): The prompt to send to the LLM
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (0.0 = deterministic)
            
        Returns:
            str: The generated completion text
        """
        provider, model = self.get_provider_and_model()
        
        if provider in self.OPENAI_COMPATIBLE_PROVIDERS:
            return await self._get_openai_completion(
                provider, model, prompt, max_tokens, temperature
            )
        elif provider in self.ANTHROPIC_PROVIDERS:
            return await self._get_anthropic_completion(
                provider, model, prompt, max_tokens, temperature
            )
        # Add other provider types as needed
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _get_openai_completion(
        self, provider: str, model: str, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """
        Get a completion from an OpenAI-compatible API
        
        Args:
            provider (str): The provider name
            model (str): The model name
            prompt (str): The prompt to send to the LLM
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: The generated completion text
        """
        await self.ensure_session()
        
        provider_config = self.config.providers.get(provider, {})
        base_url = provider_config.get("base_url", "https://api.openai.com/v1")
        api_key = provider_config.get("api_key", "")
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authentication header if API key is provided
        if api_key:
            # Different providers might use different auth header formats
            if provider == "openrouter":
                headers["Authorization"] = f"Bearer {api_key}"
                headers["HTTP-Referer"] = "https://github.com/yourname/discordllmoderator"
                headers["X-Title"] = "DiscordLLModerator"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        
        # Prepare the payload with system message and user prompt
        messages = []
        
        # Add system message if configured
        system_message = self.prepare_system_message()
        if system_message:
            messages.append(system_message)
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Get additional parameters from config
        extra_params = self.config.extra_api_parameters.copy() if hasattr(self.config, "extra_api_parameters") else {}
        
        # Override with provided parameters
        extra_params.update({
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        
        payload = {
            "model": model,
            "messages": messages,
            **extra_params
        }
        
        try:
            async with self.session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                raise_for_status=True
            ) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error calling {provider} API: {e}")
            return f"Error generating response: {str(e)}"
    
    async def _get_anthropic_completion(
        self, provider: str, model: str, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """
        Get a completion from Anthropic API
        
        Args:
            provider (str): The provider name
            model (str): The model name
            prompt (str): The prompt to send to the LLM
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: The generated completion text
        """
        await self.ensure_session()
        
        provider_config = self.config.providers.get(provider, {})
        base_url = provider_config.get("base_url", "https://api.anthropic.com/v1")
        api_key = provider_config.get("api_key", "")
        
        if not api_key:
            return "Error: Anthropic API key is required"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Prepare system message
        system = self.config.system_prompt
        
        # Prepare the payload
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add system message if present
        if system:
            payload["system"] = system
        
        try:
            async with self.session.post(
                f"{base_url}/messages",
                headers=headers,
                json=payload,
                raise_for_status=True
            ) as response:
                result = await response.json()
                return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_response(self, messages) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Generate a streaming response from the LLM
        Yields tuples of (content_delta, finish_reason)
        
        Args:
            messages: The messages to send to the LLM
            
        Yields:
            tuple: (content_delta, finish_reason)
        """
        provider, model = self.get_provider_and_model()
        
        if provider in self.OPENAI_COMPATIBLE_PROVIDERS:
            async for content, finish_reason in self._generate_openai_stream(provider, model, messages):
                yield content, finish_reason
        elif provider in self.ANTHROPIC_PROVIDERS:
            async for content, finish_reason in self._generate_anthropic_stream(provider, model, messages):
                yield content, finish_reason
        # Add other provider types as needed
        else:
            yield f"Error: Unsupported provider {provider}", "error"
    
    async def _generate_openai_stream(self, provider: str, model: str, messages) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Generate a streaming response from an OpenAI-compatible API
        
        Args:
            provider (str): The provider name
            model (str): The model name
            messages: The messages to send to the LLM
            
        Yields:
            tuple: (content_delta, finish_reason)
        """
        await self.ensure_session()
        
        provider_config = self.config.providers.get(provider, {})
        base_url = provider_config.get("base_url", "https://api.openai.com/v1")
        api_key = provider_config.get("api_key", "")
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authentication header if API key is provided
        if api_key:
            # Different providers might use different auth header formats
            if provider == "openrouter":
                headers["Authorization"] = f"Bearer {api_key}"
                headers["HTTP-Referer"] = "https://github.com/yourname/discordllmoderator"
                headers["X-Title"] = "DiscordLLModerator"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        
        # Get additional parameters from config
        extra_params = self.config.extra_api_parameters.copy() if hasattr(self.config, "extra_api_parameters") else {}
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **extra_params
        }
        
        try:
            async with self.session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error from {provider} API: {response.status} - {error_text}")
                    yield f"Error: {response.status} - {error_text}", "error"
                    return
                
                # Process streaming response
                buffer = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        break
                    
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        try:
                            chunk = json.loads(data)
                            
                            # Extract content and finish_reason
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                            content = delta.get("content", "")
                            
                            yield content, finish_reason
                            
                            if finish_reason is not None:
                                break
                                
                        except json.JSONDecodeError:
                            buffer += data
                            
                # Handle any buffered data
                if buffer:
                    try:
                        chunk = json.loads(buffer)
                        content = chunk.get("choices", [{}])[0].get("message", {}).get("content", "")
                        finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
                        yield content, finish_reason
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.error(f"Error in streaming from {provider} API: {e}")
            yield f"Error generating response: {str(e)}", "error"
    
    async def _generate_anthropic_stream(self, provider: str, model: str, messages) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Generate a streaming response from Anthropic API
        
        Args:
            provider (str): The provider name
            model (str): The model name
            messages: The messages to send to the LLM
            
        Yields:
            tuple: (content_delta, finish_reason)
        """
        await self.ensure_session()
        
        provider_config = self.config.providers.get(provider, {})
        base_url = provider_config.get("base_url", "https://api.anthropic.com/v1")
        api_key = provider_config.get("api_key", "")
        
        if not api_key:
            yield "Error: Anthropic API key is required", "error"
            return
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Get additional parameters from config
        extra_params = self.config.extra_api_parameters.copy() if hasattr(self.config, "extra_api_parameters") else {}
        
        # Extract system message if present
        system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
        system = system_message.get("content", "") if system_message else self.config.system_prompt
        
        # Filter out system messages as Anthropic handles them differently
        filtered_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        payload = {
            "model": model,
            "messages": filtered_messages,
            "stream": True,
            **extra_params
        }
        
        # Add system message if present
        if system:
            payload["system"] = system
        
        try:
            async with self.session.post(
                f"{base_url}/messages",
                headers=headers,
                json=payload,
                timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error from Anthropic API: {response.status} - {error_text}")
                    yield f"Error: {response.status} - {error_text}", "error"
                    return
                
                # Process streaming response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        break
                    
                    if line.startswith("data: "):
                        data = line[6:]  # Remove 'data: ' prefix
                        try:
                            chunk = json.loads(data)
                            
                            # Extract content and stop_reason
                            delta = chunk.get("delta", {})
                            stop_reason = chunk.get("stop_reason")
                            content = delta.get("text", "")
                            
                            # Convert Anthropic's stop_reason to OpenAI-like finish_reason
                            finish_reason = None
                            if stop_reason:
                                finish_reason = "stop" if stop_reason == "end_turn" else stop_reason
                            
                            yield content, finish_reason
                            
                            if finish_reason is not None:
                                break
                                
                        except json.JSONDecodeError:
                            pass
                        
        except Exception as e:
            logger.error(f"Error in streaming from Anthropic API: {e}")
            yield f"Error generating response: {str(e)}", "error"