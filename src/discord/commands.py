"""
Command handler for Discord slash commands
"""
import logging
import discord
from discord import app_commands

logger = logging.getLogger(__name__)

class CommandHandler:
    """
    Handles Discord slash commands for the bot
    """
    def __init__(self, bot, config):
        """
        Initialize the command handler
        
        Args:
            bot: The Discord bot client
            config: The configuration object
        """
        self.bot = bot
        self.config = config
        self.command_tree = app_commands.CommandTree(bot)
        
        # Register commands
        self._register_commands()
    
    def _register_commands(self):
        """Register all slash commands"""
        
        @self.command_tree.command(name="ping", description="Check if the bot is online")
        async def ping(interaction: discord.Interaction):
            """Simple ping command to check bot status"""
            await interaction.response.send_message("Pong! Bot is online and responding.", ephemeral=True)
        
        @self.command_tree.command(name="info", description="Get information about the bot")
        async def info(interaction: discord.Interaction):
            """Display information about the bot and its configuration"""
            provider, model = self.config.model.split("/", 1) if "/" in self.config.model else ("default", self.config.model)
            
            embed = discord.Embed(
                title="DiscordLLModerator Info",
                description="A moderation bot powered by LLMs",
                color=0x5865F2
            )
            
            embed.add_field(name="LLM Provider", value=provider, inline=True)
            embed.add_field(name="Model", value=model, inline=True)
            embed.add_field(name="Status", value="Online", inline=True)
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
            
        @self.command_tree.command(name="providers", description="List available LLM providers")
        async def providers(interaction: discord.Interaction):
            """List all configured LLM providers"""
            provider_list = list(self.config.providers.keys())
            
            if not provider_list:
                await interaction.response.send_message("No LLM providers configured.", ephemeral=True)
                return
            
            embed = discord.Embed(
                title="Available LLM Providers",
                description=f"The following {len(provider_list)} providers are configured:",
                color=0x5865F2
            )
            
            current_provider = self.config.model.split("/", 1)[0] if "/" in self.config.model else "default"
            
            for provider in provider_list:
                is_current = provider == current_provider
                embed.add_field(
                    name=f"{provider} {'(current)' if is_current else ''}",
                    value="✅ Configured" if provider in self.config.providers else "❌ Not configured",
                    inline=True
                )
            
            await interaction.response.send_message(embed=embed, ephemeral=True)
    
    async def sync_commands(self):
        """Sync commands with Discord"""
        await self.command_tree.sync()
        logger.info("Slash commands synchronized with Discord")