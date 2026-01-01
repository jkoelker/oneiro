"""Application factory for Oneiro Discord bot."""

import os
from pathlib import Path
from typing import Any

import discord

from oneiro.civitai import CivitaiClient
from oneiro.config import Config
from oneiro.discord.commands import register_commands
from oneiro.discord.handlers import create_config_change_handler, handle_reaction_delete
from oneiro.filters import ContentFilter
from oneiro.lora_detector import AutoLoraDetector, create_detector_from_config
from oneiro.pipelines import PipelineManager
from oneiro.queue import GenerationQueue


class OneiroBot(discord.Bot):
    """Oneiro Discord bot with typed state management."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config: Config | None = None
        self.pipeline_manager: PipelineManager | None = None
        self.generation_queue: GenerationQueue | None = None
        self.content_filter: ContentFilter | None = None
        self.civitai_client: CivitaiClient | None = None
        self.lora_detector: AutoLoraDetector | None = None


def create_app() -> OneiroBot:
    """Create and wire the Discord bot with all services.

    Returns:
        Configured OneiroBot instance ready to run.
    """
    activity = discord.Activity(
        name="Dreaming...",
        type=discord.ActivityType.custom,
    )

    bot = OneiroBot(activity=activity)

    # Register slash commands
    register_commands(bot)

    @bot.event
    async def on_ready() -> None:
        """Initialize config, pipeline and queue when bot connects."""
        print(f"{bot.user} is online!")

        # Load configuration
        base_config_path = Path(os.environ.get("CONFIG_PATH", "config.toml"))
        overlay_config_path = os.environ.get("CONFIG_OVERLAY_PATH")
        state_path = os.environ.get("STATE_PATH")

        bot.config = Config(
            base_path=base_config_path,
            overlay_path=Path(overlay_config_path) if overlay_config_path else None,
            state_path=Path(state_path) if state_path else None,
        )
        bot.config.load()
        print(f"Config loaded from {base_config_path}")

        # Initialize Civitai client
        bot.civitai_client = CivitaiClient.from_config(bot.config)
        print("Civitai client initialized")

        # Initialize content filter
        bot.content_filter = ContentFilter(bot.config)
        print("Content filter initialized")

        # Initialize LoRA auto-detector
        bot.lora_detector = create_detector_from_config(bot.config.data)
        print("LoRA auto-detector initialized")

        # Initialize pipeline manager with config
        bot.pipeline_manager = PipelineManager(bot.config)
        bot.pipeline_manager.set_civitai_client(bot.civitai_client)
        print("Loading default model...")
        await bot.pipeline_manager.load_model()
        print(f"Model loaded: {bot.pipeline_manager.current_model}")

        # Initialize queue with config values
        max_global = bot.config.get("queue", "max_global", default=100)
        max_per_user = bot.config.get("queue", "max_per_user", default=20)
        bot.generation_queue = GenerationQueue(max_global=max_global, max_per_user=max_per_user)
        await bot.generation_queue.start(bot.pipeline_manager)
        print(f"Queue started: {max_global} global, {max_per_user} per user")

        # Register config change callback
        bot.config.on_change(create_config_change_handler(bot))

        # Start config file watching
        await bot.config.start_watching()

        # Sync slash commands to all guilds for instant availability
        # (global commands can take up to 1 hour to propagate)
        if bot.guilds:
            guild_ids = [g.id for g in bot.guilds]
            await bot.sync_commands(guild_ids=guild_ids)
            print(f"Commands synced to {len(guild_ids)} guild(s): {[g.name for g in bot.guilds]}")

        print("Ready to generate images!")

    @bot.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
        """Handle ❌ reaction to delete generated images."""
        await handle_reaction_delete(bot, payload)

    return bot
