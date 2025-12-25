"""Oneiro Discord Bot - Image generation with Diffusers."""

import os
import time
from pathlib import Path
from typing import Any

import discord
from discord import option

from oneiro.config import Config
from oneiro.filters import ContentFilter
from oneiro.pipelines import GenerationResult, PipelineManager
from oneiro.queue import GenerationQueue, QueueStatus

# Global managers (initialized on bot ready)
config: Config | None = None
pipeline_manager: PipelineManager | None = None
generation_queue: GenerationQueue | None = None
content_filter: ContentFilter | None = None


async def get_model_choices(ctx: discord.AutocompleteContext) -> list[str]:
    """Autocomplete function for model choices."""
    global config
    if config is None:
        return ["zimage-turbo"]

    models = config.get("models", default={})
    if not isinstance(models, dict):
        return ["zimage-turbo"]

    # Filter models that start with the current input
    current = ctx.value.lower() if ctx.value else ""
    return [name for name in models.keys() if name.lower().startswith(current)]


def create_bot() -> discord.Bot:
    """Create and configure the Discord bot."""
    global config, pipeline_manager, generation_queue, content_filter

    activity = discord.Activity(
        name="Dreaming...",
        type=discord.ActivityType.custom,
    )

    bot = discord.Bot(activity=activity)

    @bot.event
    async def on_ready():
        """Initialize config, pipeline and queue when bot connects."""
        global config, pipeline_manager, generation_queue, content_filter
        print(f"{bot.user} is online!")

        # Load configuration
        base_config_path = Path(os.environ.get("CONFIG_PATH", "config.toml"))
        overlay_config_path = os.environ.get("CONFIG_OVERLAY_PATH")
        state_path = os.environ.get("STATE_PATH")

        config = Config(
            base_path=base_config_path,
            overlay_path=Path(overlay_config_path) if overlay_config_path else None,
            state_path=Path(state_path) if state_path else None,
        )
        config.load()
        print(f"Config loaded from {base_config_path}")

        # Initialize content filter
        content_filter = ContentFilter(config)
        print("Content filter initialized")

        # Initialize pipeline manager with config
        pipeline_manager = PipelineManager(config)
        print("Loading default model...")
        await pipeline_manager.load_model()
        print(f"Model loaded: {pipeline_manager.current_model}")

        # Initialize queue with config values
        max_global = config.get("queue", "max_global", default=100)
        max_per_user = config.get("queue", "max_per_user", default=20)
        generation_queue = GenerationQueue(max_global=max_global, max_per_user=max_per_user)
        await generation_queue.start(pipeline_manager)
        print(f"Queue started: {max_global} global, {max_per_user} per user")

        # Register config change callback
        async def on_config_change(new_config: dict[str, Any]) -> None:
            """Update queue limits when config changes."""
            if generation_queue is None:
                return

            queue_config = new_config.get("queue", {})
            new_max_global = queue_config.get("max_global", 100)
            new_max_per_user = queue_config.get("max_per_user", 20)

            if (
                new_max_global != generation_queue.max_global
                or new_max_per_user != generation_queue.max_per_user
            ):
                generation_queue.max_global = new_max_global
                generation_queue.max_per_user = new_max_per_user
                print(f"Queue limits updated: {new_max_global} global, {new_max_per_user} per user")

        config.on_change(on_config_change)

        # Start config file watching
        await config.start_watching()
        print("Ready to generate images!")

    @bot.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
        """Handle ❌ reaction to delete generated images."""
        # Only handle ❌ reactions
        if str(payload.emoji) != "❌":
            return

        # Don't process bot's own reactions
        if payload.user_id == bot.user.id:  # type: ignore
            return

        # Fetch the message
        channel = bot.get_channel(payload.channel_id)
        if channel is None:
            return

        try:
            message = await channel.fetch_message(payload.message_id)  # type: ignore
        except discord.errors.NotFound:
            return

        # Only process bot's messages with embeds
        if message.author != bot.user or not message.embeds:
            return

        # Delete the message - anyone can delete by clicking ❌
        try:
            await message.delete()
        except discord.errors.Forbidden:
            pass  # Can't delete in this channel

    @bot.slash_command(name="dream", description="Generate an image from a prompt")
    @option(
        "prompt",
        str,
        description="The prompt for generating the image",
        required=True,
    )
    @option(
        "negative_prompt",
        str,
        description="What to avoid in the image",
        required=False,
    )
    @option(
        "image",
        discord.Attachment,
        description="Reference image for img2img",
        required=False,
    )
    @option(
        "strength",
        float,
        description="img2img strength (0.0-1.0, higher = more change)",
        required=False,
        min_value=0.0,
        max_value=1.0,
    )
    @option(
        "width",
        int,
        description="Image width",
        required=False,
        choices=[512, 768, 1024],
    )
    @option(
        "height",
        int,
        description="Image height",
        required=False,
        choices=[512, 768, 1024],
    )
    @option(
        "seed",
        int,
        description="Random seed (-1 for random)",
        required=False,
    )
    async def dream(
        ctx: discord.ApplicationContext,
        prompt: str,
        negative_prompt: str | None = None,
        image: discord.Attachment | None = None,
        strength: float = 0.75,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
    ):
        """Generate an image from a text prompt."""
        global config, pipeline_manager, generation_queue, content_filter

        if generation_queue is None or pipeline_manager is None:
            await ctx.respond("❌ Bot is still initializing, please wait...", ephemeral=True)
            return

        # Check content filter
        if content_filter is not None:
            allowed, blocked_word = content_filter.check(prompt, negative_prompt or "")
            if not allowed:
                await ctx.respond(
                    f"❌ Your prompt contains a blocked word: `{blocked_word}`",
                    ephemeral=True,
                )
                return

        # Defer immediately to avoid 3-second timeout
        await ctx.defer()

        # Download image if provided for img2img
        init_image_bytes: bytes | None = None
        if image is not None:
            try:
                init_image_bytes = await image.read()
            except Exception as e:
                await ctx.followup.send(f"❌ Failed to read image: {e}", ephemeral=True)
                return

        # Get model-specific defaults from config
        current_model = pipeline_manager.current_model or "zimage-turbo"
        model_config = config.get("models", current_model, default={}) if config else {}
        steps = model_config.get("steps", 9)
        guidance_scale = model_config.get("guidance_scale", 0.0)

        # Handle Qwen's true_cfg_scale
        if model_config.get("true_cfg_scale"):
            guidance_scale = model_config.get("true_cfg_scale", 4.0)

        # Build generation request
        request: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }

        # Add img2img parameters if image provided
        if init_image_bytes is not None:
            request["init_image"] = init_image_bytes
            request["strength"] = strength

        start_time = time.time()
        is_img2img = init_image_bytes is not None
        status_message: discord.Message | None = None

        async def on_start() -> None:
            nonlocal status_message
            if status_message:
                try:
                    await status_message.edit(content=f"🎨 Generating for **{ctx.author.name}**...")
                except discord.errors.NotFound:
                    pass

        async def on_position_update(position: int) -> None:
            nonlocal status_message
            if status_message:
                try:
                    await status_message.edit(
                        content=f"⏳ Queued at position **{position}**. Please wait..."
                    )
                except discord.errors.NotFound:
                    pass

        async def on_complete(result: GenerationResult | Exception) -> None:
            nonlocal status_message
            if isinstance(result, Exception):
                if status_message:
                    try:
                        await status_message.edit(content=f"❌ Generation failed: {result}")
                    except discord.errors.NotFound:
                        await ctx.followup.send(f"❌ Generation failed: {result}")
                else:
                    await ctx.followup.send(f"❌ Generation failed: {result}")
                return

            elapsed = time.time() - start_time
            image_buffer = pipeline_manager.image_to_bytes(result.image)  # type: ignore
            file = discord.File(image_buffer, filename="dream.png")

            embed = discord.Embed(
                title="🎨 Dream Generated" + (" (img2img)" if is_img2img else ""),
                color=discord.Color.purple(),
            )
            embed.add_field(name="Prompt", value=prompt[:1024], inline=False)
            if negative_prompt:
                embed.add_field(
                    name="Negative Prompt",
                    value=negative_prompt[:1024],
                    inline=False,
                )
            embed.add_field(name="Size", value=f"{result.width}×{result.height}", inline=True)
            embed.add_field(name="Seed", value=str(result.seed), inline=True)
            embed.add_field(name="Time", value=f"{elapsed:.1f}s", inline=True)
            embed.add_field(name="Model", value=f"`{current_model}`", inline=True)
            if is_img2img:
                embed.add_field(name="Strength", value=f"{strength:.2f}", inline=True)
            embed.set_image(url="attachment://dream.png")
            embed.set_footer(
                text=f"Requested by {ctx.author.name} • React ❌ to delete",
                icon_url=ctx.author.avatar.url if ctx.author.avatar else None,
            )

            if status_message:
                try:
                    await status_message.edit(content=None, embed=embed, file=file)
                    await status_message.add_reaction("❌")
                except discord.errors.NotFound:
                    msg = await ctx.followup.send(embed=embed, file=file)
                    try:
                        await msg.add_reaction("❌")
                    except discord.errors.Forbidden:
                        pass
            else:
                msg = await ctx.followup.send(embed=embed, file=file)
                try:
                    await msg.add_reaction("❌")
                except discord.errors.Forbidden:
                    pass

        queue_result = generation_queue.add(
            user_id=ctx.author.id,
            request=request,
            callback=on_complete,
            on_start=on_start,
            on_position_update=on_position_update,
        )

        if queue_result.status == QueueStatus.QUEUED:
            if queue_result.position > 1:
                status_message = await ctx.followup.send(
                    f"⏳ Queued at position **{queue_result.position}**. Please wait..."
                )
        elif queue_result.status == QueueStatus.USER_LIMIT:
            await ctx.followup.send(f"❌ {queue_result.message}", ephemeral=True)
        elif queue_result.status == QueueStatus.GLOBAL_LIMIT:
            await ctx.followup.send(f"❌ {queue_result.message}", ephemeral=True)

    @bot.slash_command(name="queue", description="Check your queue status")
    async def queue_status(ctx: discord.ApplicationContext):
        """Show queue status for the user."""
        global generation_queue

        if generation_queue is None:
            await ctx.respond("❌ Bot is still initializing...", ephemeral=True)
            return

        user_count = generation_queue.user_count(ctx.author.id)
        total_count = generation_queue.size

        if user_count == 0:
            await ctx.respond("✅ You have no pending requests.", ephemeral=True)
        else:
            await ctx.respond(
                f"📊 **Queue Status**\n"
                f"Your pending requests: **{user_count}** / {generation_queue.max_per_user}\n"
                f"Total queue size: **{total_count}** / {generation_queue.max_global}",
                ephemeral=True,
            )

    @bot.slash_command(name="model", description="Switch the active model")
    @option(
        "model",
        str,
        description="Model to switch to",
        required=True,
        autocomplete=get_model_choices,
    )
    async def model_command(
        ctx: discord.ApplicationContext,
        model: str,
    ):
        """Switch the active generation model."""
        global config, pipeline_manager

        if pipeline_manager is None or config is None:
            await ctx.respond("❌ Bot is still initializing...", ephemeral=True)
            return

        # Validate model exists in config
        model_config = config.get("models", model)
        if not model_config:
            available = pipeline_manager.get_available_models()
            await ctx.respond(
                f"❌ Unknown model: `{model}`\n"
                f"Available models: {', '.join(f'`{m}`' for m in available)}",
                ephemeral=True,
            )
            return

        # Check if already loaded
        if pipeline_manager.current_model == model:
            await ctx.respond(
                f"✅ Model `{model}` is already active.",
                ephemeral=True,
            )
            return

        # Defer for model loading (can be slow)
        await ctx.defer()

        try:
            loading_msg = await ctx.followup.send(f"⏳ Loading model `{model}`...")
            await pipeline_manager.load_model(model)

            # Persist model choice for next restart
            if config.state_path:
                config.set("defaults", "model", value=model)

            await loading_msg.edit(content=f"✅ Switched to model `{model}`")
        except Exception as e:
            await ctx.followup.send(f"❌ Failed to load model: {e}", ephemeral=True)

    @bot.slash_command(name="config", description="Show current configuration")
    async def config_command(ctx: discord.ApplicationContext):
        """Show current configuration values."""
        global config, pipeline_manager

        if config is None:
            await ctx.respond("❌ Config not loaded", ephemeral=True)
            return

        defaults = config.get("defaults", default={})
        queue = config.get("queue", default={})
        models = list(config.get("models", default={}).keys())
        current_model = pipeline_manager.current_model if pipeline_manager else "N/A"

        embed = discord.Embed(
            title="⚙️ Current Configuration",
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="Active Model",
            value=f"`{current_model}`",
            inline=True,
        )
        embed.add_field(
            name="Defaults",
            value=f"Model: `{defaults.get('model', 'N/A')}`\n"
            f"Size: {defaults.get('width', 'N/A')}×{defaults.get('height', 'N/A')}",
            inline=True,
        )
        embed.add_field(
            name="Queue Limits",
            value=f"Global: {queue.get('max_global', 'N/A')}\n"
            f"Per User: {queue.get('max_per_user', 'N/A')}",
            inline=True,
        )
        embed.add_field(
            name="Available Models",
            value=", ".join(f"`{m}`" for m in models) if models else "None",
            inline=False,
        )

        await ctx.respond(embed=embed, ephemeral=True)

    return bot
