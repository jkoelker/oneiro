"""Oneiro Discord Bot - Image generation with Diffusers."""

import os
import re
import time
from pathlib import Path
from typing import Any

import discord
from discord import option

from oneiro.civitai import CivitaiClient, CivitaiError
from oneiro.config import Config
from oneiro.filters import ContentFilter
from oneiro.pipelines import (
    SCHEDULER_CHOICES,
    GenerationResult,
    LoraConfig,
    LoraSource,
    PipelineManager,
)
from oneiro.pipelines.civitai_checkpoint import CivitaiCheckpointPipeline
from oneiro.pipelines.lora import is_lora_compatible, parse_civitai_url
from oneiro.queue import GenerationQueue, QueueStatus

# Global managers (initialized on bot ready)
config: Config | None = None
pipeline_manager: PipelineManager | None = None
generation_queue: GenerationQueue | None = None
content_filter: ContentFilter | None = None
civitai_client: CivitaiClient | None = None

# LoRA weight validation limits
MIN_LORA_WEIGHT = -2.0
MAX_LORA_WEIGHT = 2.0

# Steps validation limits (for /dream and /model commands)
MIN_STEPS = 1
MAX_STEPS = 100

# Guidance scale validation limits (for /dream and /model commands)
MIN_GUIDANCE_SCALE = 0.0
MAX_GUIDANCE_SCALE = 15.0


def validate_lora_weight(weight: float, lora_name: str) -> None:
    """Validate that a LoRA weight is within acceptable bounds.

    Args:
        weight: The LoRA weight value to validate
        lora_name: Name/identifier of the LoRA (for error messages)

    Raises:
        ValueError: If weight is outside the valid range [-2.0, 2.0]
    """
    if weight < MIN_LORA_WEIGHT or weight > MAX_LORA_WEIGHT:
        raise ValueError(
            f"LoRA weight {weight} for '{lora_name}' is out of range. "
            f"Valid range is [{MIN_LORA_WEIGHT}, {MAX_LORA_WEIGHT}]."
        )


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug.

    Args:
        text: Input text (e.g., model name from Civitai)

    Returns:
        Lowercase slug with hyphens instead of spaces/special chars
    """
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().strip()
    # Remove special characters except alphanumeric and hyphens
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    # Replace spaces and multiple hyphens with single hyphen
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    return slug or "unnamed"


def parse_lora_param(lora_str: str) -> list[tuple[str, float]]:
    """Parse lora parameter string into list of (name/id, weight) tuples.

    Supports formats:
    - "lora-name" -> ("lora-name", 1.0)
    - "lora-name:0.8" -> ("lora-name", 0.8)
    - "civitai:12345" -> ("civitai:12345", 1.0)
    - "civitai:12345:0.7" -> ("civitai:12345", 0.7)
    - "lora1:0.8,lora2:0.5" -> [("lora1", 0.8), ("lora2", 0.5)]

    Args:
        lora_str: Comma-separated lora specifications

    Returns:
        List of (identifier, weight) tuples

    Raises:
        ValueError: If a weight is outside the valid range [-2.0, 2.0]
    """
    results: list[tuple[str, float]] = []
    if not lora_str:
        return results

    for part in lora_str.split(","):
        part = part.strip()
        if not part:
            continue

        # Check if it's a civitai: reference
        if part.startswith("civitai:"):
            # civitai:12345 or civitai:12345:0.8
            segments = part.split(":")
            if len(segments) == 2:
                # civitai:12345
                results.append((part, 1.0))
            elif len(segments) >= 3:
                # civitai:12345:0.8
                try:
                    weight = float(segments[2])
                except ValueError:
                    # Couldn't parse weight as float, treat whole thing as name
                    results.append((part, 1.0))
                else:
                    lora_name = f"civitai:{segments[1]}"
                    validate_lora_weight(weight, lora_name)
                    results.append((lora_name, weight))
        else:
            # Regular name or name:weight
            if ":" in part:
                name, weight_str = part.rsplit(":", 1)
                try:
                    weight = float(weight_str)
                except ValueError:
                    # Couldn't parse weight as float, treat whole thing as name
                    results.append((part, 1.0))
                else:
                    lora_name = name.strip()
                    validate_lora_weight(weight, lora_name)
                    results.append((lora_name, weight))
            else:
                results.append((part, 1.0))

    return results


async def get_lora_choices(ctx: discord.AutocompleteContext) -> list[str]:
    """Autocomplete function for lora choices."""
    global config
    if config is None:
        return []

    loras = config.get("loras", default={})
    if not isinstance(loras, dict):
        return []

    # Get current input and handle comma-separated values
    current = ctx.value or ""
    parts = [p.strip() for p in current.split(",")]
    to_match = parts[-1].lower() if parts else ""
    # Strip weight suffix for matching (e.g., "my-lora:0.8" -> "my-lora")
    if ":" in to_match:
        to_match = to_match.rsplit(":", 1)[0]
    prefix = ", ".join(parts[:-1]) + (", " if len(parts) > 1 else "")

    # Filter loras - exclude auto_load which is a list, not a lora definition
    matches = []
    for name, value in loras.items():
        if name == "auto_load":
            continue
        if isinstance(value, dict) and name.lower().startswith(to_match):
            matches.append(name)

    # Return with prefix for multi-lora support
    return [f"{prefix}{m}" for m in matches[:25]]


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
    global config, pipeline_manager, generation_queue, content_filter, civitai_client

    activity = discord.Activity(
        name="Dreaming...",
        type=discord.ActivityType.custom,
    )

    bot = discord.Bot(activity=activity)

    @bot.event
    async def on_ready():
        """Initialize config, pipeline and queue when bot connects."""
        global config, pipeline_manager, generation_queue, content_filter, civitai_client
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

        # Initialize Civitai client
        civitai_client = CivitaiClient.from_config(config)
        print("Civitai client initialized")

        # Initialize content filter
        content_filter = ContentFilter(config)
        print("Content filter initialized")

        # Initialize pipeline manager with config
        pipeline_manager = PipelineManager(config)
        pipeline_manager.set_civitai_client(civitai_client)
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

        # Sync slash commands to all guilds for instant availability
        # (global commands can take up to 1 hour to propagate)
        if bot.guilds:
            guild_ids = [g.id for g in bot.guilds]
            await bot.sync_commands(guild_ids=guild_ids)
            print(f"Commands synced to {len(guild_ids)} guild(s): {[g.name for g in bot.guilds]}")

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
    @option(
        "steps",
        int,
        description="Number of inference steps (default: model-specific)",
        required=False,
        min_value=1,
        max_value=100,
    )
    @option(
        "guidance_scale",
        float,
        description="CFG scale - prompt adherence (default: model-specific)",
        required=False,
        min_value=0.0,
        max_value=15.0,
    )
    @option(
        "lora",
        str,
        description=(
            "LoRA(s) to apply: name[:weight] or civitai:id[:weight] "
            "(comma-separated, weight defaults to 1.0)"
        ),
        required=False,
        autocomplete=get_lora_choices,
    )
    @option(
        "scheduler",
        str,
        description="Scheduler for denoising (dpm++_karras, euler_a, etc.)",
        required=False,
        choices=SCHEDULER_CHOICES,
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
        steps: int | None = None,
        guidance_scale: float | None = None,
        lora: str | None = None,
        scheduler: str | None = None,
    ):
        """Generate an image from a text prompt."""
        global config, pipeline_manager, generation_queue, content_filter, civitai_client

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
        pipeline_type = model_config.get("type") if model_config else None

        # Get model config defaults
        model_steps = model_config.get("steps", 9)
        model_guidance = model_config.get("guidance_scale", 0.0)

        # Handle Qwen's true_cfg_scale
        if model_config.get("true_cfg_scale"):
            model_guidance = model_config.get("true_cfg_scale", 4.0)

        # Check for model-specific overrides set via /model command
        model_overrides = config.get("model_overrides", current_model, default={}) if config else {}
        if model_overrides:
            if "steps" in model_overrides:
                model_steps = model_overrides["steps"]
            if "guidance_scale" in model_overrides:
                model_guidance = model_overrides["guidance_scale"]

        # User-provided values take priority over model defaults
        actual_steps = steps if steps is not None else model_steps
        actual_guidance = guidance_scale if guidance_scale is not None else model_guidance

        # Resolve LoRAs if specified
        lora_configs: list[LoraConfig] = []
        lora_warnings: list[str] = []
        if lora and config:
            parsed_loras = parse_lora_param(lora)
            loras_section = config.get("loras", default={})

            for lora_ref, weight in parsed_loras:
                try:
                    if lora_ref.startswith("civitai:"):
                        # Direct Civitai reference - on-demand download
                        civitai_id = int(lora_ref.split(":")[1])
                        lora_config = LoraConfig(
                            name=f"civitai_{civitai_id}",
                            source=LoraSource.CIVITAI,
                            civitai_id=civitai_id,
                            weight=weight,
                        )

                        # Check compatibility (soft warning)
                        if civitai_client and pipeline_type:
                            try:
                                model_info = await civitai_client.get_model(civitai_id)
                                version = model_info.latest_version
                                if version and not is_lora_compatible(
                                    pipeline_type, version.base_model
                                ):
                                    lora_warnings.append(
                                        f"⚠️ LoRA `{model_info.name}` (base: {version.base_model}) "
                                        f"may not be compatible with current model ({pipeline_type})"
                                    )
                            except CivitaiError as e:
                                lora_warnings.append(f"⚠️ Could not verify LoRA {civitai_id}: {e}")

                        lora_configs.append(lora_config)
                    else:
                        # Named reference from config
                        if lora_ref in loras_section and isinstance(loras_section[lora_ref], dict):
                            lora_def = loras_section[lora_ref]
                            source_str = lora_def.get("source", "civitai")
                            source = LoraSource(source_str)

                            lora_config = LoraConfig(
                                name=lora_ref,
                                source=source,
                                weight=weight,
                                civitai_id=lora_def.get("id") or lora_def.get("civitai_id"),
                                civitai_version=lora_def.get("version")
                                or lora_def.get("civitai_version"),
                                civitai_url=lora_def.get("url") or lora_def.get("civitai_url"),
                                repo=lora_def.get("repo"),
                                weight_name=lora_def.get("weight_name"),
                                path=lora_def.get("path"),
                            )

                            # Check compatibility for Civitai LoRAs
                            if source == LoraSource.CIVITAI and civitai_client and pipeline_type:
                                civitai_id = lora_config.civitai_id
                                if civitai_id:
                                    try:
                                        model_info = await civitai_client.get_model(civitai_id)
                                        version = model_info.latest_version
                                        if version and not is_lora_compatible(
                                            pipeline_type, version.base_model
                                        ):
                                            lora_warnings.append(
                                                f"⚠️ LoRA `{lora_ref}` (base: {version.base_model}) "
                                                f"may not be compatible with current model ({pipeline_type})"
                                            )
                                    except CivitaiError as e:
                                        lora_warnings.append(
                                            f"⚠️ Could not verify LoRA `{lora_ref}`: {e}"
                                        )

                            lora_configs.append(lora_config)
                        else:
                            await ctx.followup.send(
                                f"❌ Unknown LoRA: `{lora_ref}`. "
                                f"Use `/fetch` to download it first, or use `civitai:<id>` format.",
                                ephemeral=True,
                            )
                            return
                except ValueError as e:
                    await ctx.followup.send(f"❌ Invalid LoRA specification: {e}", ephemeral=True)
                    return

        request: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "steps": actual_steps,
            "guidance_scale": actual_guidance,
        }

        if scheduler:
            request["scheduler"] = scheduler

        if lora_configs:
            request["loras"] = lora_configs

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
            embed.add_field(name="Steps", value=str(result.steps), inline=True)
            embed.add_field(name="CFG", value=f"{result.guidance_scale:.1f}", inline=True)
            if is_img2img:
                embed.add_field(name="Strength", value=f"{strength:.2f}", inline=True)
            if lora_configs:
                lora_display = ", ".join(f"`{lc.name}`:{lc.weight}" for lc in lora_configs)
                if len(lora_display) > 1024:
                    lora_display = lora_display[:1021] + "..."
                embed.add_field(name="LoRA", value=lora_display, inline=True)
            if scheduler:
                embed.add_field(name="Scheduler", value=f"`{scheduler}`", inline=True)
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

        # Send LoRA compatibility warnings (soft warnings, don't block)
        if lora_warnings:
            warning_text = "\n".join(lora_warnings)
            await ctx.followup.send(warning_text, ephemeral=True)

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
    @option(
        "scheduler",
        str,
        description="Override default scheduler for this model",
        required=False,
        choices=SCHEDULER_CHOICES,
    )
    @option(
        "steps",
        int,
        description="Override default steps for this model",
        required=False,
        min_value=1,
        max_value=100,
    )
    @option(
        "guidance_scale",
        float,
        description="Override default CFG scale for this model",
        required=False,
        min_value=0.0,
        max_value=15.0,
    )
    async def model_command(
        ctx: discord.ApplicationContext,
        model: str,
        scheduler: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ):
        """Switch the active diffusion model and optionally override its scheduler.

        This command changes which configured model is used for image generation.
        If a scheduler is provided and supported by the loaded pipeline, it will be
        applied after the model is loaded. The selected model may also be stored as
        the default in the persistent configuration if a state path is configured.
        """
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
            # Model is already active - handle overrides only
            overrides_applied = []

            if scheduler and pipeline_manager.pipeline is not None:
                if isinstance(pipeline_manager.pipeline, CivitaiCheckpointPipeline):
                    pipeline_manager.pipeline.configure_scheduler(scheduler)
                    overrides_applied.append(f"scheduler=`{scheduler}`")
                else:
                    await ctx.respond(
                        f"✅ Model `{model}` is already active.\n"
                        f"⚠️ Scheduler override is not supported for this pipeline type.",
                        ephemeral=True,
                    )
                    return

            # Save steps/guidance_scale overrides to state
            if config.state_path:
                if steps is not None:
                    config.set("model_overrides", model, "steps", value=steps)
                    overrides_applied.append(f"steps={steps}")
                if guidance_scale is not None:
                    config.set("model_overrides", model, "guidance_scale", value=guidance_scale)
                    overrides_applied.append(f"guidance_scale={guidance_scale}")

            if overrides_applied:
                await ctx.respond(
                    f"✅ Model `{model}` already active. Set: {', '.join(overrides_applied)}",
                    ephemeral=True,
                )
            else:
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

            if scheduler and pipeline_manager.pipeline is not None:
                if isinstance(pipeline_manager.pipeline, CivitaiCheckpointPipeline):
                    pipeline_manager.pipeline.configure_scheduler(scheduler)

            if config.state_path:
                config.set("defaults", "model", value=model)
                # Save steps/guidance_scale overrides to state
                if steps is not None:
                    config.set("model_overrides", model, "steps", value=steps)
                if guidance_scale is not None:
                    config.set("model_overrides", model, "guidance_scale", value=guidance_scale)

            msg = f"✅ Switched to model `{model}`"
            overrides = []
            if scheduler:
                overrides.append(f"scheduler=`{scheduler}`")
            if steps is not None:
                overrides.append(f"steps={steps}")
            if guidance_scale is not None:
                overrides.append(f"guidance_scale={guidance_scale}")
            if overrides:
                msg += f" with {', '.join(overrides)}"
            await loading_msg.edit(content=msg)
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

    @bot.slash_command(name="fetch", description="Fetch a model from Civitai URL")
    @option(
        "url",
        str,
        description="Civitai URL (e.g., https://civitai.com/models/12345)",
        required=True,
    )
    @option(
        "name",
        str,
        description="Custom name for the resource (default: slugified model name)",
        required=False,
    )
    async def fetch_command(
        ctx: discord.ApplicationContext,
        url: str,
        name: str | None = None,
    ):
        """Fetch a model from Civitai and auto-configure it."""
        global config, civitai_client, pipeline_manager

        if config is None or civitai_client is None:
            await ctx.respond("❌ Bot is still initializing...", ephemeral=True)
            return

        if not config.state_path:
            await ctx.respond(
                "❌ State persistence not configured. Cannot save fetched resources.",
                ephemeral=True,
            )
            return

        # Validate URL format
        if "civitai.com" not in url:
            await ctx.respond(
                "❌ Invalid URL. Please provide a Civitai URL "
                "(e.g., https://civitai.com/models/12345)",
                ephemeral=True,
            )
            return

        await ctx.defer()

        try:
            # Parse URL to get model ID and optional version
            model_id, version_id = parse_civitai_url(url)

            # Fetch model info
            status_msg = await ctx.followup.send(f"⏳ Fetching model info for ID {model_id}...")

            model = await civitai_client.get_model(model_id)

            # Get the specific version or latest
            if version_id:
                version = None
                for v in model.versions:
                    if v.id == version_id:
                        version = v
                        break
                if version is None:
                    version = await civitai_client.get_model_version(version_id)
            else:
                version = model.latest_version

            if version is None:
                await status_msg.edit(content=f"❌ No versions found for model {model_id}")
                return

            # Generate name: custom name, or slugified model name
            resource_name = name if name else slugify(model.name)

            # Ensure unique name by appending number if needed
            base_name = resource_name
            counter = 1
            existing_loras = config.get("loras", default={})
            existing_models = config.get("models", default={})
            existing_embeddings = config.get("embeddings", default={})
            while (
                resource_name in existing_loras
                or resource_name in existing_models
                or resource_name in existing_embeddings
            ):
                resource_name = f"{base_name}-{counter}"
                counter += 1

            # Determine resource type and configure accordingly
            model_type = model.type.upper() if model.type else "UNKNOWN"

            await status_msg.edit(
                content=f"⏳ Downloading {model_type}: {model.name} ({version.name})..."
            )

            # Download the model
            downloaded_path = await civitai_client.download_model_version(version)

            # Get current pipeline type for compatibility info
            pipeline_type = None
            if pipeline_manager and pipeline_manager.current_model:
                model_config = config.get("models", pipeline_manager.current_model)
                if model_config:
                    pipeline_type = model_config.get("type")

            # Check compatibility and prepare warning
            compatibility_warning = ""
            if model_type == "LORA" and pipeline_type and version.base_model:
                if not is_lora_compatible(pipeline_type, version.base_model):
                    compatibility_warning = (
                        f"\n⚠️ **Note**: This LoRA (base: {version.base_model}) may not be "
                        f"compatible with the current model ({pipeline_type})"
                    )

            # Save to config based on type
            if model_type == "LORA":
                config.set(
                    "loras",
                    resource_name,
                    value={
                        "source": "civitai",
                        "id": model_id,
                        "version": version.id,
                        "name": model.name,
                        "base_model": version.base_model,
                        "path": str(downloaded_path),
                        "weight": 1.0,
                    },
                )

                # Build response embed
                embed = discord.Embed(
                    title="✅ LoRA Fetched",
                    description=f"**{model.name}** ({version.name})",
                    color=discord.Color.green(),
                )
                embed.add_field(name="Name", value=f"`{resource_name}`", inline=True)
                embed.add_field(
                    name="Base Model", value=version.base_model or "Unknown", inline=True
                )
                embed.add_field(name="Type", value="LoRA", inline=True)
                embed.add_field(
                    name="Usage",
                    value=f'`/dream prompt:"your prompt" lora:{resource_name}:0.8`',
                    inline=False,
                )
                if compatibility_warning:
                    embed.add_field(
                        name="⚠️ Compatibility",
                        value=compatibility_warning.strip(),
                        inline=False,
                    )

            elif model_type == "CHECKPOINT":
                # For checkpoints, we need to add to models section
                config.set(
                    "models",
                    resource_name,
                    value={
                        "type": "civitai",
                        "civitai_id": model_id,
                        "civitai_version": version.id,
                        "name": model.name,
                        "base_model": version.base_model,
                        "checkpoint_path": str(downloaded_path),
                    },
                )

                embed = discord.Embed(
                    title="✅ Checkpoint Fetched",
                    description=f"**{model.name}** ({version.name})",
                    color=discord.Color.green(),
                )
                embed.add_field(name="Name", value=f"`{resource_name}`", inline=True)
                embed.add_field(
                    name="Base Model", value=version.base_model or "Unknown", inline=True
                )
                embed.add_field(name="Type", value="Checkpoint", inline=True)
                embed.add_field(
                    name="Usage",
                    value=f"`/model {resource_name}`",
                    inline=False,
                )

            elif model_type == "TEXTUALINVERSION":
                config.set(
                    "embeddings",
                    resource_name,
                    value={
                        "source": "civitai",
                        "id": model_id,
                        "version": version.id,
                        "name": model.name,
                        "base_model": version.base_model,
                        "path": str(downloaded_path),
                    },
                )

                embed = discord.Embed(
                    title="✅ Embedding Fetched",
                    description=f"**{model.name}** ({version.name})",
                    color=discord.Color.green(),
                )
                embed.add_field(name="Name", value=f"`{resource_name}`", inline=True)
                embed.add_field(
                    name="Base Model", value=version.base_model or "Unknown", inline=True
                )
                embed.add_field(name="Type", value="Textual Inversion", inline=True)

            else:
                # Unknown type - just report the download
                embed = discord.Embed(
                    title="✅ Resource Fetched",
                    description=f"**{model.name}** ({version.name})",
                    color=discord.Color.yellow(),
                )
                embed.add_field(name="Type", value=model_type, inline=True)
                embed.add_field(
                    name="Base Model", value=version.base_model or "Unknown", inline=True
                )
                embed.add_field(
                    name="Note",
                    value=f"Downloaded to: `{downloaded_path}`\nManual configuration may be required.",
                    inline=False,
                )

            # Add common fields
            primary_file = version.primary_file
            if primary_file:
                size_mb = primary_file.size_kb / 1024
                embed.add_field(name="Size", value=f"{size_mb:.1f} MB", inline=True)

            embed.set_footer(text=f"Civitai Model ID: {model_id} | Version: {version.id}")

            await status_msg.edit(content=None, embed=embed)

        except ValueError as e:
            await ctx.followup.send(f"❌ Invalid URL: {e}", ephemeral=True)
        except CivitaiError as e:
            await ctx.followup.send(f"❌ Civitai error: {e}", ephemeral=True)
        except Exception as e:
            await ctx.followup.send(f"❌ Failed to fetch: {e}", ephemeral=True)

    return bot
