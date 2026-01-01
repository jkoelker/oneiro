"""Slash command definitions for Oneiro Discord bot."""

import re
from typing import TYPE_CHECKING, Any

import discord
from discord import option

from oneiro.civitai import CivitaiError, parse_civitai_url
from oneiro.discord.handlers import DreamContext, create_dream_callbacks
from oneiro.pipelines import SCHEDULER_CHOICES
from oneiro.pipelines.civitai_checkpoint import CivitaiCheckpointPipeline
from oneiro.pipelines.lora import is_resource_compatible
from oneiro.queue import QueueStatus
from oneiro.services.generation import (
    MAX_GUIDANCE_SCALE,
    MAX_STEPS,
    MIN_GUIDANCE_SCALE,
    MIN_STEPS,
    LoraNotFoundError,
    resolve_loras,
)

if TYPE_CHECKING:
    from oneiro.app import OneiroBot


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


async def get_lora_choices(ctx: discord.AutocompleteContext) -> list[str]:
    """Autocomplete function for lora choices."""
    if ctx.bot.config is None:
        return []

    loras = ctx.bot.config.get("loras", default={})
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
    if ctx.bot.config is None:
        return ["zimage-turbo"]

    models = ctx.bot.config.get("models", default={})
    if not isinstance(models, dict):
        return ["zimage-turbo"]

    # Filter models that start with the current input
    current = ctx.value.lower() if ctx.value else ""
    return [name for name in models.keys() if name.lower().startswith(current)]


def register_commands(bot: "OneiroBot") -> None:
    """Register all slash commands on the bot.

    Args:
        bot: The OneiroBot instance to register commands on
    """

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
        min_value=MIN_STEPS,
        max_value=MAX_STEPS,
    )
    @option(
        "guidance_scale",
        float,
        description="CFG scale - prompt adherence (default: model-specific)",
        required=False,
        min_value=MIN_GUIDANCE_SCALE,
        max_value=MAX_GUIDANCE_SCALE,
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
    ) -> None:
        """Generate an image from a text prompt."""
        if ctx.bot.generation_queue is None or ctx.bot.pipeline_manager is None:
            await ctx.respond("❌ Bot is still initializing, please wait...", ephemeral=True)
            return

        # Check content filter
        if ctx.bot.content_filter is not None:
            allowed, blocked_word = ctx.bot.content_filter.check(prompt, negative_prompt or "")
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
        current_model = ctx.bot.pipeline_manager.current_model or "zimage-turbo"
        model_config = (
            ctx.bot.config.get("models", current_model, default={}) if ctx.bot.config else {}
        )
        pipeline_type = model_config.get("type") if model_config else None

        # Get model config defaults
        model_steps = model_config.get("steps", 9)
        model_guidance = model_config.get("guidance_scale", 0.0)

        # Handle Qwen's true_cfg_scale
        if model_config.get("true_cfg_scale"):
            model_guidance = model_config["true_cfg_scale"]

        # Check for model-specific overrides set via /model command
        model_overrides = (
            ctx.bot.config.get("model_overrides", current_model, default={})
            if ctx.bot.config
            else {}
        )
        if model_overrides:
            if "steps" in model_overrides:
                model_steps = model_overrides["steps"]
            if "guidance_scale" in model_overrides:
                model_guidance = model_overrides["guidance_scale"]

        # User-provided values take priority over model defaults
        actual_steps = steps if steps is not None else model_steps
        actual_guidance = guidance_scale if guidance_scale is not None else model_guidance

        # Resolve LoRAs: explicit param OR auto-detect (not both)
        try:
            lora_result = await resolve_loras(
                lora_param=lora,
                prompt=prompt,
                config=ctx.bot.config,
                civitai_client=ctx.bot.civitai_client,
                lora_detector=ctx.bot.lora_detector,
                pipeline_type=pipeline_type,
            )
        except LoraNotFoundError as e:
            await ctx.followup.send(
                f"❌ {e}. Use `/fetch` to download it first, or use `civitai:<id>` format.",
                ephemeral=True,
            )
            return
        except ValueError as e:
            await ctx.followup.send(f"❌ Invalid LoRA specification: {e}", ephemeral=True)
            return

        lora_configs = lora_result.configs
        lora_warnings = lora_result.warnings
        auto_detected_loras = lora_result.auto_detected

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

        dream_context = DreamContext(
            ctx=ctx,
            prompt=prompt,
            negative_prompt=negative_prompt,
            current_model=current_model,
            scheduler=scheduler,
            lora_configs=lora_configs,
            auto_detected_loras=auto_detected_loras,
            is_img2img=init_image_bytes is not None,
            strength=strength,
            pipeline_manager=ctx.bot.pipeline_manager,  # type: ignore[arg-type]
        )
        on_start, on_position_update, on_complete = create_dream_callbacks(dream_context)

        # Send LoRA compatibility warnings (soft warnings, don't block)
        if lora_warnings:
            warning_text = "\n".join(lora_warnings)
            await ctx.followup.send(warning_text, ephemeral=True)

        queue_result = ctx.bot.generation_queue.add(
            user_id=ctx.author.id,
            request=request,
            callback=on_complete,
            on_start=on_start,
            on_position_update=on_position_update,
        )

        if queue_result.status == QueueStatus.QUEUED:
            if queue_result.position > 1:
                dream_context.status_message = await ctx.followup.send(
                    f"⏳ Queued at position **{queue_result.position}**. Please wait..."
                )
        elif queue_result.status == QueueStatus.USER_LIMIT:
            await ctx.followup.send(f"❌ {queue_result.message}", ephemeral=True)
        elif queue_result.status == QueueStatus.GLOBAL_LIMIT:
            await ctx.followup.send(f"❌ {queue_result.message}", ephemeral=True)

    @bot.slash_command(name="queue", description="Check your queue status")
    async def queue_status(ctx: discord.ApplicationContext) -> None:
        """Show queue status for the user."""
        if ctx.bot.generation_queue is None:
            await ctx.respond("❌ Bot is still initializing...", ephemeral=True)
            return

        user_count = ctx.bot.generation_queue.user_count(ctx.author.id)
        total_count = ctx.bot.generation_queue.size

        if user_count == 0:
            await ctx.respond("✅ You have no pending requests.", ephemeral=True)
        else:
            await ctx.respond(
                f"📊 **Queue Status**\n"
                f"Your pending requests: **{user_count}** / "
                f"{ctx.bot.generation_queue.max_per_user}\n"
                f"Total queue size: **{total_count}** / {ctx.bot.generation_queue.max_global}",
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
        min_value=MIN_STEPS,
        max_value=MAX_STEPS,
    )
    @option(
        "guidance_scale",
        float,
        description="Override default CFG scale for this model",
        required=False,
        min_value=MIN_GUIDANCE_SCALE,
        max_value=MAX_GUIDANCE_SCALE,
    )
    async def model_command(
        ctx: discord.ApplicationContext,
        model: str,
        scheduler: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> None:
        """Switch the active diffusion model and optionally override its scheduler.

        This command changes which configured model is used for image generation.
        If a scheduler is provided and supported by the loaded pipeline, it will be
        applied after the model is loaded. The selected model may also be stored as
        the default in the persistent configuration if a state path is configured.
        """
        if ctx.bot.pipeline_manager is None or ctx.bot.config is None:
            await ctx.respond("❌ Bot is still initializing...", ephemeral=True)
            return

        # Validate model exists in config
        model_config = ctx.bot.config.get("models", model)
        if not model_config:
            available = ctx.bot.pipeline_manager.get_available_models()
            await ctx.respond(
                f"❌ Unknown model: `{model}`\n"
                f"Available models: {', '.join(f'`{m}`' for m in available)}",
                ephemeral=True,
            )
            return

        # Check if already loaded
        if ctx.bot.pipeline_manager.current_model == model:
            # Model is already active - handle overrides only
            overrides_applied = []

            if scheduler and ctx.bot.pipeline_manager.pipeline is not None:
                if isinstance(ctx.bot.pipeline_manager.pipeline, CivitaiCheckpointPipeline):
                    ctx.bot.pipeline_manager.pipeline.configure_scheduler(scheduler)
                    overrides_applied.append(f"scheduler=`{scheduler}`")
                else:
                    await ctx.respond(
                        f"✅ Model `{model}` is already active.\n"
                        f"⚠️ Scheduler override is not supported for this pipeline type.",
                        ephemeral=True,
                    )
                    return

            # Save steps/guidance_scale overrides to state
            if ctx.bot.config.state_path:
                if steps is not None:
                    ctx.bot.config.set("model_overrides", model, "steps", value=steps)
                    overrides_applied.append(f"steps={steps}")
                if guidance_scale is not None:
                    ctx.bot.config.set(
                        "model_overrides", model, "guidance_scale", value=guidance_scale
                    )
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
            await ctx.bot.pipeline_manager.load_model(model)

            if scheduler and ctx.bot.pipeline_manager.pipeline is not None:
                if isinstance(ctx.bot.pipeline_manager.pipeline, CivitaiCheckpointPipeline):
                    ctx.bot.pipeline_manager.pipeline.configure_scheduler(scheduler)

            if ctx.bot.config.state_path:
                ctx.bot.config.set("defaults", "model", value=model)
                # Save steps/guidance_scale overrides to state
                if steps is not None:
                    ctx.bot.config.set("model_overrides", model, "steps", value=steps)
                if guidance_scale is not None:
                    ctx.bot.config.set(
                        "model_overrides", model, "guidance_scale", value=guidance_scale
                    )

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
    async def config_command(ctx: discord.ApplicationContext) -> None:
        """Show current configuration values."""
        if ctx.bot.config is None:
            await ctx.respond("❌ Config not loaded", ephemeral=True)
            return

        defaults = ctx.bot.config.get("defaults", default={})
        queue = ctx.bot.config.get("queue", default={})
        models = list(ctx.bot.config.get("models", default={}).keys())
        current_model = (
            ctx.bot.pipeline_manager.current_model if ctx.bot.pipeline_manager else "N/A"
        )

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
    ) -> None:
        """Fetch a model from Civitai and auto-configure it."""
        if ctx.bot.config is None or ctx.bot.civitai_client is None:
            await ctx.respond("❌ Bot is still initializing...", ephemeral=True)
            return

        if not ctx.bot.config.state_path:
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

            model = await ctx.bot.civitai_client.get_model(model_id)

            # Get the specific version or latest
            if version_id:
                version = None
                for v in model.versions:
                    if v.id == version_id:
                        version = v
                        break
                if version is None:
                    version = await ctx.bot.civitai_client.get_model_version(version_id)
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
            existing_loras = ctx.bot.config.get("loras", default={})
            existing_models = ctx.bot.config.get("models", default={})
            existing_embeddings = ctx.bot.config.get("embeddings", default={})
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
            downloaded_path = await ctx.bot.civitai_client.download_model_version(version)

            # Get current pipeline type for compatibility info
            pipeline_type = None
            if ctx.bot.pipeline_manager and ctx.bot.pipeline_manager.current_model:
                model_config = ctx.bot.config.get("models", ctx.bot.pipeline_manager.current_model)
                if model_config:
                    pipeline_type = model_config.get("type")

            # Check compatibility and prepare warning
            compatibility_warning = ""
            if model_type == "LORA" and pipeline_type and version.base_model:
                if not is_resource_compatible(pipeline_type, version.base_model):
                    compatibility_warning = (
                        f"\n⚠️ **Note**: This LoRA (base: {version.base_model}) may not be "
                        f"compatible with the current model ({pipeline_type})"
                    )

            # Save to config based on type
            if model_type == "LORA":
                ctx.bot.config.set(
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
                ctx.bot.config.set(
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
                ctx.bot.config.set(
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
