"""Event handlers and callback factories for Discord bot."""

import asyncio
import io
import re
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import discord
from discord.components import Thumbnail as ThumbnailComponent
from PIL import Image, ImageOps

from oneiro.pipelines import GenerationResult, LoraConfig

if TYPE_CHECKING:
    from oneiro.app import OneiroBot
    from oneiro.pipelines import PipelineManager


_TOKEN_PATTERN = re.compile(r"([?&]token=)[^&\s'\"`]+", re.IGNORECASE)


class _SpoilerThumbnail(discord.ui.Thumbnail):
    """Preserve the spoiler setting when Pycord serializes a nested thumbnail."""

    def _generate_underlying(self, **kwargs: Any) -> ThumbnailComponent:
        """Keep Pycord's saved state rather than its default False override."""
        # Source: https://github.com/Pycord-Development/pycord/blob/v2.8.1/discord/ui/thumbnail.py#L92
        # Local tracking: https://github.com/jkoelker/oneiro/pull/154 (upstream filing deferred).
        # ponytail: private Pycord hook; remove when upstream preserves the spoiler flag.
        kwargs.setdefault("spoiler", None)
        return super()._generate_underlying(**kwargs)


def _sanitize_error_text(text: str) -> str:
    """Redact CivitAI API tokens from exception text."""
    return _TOKEN_PATTERN.sub(r"\1<redacted>", text)


def format_exception_response(prefix: str, error: BaseException) -> dict[str, Any]:
    """Build Discord send arguments for a sanitized traceback response."""
    trace = _sanitize_error_text("".join(traceback.format_exception(error)))
    message = f"{prefix}\n```py\n{trace}\n```"
    response: dict[str, Any] = {"content": message}
    if len(message) > 2000:
        preview = 2000 - len(prefix) - len("\n```py\n...\n```")
        response["content"] = f"{prefix}\n```py\n{trace[:preview]}...\n```"
        response["file"] = discord.File(io.BytesIO(trace.encode()), filename="traceback.txt")
    return response


def _input_thumbnail(image_data: bytes) -> bytes:
    """Encode a small preview of the input already validated during generation."""
    with Image.open(io.BytesIO(image_data)) as image:
        image.load()
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass  # Keep the same best-effort orientation as the model input.
        image = image.convert("RGBA")
        image.thumbnail((256, 256))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@dataclass
class DreamContext:
    """Context for dream command callbacks.

    Captures all state needed by on_start, on_position_update, on_complete.
    Holds mutable state (status_message) that callbacks can modify.
    """

    ctx: discord.ApplicationContext
    prompt: str
    negative_prompt: str | None
    current_model: str
    scheduler: str | None
    lora_configs: list[LoraConfig]
    auto_detected_loras: list[tuple[str, str]]
    is_img2img: bool
    is_inpaint: bool
    strength: float | None
    pipeline_manager: "PipelineManager"
    input_image: bytes | None = None
    start_time: float = field(default_factory=time.time)
    status_message: discord.Message | None = None


def create_dream_callbacks(
    context: DreamContext,
) -> tuple[
    Callable[[], Awaitable[None]],
    Callable[[int], Awaitable[None]],
    Callable[[GenerationResult | Exception], Awaitable[None]],
]:
    """Create callback closures for dream command queue submission.

    Args:
        context: DreamContext with all state needed by callbacks

    Returns:
        Tuple of (on_start, on_position_update, on_complete) callbacks
    """

    async def on_start() -> None:
        if context.status_message:
            try:
                await context.status_message.edit(
                    content=f"🎨 Generating for **{context.ctx.author.name}**..."
                )
            except discord.errors.NotFound:
                pass

    async def on_position_update(position: int) -> None:
        if context.status_message:
            try:
                await context.status_message.edit(
                    content=f"⏳ Queued at position **{position}**. Please wait..."
                )
            except discord.errors.NotFound:
                pass

    async def on_complete(result: GenerationResult | Exception) -> None:
        if isinstance(result, Exception):
            prefix = "❌ Generation failed"
            separator = ": "
            ellipsis = "..."
            max_reason_length = 2000 - len(prefix) - len(separator) - len(ellipsis)
            reason = _sanitize_error_text(str(result))
            if len(reason) > max_reason_length:
                reason = f"{reason[:max_reason_length]}{ellipsis}"
            public_message = f"{prefix}{separator}{reason}"
            if context.status_message:
                try:
                    await context.status_message.edit(content=public_message)
                except discord.errors.NotFound:
                    await context.ctx.followup.send(public_message)
            else:
                await context.ctx.followup.send(public_message)

            await context.ctx.followup.send(
                **format_exception_response(prefix, result), ephemeral=True
            )
            return

        elapsed = time.time() - context.start_time
        image_buffer = context.pipeline_manager.image_to_bytes(result.image)
        attachments = [("SPOILER_dream.png", image_buffer.getvalue(), "Generated image")]
        thumbnail: bytes | None = None
        if context.input_image is not None:
            try:
                thumbnail = await asyncio.to_thread(_input_thumbnail, context.input_image)
            except Exception as error:
                print(
                    f"Warning: Failed to create input thumbnail: {_sanitize_error_text(str(error))}"
                )
            else:
                attachments.append(
                    ("SPOILER_input.png", thumbnail, "Input image used for generation")
                )

        def create_files() -> list[discord.File]:
            """Discord upload streams are single-use, including failed message edits."""
            return [
                discord.File(io.BytesIO(data), filename=name, description=description, spoiler=True)
                for name, data, description in attachments
            ]

        mode = {
            "image2image": " (img2img)",
            "inpainting": " (inpaint)",
            "reference": " (reference)",
            "image_conditioned": " (image conditioned)",
        }.get(result.workflow, "")
        heading = f"## 🎨 Dream Generated{mode}\n**Prompt:**\n{context.prompt[:1024]}"
        if context.negative_prompt:
            heading += f"\n**Negative Prompt:**\n{context.negative_prompt[:1024]}"
        details = [
            f"**Size:** {result.width}×{result.height} · **Seed:** {result.seed} · **Time:** {elapsed:.1f}s",
            f"**Model:** `{(result.model_name or context.current_model)[:128]}`",
            f"**Steps:** {result.steps} · **CFG:** {result.guidance_scale:.1f}",
        ]
        if result.workflow in {"image2image", "inpainting"} and result.strength is not None:
            details.append(f"**Strength:** {result.strength:.2f}")
        if context.lora_configs:
            lora_display = ", ".join(f"`{lc.name}`:{lc.weight}" for lc in context.lora_configs)
            if len(lora_display) > 512:
                lora_display = lora_display[:509] + "..."
            details.append(f"**LoRA:** {lora_display}")
        if context.auto_detected_loras:
            auto_display = ", ".join(
                f'`{name}` (matched "{trigger}")' for name, trigger in context.auto_detected_loras
            )
            if len(auto_display) > 512:
                auto_display = auto_display[:509] + "..."
            details.append(f"**Auto LoRAs:** {auto_display}")
        if context.scheduler:
            details.append(f"**Scheduler:** `{context.scheduler[:128]}`")
        footer = f"-# Requested by {context.ctx.author.name[:128]} • React ❌ to delete"
        header = discord.ui.TextDisplay(heading)
        card = discord.ui.Container(colour=discord.Colour.purple())
        if thumbnail is not None:
            card.add_item(
                discord.ui.Section(
                    header,
                    accessory=_SpoilerThumbnail(
                        "attachment://SPOILER_input.png",
                        description="Input image used for generation",
                        spoiler=True,
                    ),
                )
            )
        else:
            card.add_item(header)
        # Discord permits 4000 characters across all TextDisplay components combined.
        card.add_item(
            discord.ui.TextDisplay("\n".join(details)[: 4000 - len(heading) - len(footer)])
        )
        card.add_item(
            discord.ui.MediaGallery(
                discord.MediaGalleryItem(
                    "attachment://SPOILER_dream.png", description="Generated image", spoiler=True
                )
            )
        )
        card.add_item(discord.ui.TextDisplay(footer))
        view = discord.ui.DesignerView(card, timeout=None, store=False)
        allowed_mentions = discord.AllowedMentions.none()

        if context.status_message:
            try:
                await context.status_message.edit(
                    content=None,
                    embeds=[],
                    view=view,
                    files=create_files(),
                    allowed_mentions=allowed_mentions,
                )
                await context.status_message.add_reaction("❌")
            except discord.errors.NotFound:
                msg = await context.ctx.followup.send(
                    view=view, files=create_files(), allowed_mentions=allowed_mentions
                )
                try:
                    await msg.add_reaction("❌")  # type: ignore[union-attr]
                except discord.errors.Forbidden:
                    pass
        else:
            msg = await context.ctx.followup.send(
                view=view, files=create_files(), allowed_mentions=allowed_mentions
            )
            try:
                await msg.add_reaction("❌")  # type: ignore[union-attr]
            except discord.errors.Forbidden:
                pass

    return on_start, on_position_update, on_complete


def create_config_change_handler(
    bot: "OneiroBot",
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Create a config change callback for the given bot.

    Args:
        bot: The OneiroBot instance to update on config changes

    Returns:
        Async callback function for config.on_change()
    """
    from oneiro.lora_detector import create_detector_from_config

    async def on_config_change(new_config: dict[str, Any]) -> None:
        if bot.generation_queue is None:
            return

        queue_config = new_config.get("queue", {})
        new_max_global = queue_config.get("max_global", 100)
        new_max_per_user = queue_config.get("max_per_user", 20)

        if (
            new_max_global != bot.generation_queue.max_global
            or new_max_per_user != bot.generation_queue.max_per_user
        ):
            bot.generation_queue.max_global = new_max_global
            bot.generation_queue.max_per_user = new_max_per_user
            print(f"Queue limits updated: {new_max_global} global, {new_max_per_user} per user")

        bot.lora_detector = create_detector_from_config(new_config)
        print("LoRA auto-detector rebuilt")

    return on_config_change


async def handle_reaction_delete(
    bot: "OneiroBot",
    payload: discord.RawReactionActionEvent,
) -> None:
    """Handle ❌ reaction to delete generated images.

    Args:
        bot: The OneiroBot instance
        payload: The reaction event payload
    """
    if str(payload.emoji) != "❌":
        return

    if payload.user_id == bot.user.id:  # type: ignore[union-attr]
        return

    channel = bot.get_channel(payload.channel_id)
    if channel is None:
        return

    try:
        message = await channel.fetch_message(payload.message_id)  # type: ignore[union-attr]
    except discord.errors.NotFound:
        return

    if message.author != bot.user or not (
        message.embeds
        or (
            message.flags.is_components_v2
            and any(
                attachment.filename == "SPOILER_dream.png" for attachment in message.attachments
            )
        )
    ):
        return

    try:
        await message.delete()
    except discord.errors.Forbidden:
        pass
