"""Event handlers and callback factories for Discord bot."""

import io
import re
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import discord

from oneiro.pipelines import GenerationResult, LoraConfig

if TYPE_CHECKING:
    from oneiro.app import OneiroBot
    from oneiro.pipelines import PipelineManager


_TOKEN_PATTERN = re.compile(r"([?&]token=)[^&\s'\"`]+", re.IGNORECASE)


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
    strength: float
    pipeline_manager: "PipelineManager"
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
        file = discord.File(image_buffer, filename="dream.png")

        mode = " (inpaint)" if context.is_inpaint else " (img2img)" if context.is_img2img else ""
        embed = discord.Embed(title="🎨 Dream Generated" + mode, color=discord.Color.purple())
        embed.add_field(name="Prompt", value=context.prompt[:1024], inline=False)
        if context.negative_prompt:
            embed.add_field(
                name="Negative Prompt",
                value=context.negative_prompt[:1024],
                inline=False,
            )
        embed.add_field(name="Size", value=f"{result.width}×{result.height}", inline=True)
        embed.add_field(name="Seed", value=str(result.seed), inline=True)
        embed.add_field(name="Time", value=f"{elapsed:.1f}s", inline=True)
        embed.add_field(name="Model", value=f"`{context.current_model}`", inline=True)
        embed.add_field(name="Steps", value=str(result.steps), inline=True)
        embed.add_field(name="CFG", value=f"{result.guidance_scale:.1f}", inline=True)
        if context.is_img2img:
            embed.add_field(name="Strength", value=f"{context.strength:.2f}", inline=True)
        if context.lora_configs:
            lora_display = ", ".join(f"`{lc.name}`:{lc.weight}" for lc in context.lora_configs)
            if len(lora_display) > 1024:
                lora_display = lora_display[:1021] + "..."
            embed.add_field(name="LoRA", value=lora_display, inline=True)
        if context.auto_detected_loras:
            auto_display = ", ".join(
                f'`{name}` (matched "{trigger}")' for name, trigger in context.auto_detected_loras
            )
            if len(auto_display) > 1024:
                auto_display = auto_display[:1021] + "..."
            embed.add_field(name="Auto LoRAs", value=auto_display, inline=False)
        if context.scheduler:
            embed.add_field(name="Scheduler", value=f"`{context.scheduler}`", inline=True)
        embed.set_image(url="attachment://dream.png")
        embed.set_footer(
            text=f"Requested by {context.ctx.author.name} • React ❌ to delete",
            icon_url=context.ctx.author.avatar.url if context.ctx.author.avatar else None,
        )

        if context.status_message:
            try:
                await context.status_message.edit(content=None, embed=embed, file=file)
                await context.status_message.add_reaction("❌")
            except discord.errors.NotFound:
                msg = await context.ctx.followup.send(embed=embed, file=file)
                try:
                    await msg.add_reaction("❌")  # type: ignore[union-attr]
                except discord.errors.Forbidden:
                    pass
        else:
            msg = await context.ctx.followup.send(embed=embed, file=file)
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

    if message.author != bot.user or not message.embeds:
        return

    try:
        await message.delete()
    except discord.errors.Forbidden:
        pass
