"""Tests for the /dream option schema, attachments and native defaults."""

from types import SimpleNamespace

import discord
import pytest

from oneiro.discord.commands import (
    MAX_DREAM_ATTACHMENT_BYTES,
    get_generation_defaults,
    register_commands,
    validate_image_attachment,
)


async def test_dream_registers_one_optional_reference_attachment() -> None:
    """The actual slash schema exposes one optional reference and no strength default."""
    bot = discord.Bot(intents=discord.Intents.none())
    register_commands(bot)
    command = next(
        command for command in bot.pending_application_commands if command.name == "dream"
    )
    options = {option.name: option for option in command.options}
    assert options["reference_image"].input_type == discord.SlashCommandOptionType.attachment
    assert options["reference_image"].required is False
    assert options["strength"].required is False
    assert options["strength"].default is None
    assert not options["strength"].min_value <= 0.0 <= options["strength"].max_value
    assert options["strength"].min_value <= 0.001 < 1.0 <= options["strength"].max_value
    assert sum(option.name.startswith("reference") for option in command.options) == 1


class TestDreamAttachmentValidation:
    """Tests for /dream image attachment validation."""

    def test_accepts_image_content_type(self):
        """Image attachments with valid content types are accepted."""
        attachment = SimpleNamespace(size=1024, filename="upload.bin", content_type="image/png")

        assert validate_image_attachment(attachment, "image") is None

    def test_accepts_image_extension_without_content_type(self):
        """Image attachments can fall back to filename extension."""
        attachment = SimpleNamespace(size=1024, filename="mask.webp", content_type=None)

        assert validate_image_attachment(attachment, "mask") is None

    def test_rejects_large_attachment(self):
        """Oversized attachments are rejected before reading bytes."""
        attachment = SimpleNamespace(
            size=MAX_DREAM_ATTACHMENT_BYTES + 1,
            filename="image.png",
            content_type="image/png",
        )

        error = validate_image_attachment(attachment, "image")

        assert error is not None
        assert "MiB or smaller" in error

    def test_rejects_non_image_attachment(self):
        """Non-image attachments are rejected."""
        attachment = SimpleNamespace(size=1024, filename="notes.txt", content_type="text/plain")

        error = validate_image_attachment(attachment, "mask")

        assert error is not None
        assert "HEIC/HEIF" in error


@pytest.mark.parametrize("steps,guidance", [(28, 4.5), (4, 1.0), (8, 0.0)])
def test_native_recipe_defaults_override_generic_signature(steps: int, guidance: float) -> None:
    """Hosted Raw/Klein defaults must not fall back to the shared None signature."""
    pipeline = SimpleNamespace(
        default_steps=steps,
        default_guidance_scale=guidance,
        generate=lambda prompt, steps=None, guidance_scale=None: None,
    )
    assert get_generation_defaults(pipeline) == (steps, guidance)
