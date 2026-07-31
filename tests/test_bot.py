"""Tests for bot helper functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oneiro.civitai import ModelVersion
from oneiro.discord.commands import (
    is_krea2_base_model,
    krea2_component_repo,
    register_commands,
    slugify,
)
from oneiro.queue import QueueStatus
from oneiro.services.generation import (
    MAX_LORA_WEIGHT,
    MIN_LORA_WEIGHT,
    parse_lora_param,
    validate_lora_weight,
)


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_text(self):
        """Slugify basic text."""
        assert slugify("My Cool Model") == "my-cool-model"

    def test_special_characters_removed(self):
        """Special characters are removed."""
        assert slugify("Model (v1.2)") == "model-v12"

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces become single hyphen."""
        assert slugify("model   with   spaces") == "model-with-spaces"

    def test_underscores_removed(self):
        """Underscores are removed (filtered out by special char regex)."""
        # Underscores are removed by the [^a-z0-9\s-] regex before space handling
        assert slugify("model_name") == "modelname"

    def test_empty_string_returns_unnamed(self):
        """Empty string returns 'unnamed'."""
        assert slugify("") == "unnamed"

    def test_only_special_chars_returns_unnamed(self):
        """String with only special chars returns 'unnamed'."""
        assert slugify("!!!@@@") == "unnamed"


@pytest.mark.parametrize(
    ("base_model", "expected"),
    [
        ("Krea 2", True),
        ("Krea-2", True),
        ("Krea2", True),
        ("Krea 2 Turbo", True),
        ("Flux.1 Krea", False),
        (None, False),
    ],
)
def test_is_krea2_base_model_uses_pipeline_resolution(base_model, expected):
    """Fetch precision selection uses the same architecture resolution as model loading."""
    assert is_krea2_base_model(base_model) is expected


@pytest.mark.parametrize(
    ("model_name", "version_name", "expected"),
    [
        ("Krea 2 Raw", "v1", "krea/Krea-2-Raw"),
        ("Krea 2", "raw v1", "krea/Krea-2-Raw"),
        ("Krea 2 Turbo", "v1", "krea/Krea-2-Turbo"),
        ("Krea 2 Turbo Raw", "v1 Turbo", "krea/Krea-2-Turbo"),
    ],
)
def test_krea2_component_repo_follows_civitai_names(model_name, version_name, expected):
    """Fetched Krea checkpoints carry the matching hosted component recipe."""
    assert krea2_component_repo(model_name, version_name) == expected


@pytest.mark.parametrize(
    ("checkpoint_kind", "writes_config"),
    [
        ("krea", True),
        ("quantized", False),
        ("lora", False),
        ("malformed", False),
        ("cached", True),
        ("cached_malformed", False),
    ],
)
async def test_fetch_krea2_validates_header_before_writing_config(
    checkpoint_kind, writes_config, tmp_path
):
    """Fetch persists only validated Krea checkpoints and reports header precision."""
    commands = {}
    bot = MagicMock()

    def slash_command(**kwargs):
        def register(command):
            commands[kwargs["name"]] = command
            return command

        return register

    bot.slash_command.side_effect = slash_command
    register_commands(bot)

    version = ModelVersion.from_dict(
        {
            "id": 2,
            "modelId": 1,
            "name": "v1 Turbo",
            "baseModel": "Krea 2",
            "files": [
                {
                    "id": 3,
                    "name": "krea.safetensors",
                    "type": "Model",
                    "downloadUrl": "https://civitai.com/download/3",
                    "metadata": {"format": "SafeTensor", "fp": "bf16"},
                    "hashes": {"SHA256": "ABC123"},
                    "primary": True,
                }
            ],
        }
    )
    model = MagicMock()
    model.name = "Krea 2 Turbo Raw"
    model.type = "Checkpoint"
    model.latest_version = version
    model.versions = [version]

    status = MagicMock()
    status.edit = AsyncMock()
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=status)
    ctx.bot.config.state_path = "state.toml"
    ctx.bot.config.get.return_value = {}
    ctx.bot.config.set = MagicMock()
    ctx.bot.pipeline_manager = None
    ctx.bot.civitai_client.get_model = AsyncMock(return_value=model)
    downloaded_path = tmp_path / "krea.safetensors"
    import torch
    from safetensors.torch import save_file

    dtype = torch.int8 if checkpoint_kind == "quantized" else torch.bfloat16
    malformed = checkpoint_kind in {"malformed", "cached_malformed"}
    tensors = (
        {
            "first.weight": torch.ones(1, dtype=dtype),
            "last.linear.weight": torch.ones(1, dtype=dtype),
            "blocks.0.attn.wq.weight": torch.ones(1, dtype=dtype),
        }
        if checkpoint_kind != "lora"
        else {"lora_unet_blocks_0_attn_wq.alpha": torch.ones(1, dtype=torch.float16)}
    )
    if malformed:
        downloaded_path.write_bytes(b"not a safetensor")
    elif checkpoint_kind == "lora":
        save_file(tensors, downloaded_path)
    elif checkpoint_kind == "krea":
        tensors["last.norm.scale"] = torch.ones(1, dtype=torch.float32)
        save_file(tensors, downloaded_path)
    else:
        save_file(tensors, downloaded_path)

    header_kind = "krea" if malformed or checkpoint_kind == "cached" else checkpoint_kind
    header_dtype = "I8" if header_kind == "quantized" else "BF16"
    header_keys = (
        ["lora_unet_blocks_0_attn_wq.alpha"]
        if header_kind == "lora"
        else ["first.weight", "last.linear.weight", "blocks.0.attn.wq.weight"]
    )
    ctx.bot.civitai_client.get_safetensor_header = AsyncMock(
        return_value={
            key: {"dtype": header_dtype, "shape": [1], "data_offsets": [0, 2]}
            for key in header_keys
        }
    )
    ctx.bot.civitai_client.cache.get.return_value = (
        downloaded_path if checkpoint_kind.startswith("cached") else None
    )
    ctx.bot.civitai_client.download_model_version = AsyncMock(return_value=downloaded_path)

    with patch("oneiro.discord.commands.DevicePolicy.auto_detect") as auto_detect:
        auto_detect.return_value.dtype = "bfloat16"
        await commands["fetch"](
            ctx,
            "https://civitai.com/models/1",
            krea2_variant="raw",
        )

    assert ctx.bot.config.set.called is writes_config
    if writes_config:
        ctx.bot.civitai_client.download_model_version.assert_awaited_once()
        if checkpoint_kind == "cached":
            ctx.bot.civitai_client.get_safetensor_header.assert_not_awaited()
        else:
            ctx.bot.civitai_client.get_safetensor_header.assert_awaited_once()
        checkpoint_config = ctx.bot.config.set.call_args.kwargs["value"]
        assert checkpoint_config["krea2_component_repo"] == "krea/Krea-2-Raw"
        assert checkpoint_config["steps"] == 28
        assert checkpoint_config["guidance_scale"] == 4.5
        embed = status.edit.call_args.kwargs["embed"]
        assert next(field.value for field in embed.fields if field.name == "Precision") == "`bf16`"

        ctx.bot.pipeline_manager = MagicMock(current_model="fetched-krea")
        ctx.bot.content_filter = None
        ctx.bot.config.get.side_effect = lambda *keys, default=None: (
            checkpoint_config if keys == ("models", "fetched-krea") else {}
        )
        queue_result = MagicMock(status=QueueStatus.QUEUED, position=1)
        ctx.bot.generation_queue.add = MagicMock(return_value=queue_result)
        ctx.author.id = 1
        lora_result = MagicMock(configs=[], warnings=[], auto_detected=False)

        with (
            patch(
                "oneiro.discord.commands.resolve_loras",
                new=AsyncMock(return_value=lora_result),
            ),
            patch(
                "oneiro.discord.commands.create_dream_callbacks",
                return_value=(MagicMock(), MagicMock(), MagicMock()),
            ),
        ):
            await commands["dream"](ctx, "test prompt")

        request = ctx.bot.generation_queue.add.call_args.kwargs["request"]
        assert request["steps"] == 28
        assert request["guidance_scale"] == 4.5
    elif checkpoint_kind in {"quantized", "lora"}:
        ctx.bot.civitai_client.download_model_version.assert_not_awaited()
        ctx.bot.civitai_client.get_safetensor_header.assert_awaited_once()
    else:
        if checkpoint_kind == "cached_malformed":
            ctx.bot.civitai_client.download_model_version.assert_not_awaited()
            ctx.bot.civitai_client.get_safetensor_header.assert_not_awaited()
        else:
            ctx.bot.civitai_client.download_model_version.assert_awaited_once()
            ctx.bot.civitai_client.get_safetensor_header.assert_awaited_once()
        assert not downloaded_path.exists()
        ctx.bot.civitai_client.cache.remove.assert_called_once_with("ABC123")


class TestValidateLoraWeight:
    """Tests for validate_lora_weight function."""

    def test_valid_weight_zero(self):
        """Weight 0.0 is valid."""
        validate_lora_weight(0.0, "test-lora")  # Should not raise

    def test_valid_weight_one(self):
        """Weight 1.0 is valid."""
        validate_lora_weight(1.0, "test-lora")  # Should not raise

    def test_valid_weight_max(self):
        """Weight at MAX_LORA_WEIGHT is valid."""
        validate_lora_weight(MAX_LORA_WEIGHT, "test-lora")  # Should not raise

    def test_valid_weight_min(self):
        """Weight at MIN_LORA_WEIGHT is valid."""
        validate_lora_weight(MIN_LORA_WEIGHT, "test-lora")  # Should not raise

    def test_valid_weight_negative(self):
        """Negative weight within range is valid."""
        validate_lora_weight(-1.5, "test-lora")  # Should not raise

    def test_invalid_weight_too_high(self):
        """Weight above MAX_LORA_WEIGHT raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_lora_weight(2.5, "test-lora")

    def test_invalid_weight_too_low(self):
        """Weight below MIN_LORA_WEIGHT raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_lora_weight(-3.0, "test-lora")

    def test_invalid_weight_extreme(self):
        """Extremely large weight raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_lora_weight(100.0, "test-lora")

    def test_error_message_includes_lora_name(self):
        """Error message includes the LoRA name."""
        with pytest.raises(ValueError, match="my-custom-lora"):
            validate_lora_weight(5.0, "my-custom-lora")

    def test_error_message_includes_weight(self):
        """Error message includes the invalid weight value."""
        with pytest.raises(ValueError, match="5.0"):
            validate_lora_weight(5.0, "test-lora")


class TestParseLoraParam:
    """Tests for parse_lora_param function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert parse_lora_param("") == []

    def test_single_lora_name_only(self):
        """Single LoRA name without weight defaults to 1.0."""
        result = parse_lora_param("my-lora")
        assert result == [("my-lora", 1.0)]

    def test_single_lora_with_weight(self):
        """Single LoRA with weight parses correctly."""
        result = parse_lora_param("my-lora:0.8")
        assert result == [("my-lora", 0.8)]

    def test_civitai_id_only(self):
        """Civitai ID without weight defaults to 1.0."""
        result = parse_lora_param("civitai:12345")
        assert result == [("civitai:12345", 1.0)]

    def test_civitai_id_with_weight(self):
        """Civitai ID with weight parses correctly."""
        result = parse_lora_param("civitai:12345:0.7")
        assert result == [("civitai:12345", 0.7)]

    def test_multiple_loras(self):
        """Multiple comma-separated LoRAs parse correctly."""
        result = parse_lora_param("lora1:0.8,lora2:0.5")
        assert result == [("lora1", 0.8), ("lora2", 0.5)]

    def test_multiple_loras_mixed_formats(self):
        """Multiple LoRAs with mixed formats parse correctly."""
        result = parse_lora_param("my-lora,civitai:12345:0.7,another:0.5")
        assert result == [
            ("my-lora", 1.0),
            ("civitai:12345", 0.7),
            ("another", 0.5),
        ]

    def test_spaces_trimmed(self):
        """Spaces around entries are trimmed."""
        result = parse_lora_param("  lora1:0.8  ,  lora2:0.5  ")
        assert result == [("lora1", 0.8), ("lora2", 0.5)]

    def test_invalid_weight_falls_back_to_name(self):
        """Non-numeric weight treats whole thing as name."""
        result = parse_lora_param("lora-with-invalid:abc")
        assert result == [("lora-with-invalid:abc", 1.0)]

    def test_negative_weight_valid(self):
        """Negative weight within range is accepted."""
        result = parse_lora_param("my-lora:-1.5")
        assert result == [("my-lora", -1.5)]

    def test_weight_at_max_bound(self):
        """Weight at maximum bound is accepted."""
        result = parse_lora_param("my-lora:2.0")
        assert result == [("my-lora", 2.0)]

    def test_weight_at_min_bound(self):
        """Weight at minimum bound is accepted."""
        result = parse_lora_param("my-lora:-2.0")
        assert result == [("my-lora", -2.0)]

    def test_weight_exceeds_max_raises(self):
        """Weight exceeding maximum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("my-lora:2.5")

    def test_weight_below_min_raises(self):
        """Weight below minimum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("my-lora:-3.0")

    def test_civitai_weight_exceeds_max_raises(self):
        """Civitai LoRA weight exceeding maximum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("civitai:12345:5.0")

    def test_civitai_weight_below_min_raises(self):
        """Civitai LoRA weight below minimum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("civitai:12345:-10.0")

    def test_error_message_includes_lora_name(self):
        """Error for invalid weight includes LoRA name."""
        with pytest.raises(ValueError, match="test-lora"):
            parse_lora_param("test-lora:10.0")

    def test_mixed_valid_and_invalid_raises_on_invalid(self):
        """Parsing stops with error when invalid weight encountered."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("valid-lora:0.8,invalid-lora:100.0")

    def test_empty_parts_ignored(self):
        """Empty parts from multiple commas are ignored."""
        result = parse_lora_param("lora1,,lora2")
        assert result == [("lora1", 1.0), ("lora2", 1.0)]
