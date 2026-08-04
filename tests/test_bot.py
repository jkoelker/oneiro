"""Tests for bot helper functions."""

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import httpx
import pytest
import respx
from discord.webhook.async_ import handle_message_parameters

from oneiro.civitai import CivitaiClient, CivitaiError, ModelVersion
from oneiro.discord.commands import (
    get_generation_defaults,
    is_krea2_base_model,
    krea2_component_repo,
    register_commands,
    slugify,
)
from oneiro.discord.handlers import (
    DreamContext,
    create_dream_callbacks,
    format_exception_response,
)
from oneiro.pipelines.civitai_checkpoint import get_krea2_checkpoint_precision
from oneiro.queue import QueueStatus
from oneiro.services.generation import (
    MAX_LORA_WEIGHT,
    MIN_LORA_WEIGHT,
    parse_lora_param,
    validate_lora_weight,
)


def _register_test_commands() -> dict[str, Any]:
    commands = {}
    bot = MagicMock()

    def slash_command(**kwargs: Any) -> Callable[[Any], Any]:
        def register(command: Any) -> Any:
            commands[kwargs["name"]] = command
            return command

        return register

    bot.slash_command.side_effect = slash_command
    register_commands(bot)
    return commands


def test_short_exception_response_omits_file_and_is_valid_for_pycord():
    """Short traceback responses omit py-cord's invalid file=None value."""
    response = format_exception_response("❌ Failed", RuntimeError("short"))

    assert "file" not in response
    handle_message_parameters(**response)


def test_exception_response_redacts_tokens_from_preview_and_attachment():
    """Tokenized URLs are redacted from every traceback representation."""
    request = httpx.Request("GET", "https://civitai.com/download?token=SUPER_SECRET")
    http_response = httpx.Response(401, request=request)

    try:
        http_response.raise_for_status()
    except httpx.HTTPStatusError as cause:
        try:
            raise RuntimeError("x" * 3000) from cause
        except RuntimeError as error:
            response = format_exception_response("❌ Failed", error)

    assert "SUPER_SECRET" not in response["content"]
    assert "SUPER_SECRET" not in response["file"].fp.getvalue().decode()


async def test_generation_traceback_is_ephemeral_and_attaches_full_trace_when_truncated():
    """Queued generation failures keep details private and preserve the full traceback."""
    ctx = MagicMock()
    ctx.followup.send = AsyncMock()
    status_message = MagicMock()
    status_message.edit = AsyncMock()
    context = DreamContext(
        ctx=ctx,
        prompt="test prompt",
        negative_prompt=None,
        current_model="test-model",
        scheduler=None,
        lora_configs=[],
        auto_detected_loras=[],
        is_img2img=False,
        is_inpaint=False,
        strength=0.0,
        pipeline_manager=MagicMock(),
        status_message=status_message,
    )

    _, _, on_complete = create_dream_callbacks(context)
    error_message = "x" * 3000
    try:
        raise RuntimeError(error_message)
    except RuntimeError as cause:
        try:
            raise CivitaiError("Download failed: HTTP 401") from cause
        except CivitaiError as error:
            await on_complete(error)

    context.status_message.edit.assert_awaited_once_with(
        content="❌ Generation failed: Download failed: HTTP 401"
    )
    call = context.ctx.followup.send.await_args
    assert call.kwargs["content"].startswith("❌ Generation failed")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert len(call.kwargs["content"]) <= 2000
    assert call.kwargs["ephemeral"] is True
    assert call.kwargs["file"].filename == "traceback.txt"
    assert error_message in call.kwargs["file"].fp.getvalue().decode()


async def test_generation_public_summary_is_bounded_for_long_top_level_error():
    """Long top-level errors keep the public summary within Discord's limit."""
    ctx = MagicMock()
    ctx.followup.send = AsyncMock()
    status_message = MagicMock()
    status_message.edit = AsyncMock()
    context = DreamContext(
        ctx=ctx,
        prompt="test prompt",
        negative_prompt=None,
        current_model="test-model",
        scheduler=None,
        lora_configs=[],
        auto_detected_loras=[],
        is_img2img=False,
        is_inpaint=False,
        strength=0.0,
        pipeline_manager=MagicMock(),
        status_message=status_message,
    )

    _, _, on_complete = create_dream_callbacks(context)
    await on_complete(RuntimeError("useful context " + "x" * 3000))

    public_message = status_message.edit.await_args.kwargs["content"]
    assert len(public_message) <= 2000
    assert public_message.startswith("❌ Generation failed: useful context ")
    assert public_message.endswith("...")
    private_call = context.ctx.followup.send.await_args
    assert private_call.kwargs["ephemeral"] is True
    assert "RuntimeError: useful context" in private_call.kwargs["content"]
    assert private_call.kwargs["file"].filename == "traceback.txt"


async def test_generation_traceback_not_found_sends_distinct_public_and_private_messages():
    """A missing status message gets a sanitized summary and separate private traceback."""
    ctx = MagicMock()
    ctx.followup.send = AsyncMock()
    status_message = MagicMock()
    status_message.edit = AsyncMock(
        side_effect=discord.errors.NotFound(
            MagicMock(status=404, reason="Not Found"),
            {"message": "Unknown Message", "code": 10008},
        )
    )
    context = DreamContext(
        ctx=ctx,
        prompt="test prompt",
        negative_prompt=None,
        current_model="test-model",
        scheduler=None,
        lora_configs=[],
        auto_detected_loras=[],
        is_img2img=False,
        is_inpaint=False,
        strength=0.0,
        pipeline_manager=MagicMock(),
        status_message=status_message,
    )

    _, _, on_complete = create_dream_callbacks(context)
    await on_complete(
        RuntimeError("Download failed: https://civitai.com/download?token=SUPER_SECRET")
    )

    public_call, private_call = context.ctx.followup.send.await_args_list
    public_message = public_call.args[0]
    assert "SUPER_SECRET" not in status_message.edit.await_args.kwargs["content"]
    assert "SUPER_SECRET" not in public_message
    assert "token=<redacted>" in public_message
    assert private_call.kwargs["content"] != public_call.args[0]
    assert private_call.kwargs["ephemeral"] is True


async def test_model_error_includes_traceback_and_attaches_full_trace_when_truncated():
    """Unexpected model failures retain the complete traceback without exceeding Discord limits."""
    commands = _register_test_commands()
    error = RuntimeError("x" * 3000)
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=MagicMock())
    ctx.bot.config.get.return_value = {"type": "test"}
    ctx.bot.pipeline_manager.current_model = "working"
    ctx.bot.pipeline_manager.load_model = AsyncMock(side_effect=error)

    await commands["model"](ctx, "broken")

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Failed to load model")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert len(call.kwargs["content"]) <= 2000
    assert call.kwargs["ephemeral"] is True
    assert call.kwargs["file"].filename == "traceback.txt"
    assert str(error) in call.kwargs["file"].fp.getvalue().decode()


@pytest.mark.parametrize(
    "error",
    [CivitaiError("checkpoint download failed"), ValueError("invalid model configuration")],
)
async def test_model_expected_error_is_concise_and_ephemeral(error):
    """Expected model failures omit traceback details and attachments."""
    commands = _register_test_commands()
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=MagicMock())
    ctx.bot.config.get.return_value = {"type": "test"}
    ctx.bot.pipeline_manager.current_model = "working"
    ctx.bot.pipeline_manager.load_model = AsyncMock(side_effect=error)

    await commands["model"](ctx, "broken")

    call = ctx.followup.send.await_args_list[-1]
    assert call.args == (f"❌ Failed to load model: {error}",)
    assert call.kwargs == {"ephemeral": True}


async def test_model_post_load_value_error_includes_traceback():
    """Unexpected state persistence failures retain their traceback."""
    commands = _register_test_commands()
    error = ValueError("state persistence failed")
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=MagicMock())
    ctx.bot.config.get.return_value = {"type": "test"}
    ctx.bot.config.state_path = "state.toml"
    ctx.bot.config.set.side_effect = error
    ctx.bot.pipeline_manager.current_model = "working"
    ctx.bot.pipeline_manager.load_model = AsyncMock()

    await commands["model"](ctx, "broken")

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Failed to load model")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert "ValueError: state persistence failed" in call.kwargs["content"]
    assert call.kwargs.keys() == {"content", "ephemeral"}


async def test_fetch_error_includes_traceback():
    """Unexpected fetch failures include their traceback in the response."""
    commands = _register_test_commands()
    error = RuntimeError("fetch exploded")
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=MagicMock())
    ctx.bot.config.state_path = "state.toml"
    ctx.bot.civitai_client.get_model = AsyncMock(side_effect=error)

    await commands["fetch"](ctx, "https://civitai.com/models/1")

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Failed to fetch")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert "RuntimeError: fetch exploded" in call.kwargs["content"]
    assert call.kwargs.keys() == {"content", "ephemeral"}


async def test_fetch_long_civitai_error_attaches_full_details():
    """Expected Civitai failures preserve diagnostics beyond Discord's limit."""
    commands = _register_test_commands()
    error = CivitaiError("x" * 2500)
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=MagicMock())
    ctx.bot.config.state_path = "state.toml"
    ctx.bot.civitai_client.get_model = AsyncMock(side_effect=error)

    await commands["fetch"](ctx, "https://civitai.com/models/1")

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Civitai error")
    assert len(call.kwargs["content"]) <= 2000
    assert call.kwargs["file"].filename == "traceback.txt"
    assert str(error) in call.kwargs["file"].fp.getvalue().decode()


async def test_fetch_civitai_error_includes_traceback():
    """Expected Civitai failures include their traceback for diagnosis."""
    commands = _register_test_commands()
    error = CivitaiError("checkpoint rejected")
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock(return_value=MagicMock())
    ctx.bot.config.state_path = "state.toml"
    ctx.bot.civitai_client.get_model = AsyncMock(side_effect=error)

    await commands["fetch"](ctx, "https://civitai.com/models/1")

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Civitai error")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert "CivitaiError: checkpoint rejected" in call.kwargs["content"]
    assert call.kwargs.keys() == {"content", "ephemeral"}


async def test_image_read_error_includes_traceback():
    """Unexpected image read failures include their traceback in the response."""
    commands = _register_test_commands()
    error = RuntimeError("image exploded")
    image = MagicMock(filename="image.png", content_type="image/png", size=1)
    image.read = AsyncMock(side_effect=error)
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock()
    ctx.bot.content_filter = None
    ctx.bot.config = None

    await commands["dream"](ctx, "prompt", image=image)

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Failed to read image")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert "RuntimeError: image exploded" in call.kwargs["content"]
    assert call.kwargs.keys() == {"content", "ephemeral"}


async def test_mask_read_error_includes_traceback():
    """Unexpected mask read failures include their traceback in the response."""
    commands = _register_test_commands()
    error = RuntimeError("mask exploded")
    image = MagicMock(filename="image.png", content_type="image/png", size=1)
    image.read = AsyncMock(return_value=b"image")
    mask = MagicMock(filename="mask.png", content_type="image/png", size=1)
    mask.read = AsyncMock(side_effect=error)
    ctx = MagicMock()
    ctx.defer = AsyncMock()
    ctx.followup.send = AsyncMock()
    ctx.bot.content_filter = None
    ctx.bot.config = None
    ctx.bot.pipeline_manager.pipeline.supports_inpaint = True

    await commands["dream"](ctx, "prompt", image=image, mask=mask)

    call = ctx.followup.send.await_args_list[-1]
    assert call.kwargs["content"].startswith("❌ Failed to read mask image")
    assert "Traceback (most recent call last)" in call.kwargs["content"]
    assert "RuntimeError: mask exploded" in call.kwargs["content"]
    assert call.kwargs.keys() == {"content", "ephemeral"}


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


def test_get_generation_defaults_uses_wrapper_signature():
    """Native wrappers retain their declared generation defaults."""

    class Pipeline:
        def generate(self, prompt, steps=28, guidance_scale=3.5):
            pass

    assert get_generation_defaults(Pipeline()) == (28, 3.5)


def test_get_generation_defaults_handles_unloaded_pipeline():
    """A failed model load can still queue a request for automatic reload."""
    assert get_generation_defaults(None) == (9, 0.0)


@pytest.mark.parametrize(
    ("checkpoint_kind", "writes_config"),
    [
        ("krea", True),
        ("quantized", False),
        ("lora", False),
        ("malformed", False),
        ("cached", True),
        ("cached_malformed", True),
        ("cached_io_error", False),
        ("fallback", True),
        ("precision_mismatch", True),
        ("local_fallback", True),
        ("transport", False),
    ],
)
@respx.mock
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

    files = [
        {
            "id": 3,
            "name": "krea.safetensors",
            "sizeKB": 200,
            "type": "Model",
            "downloadUrl": "https://civitai.com/download/3",
            "metadata": {"format": "SafeTensor", "fp": "bf16"},
            "hashes": {"SHA256": "ABC123"},
            "primary": True,
        }
    ]
    if checkpoint_kind in {"fallback", "precision_mismatch", "local_fallback", "transport"}:
        files.append(
            {
                "id": 4,
                "name": "krea.safetensors",
                "sizeKB": 100,
                "type": "Model",
                "downloadUrl": "https://civitai.com/download/4",
                "metadata": {
                    "format": "SafeTensor",
                    "fp": "fp16" if checkpoint_kind == "precision_mismatch" else "bf16",
                },
                "hashes": {"SHA256": "DEF456"},
            }
        )
    version = ModelVersion.from_dict(
        {
            "id": 2,
            "modelId": 1,
            "name": "v1 Turbo",
            "baseModel": "Krea 2",
            "files": files,
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

    header_kind = (
        "krea"
        if malformed or checkpoint_kind in {"cached", "precision_mismatch"}
        else checkpoint_kind
    )
    header_dtype = "I8" if header_kind == "quantized" else "BF16"
    header_keys = (
        ["lora_unet_blocks_0_attn_wq.alpha"]
        if header_kind == "lora"
        else ["first.weight", "last.linear.weight", "blocks.0.attn.wq.weight"]
    )
    header = {
        key: {"dtype": header_dtype, "shape": [1], "data_offsets": [0, 2]} for key in header_keys
    }
    header_client = None
    header_route = None
    if checkpoint_kind == "krea":
        contents = downloaded_path.read_bytes()
        header_size = int.from_bytes(contents[:8], "little")
        header_route = respx.get("https://civitai.com/download/3")
        header_route.side_effect = [
            httpx.Response(
                206,
                content=contents[:8],
                headers={"Content-Range": f"bytes 0-7/{len(contents)}"},
            ),
            httpx.Response(
                206,
                content=contents[: 8 + header_size],
                headers={"Content-Range": f"bytes 0-{7 + header_size}/{len(contents)}"},
            ),
        ]
        header_client = CivitaiClient(cache_dir=tmp_path)
        ctx.bot.civitai_client.get_safetensor_header = header_client.get_safetensor_header
    elif checkpoint_kind in {"fallback", "precision_mismatch"}:
        first_dtype = "F16" if checkpoint_kind == "precision_mismatch" else "BF16"
        ctx.bot.civitai_client.get_safetensor_header = AsyncMock(
            side_effect=[
                (
                    {
                        key: {
                            "dtype": first_dtype,
                            "shape": [1],
                            "data_offsets": [0, 2],
                        }
                        for key in header_keys
                    }
                    if checkpoint_kind == "precision_mismatch"
                    else {
                        "lora.alpha": {
                            "dtype": "BF16",
                            "shape": [1],
                            "data_offsets": [0, 2],
                        }
                    }
                ),
                header,
            ]
        )
    elif checkpoint_kind == "local_fallback":
        ctx.bot.civitai_client.get_safetensor_header = AsyncMock(return_value=header)
    elif checkpoint_kind == "transport":
        ctx.bot.civitai_client.get_safetensor_header = AsyncMock(
            side_effect=CivitaiError("header request failed")
        )
    else:
        ctx.bot.civitai_client.get_safetensor_header = AsyncMock(return_value=header)
    ctx.bot.civitai_client.cache.get.return_value = (
        downloaded_path if checkpoint_kind.startswith("cached") else None
    )

    async def download_model_version(*args, **kwargs):
        if checkpoint_kind == "cached_malformed":
            save_file(
                {
                    "first.weight": torch.ones(1, dtype=torch.bfloat16),
                    "last.linear.weight": torch.ones(1, dtype=torch.bfloat16),
                    "blocks.0.attn.wq.weight": torch.ones(1, dtype=torch.bfloat16),
                },
                downloaded_path,
            )
        elif checkpoint_kind == "local_fallback":
            if kwargs["model_file"].id == 3:
                downloaded_path.write_bytes(b"not a safetensor")
            else:
                save_file(
                    {
                        "first.weight": torch.ones(1, dtype=torch.bfloat16),
                        "last.linear.weight": torch.ones(1, dtype=torch.bfloat16),
                        "blocks.0.attn.wq.weight": torch.ones(1, dtype=torch.bfloat16),
                    },
                    downloaded_path,
                )
        return downloaded_path

    ctx.bot.civitai_client.download_model_version = AsyncMock(side_effect=download_model_version)

    with (
        patch("oneiro.discord.commands.DevicePolicy.auto_detect") as auto_detect,
        patch(
            "oneiro.discord.commands.get_krea2_checkpoint_precision",
            wraps=get_krea2_checkpoint_precision,
        ) as inspect_local_precision,
    ):
        auto_detect.return_value.dtype = "bfloat16"
        if checkpoint_kind == "cached_io_error":
            inspect_local_precision.side_effect = OSError("cached checkpoint read failed")
        await commands["fetch"](
            ctx,
            "https://civitai.com/models/1",
            precision="bf16" if checkpoint_kind == "precision_mismatch" else "auto",
            krea2_variant="raw",
        )
    if header_client:
        await header_client.close()

    assert ctx.bot.config.set.called is writes_config
    if writes_config:
        if checkpoint_kind == "cached":
            inspect_local_precision.assert_called_once_with(downloaded_path)
            ctx.bot.civitai_client.download_model_version.assert_not_awaited()
            ctx.bot.civitai_client.get_safetensor_header.assert_not_awaited()
        else:
            expected_downloads = 2 if checkpoint_kind == "local_fallback" else 1
            assert ctx.bot.civitai_client.download_model_version.await_count == expected_downloads
        if checkpoint_kind == "krea":
            assert header_route is not None and header_route.call_count == 2
        elif checkpoint_kind != "cached":
            expected_calls = (
                2 if checkpoint_kind in {"fallback", "precision_mismatch", "local_fallback"} else 1
            )
            assert ctx.bot.civitai_client.get_safetensor_header.await_count == expected_calls
        if checkpoint_kind in {"fallback", "precision_mismatch", "local_fallback"}:
            selected = ctx.bot.civitai_client.download_model_version.call_args.kwargs["model_file"]
            assert selected.id == 4
        checkpoint_config = ctx.bot.config.set.call_args.kwargs["value"]
        assert checkpoint_config["krea2_component_repo"] == "krea/Krea-2-Raw"
        assert checkpoint_config["steps"] == 28
        assert checkpoint_config["guidance_scale"] == 4.5
        embed = status.edit.call_args.kwargs["embed"]
        assert next(field.value for field in embed.fields if field.name == "Precision") == "`bf16`"
        recipe = next(field.value for field in embed.fields if field.name == "Krea 2 Recipe")
        assert "krea/Krea-2-Raw" in recipe
        assert "28 steps" in recipe
        assert "guidance 4.5" in recipe

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

        operator_config = {
            key: value
            for key, value in checkpoint_config.items()
            if key not in {"steps", "guidance_scale"}
        }
        ctx.bot.pipeline_manager.pipeline.pipeline_config.default_steps = 28
        ctx.bot.pipeline_manager.pipeline.pipeline_config.default_guidance_scale = 4.5
        ctx.bot.config.get.side_effect = lambda *keys, default=None: (
            operator_config if keys == ("models", "fetched-krea") else {}
        )
        ctx.bot.generation_queue.add.reset_mock()
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
        operator_request = ctx.bot.generation_queue.add.call_args.kwargs["request"]
        assert operator_request["steps"] == 28
        assert operator_request["guidance_scale"] == 4.5
    elif checkpoint_kind == "cached_io_error":
        ctx.bot.civitai_client.download_model_version.assert_not_awaited()
        ctx.bot.civitai_client.get_safetensor_header.assert_not_awaited()
        ctx.bot.civitai_client.cache.remove.assert_not_called()
        failure = ctx.followup.send.await_args_list[-1]
        assert "OSError: cached checkpoint read failed" in failure.kwargs["content"]
    elif checkpoint_kind in {"quantized", "lora", "transport"}:
        ctx.bot.civitai_client.download_model_version.assert_not_awaited()
        ctx.bot.civitai_client.get_safetensor_header.assert_awaited_once()
        if checkpoint_kind == "quantized":
            failure = ctx.followup.send.await_args_list[-1].kwargs["content"]
            assert "krea.safetensors (bf16, file 3)" in failure
            assert "I8" in failure
    else:
        ctx.bot.civitai_client.download_model_version.assert_awaited_once()
        ctx.bot.civitai_client.get_safetensor_header.assert_awaited_once()
        assert not downloaded_path.exists()
        assert ctx.bot.civitai_client.cache.remove.call_count == 1
        ctx.bot.civitai_client.cache.remove.assert_called_with("ABC123")


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
