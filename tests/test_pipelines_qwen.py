"""Offline Qwen native CFG, scheduler, and GGUF contracts."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from oneiro.pipelines.qwen import QwenPipelineWrapper
from tests.test_pipelines_modular import capture_generation, image_bytes, load_hosted
from tests.test_pipelines_modular import offline as offline


@pytest.mark.parametrize("filename,is_gguf", [("model.gguf", True), ("model.safetensors", False)])
def test_local_transformer_path(tmp_path: Path, filename: str, is_gguf: bool) -> None:
    """Both local checkpoint formats preserve the parsed path."""
    path = tmp_path / filename
    path.touch()
    assert QwenPipelineWrapper()._parse_transformer_path(str(path)) == (str(path), is_gguf)


def test_hub_transformer_path() -> None:
    """Repository:file syntax resolves the requested file, not a guessed repo."""
    with patch("huggingface_hub.hf_hub_download", return_value="/cache/model.gguf") as download:
        assert QwenPipelineWrapper()._parse_transformer_path("owner/repo:model.gguf") == (
            "/cache/model.gguf",
            True,
        )
    download.assert_called_once_with(repo_id="owner/repo", filename="model.gguf")


def test_invalid_transformer_path() -> None:
    """Ambiguous transformer identifiers still fail."""
    with pytest.raises(ValueError, match="must be"):
        QwenPipelineWrapper()._parse_transformer_path("invalid")


@pytest.mark.parametrize("extension", ["gguf", "safetensors"])
def test_injected_quantized_component_is_not_reloaded(
    monkeypatch: pytest.MonkeyPatch, extension: str
) -> None:
    """The actual single-file loader result is registered once with GGUF config intact."""
    asset = torch.nn.Linear(2, 2)
    with patch(
        "diffusers.QwenImageTransformer2DModel.from_single_file", return_value=asset
    ) as load:
        wrapper, records = load_hosted(
            QwenPipelineWrapper, monkeypatch, {"transformer": f"/local/model.{extension}"}
        )
    assert wrapper.pipe.transformer is asset
    assert "transformer" not in records["spec_loads"]
    assert any(value is asset for value in wrapper.components_manager.components.values())
    kwargs = load.call_args.kwargs
    assert kwargs["config"] == "Qwen/Qwen-Image"
    assert kwargs["subfolder"] == "transformer"
    assert kwargs["torch_dtype"] == wrapper.policy.dtype
    if extension == "gguf":
        assert kwargs["quantization_config"].compute_dtype == wrapper.policy.dtype
    else:
        assert "quantization_config" not in kwargs


def test_native_cfg_and_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy true_cfg_scale is consumed into a native request-local guider."""
    wrapper, _ = load_hosted(QwenPipelineWrapper, monkeypatch)
    scheduler = wrapper.pipe.scheduler.config
    assert scheduler.base_shift == scheduler.max_shift == math.log(3)
    assert scheduler.max_image_seq_len == 8192 and scheduler.use_dynamic_shifting
    assert "scheduler" not in wrapper.blocks.get_workflow("text2image").input_names
    original = wrapper.pipe.guider
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate(
        "test", negative_prompt="blurry", true_cfg_scale=7.0, width=64, height=32, seed=123
    )
    assert calls[0]["guider"] is not original
    assert calls[0]["guider"].config.guidance_scale == 7.0
    assert "true_cfg_scale" not in calls[0] and "guidance_scale" not in calls[0]
    assert calls[0]["negative_prompt"] == "blurry"
    assert result.guidance_scale == 7.0 and result.image.size == (64, 32)
    assert wrapper.pipe.guider is original
    wrapper.generate("test", init_image=image_bytes(), strength=0.5)
    assert calls[-1]["strength"] == 0.5
    assert calls[-1]["guider"].config.guidance_scale == 4.0


def test_custom_source_needs_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom models do not silently get Qwen Image instead of an edit recipe."""
    with pytest.raises(ValueError, match="variant"):
        load_hosted(QwenPipelineWrapper, monkeypatch, {"repo": "custom/qwen"})
    wrapper, _ = load_hosted(
        QwenPipelineWrapper, monkeypatch, {"repo": "custom/qwen", "variant": "image"}
    )
    assert wrapper.family == "qwen"
