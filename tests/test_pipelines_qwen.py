"""Offline Qwen native CFG, scheduler, and GGUF contracts."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from diffusers.modular_pipelines.modular_pipeline import BlockState
from diffusers.modular_pipelines.qwenimage.denoise import QwenImageLoopDenoiser
from PIL import Image

from oneiro.pipelines.qwen import QwenPipelineWrapper
from tests.test_civitai_checkpoint import checkpoint_wrapper
from tests.test_pipelines_modular import capture_generation, image_bytes, load_hosted
from tests.test_pipelines_modular import offline as offline


@pytest.mark.parametrize("source", ["hosted", "checkpoint"])
@pytest.mark.parametrize("scale", [0.0, 0.25, 1.0, 3.0])
@pytest.mark.parametrize("failure", [False, True])
def test_native_qwen_positive_only_threshold_and_cfg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, scale: float, failure: bool
) -> None:
    """Real Qwen denoiser normalization must retain <=1 positive-only sampling for both sources."""
    if source == "hosted":
        wrapper, _ = load_hosted(QwenPipelineWrapper, monkeypatch)
    else:
        wrapper, config, _ = checkpoint_wrapper(tmp_path, monkeypatch, "Qwen", "image")
        wrapper.load(config)
    original = wrapper.pipe.guider
    original_config, original_state = dict(original.config), original.get_state()
    predictions = []
    seen = []

    class Prediction(torch.nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, **kwargs: Any
        ) -> tuple[torch.Tensor]:
            value = encoder_hidden_states.clone()
            predictions.append(value.tolist())
            return (value,)

    wrapper.pipe.register_components(transformer=Prediction())

    def inference(values: dict[str, Any], is_img2img: bool) -> dict[str, Any]:
        state = BlockState(
            prompt_embeds=torch.tensor([[[10.0, 0.0]]]),
            negative_prompt_embeds=torch.tensor([[[0.0, -10.0]]]),
            prompt_embeds_mask=None,
            negative_prompt_embeds_mask=None,
            denoiser_input_fields={},
            additional_cond_kwargs={},
            latent_model_input=torch.zeros(1, 1, 2),
            timestep=torch.tensor([1000.0]),
            num_inference_steps=1,
            attention_kwargs=None,
        )
        QwenImageLoopDenoiser()(wrapper.pipe, state, 0, torch.tensor(1000.0))
        seen.append(SimpleNamespace(noise=state.noise_pred, guider=wrapper.pipe.guider))
        if failure:
            raise RuntimeError("after native Qwen denoising")
        return {"images": [Image.new("RGB", (32, 32))]}

    monkeypatch.setattr(wrapper, "run_inference", inference)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="after native Qwen"):
                wrapper.generate("positive", negative_prompt="negative", guidance_scale=scale)
        else:
            result = wrapper.generate("positive", negative_prompt="negative", guidance_scale=scale)
            assert result.guidance_scale == scale
        expected = [10.0, 0.0] if scale <= 1 else [300 / math.sqrt(1300), 200 / math.sqrt(1300)]
        torch.testing.assert_close(seen[0].noise, torch.tensor([[expected]]))
        assert predictions == (
            [[[[10.0, 0.0]]]] if scale <= 1 else [[[[10.0, 0.0]]], [[[0.0, -10.0]]]]
        )
        assert seen[0].guider is not original
        assert wrapper.pipe.guider is original
        assert dict(original.config) == original_config and original.get_state() == original_state
    finally:
        wrapper.unload()


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
