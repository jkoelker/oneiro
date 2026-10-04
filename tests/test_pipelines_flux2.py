"""Offline FLUX.2 loading and reference-conditioning contracts."""

import pytest

from oneiro.pipelines.flux2 import Flux2PipelineWrapper
from tests.test_pipelines_modular import capture_generation, image_bytes, load_hosted
from tests.test_pipelines_modular import offline as offline


def test_injected_quantized_component_is_not_reloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted BNB metadata and both loaded identities survive native registration."""
    wrapper, records = load_hosted(Flux2PipelineWrapper, monkeypatch)
    assert {name for name, _, _, _ in records["pretrained"]} == {"transformer", "text_encoder"}
    for name, args, kwargs, asset in records["pretrained"]:
        assert args == ("diffusers/FLUX.2-dev-bnb-4bit",)
        assert kwargs["subfolder"] == name and "device_map" not in kwargs
        assert "quantization_config" not in kwargs
        assert wrapper.pipe.components[name] is asset
        assert asset.quantization_config["load_in_4bit"] is True
        assert name not in records["spec_loads"]
        assert any(value is asset for value in wrapper.components_manager.components.values())


def test_reference_conditioning_rejects_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    """FLUX.2 image conditioning is not img2img denoising."""
    wrapper, _ = load_hosted(Flux2PipelineWrapper, monkeypatch)
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate(
        "test", width=64, height=32, seed=12, reference_image=[image_bytes(), image_bytes()]
    )
    assert result.workflow == "image_conditioned" and result.strength is None
    assert len(calls[0]["image"]) == 2
    assert "strength" not in calls[0]
    with pytest.raises(ValueError, match="strength"):
        wrapper.generate("test", init_image=image_bytes(), strength=0.5)


def test_custom_source_and_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit dev metadata permits custom repositories and request guidance."""
    wrapper, _ = load_hosted(
        Flux2PipelineWrapper, monkeypatch, {"repo": "custom/model", "variant": "dev"}
    )
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate("test", width=64, height=32, steps=20, guidance_scale=5.0)
    assert result.steps == 20 and result.guidance_scale == 5.0
    assert calls[0]["guidance_scale"] == 5.0
    assert 0 <= result.seed < 2**32
    with pytest.raises(ValueError):
        wrapper.generate("test", negative_prompt="ignored before")


def test_ambiguous_source_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """No source-name substring guessing."""
    with pytest.raises(ValueError, match="variant"):
        load_hosted(Flux2PipelineWrapper, monkeypatch, {"repo": "custom/dev"})
