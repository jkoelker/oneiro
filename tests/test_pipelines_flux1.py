"""Offline FLUX.1 recipe and native input contracts."""

from typing import Any

import pytest

from oneiro.pipelines.flux1 import Flux1PipelineWrapper
from tests.test_pipelines_modular import capture_generation, image_bytes, load_hosted
from tests.test_pipelines_modular import offline as offline


@pytest.mark.parametrize(
    "repo,steps,guidance",
    [
        ("black-forest-labs/FLUX.1-dev", 28, 3.5),
        ("black-forest-labs/FLUX.1-schnell", 4, 0.0),
    ],
)
def test_recipe_defaults(
    monkeypatch: pytest.MonkeyPatch, repo: str, steps: int, guidance: float
) -> None:
    """Schnell must not inherit dev sampling defaults."""
    wrapper, _ = load_hosted(Flux1PipelineWrapper, monkeypatch, {"repo": repo})
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate("test", seed=42)
    assert result.steps == steps
    assert calls[0].get("guidance_scale", 0.0) == guidance
    assert wrapper.pipe.vae.use_tiling and wrapper.pipe.vae.use_slicing


def test_custom_generation_and_img2img(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real native declarations receive seeded, sized img2img controls."""
    wrapper, _ = load_hosted(Flux1PipelineWrapper, monkeypatch)
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate(
        "test",
        width=64,
        height=32,
        seed=123,
        steps=4,
        guidance_scale=2.0,
        init_image=image_bytes(),
        strength=0.5,
        max_sequence_length=256,
    )
    assert result.image.size == (64, 32)
    assert result.workflow == "image2image" and result.strength == 0.5
    assert calls[0]["max_sequence_length"] == 256
    assert calls[0]["generator"].initial_seed() == 123


@pytest.mark.parametrize("config", [{"repo": "custom/dev-ish"}, {"variant": "bad"}])
def test_ambiguous_variant_rejected(
    monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]
) -> None:
    """An arbitrary repo name is not dev/schnell metadata."""
    with pytest.raises(ValueError, match="variant"):
        load_hosted(Flux1PipelineWrapper, monkeypatch, config)


def test_custom_source_with_explicit_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit metadata selects the recipe for a custom source."""
    wrapper, _ = load_hosted(
        Flux1PipelineWrapper,
        monkeypatch,
        {"repo": "custom/model", "variant": "schnell", "cpu_offload": False},
    )
    calls = capture_generation(wrapper, monkeypatch)
    assert wrapper.generate("test").steps == 4
    assert "guidance_scale" not in calls[0]


def test_unsupported_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Controls that classic FLUX silently dropped now fail explicitly."""
    wrapper, _ = load_hosted(Flux1PipelineWrapper, monkeypatch)
    for request in (
        {"negative_prompt": "bad"},
        {"mask_image": image_bytes()},
        {"control_image": image_bytes()},
        {"embeddings": ["style"]},
    ):
        with pytest.raises(ValueError):
            wrapper.generate("test", **request)


def test_unloaded_generation_rejected() -> None:
    """Loading is still required."""
    with pytest.raises(RuntimeError, match="not loaded"):
        Flux1PipelineWrapper().generate("test")


def test_schnell_guidance_is_recipe_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Schnell's transformer has no guidance embedding, so nonzero controls must fail."""
    wrapper, _ = load_hosted(
        Flux1PipelineWrapper, monkeypatch, {"repo": "black-forest-labs/FLUX.1-schnell"}
    )
    calls = capture_generation(wrapper, monkeypatch)
    with pytest.raises(ValueError, match="recipe"):
        wrapper.generate("test", guidance_scale=7.0)
    assert calls == []
