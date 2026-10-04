"""Offline Klein distilled/base recipes."""

import pytest

from oneiro.pipelines.flux2_klein import Flux2KleinPipelineWrapper
from tests.test_pipelines_modular import capture_generation, image_bytes, load_hosted
from tests.test_pipelines_modular import offline as offline


@pytest.mark.parametrize(
    "variant,graph,steps,guidance",
    [
        ("distilled", "Flux2KleinAutoBlocks", 4, 1.0),
        ("base", "Flux2KleinBaseAutoBlocks", 50, 4.0),
    ],
)
def test_klein_variants(
    monkeypatch: pytest.MonkeyPatch, variant: str, graph: str, steps: int, guidance: float
) -> None:
    """Base models use native CFG; distilled models never enable it accidentally."""
    wrapper, _ = load_hosted(
        Flux2KleinPipelineWrapper, monkeypatch, {"repo": "custom/model", "variant": variant}
    )
    assert type(wrapper.blocks).__name__ == graph
    assert wrapper.pipe.config.is_distilled is (variant == "distilled")
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate("test", init_image=image_bytes())
    assert result.steps == steps and result.guidance_scale == guidance
    assert "strength" not in calls[0] and "guidance_scale" not in calls[0]
    if variant == "base":
        assert calls[0]["guider"].config.guidance_scale == 4.0
    else:
        with pytest.raises(ValueError, match="recipe"):
            wrapper.generate("test", guidance_scale=5.0)


@pytest.mark.parametrize(
    "repo", ["black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B"]
)
def test_official_distilled_sources(monkeypatch: pytest.MonkeyPatch, repo: str) -> None:
    """Both published distilled sizes resolve without extra metadata."""
    wrapper, _ = load_hosted(Flux2KleinPipelineWrapper, monkeypatch, {"repo": repo})
    assert wrapper.default_steps == 4
    with pytest.raises(ValueError):
        wrapper.generate("test", init_image=image_bytes(), strength=0.5)


def test_custom_source_needs_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambiguous custom sources are not assumed distilled."""
    with pytest.raises(ValueError, match="variant"):
        load_hosted(Flux2KleinPipelineWrapper, monkeypatch, {"repo": "custom/base"})
