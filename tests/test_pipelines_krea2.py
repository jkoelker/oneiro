"""Offline hosted Krea recipes and actual tiny backported workflows."""

from pathlib import Path
from unittest.mock import patch

import pytest

from oneiro.pipelines.backports.krea2 import Krea2AutoBlocks, Krea2TurboAutoBlocks
from oneiro.pipelines.krea2 import Krea2PipelineWrapper, load_krea2_tokenizer
from tests.test_pipelines_modular import (
    capture_generation,
    image_bytes,
    load_hosted,
    local_wrapper,
)
from tests.test_pipelines_modular import offline as offline


def test_published_fast_tokenizer() -> None:
    """Retain the published fast asset, never legacy tiktoken conversion."""
    with patch("transformers.AutoTokenizer.from_pretrained") as load:
        assert load_krea2_tokenizer("krea/Krea-2-Turbo") is load.return_value
    load.assert_called_once_with("krea/Krea-2-Turbo", subfolder="tokenizer", use_fast=True)


@pytest.mark.parametrize(
    "repo,graph,steps,guidance",
    [
        ("krea/Krea-2-Turbo", Krea2TurboAutoBlocks, 8, 0.0),
        ("krea/Krea-2-Raw", Krea2AutoBlocks, 28, 4.5),
    ],
)
def test_local_graph_and_transformer(
    monkeypatch: pytest.MonkeyPatch, repo: str, graph: type, steps: int, guidance: float
) -> None:
    """Hosted recipes inject the backported transformer and fast tokenizer exactly once."""
    wrapper, records = load_hosted(Krea2PipelineWrapper, monkeypatch, {"repo": repo})
    assert type(wrapper.blocks) is graph
    for name, args, kwargs, asset in records["pretrained"]:
        assert args == (repo,)
        if name == "tokenizer":
            assert kwargs == {"subfolder": "tokenizer", "use_fast": True}
            assert wrapper.pipe.tokenizer is asset
        else:
            assert name == "backported_transformer"
            assert kwargs["subfolder"] == "transformer"
            assert wrapper.pipe.transformer is asset
    assert "transformer" not in records["spec_loads"]
    assert "tokenizer" not in records["spec_loads"]
    capture_generation(wrapper, monkeypatch)
    result = wrapper.generate("test")
    assert result.steps == steps and result.guidance_scale == guidance


@pytest.mark.parametrize("graph", [Krea2AutoBlocks, Krea2TurboAutoBlocks])
@pytest.mark.parametrize("workflow", ["text2image", "image2image", "inpainting", "reference"])
def test_tiny_krea_routes(tmp_path: Path, graph: type, workflow: str) -> None:
    """Actual backported blocks compute tiny CPU images for all published workflows."""
    tiny = local_wrapper(tmp_path, graph())
    wrapper = Krea2PipelineWrapper()
    wrapper.blocks, wrapper.pipe = tiny.blocks, tiny.pipe
    wrapper.components_manager = tiny.components_manager
    controls = {"max_sequence_length": 8}
    if workflow in {"image2image", "inpainting"}:
        controls["init_image"] = image_bytes()
        controls["strength"] = 0.5
    if workflow == "inpainting":
        controls["mask_image"] = image_bytes()
    if workflow == "reference":
        controls.pop("max_sequence_length")
        controls["reference_image"] = [image_bytes()]
        controls["reference_image_encoder_resolution"] = 32
    first = wrapper.generate("a cat", width=32, height=32, steps=2, seed=9, **controls)
    second = wrapper.generate("a cat", width=32, height=32, steps=2, seed=9, **controls)
    assert first.image.size == (32, 32) and first.workflow == workflow
    assert first.image.tobytes() == second.image.tobytes()
    wrapper.unload()


def test_custom_source_needs_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw/Turbo cannot be guessed from a custom source's name."""
    with pytest.raises(ValueError, match="variant"):
        load_hosted(Krea2PipelineWrapper, monkeypatch, {"repo": "custom/turbo"})
