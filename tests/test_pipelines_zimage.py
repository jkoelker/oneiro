"""Offline Z-Image modular workflows and retained native mask exception."""

from functools import wraps
from pathlib import Path
from typing import Any

import pytest
from diffusers import ZImageInpaintPipeline
from PIL import Image

from oneiro.device import DevicePolicy, OffloadType
from oneiro.pipelines.zimage import ZImagePipelineWrapper
from tests.test_pipelines_modular import (
    capture_generation,
    image_bytes,
    load_hosted,
    place_embedding_wrapper,
)
from tests.test_pipelines_modular import offline as offline


def test_native_inpaint_sees_real_adapters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The native exception observes real PEFT changes applied by the modular owner."""
    from diffusers import ZImageModularPipeline, ZImageTransformer2DModel
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict

    from oneiro.pipelines.lora import LoraConfig, LoraSource

    transformer = ZImageTransformer2DModel(
        in_channels=4,
        dim=16,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=8,
        axes_dims=[4, 2, 2],
        axes_lens=[32, 32, 32],
    )
    transformer.add_adapter(PeftLoraConfig(r=2, target_modules=["to_q"]))
    ZImageModularPipeline.save_lora_weights(
        tmp_path / "adapter", transformer_lora_layers=get_peft_model_state_dict(transformer)
    )
    transformer.delete_adapters("default")
    wrapper, _ = load_hosted(
        ZImagePipelineWrapper, monkeypatch, assets={"transformer": transformer}
    )
    lora = LoraConfig(name="style", source=LoraSource.LOCAL, path="unused", weight=0.6)
    lora._resolved_path = tmp_path / "adapter" / "pytorch_lora_weights.safetensors"
    wrapper.load_loras_sync([lora])
    assert wrapper.inpaint_pipe.transformer is wrapper.pipe.transformer is transformer
    assert set(wrapper.inpaint_pipe.transformer.peft_config) == {"style"}
    assert transformer.layers[0].attention.to_q.scaling["style"] == 0.6
    wrapper.unload_loras()
    assert not getattr(wrapper.inpaint_pipe.transformer, "peft_config", {})


def test_zimage_native_inpaint_shares_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exception is built before the one shared placement, using only native args."""
    original = DevicePolicy.apply_to_modular_pipeline
    placed = []
    wrappers = []
    original_init = ZImagePipelineWrapper.__init__

    def init(wrapper: Any) -> None:
        original_init(wrapper)
        wrappers.append(wrapper)

    def place(policy: DevicePolicy, pipe: Any, manager: Any) -> None:
        wrapper = wrappers[-1]
        assert isinstance(wrapper.inpaint_pipe, ZImageInpaintPipeline)
        assert set(wrapper.inpaint_pipe.components) == {
            "transformer",
            "vae",
            "text_encoder",
            "tokenizer",
            "scheduler",
        }
        assert all(
            value is pipe.components[name]
            for name, value in wrapper.inpaint_pipe.components.items()
        )
        placed.append(pipe)
        original(policy, pipe, manager)

    monkeypatch.setattr(ZImagePipelineWrapper, "__init__", init)
    monkeypatch.setattr(DevicePolicy, "apply_to_modular_pipeline", place)
    wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch)
    assert placed == [wrapper.pipe]
    assert "inpainting" not in wrapper.blocks.available_workflows
    assert "mask_image" not in wrapper.blocks.get_workflow("text2image").input_names
    assert "mask_image" in wrapper.workflow_inputs("inpainting")
    manager = wrapper.components_manager
    wrapper.unload()
    assert wrapper.inpaint_pipe is None and manager.components == {}


def test_native_constructor_failure_releases_shared_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure before placement cannot retain loaded components or a second native view."""
    from oneiro.pipelines import modular

    managers = []
    original = modular.ComponentsManager

    def manager() -> Any:
        instance = original()
        managers.append(instance)
        return instance

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("native constructor failed")

    monkeypatch.setattr(modular, "ComponentsManager", manager)
    monkeypatch.setattr(ZImageInpaintPipeline, "__init__", fail)
    with pytest.raises(RuntimeError, match="native constructor failed"):
        load_hosted(ZImagePipelineWrapper, monkeypatch)
    assert managers[0].components == {} and managers[0].model_hooks is None


def test_native_mask_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only masks use the native exception, with decoded image and sized output."""
    wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch)
    calls = []

    @wraps(ZImageInpaintPipeline.__call__)
    def native(pipe: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        assert pipe.transformer is wrapper.pipe.transformer
        calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))])

    monkeypatch.setattr(ZImageInpaintPipeline, "__call__", native)
    result = wrapper.generate(
        "test",
        init_image=image_bytes(),
        mask_image=image_bytes(),
        strength=0.4,
        width=64,
        height=32,
        seed=42,
    )
    assert result.workflow == "inpainting" and result.strength == 0.4
    assert result.image.size == (64, 32) and result.guidance_scale == 0.0
    assert calls[0]["guidance_scale"] == 0.0
    assert isinstance(calls[0]["mask_image"], Image.Image)
    assert calls[0]["generator"].initial_seed() == 42
    modular_calls = capture_generation(wrapper, monkeypatch)
    wrapper.generate("test", init_image=image_bytes(), strength=0.5)
    assert modular_calls[0]["strength"] == 0.5


@pytest.mark.parametrize("offload_type", list(OffloadType))
def test_native_exception_owns_no_extra_hooks(
    monkeypatch: pytest.MonkeyPatch, offload_type: OffloadType
) -> None:
    """All native views observe the same hooked modules, and unload releases them once."""
    wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch)
    place_embedding_wrapper(wrapper, offload_type, monkeypatch)
    model = wrapper.pipe.transformer
    assert wrapper.inpaint_pipe.transformer is model
    if offload_type == OffloadType.GROUP:
        owned_hook = model._diffusers_hook.get_hook("group_offloading")
        assert owned_hook is not None
    else:
        owned_hook = model._hf_hook
    wrapper.inpaint_pipe.maybe_free_model_hooks()
    wrapper.post_generate()
    if offload_type == OffloadType.GROUP:
        assert model._diffusers_hook.get_hook("group_offloading") is owned_hook
    else:
        assert model._hf_hook is owned_hook
    manager = wrapper.components_manager
    wrapper.unload()
    assert not hasattr(model, "_hf_hook") and manager.components == {}
    if hasattr(model, "_diffusers_hook"):
        assert model._diffusers_hook.hooks == {}


@pytest.mark.parametrize(
    "controls",
    [
        {"mask_image": image_bytes()},
        {"guidance_scale": 7.5},
        {"init_image": image_bytes(), "strength": float("nan")},
        {"control_image": image_bytes()},
    ],
)
def test_invalid_native_controls(monkeypatch: pytest.MonkeyPatch, controls: dict[str, Any]) -> None:
    """Missing image, unsupported CFG, and invalid controls fail before inference."""
    wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch)
    with pytest.raises(ValueError):
        wrapper.generate("test", **controls)


def test_custom_source_needs_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown sources must explicitly declare Turbo."""
    with pytest.raises(ValueError, match="variant"):
        load_hosted(ZImagePipelineWrapper, monkeypatch, {"repo": "custom/turbo"})
