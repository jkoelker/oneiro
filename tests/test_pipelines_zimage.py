"""Offline Z-Image modular workflows and retained native mask exception."""

import io
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
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


@pytest.mark.parametrize("scale,encoded", [(0.0, ["positive"]), (5.0, ["positive", "negative"])])
def test_native_negative_encoding_uses_actual_cfg_boundary(
    monkeypatch: pytest.MonkeyPatch, scale: float, encoded: list[str]
) -> None:
    """Native Turbo guidance 0 encodes no negative text; the native CFG path still does."""
    from tests.test_krea2_backport import TinyTextEncoder, TinyTokenizer

    observed = []

    class Tokenizer(TinyTokenizer):
        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            text = messages[0]["content"]
            observed.append(text)
            return text

    wrapper, _ = load_hosted(
        ZImagePipelineWrapper,
        monkeypatch,
        assets={"tokenizer": Tokenizer(), "text_encoder": TinyTextEncoder()},
    )
    native = wrapper.inpaint_pipe
    native.transformer.in_channels = 4
    monkeypatch.setattr(
        ZImageInpaintPipeline, "_execution_device", property(lambda self: torch.device("cpu"))
    )

    def stop(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("native model computation boundary")

    monkeypatch.setattr(native, "prepare_latents", stop)
    with pytest.raises(RuntimeError, match="native model computation boundary"):
        native(
            "positive",
            negative_prompt="negative",
            guidance_scale=scale,
            image=Image.new("RGB", (32, 32)),
            mask_image=Image.new("L", (32, 32)),
            num_inference_steps=2,
            strength=1.0,
        )
    assert native.do_classifier_free_guidance is (scale > 0)
    assert observed == encoded
    wrapper.unload()


@pytest.mark.parametrize("source", ["hosted", "checkpoint"])
@pytest.mark.parametrize("workflow", ["text2image", "image2image", "inpainting"])
def test_turbo_rejects_ineffective_negative_before_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, workflow: str
) -> None:
    """A native input slot must not let Turbo report negative conditioning it never applies."""
    from unittest.mock import Mock

    from tests.test_civitai_checkpoint import checkpoint_wrapper

    if source == "hosted":
        wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch)
    else:
        wrapper, config, components = checkpoint_wrapper(
            tmp_path, monkeypatch, "Z-Image Turbo", "turbo"
        )
        components["vae"].config = SimpleNamespace(block_out_channels=[8, 8])
        wrapper.load(config)
    controls = {} if workflow == "text2image" else {"init_image": image_bytes()}
    if workflow == "inpainting":
        controls["mask_image"] = image_bytes()
    capture_generation(wrapper, monkeypatch)
    decode, setup = Mock(wraps=wrapper._load_init_image), Mock(wraps=wrapper.pre_generate)
    monkeypatch.setattr(wrapper, "_load_init_image", decode)
    monkeypatch.setattr(wrapper, "pre_generate", setup)
    with pytest.raises(ValueError, match="Negative prompts"):
        wrapper.generate("positive", negative_prompt="negative", **controls)
    decode.assert_not_called()
    setup.assert_not_called()
    wrapper.unload()


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


@pytest.mark.parametrize(
    "size,region",
    [
        ((32, 32), (8, 4, 24, 20)),
        ((64, 32), (16, 4, 48, 20)),
        ((32, 64), (8, 8, 24, 40)),
        ((64, 64), (16, 8, 48, 40)),
    ],
    ids=["same-size", "wider", "taller", "larger-square"],
)
def test_native_inpaint_resizes_source_and_aligned_mask(
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    region: tuple[int, int, int, int],
) -> None:
    """Real native VAE/denoising must use requested, mutually aligned image coordinates."""
    from diffusers import AutoencoderKL, ZImageTransformer2DModel

    from tests.test_krea2_backport import TinyTextEncoder

    transformer = ZImageTransformer2DModel(
        in_channels=4,
        dim=16,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=8,
        axes_dims=[4, 2, 2],
        axes_lens=[128, 128, 128],
    ).eval()
    vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8,) * 4,
        norm_num_groups=4,
        latent_channels=4,
        shift_factor=0.0,
    ).eval()
    wrapper, _ = load_hosted(
        ZImagePipelineWrapper,
        monkeypatch,
        assets={"transformer": transformer, "vae": vae, "text_encoder": TinyTextEncoder()},
    )
    native = wrapper.inpaint_pipe
    native.set_progress_bar_config(disable=True)
    resources = dict(native.components)
    seen = {}

    def prompt(**kwargs: Any) -> tuple[list[torch.Tensor], None]:
        return [torch.zeros(4, 8)], None

    monkeypatch.setattr(native, "encode_prompt", prompt)
    original_latents = native.prepare_latents
    original_mask_latents = native.prepare_mask_latents

    def record_latents(image: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        seen["source"] = image.detach().clone()
        return original_latents(image, *args, **kwargs)

    def record_mask_latents(
        mask: torch.Tensor, masked_image: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        seen["mask"] = mask.detach().clone()
        seen["masked_source"] = masked_image.detach().clone()
        return original_mask_latents(mask, masked_image, *args, **kwargs)

    # Observation only: all latent preparation, VAE encode/decode and denoising stay native.
    monkeypatch.setattr(native, "prepare_latents", record_latents)
    monkeypatch.setattr(native, "prepare_mask_latents", record_mask_latents)
    source = Image.new("RGB", (32, 32), "black")
    source.paste("white", (8, 4, 24, 20))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    result = wrapper.generate(
        "test",
        init_image=buffer.getvalue(),
        mask_image=buffer.getvalue(),
        width=size[0],
        height=size[1],
        steps=2,
        strength=1.0,
        seed=42,
    )
    assert result.image.size == size
    assert (result.width, result.height) == size
    assert result.workflow == "inpainting" and result.guidance_scale == 0.0
    assert seen["source"].shape == (1, 3, size[1], size[0])
    assert seen["mask"].shape == (1, 1, size[1], size[0])
    expected_mask = torch.zeros(size[1], size[0])
    left, top, right, bottom = region
    expected_mask[top:bottom, left:right] = 1
    torch.testing.assert_close(seen["mask"][0, 0], expected_mask)
    assert torch.equal(seen["source"][0, 0] > 0, expected_mask.bool())
    assert seen["masked_source"][0, :, expected_mask.bool()].count_nonzero() == 0
    assert all(
        wrapper.pipe.components[name] is native.components[name] is resource
        for name, resource in resources.items()
    )
    wrapper.unload()


@pytest.mark.parametrize("image_only", [False, True], ids=["default-text", "image-only"])
def test_non_mask_routing_keeps_source_coordinates(
    monkeypatch: pytest.MonkeyPatch, image_only: bool
) -> None:
    """Native-mask normalization must not resize inputs to the shared modular paths."""
    from oneiro.pipelines.base import BasePipeline

    wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch)
    seen = []

    def modular_boundary(owner: Any, gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        seen.append(gen_kwargs)
        assert is_img2img is image_only
        raise RuntimeError("modular boundary reached")

    monkeypatch.setattr(BasePipeline, "run_inference", modular_boundary)
    controls = {"init_image": image_bytes(), "width": 64, "height": 32} if image_only else {}
    with pytest.raises(RuntimeError, match="modular boundary reached"):
        wrapper.generate("test", **controls)
    assert seen[0]["num_inference_steps"] == 9
    assert "guidance_scale" not in seen[0]
    assert wrapper.pipe.guider.config.enabled is False
    assert (seen[0]["width"], seen[0]["height"]) == ((64, 32) if image_only else (1024, 1024))
    if image_only:
        assert seen[0]["image"].size == (32, 32)
        assert seen[0]["strength"] == 0.75
    else:
        assert "image" not in seen[0] and "strength" not in seen[0]
    assert "mask_image" not in seen[0]
    wrapper.unload()
