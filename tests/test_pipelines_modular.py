"""Offline checks for the shared native modular lifecycle."""

import io
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Flux2AutoBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
from PIL import Image

from oneiro.device import DevicePolicy, OffloadType
from oneiro.pipelines.backports.krea2 import (
    BackportedKrea2Transformer2DModel,
    Krea2AutoBlocks,
    Krea2TurboAutoBlocks,
)
from oneiro.pipelines.lora import LoraConfig, LoraSource
from tests.test_krea2_backport import TINY_CONFIG, TinyTextEncoder, TinyTokenizer


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forbid downloads and keep tiny CPU work bounded."""

    def disallow_connection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access is forbidden in the modular gate")

    monkeypatch.setattr(socket.socket, "connect", disallow_connection)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def local_wrapper(tmp_path: Path, blocks: Any = None, omit: str | None = None) -> Any:
    """Inject local components into the real family graph and native manager."""
    from oneiro.pipelines.modular import ModularPipelineWrapper

    class Wrapper(ModularPipelineWrapper):
        family = "krea2"

        def load(
            self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
        ) -> None:
            """The test supplies already-created tiny components."""

    blocks = blocks or Krea2TurboAutoBlocks()
    repo = tmp_path / "components"
    repo.mkdir(exist_ok=True)
    (repo / "model_index.json").write_text("{}")
    components = {
        spec.name: TinyTokenizer() if "tokenizer" in spec.name else torch.nn.Linear(2, 2)
        for spec in blocks.expected_components
        if spec.default_creation_method == "from_pretrained"
    }
    if isinstance(blocks, Krea2AutoBlocks):
        components.update(
            scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
            text_encoder=TinyTextEncoder(),
            transformer=BackportedKrea2Transformer2DModel(
                **{**TINY_CONFIG, "num_text_layers": 12}
            ).eval(),
            vae=AutoencoderKLQwenImage(
                base_dim=24,
                z_dim=4,
                dim_mult=[1, 2, 4],
                num_res_blocks=1,
                temperal_downsample=[False, True],
                latents_mean=[0.0] * 4,
                latents_std=[1.0] * 4,
            ).eval(),
        )
    if omit:
        components.pop(omit)
    wrapper = Wrapper()
    wrapper.initialize_pipeline(str(repo), blocks, components)
    wrapper.pipe.set_progress_bar_config(disable=True)
    return wrapper


@pytest.mark.parametrize("blocks_class", [Krea2AutoBlocks, Krea2TurboAutoBlocks])
def test_modular_workflow_validation(tmp_path: Path, blocks_class: type) -> None:
    """Strength denotes denoising, not text or reference conditioning."""
    wrapper = local_wrapper(tmp_path, blocks_class())
    assert wrapper.validate_request(has_image=True) == "image2image"
    assert wrapper.validate_request(has_image=True, has_mask=True) == "inpainting"
    assert wrapper.validate_request(has_reference=True) == "reference"
    assert wrapper.supports_inpaint is True
    for request in (
        {"has_mask": True},
        {"has_image": True, "has_mask": True, "has_reference": True},
        {"has_image": True, "has_reference": True},
        {"strength": 0.75},
        {"has_reference": True, "strength": 0.75},
        *(
            {"has_image": True, "strength": value}
            for value in (float("nan"), float("inf"), -0.1, 1.1)
        ),
    ):
        with pytest.raises(ValueError):
            wrapper.validate_request(**request)


def test_conditioned_workflow_rejects_strength(tmp_path: Path) -> None:
    """FLUX.2 images condition generation; they do not select denoising strength."""
    wrapper = local_wrapper(tmp_path, Flux2AutoBlocks())
    assert wrapper.validate_request(has_image=True) == "image_conditioned"
    assert wrapper.supports_inpaint is False
    with pytest.raises(ValueError):
        wrapper.validate_request(has_image=True, strength=0.75)


def test_partial_component_load_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A native warning-only load failure cannot leave a usable partial wrapper."""

    def fail_load(self: ComponentSpec, **kwargs: Any) -> Any:
        raise OSError("local component unavailable")

    monkeypatch.setattr(ComponentSpec, "load", fail_load)
    with pytest.raises(RuntimeError, match="transformer"):
        local_wrapper(tmp_path, omit="transformer")


def test_request_state_is_fresh(tmp_path: Path) -> None:
    """Each request executes native blocks with a distinct PipelineState."""
    wrapper = local_wrapper(tmp_path)
    states = []
    original = wrapper.run_inference

    def record(gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        state = original(gen_kwargs, is_img2img)
        states.append(state)
        return state

    wrapper.run_inference = record
    first = wrapper.generate("a cat", width=32, height=32, steps=2, seed=7, max_sequence_length=8)
    second = wrapper.generate("a cat", width=32, height=32, steps=2, seed=7, max_sequence_length=8)
    assert states[0] is not states[1]
    assert first.image.tobytes() == second.image.tobytes()
    assert first.workflow == "text2image"
    assert first.strength is None


def image_bytes(format: str = "PNG") -> bytes:
    """Encode a small attachment without external assets."""
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buffer, format=format)
    return buffer.getvalue()


@pytest.mark.parametrize("failure", ["second_adapter", "same_name", "inference"])
@pytest.mark.parametrize("offload_type", [None, *OffloadType])
def test_dynamic_lora_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    offload_type: OffloadType | None,
) -> None:
    """Real PEFT/native weights must return to the static baseline on every failure."""
    from diffusers import Krea2ModularPipeline
    from peft import LoraConfig as PeftLoraConfig
    from peft.utils import get_peft_model_state_dict

    wrapper = local_wrapper(tmp_path)
    transformer = wrapper.pipe.transformer
    transformer.add_adapter(PeftLoraConfig(r=2, target_modules=["img_in"]))
    Krea2ModularPipeline.save_lora_weights(
        tmp_path / "adapter", transformer_lora_layers=get_peft_model_state_dict(transformer)
    )
    transformer.delete_adapters("default")
    if offload_type is not None:
        if offload_type == OffloadType.GROUP:
            from diffusers.hooks import apply_group_offloading

            def cpu_group(module: torch.nn.Module, **kwargs: Any) -> None:
                assert kwargs["onload_device"] == torch.device("cuda")
                apply_group_offloading(module, **{**kwargs, "onload_device": torch.device("cpu")})

            monkeypatch.setattr(
                torch.accelerator, "current_accelerator", lambda: torch.device("cuda")
            )
            monkeypatch.setattr("diffusers.hooks.apply_group_offloading", cpu_group)
        wrapper.policy = DevicePolicy(
            device="cuda",
            dtype=torch.float32,
            offload_type=offload_type,
            group_offload_use_stream=False,
        )
        wrapper.policy.apply_to_modular_pipeline(wrapper.pipe, wrapper.components_manager)

    def config(name: str) -> LoraConfig:
        lora = LoraConfig(name=name, source=LoraSource.LOCAL, path="unused")
        lora._resolved_path = tmp_path / "adapter" / "pytorch_lora_weights.safetensors"
        return lora

    static = config("static")
    static.weight = 0.6
    wrapper.load_loras_sync([static])
    wrapper.set_static_loras([static])
    original = wrapper.pipe.load_lora_weights
    names = ["dynamic", "broken"] if failure == "second_adapter" else ["static"]
    failed = False

    def fail_loader(*args: Any, **kwargs: Any) -> None:
        nonlocal failed
        original(*args, **kwargs)
        if not failed and kwargs["adapter_name"] == (
            "broken" if failure == "second_adapter" else "static"
        ):
            failed = True
            raise RuntimeError("adapter failed after mutation")

    if failure != "inference":
        monkeypatch.setattr(wrapper.pipe, "load_lora_weights", fail_loader)

    def fail_inference(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("inference failed")

    monkeypatch.setattr(wrapper, "run_inference", fail_inference)
    if failure == "second_adapter":
        with pytest.warns(UserWarning, match="Already found a `peft_config`"):
            with pytest.raises(RuntimeError):
                wrapper.generate("test", loras=[config(name) for name in names])
    else:
        with pytest.raises(RuntimeError):
            wrapper.generate("test", loras=[config(name) for name in names])
    assert wrapper.active_loras == ["static"]
    assert set(transformer.peft_config) == {"static"}
    assert wrapper.pipe.get_active_adapters() == ["static"]
    assert transformer.img_in.scaling["static"] == 0.6
    if offload_type in {OffloadType.MODEL, OffloadType.SEQUENTIAL}:
        assert hasattr(transformer, "_hf_hook")
    elif offload_type == OffloadType.GROUP:
        assert any(hasattr(child, "_diffusers_hook") for child in transformer.modules())


def test_invalid_controls_precede_lora_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bad workflows, bytes, and unsupported controls never enter resource setup."""
    wrapper = local_wrapper(tmp_path)

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("resource mutation preceded validation")

    monkeypatch.setattr(wrapper, "pre_generate", forbidden)
    for request in (
        {"init_image": b"bad"},
        {"mask_image": image_bytes()},
        {"strength": 0.75},
        {"control_image": image_bytes()},
        {"unknown_option": True},
        {"reference_image": [image_bytes(), b"bad"]},
        {"reference_image": [image_bytes(), None]},
        {"reference_image": []},
        {"negative_prompt": "bad"},
        {"guidance_scale": 7.0},
        {"embeddings": ["unsupported"]},
    ):
        with pytest.raises(ValueError):
            wrapper.generate("test", loras=["unused"], **request)


def test_modular_unload_releases_ownership(tmp_path: Path) -> None:
    """Unloading clears native manager entries and drops the wrapper's component refs."""
    import gc
    import weakref

    from accelerate import cpu_offload

    wrapper = local_wrapper(tmp_path)
    manager = wrapper.components_manager
    pipe = wrapper.pipe
    transformer = weakref.ref(pipe.transformer)
    cpu_offload(pipe.transformer, execution_device=torch.device("cpu"))
    wrapper.unload()
    gc.collect()
    assert wrapper.pipe is None
    assert wrapper.components_manager is None
    assert manager.components == {}
    assert all(component is None for component in pipe.components.values())
    assert transformer() is None


def test_cfg_guider_is_copied_and_restored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Request CFG cannot mutate the guider shared by other native workflows."""
    wrapper = local_wrapper(tmp_path, Krea2AutoBlocks())
    original = wrapper.pipe.guider
    seen = []

    def fail(gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        seen.append(wrapper.pipe.guider)
        raise RuntimeError("inference failed")

    monkeypatch.setattr(wrapper, "run_inference", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        wrapper.generate("test", guidance_scale=2.5)
    assert seen[0] is not original
    assert seen[0].config.guidance_scale == 2.5
    assert wrapper.pipe.guider is original


def test_unsupported_explicit_embedding_uses_clear_error(tmp_path: Path) -> None:
    """A direct explicit load fails at the mixin boundary, not with AttributeError."""
    from oneiro.pipelines.embedding import EmbeddingConfig, EmbeddingSource

    wrapper = local_wrapper(tmp_path)
    with pytest.raises(ValueError, match="does not support.*embeddings"):
        wrapper.load_single_embedding(
            EmbeddingConfig(name="style", source=EmbeddingSource.HUGGINGFACE, repo="unused")
        )


@pytest.mark.parametrize("offload_type", [None, OffloadType.SEQUENTIAL])
def test_supported_native_embedding_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offload_type: OffloadType | None
) -> None:
    """SDXL's real native textual-inversion loader remains reachable through the mixin."""
    from diffusers import StableDiffusionXLAutoBlocks
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    from oneiro.pipelines.embedding import EmbeddingConfig, EmbeddingSource

    wrapper = local_wrapper(tmp_path, StableDiffusionXLAutoBlocks())
    encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=2,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    )
    tokenizer = CLIPTokenizer(vocab={"<|startoftext|>": 0, "<|endoftext|>": 1}, merges=[])
    wrapper.pipe.register_components(text_encoder=encoder, tokenizer=tokenizer)
    if offload_type is not None:
        from accelerate import cpu_offload

        def cpu_sequential(module: torch.nn.Module, **kwargs: Any) -> Any:
            assert kwargs["execution_device"] == torch.device("cuda")
            return cpu_offload(module, **{**kwargs, "execution_device": torch.device("cpu")})

        monkeypatch.setattr("accelerate.cpu_offload", cpu_sequential)
        wrapper.policy = DevicePolicy(device="cuda", dtype=torch.float32, offload_type=offload_type)
        wrapper.policy.apply_to_modular_pipeline(wrapper.pipe, wrapper.components_manager)
    path = tmp_path / "embedding.pt"
    torch.save({"<style>": torch.arange(8).float()}, path)
    embedding = EmbeddingConfig(
        name="style", token="<style>", source=EmbeddingSource.LOCAL, path=str(path)
    )
    embedding._resolved_path = path
    wrapper.load_single_embedding(embedding)
    if offload_type is not None:
        from accelerate.hooks import remove_hook_from_module

        assert hasattr(encoder, "_hf_hook")
        remove_hook_from_module(encoder, recurse=True)
    token_id = tokenizer.convert_tokens_to_ids("<style>")
    torch.testing.assert_close(
        encoder.get_input_embeddings().weight[token_id], torch.arange(8).float()
    )
    assert wrapper.active_embeddings == ["<style>"]


@pytest.mark.parametrize("workflow", ["image2image", "inpainting", "reference"])
def test_shared_image_workflows(tmp_path: Path, workflow: str) -> None:
    """Actual native blocks receive decoded images and workflow-appropriate strength."""
    wrapper = local_wrapper(tmp_path)
    controls = (
        {
            "reference_image": [image_bytes(), image_bytes()],
            "reference_image_encoder_resolution": 32,
        }
        if workflow == "reference"
        else {"init_image": image_bytes()}
    )
    if workflow == "inpainting":
        controls["mask_image"] = image_bytes()
    if workflow != "reference":
        controls["max_sequence_length"] = 8
    result = wrapper.generate("a cat", width=32, height=32, steps=2, seed=7, **controls)
    assert result.image.size == (32, 32)
    assert result.workflow == workflow
    assert result.strength == (None if workflow == "reference" else 0.75)


def test_partial_load_releases_owned_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native manager ownership is empty even when a warning-only component load fails."""
    from oneiro.pipelines import modular

    managers = []
    original = modular.ComponentsManager

    def create_manager() -> Any:
        manager = original()
        managers.append(manager)
        return manager

    monkeypatch.setattr(modular, "ComponentsManager", create_manager)

    def fail_load(self: ComponentSpec, **kwargs: Any) -> Any:
        raise OSError("local component unavailable")

    monkeypatch.setattr(ComponentSpec, "load", fail_load)
    with pytest.raises(RuntimeError, match="transformer"):
        local_wrapper(tmp_path, omit="transformer")
    assert managers[0].components == {}
    assert managers[0].model_hooks is None


def test_native_group_hooks_survive_requests_and_release_on_unload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native hook wrappers stay shared between requests, but release owned forwards on unload."""
    from diffusers.hooks import apply_group_offloading

    wrapper = local_wrapper(tmp_path)
    model = wrapper.pipe.transformer
    # Native ModuleGroup queries accelerator metadata even for non-streamed CPU transfers.
    # Stub discovery only; real hooks and all weights/computation still run on CPU.
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
    apply_group_offloading(model, onload_device="cpu", offload_type="leaf_level")
    hooks = [
        (child, name, hook)
        for child in model.modules()
        if hasattr(child, "_diffusers_hook")
        for name, hook in child._diffusers_hook.hooks.items()
    ]
    assert hooks
    wrapper.generate("a cat", width=32, height=32, steps=2, max_sequence_length=8)
    assert all(child._diffusers_hook.get_hook(name) is hook for child, name, hook in hooks)
    wrapper.unload()
    assert all(child._diffusers_hook.get_hook(name) is None for child, name, hook in hooks)
    assert all(not hasattr(child.forward, "__wrapped__") for child, name, hook in hooks)


def test_cfg_scale_alias_is_consumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native CFG receives the legacy true_cfg_scale control without warn-and-ignore."""
    wrapper = local_wrapper(tmp_path, Krea2AutoBlocks())
    original = wrapper.pipe.guider

    def fail(gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        assert "true_cfg_scale" not in gen_kwargs
        assert wrapper.pipe.guider.config.guidance_scale == 2.5
        raise RuntimeError("inference failed")

    monkeypatch.setattr(wrapper, "run_inference", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        wrapper.generate("test", true_cfg_scale=2.5)
    assert wrapper.pipe.guider is original


def test_distilled_disabled_guider_is_recipe_controlled(tmp_path: Path) -> None:
    """A disabled recipe guider must not be turned into request-time CFG."""
    wrapper = local_wrapper(tmp_path, Krea2AutoBlocks())
    guider = wrapper.pipe.guider.new(enabled=False)
    wrapper.pipe.register_components(guider=guider)
    with pytest.raises(ValueError, match="recipe"):
        wrapper.generate("test", guidance_scale=7.0, width=32, height=32, steps=2)
    assert wrapper.pipe.guider is guider


def test_build_kwargs_rejects_unsupported_controls(tmp_path: Path) -> None:
    """The public kwargs boundary also rejects controls the selected workflow ignores."""
    wrapper = local_wrapper(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        wrapper.build_generation_kwargs(
            "test", None, 32, 32, 2, 0.0, torch.Generator(), None, None, unknown_option=True
        )


def test_original_graph_initialized_before_workflow_extraction(tmp_path: Path) -> None:
    """Workflow extraction must not erase Turbo's native initialization override."""
    from diffusers.modular_pipelines.krea2 import Krea2TurboModularPipeline

    class OriginalTurbo(Krea2TurboAutoBlocks):
        initialized = False

        def init_pipeline(self, *args: Any, **kwargs: Any) -> Any:
            self.initialized = True
            return super().init_pipeline(*args, **kwargs)

        def get_workflow(self, workflow_name: str) -> Any:
            assert self.initialized, "Workflow extraction happened before original init"
            return super().get_workflow(workflow_name)

    wrapper = local_wrapper(tmp_path, OriginalTurbo())
    assert type(wrapper.pipe) is Krea2TurboModularPipeline


def test_capability_validation_does_not_require_manager(tmp_path: Path) -> None:
    """Execution capability can be checked from declarations before loading weights."""
    loaded = local_wrapper(tmp_path)
    wrapper = type(loaded)()
    wrapper.blocks = Krea2AutoBlocks()
    assert wrapper.pipe is None
    assert wrapper.components_manager is None
    assert wrapper.supports_inpaint is True
    assert wrapper.validate_request(has_image=True, has_mask=True) == "inpainting"


def test_optional_controlnet_is_not_required(tmp_path: Path) -> None:
    """This product's supported SDXL workflows do not require ControlNet weights."""
    from diffusers import StableDiffusionXLAutoBlocks

    wrapper = local_wrapper(tmp_path, StableDiffusionXLAutoBlocks(), omit="controlnet")
    assert wrapper.pipe.controlnet is None
    assert wrapper.validate_request(has_image=True, has_mask=True) == "inpainting"
