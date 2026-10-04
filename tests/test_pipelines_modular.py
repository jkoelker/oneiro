"""Offline checks for the shared native modular lifecycle."""

import io
import socket
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

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
from oneiro.pipelines.civitai_checkpoint import CivitaiCheckpointPipeline
from oneiro.pipelines.flux2 import Flux2PipelineWrapper
from oneiro.pipelines.krea2 import Krea2PipelineWrapper
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
def test_modular_workflow_validation(blocks_class: type) -> None:
    """Strength denotes denoising, not text or reference conditioning."""
    wrapper = Krea2PipelineWrapper()
    wrapper.blocks = blocks_class()
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
            for value in (0.0, float("nan"), float("inf"), -0.1, 1.1)
        ),
    ):
        with pytest.raises(ValueError):
            wrapper.validate_request(**request)


def test_conditioned_workflow_rejects_strength() -> None:
    """FLUX.2 images condition generation; they do not select denoising strength."""
    wrapper = Flux2PipelineWrapper()
    wrapper.blocks = Flux2AutoBlocks()
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


def test_invalid_controls_precede_lora_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad workflows, bytes, and unsupported controls never enter resource setup."""
    wrapper = Krea2PipelineWrapper()
    wrapper.blocks = Krea2TurboAutoBlocks()
    wrapper.pipe = SimpleNamespace(components={"guider": None}, load_lora_weights=lambda *_: None)

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("resource mutation preceded validation")

    monkeypatch.setattr(wrapper, "pre_generate", forbidden)
    for request in (
        {"init_image": b"bad"},
        {"init_image": image_bytes(), "strength": 0.0},
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


@pytest.mark.parametrize(
    ("base_model", "component_repo", "has_mask"),
    [
        ("Krea 2", "krea/Krea-2-Raw", False),
        ("Krea 2", "krea/Krea-2-Raw", True),
        ("Krea 2", "krea/Krea-2-Turbo", False),
        ("Krea 2", "krea/Krea-2-Turbo", True),
        ("Pony", None, False),
        ("Pony", None, True),
        ("Flux.1 D", None, False),
        ("SD 3.5", None, False),
        ("Qwen", None, False),
        ("Qwen", None, True),
        ("Z-Image Turbo", None, False),
        ("Z-Image Turbo", None, True),
    ],
)
@pytest.mark.parametrize("strength", [0.0, float("nan"), float("inf")])
async def test_invalid_denoising_strength_precedes_decode_and_resource_mutation(
    base_model: str,
    component_repo: str | None,
    has_mask: bool,
    strength: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real native denoising declarations reject invalid strength before backend side effects."""
    wrapper = CivitaiCheckpointPipeline()
    await wrapper.resolve_config(
        {"checkpoint_path": "unused", "base_model": base_model, "component_repo": component_repo},
        None,
    )
    assert wrapper.validate_request(has_image=True, has_mask=has_mask) == (
        "inpainting" if has_mask else "image2image"
    )
    wrapper.pipe = SimpleNamespace(
        components={"scheduler": object(), "guider": None}, load_lora_weights=lambda *_: None
    )
    decode = Mock(wraps=wrapper._load_init_image)
    mutate = Mock(side_effect=AssertionError("resource mutation preceded strength validation"))
    monkeypatch.setattr(wrapper, "_load_init_image", decode)
    monkeypatch.setattr(wrapper, "pre_generate", mutate)
    request = {"init_image": image_bytes(), "strength": strength}
    if has_mask:
        request["mask_image"] = image_bytes()
    lora = LoraConfig(name="unused", source=LoraSource.LOCAL, path="unused")
    with pytest.raises(ValueError, match="Strength"):
        wrapper.generate("test", loras=[lora], **request)
    decode.assert_not_called()
    mutate.assert_not_called()


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


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.5])
def test_cfg_guider_is_copied_and_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scale: float
) -> None:
    """Request CFG cannot mutate the guider shared by other native workflows."""
    wrapper = local_wrapper(tmp_path, Krea2AutoBlocks())
    original = wrapper.pipe.guider
    seen = []

    def fail(gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        seen.append(wrapper.pipe.guider)
        raise RuntimeError("inference failed")

    monkeypatch.setattr(wrapper, "run_inference", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        wrapper.generate("test", guidance_scale=scale)
    assert seen[0] is not original
    assert seen[0].config.guidance_scale == scale
    assert seen[0].config.use_original_formulation is True
    assert wrapper.pipe.guider is original


def test_unsupported_explicit_embedding_uses_clear_error(tmp_path: Path) -> None:
    """A direct explicit load fails at the mixin boundary, not with AttributeError."""
    from oneiro.pipelines.embedding import EmbeddingConfig, EmbeddingSource

    wrapper = local_wrapper(tmp_path)
    with pytest.raises(ValueError, match="does not support.*embeddings"):
        wrapper.load_single_embedding(
            EmbeddingConfig(name="style", source=EmbeddingSource.HUGGINGFACE, repo="unused")
        )


def local_embedding_wrapper(tmp_path: Path) -> tuple[Any, Any]:
    """Supply native SDXL textual inversion with a tiny local CLIP and embedding."""
    from diffusers import StableDiffusionXLAutoBlocks
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    from oneiro.pipelines.embedding import EmbeddingConfig, EmbeddingSource

    wrapper = local_wrapper(tmp_path, StableDiffusionXLAutoBlocks())
    encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=2,
            bos_token_id=0,
            eos_token_id=1,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        )
    )
    tokenizer = CLIPTokenizer(vocab={"<|startoftext|>": 0, "<|endoftext|>": 1}, merges=[])
    wrapper.pipe.register_components(text_encoder=encoder, tokenizer=tokenizer)
    path = tmp_path / "embedding.pt"
    torch.save({"<style>": torch.arange(8).float()}, path)
    embedding = EmbeddingConfig(
        name="style", token="<style>", source=EmbeddingSource.LOCAL, path=str(path)
    )
    embedding._resolved_path = path
    return wrapper, embedding


def place_embedding_wrapper(
    wrapper: Any, offload_type: OffloadType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain native hooks while substituting only accelerator transfer boundaries."""
    from accelerate import cpu_offload
    from diffusers.hooks import apply_group_offloading

    def cpu_sequential(module: torch.nn.Module, **kwargs: Any) -> Any:
        assert kwargs["execution_device"] == torch.device("cuda")
        return cpu_offload(module, **{**kwargs, "execution_device": torch.device("cpu")})

    def cpu_group(module: torch.nn.Module, **kwargs: Any) -> None:
        assert kwargs["onload_device"] == torch.device("cuda")
        apply_group_offloading(module, **{**kwargs, "onload_device": torch.device("cpu")})

    monkeypatch.setattr("accelerate.cpu_offload", cpu_sequential)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
    monkeypatch.setattr("diffusers.hooks.apply_group_offloading", cpu_group)
    wrapper.policy = DevicePolicy(
        device="cuda",
        dtype=torch.float32,
        offload_type=offload_type,
        group_offload_use_stream=False,
    )
    wrapper.policy.apply_to_modular_pipeline(wrapper.pipe, wrapper.components_manager)


@pytest.mark.parametrize("offload_type", [None, OffloadType.SEQUENTIAL])
def test_supported_native_embedding_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offload_type: OffloadType | None
) -> None:
    """SDXL's real native textual-inversion loader remains reachable through the mixin."""
    wrapper, embedding = local_embedding_wrapper(tmp_path)
    encoder, tokenizer = wrapper.pipe.text_encoder, wrapper.pipe.tokenizer
    if offload_type is not None:
        place_embedding_wrapper(wrapper, offload_type, monkeypatch)
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


@pytest.mark.parametrize("removal", ["single", "all"])
def test_group_embedding_mutations_refresh_native_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, removal: str
) -> None:
    """Native snapshot transfers retain added rows and hook the replacement on removal."""
    from diffusers.hooks.group_offloading import ModuleGroup

    wrapper, embedding = local_embedding_wrapper(tmp_path)
    encoder, tokenizer = wrapper.pipe.text_encoder, wrapper.pipe.tokenizer
    original = ModuleGroup._init_cpu_param_dict

    def cpu_snapshot(group: ModuleGroup) -> dict[Any, torch.Tensor]:
        # Exercise native streamed snapshot construction without CUDA streams/pinning.
        stream = group.stream
        group.stream = object()
        try:
            return original(group)
        finally:
            group.stream = stream

    monkeypatch.setattr(ModuleGroup, "_init_cpu_param_dict", cpu_snapshot)
    monkeypatch.setattr(
        ModuleGroup, "_to_cpu", staticmethod(lambda tensor, _: tensor.detach().cpu().clone())
    )
    place_embedding_wrapper(wrapper, OffloadType.GROUP, monkeypatch)
    baseline = encoder.get_input_embeddings().weight.detach().clone()
    wrapper.load_embeddings_sync([embedding])
    matrix = encoder.get_input_embeddings()
    group = matrix._diffusers_hook.get_hook("group_offloading").group
    group._process_tensors_from_modules(pinned_memory=group.cpu_param_dict)
    assert matrix.weight.shape == (3, 8)
    token_id = tokenizer.convert_tokens_to_ids("<style>")
    torch.testing.assert_close(matrix(torch.tensor([token_id]))[0], torch.arange(8).float())
    if removal == "single":
        wrapper.unload_single_embedding("<style>")
    else:
        wrapper.unload_embeddings()
    matrix = encoder.get_input_embeddings()
    assert matrix.weight.shape == (2, 8)
    group = matrix._diffusers_hook.get_hook("group_offloading").group
    group._process_tensors_from_modules(pinned_memory=group.cpu_param_dict)
    torch.testing.assert_close(matrix(torch.tensor([0, 1])), baseline)
    assert "<style>" not in tokenizer.get_vocab()
    assert wrapper.active_embeddings == []
    wrapper.unload()


@pytest.mark.parametrize("removal", ["single", "all"])
def test_sequential_embedding_removal_restores_owned_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, removal: str
) -> None:
    """Both native removals resize real weights and reinstall working sequential hooks."""
    wrapper, embedding = local_embedding_wrapper(tmp_path)
    encoder, tokenizer = wrapper.pipe.text_encoder, wrapper.pipe.tokenizer
    baseline = encoder.get_input_embeddings().weight.detach().clone()
    place_embedding_wrapper(wrapper, OffloadType.SEQUENTIAL, monkeypatch)
    wrapper.load_single_embedding(embedding)
    if removal == "single":
        wrapper.unload_single_embedding("<style>")
    else:
        wrapper.unload_embeddings()
    matrix = encoder.get_input_embeddings()
    assert matrix.weight.shape == (2, 8)
    assert matrix.weight.device.type == "meta"
    torch.testing.assert_close(matrix(torch.tensor([0, 1])), baseline)
    assert hasattr(encoder, "_hf_hook")
    assert matrix.weight.device.type == "meta"
    assert "<style>" not in tokenizer.get_vocab()
    assert wrapper.active_embeddings == []
    assert wrapper._embedding_configs == []
    wrapper.unload()


@pytest.mark.parametrize("offload_type", [OffloadType.GROUP, OffloadType.SEQUENTIAL])
@pytest.mark.parametrize("removal", ["single", "all"])
def test_embedding_removal_failure_restores_owned_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offload_type: OffloadType,
    removal: str,
) -> None:
    """Even failure after native mutation restores placement without clearing tracking."""
    wrapper, embedding = local_embedding_wrapper(tmp_path)
    encoder = wrapper.pipe.text_encoder
    baseline = encoder.get_input_embeddings().weight.detach().clone()
    place_embedding_wrapper(wrapper, offload_type, monkeypatch)
    wrapper.load_single_embedding(embedding)
    original = wrapper.pipe.unload_textual_inversion

    def fail_after_removal(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("failed after native removal")

    monkeypatch.setattr(wrapper.pipe, "unload_textual_inversion", fail_after_removal)
    with pytest.raises(RuntimeError, match="failed after native removal"):
        if removal == "single":
            wrapper.unload_single_embedding("<style>")
        else:
            wrapper.unload_embeddings()
    assert wrapper.active_embeddings == ["<style>"]
    assert wrapper._embedding_configs == [embedding]
    matrix = encoder.get_input_embeddings()
    assert matrix.weight.shape == (2, 8)
    torch.testing.assert_close(matrix(torch.tensor([0, 1])), baseline)
    if offload_type == OffloadType.SEQUENTIAL:
        assert hasattr(encoder, "_hf_hook")
        assert matrix.weight.device.type == "meta"
    else:
        assert matrix._diffusers_hook.get_hook("group_offloading") is not None
    wrapper.unload()


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


def load_hosted(
    wrapper_class: type,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any] | None = None,
    full_config: dict[str, Any] | None = None,
    assets: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Keep real native initialization/ownership; replace only asset-loading boundaries."""
    import diffusers
    from diffusers import Flux2Transformer2DModel, ModularPipeline
    from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

    from oneiro.pipelines.base import BasePipeline
    from oneiro.pipelines.modular import ModularPipelineWrapper

    assert issubclass(wrapper_class, ModularPipelineWrapper), "Hosted loading is still classic"
    records: dict[str, Any] = {"spec_loads": [], "pretrained": []}

    def hosted_index(cls: type, *args: Any, **kwargs: Any) -> tuple[None, dict[str, Any]]:
        blocks = getattr(diffusers, cls.default_blocks_name)()
        return None, {
            spec.name: (spec.type_hint.__module__.split(".")[0], spec.type_hint.__name__)
            for spec in blocks.expected_components
            if spec.default_creation_method == "from_pretrained"
        }

    monkeypatch.setattr(ModularPipeline, "_load_pipeline_config", classmethod(hosted_index))
    monkeypatch.setattr(BasePipeline, "_configure_cpu_threads", lambda *a: 1)

    def component(name: str) -> Any:
        if assets and name in assets:
            return assets[name]
        if "tokenizer" in name:
            return TinyTokenizer()
        if name == "scheduler":
            return FlowMatchEulerDiscreteScheduler()
        module = torch.nn.Linear(2, 2)
        module.config = SimpleNamespace(block_out_channels=[8, 8])
        if name == "vae":
            module.enable_tiling = lambda: setattr(module, "use_tiling", True)
            module.enable_slicing = lambda: setattr(module, "use_slicing", True)
        return module

    def load_spec(spec: ComponentSpec, **kwargs: Any) -> Any:
        records["spec_loads"].append(spec.name)
        return component(spec.name)

    monkeypatch.setattr(ComponentSpec, "load", load_spec)
    for cls, name in (
        (Flux2Transformer2DModel, "transformer"),
        (Mistral3ForConditionalGeneration, "text_encoder"),
        (BackportedKrea2Transformer2DModel, "backported_transformer"),
        (AutoTokenizer, "tokenizer"),
    ):

        def load_asset(*args: Any, _name: str = name, **kwargs: Any) -> Any:
            asset = component(_name)
            if _name in {"transformer", "text_encoder"}:
                asset.quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True}
            records["pretrained"].append((_name, args, kwargs, asset))
            return asset

        monkeypatch.setattr(cls, "from_pretrained", load_asset)
    wrapper = wrapper_class()
    wrapper.load(config or {}, full_config)
    return wrapper, records


def capture_generation(wrapper: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Observe real wrapper routing at the model-computation boundary."""
    calls = []

    def infer(gen_kwargs: dict[str, Any], is_img2img: bool) -> dict[str, Any]:
        calls.append({**gen_kwargs, "guider": wrapper.pipe.components.get("guider")})
        return {"images": [Image.new("RGB", (gen_kwargs["width"], gen_kwargs["height"]))]}

    monkeypatch.setattr(wrapper, "run_inference", infer)
    return calls


@pytest.mark.parametrize(
    "module,class_name,config,graph,steps,guidance,image_workflow,mask",
    [
        ("flux1", "Flux1PipelineWrapper", {}, "FluxAutoBlocks", 28, 3.5, "image2image", False),
        (
            "flux1",
            "Flux1PipelineWrapper",
            {"repo": "black-forest-labs/FLUX.1-schnell"},
            "FluxAutoBlocks",
            4,
            0.0,
            "image2image",
            False,
        ),
        (
            "flux2",
            "Flux2PipelineWrapper",
            {},
            "Flux2AutoBlocks",
            28,
            4.0,
            "image_conditioned",
            False,
        ),
        (
            "flux2_klein",
            "Flux2KleinPipelineWrapper",
            {},
            "Flux2KleinAutoBlocks",
            4,
            1.0,
            "image_conditioned",
            False,
        ),
        (
            "flux2_klein",
            "Flux2KleinPipelineWrapper",
            {"repo": "black-forest-labs/FLUX.2-klein-base-9B"},
            "Flux2KleinBaseAutoBlocks",
            50,
            4.0,
            "image_conditioned",
            False,
        ),
        ("qwen", "QwenPipelineWrapper", {}, "QwenImageAutoBlocks", 8, 4.0, "image2image", True),
        ("krea2", "Krea2PipelineWrapper", {}, "Krea2TurboAutoBlocks", 8, 0.0, "image2image", True),
        (
            "krea2",
            "Krea2PipelineWrapper",
            {"repo": "krea/Krea-2-Raw"},
            "Krea2AutoBlocks",
            28,
            4.5,
            "image2image",
            True,
        ),
        ("zimage", "ZImagePipelineWrapper", {}, "ZImageAutoBlocks", 9, 0.0, "image2image", True),
    ],
)
def test_hosted_family_workflows(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    class_name: str,
    config: dict[str, Any],
    graph: str,
    steps: int,
    guidance: float,
    image_workflow: str,
    mask: bool,
) -> None:
    """Recipes choose real graphs and consumer-visible defaults without loading weights."""
    import importlib

    from oneiro.pipelines.modular import ModularPipelineWrapper

    cls = getattr(importlib.import_module(f"oneiro.pipelines.{module}"), class_name)
    assert issubclass(cls, ModularPipelineWrapper)
    wrapper, _ = load_hosted(cls, monkeypatch, config)
    default_repos = {
        "flux1": "black-forest-labs/FLUX.1-dev",
        "flux2": "diffusers/FLUX.2-dev-bnb-4bit",
        "flux2_klein": "black-forest-labs/FLUX.2-klein-9B",
        "qwen": "Qwen/Qwen-Image",
        "krea2": "krea/Krea-2-Turbo",
        "zimage": "Tongyi-MAI/Z-Image-Turbo",
    }
    assert wrapper.pipe._pretrained_model_name_or_path == config.get("repo", default_repos[module])
    assert type(wrapper.blocks).__name__ == graph
    assert wrapper.validate_request(has_image=True) == image_workflow
    assert wrapper.supports_inpaint is mask
    calls = capture_generation(wrapper, monkeypatch)
    result = wrapper.generate("a cat", seed=7, width=64, height=32)
    assert result.steps == steps
    assert result.guidance_scale == guidance
    assert result.image.size == (64, 32)
    assert calls[0]["generator"].initial_seed() == 7
    assert calls[0]["num_inference_steps"] == steps
    if calls[0]["guider"] is not None and calls[0]["guider"].config.enabled:
        assert calls[0]["guider"].config.guidance_scale == guidance
    elif "guidance_scale" in calls[0]:
        assert calls[0]["guidance_scale"] == guidance
    wrapper.generate("a cat", width=64, height=32, init_image=image_bytes())
    assert isinstance(calls[-1]["image"], Image.Image)
    assert ("strength" in calls[-1]) is (image_workflow == "image2image")
    if image_workflow == "image_conditioned":
        with pytest.raises(ValueError, match="strength"):
            wrapper.generate("a cat", init_image=image_bytes(), strength=0.5)
    manager = wrapper.components_manager
    wrapper.unload()
    assert manager.components == {}
    assert wrapper.pipe is None


@pytest.mark.parametrize(
    "module,class_name,config",
    [
        ("qwen", "QwenPipelineWrapper", {"variant": "image"}),
        ("krea2", "Krea2PipelineWrapper", {"variant": "raw"}),
        ("flux2_klein", "Flux2KleinPipelineWrapper", {"variant": "base"}),
    ],
)
def test_guidance_override_does_not_leak(
    monkeypatch: pytest.MonkeyPatch, module: str, class_name: str, config: dict[str, Any]
) -> None:
    """Each CFG request copies the native guider and cleanup restores its identity."""
    import importlib

    cls = getattr(importlib.import_module(f"oneiro.pipelines.{module}"), class_name)
    # Explicit variant metadata is valid for a custom source, not a conflicting official one.
    wrapper, _ = load_hosted(cls, monkeypatch, {"repo": "custom/model", **config})
    original = wrapper.pipe.guider
    calls = capture_generation(wrapper, monkeypatch)
    wrapper.generate("test", guidance_scale=2.5)
    assert calls[0]["guider"] is not original
    assert calls[0]["guider"].config.guidance_scale == 2.5
    assert wrapper.pipe.guider is original
    wrapper.generate("test")
    assert calls[1]["guider"].config.guidance_scale == wrapper.default_guidance_scale
    assert wrapper.pipe.guider is original


@pytest.mark.parametrize(
    "module,class_name",
    [
        ("flux1", "Flux1PipelineWrapper"),
        ("flux2", "Flux2PipelineWrapper"),
        ("flux2_klein", "Flux2KleinPipelineWrapper"),
        ("qwen", "QwenPipelineWrapper"),
        ("krea2", "Krea2PipelineWrapper"),
        ("zimage", "ZImagePipelineWrapper"),
    ],
)
def test_unsupported_embedding_is_not_ignored(
    monkeypatch: pytest.MonkeyPatch, module: str, class_name: str
) -> None:
    """Explicit named/inline embedding requests cannot become decorative mixins/warnings."""
    import importlib

    cls = getattr(importlib.import_module(f"oneiro.pipelines.{module}"), class_name)
    with pytest.raises(ValueError, match="embeddings"):
        load_hosted(cls, monkeypatch, {"embeddings": ["missing"]})
    with pytest.raises(ValueError, match="embeddings"):
        load_hosted(cls, monkeypatch, {"inline_embeddings": [{"path": "unused"}]})
    wrapper, _ = load_hosted(cls, monkeypatch, full_config={"embeddings": {"auto_load": ["style"]}})
    assert wrapper.active_embeddings == []


@pytest.mark.parametrize(
    "module,class_name",
    [
        ("flux1", "Flux1PipelineWrapper"),
        ("flux2", "Flux2PipelineWrapper"),
        ("flux2_klein", "Flux2KleinPipelineWrapper"),
        ("qwen", "QwenPipelineWrapper"),
        ("krea2", "Krea2PipelineWrapper"),
        ("zimage", "ZImagePipelineWrapper"),
    ],
)
def test_hosted_placement_configuration(
    monkeypatch: pytest.MonkeyPatch, module: str, class_name: str
) -> None:
    """All recipes retain offload overrides without installing classic pipeline hooks."""
    import importlib

    from oneiro.device import OffloadMode

    cls = getattr(importlib.import_module(f"oneiro.pipelines.{module}"), class_name)
    wrapper, _ = load_hosted(
        cls,
        monkeypatch,
        {
            "cpu_offload": False,
            "offload_type": "sequential",
            "group_offload_type": "block_level",
            "group_offload_use_stream": False,
            "group_offload_num_blocks_per_group": 2,
        },
    )
    assert wrapper.policy.offload == OffloadMode.NEVER
    assert wrapper.policy.offload_type == OffloadType.SEQUENTIAL
    assert wrapper.policy.group_offload_type == "block_level"
    assert wrapper.policy.group_offload_use_stream is False
    assert wrapper.policy.group_offload_num_blocks_per_group == 2
    assert wrapper.components_manager.model_hooks is None
    assert all(not hasattr(value, "_hf_hook") for value in wrapper.pipe.components.values())
