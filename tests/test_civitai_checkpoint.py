"""Bounded offline component, workflow, scheduler and Krea conversion gates."""

import json
import socket
import threading
import weakref
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
from PIL import Image
from safetensors.torch import save_file

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.backports.krea2 import BackportedKrea2Transformer2DModel
from oneiro.pipelines.base import BasePipeline
from oneiro.pipelines.civitai_checkpoint import (
    CIVITAI_BASE_MODEL_PIPELINE_MAP,
    CivitaiCheckpointPipeline,
    PipelineConfig,
    get_pipeline_config_for_base_model,
)
from oneiro.pipelines.krea2_checkpoint import (
    convert_krea2_checkpoint_tensor,
    get_krea2_checkpoint_precision,
    get_krea2_checkpoint_precision_from_header,
    load_krea2_transformer,
)
from oneiro.pipelines.modular import ModularPipelineWrapper
from tests.test_krea2_backport import TinyTokenizer
from tests.test_pipelines_modular import image_bytes, local_embedding_wrapper, local_wrapper


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make accidental model downloads fail immediately."""

    def fail_network(*args: Any, **kwargs: Any) -> None:
        """Fail any accidental external connection in these offline gates."""
        raise AssertionError("No downloads allowed in checkpoint gates")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TestCheckpointPreflight:
    """Only known family metadata and explicit source variants reach weight loaders."""

    @pytest.mark.parametrize("source", ["local", "remote"])
    @pytest.mark.parametrize(
        "label,family,variant",
        [
            ("ZImageTurbo", "zimage", "turbo"),
            ("Flux.2 D", "flux2", None),
            ("Flux.1 Krea", "flux1", "dev"),
        ],
    )
    async def test_observed_civitai_labels_resolve_native_recipes(
        self, source: str, label: str, family: str, variant: str | None
    ) -> None:
        """Actual API labels must reach their retained family, never a substring fallback."""
        pipeline = CivitaiCheckpointPipeline()
        client = AsyncMock()
        client.get_model.return_value = SimpleNamespace(
            latest_version=SimpleNamespace(base_model=label)
        )
        profile = {"checkpoint_path": "unused", "base_model": label}
        if source == "remote":
            profile = {"civitai_model_id": 1}
        resolved = await pipeline.resolve_config(profile, client)
        assert resolved["family"] == family
        assert resolved["variant"] == variant
        assert pipeline.validate_request(has_image=True) == (
            "image_conditioned" if family == "flux2" else "image2image"
        )

    @pytest.mark.parametrize(
        "base_model",
        [
            None,
            "Other",
            "SD 1.5",
            "SD 2.1",
            "PixArt Sigma",
            "Kolors",
            "Hunyuan DiT",
            "Lumina",
            "AuraFlow",
            "unknown XL",
            "Flux.9000",
            "Krea Unknown",
        ],
    )
    async def test_civitai_rejects_removed_family_before_download(
        self, base_model: str | None
    ) -> None:
        """Unsupported remote metadata cannot reach any weight loader or download."""
        pipeline = CivitaiCheckpointPipeline()
        client = AsyncMock()
        client.get_model_version.return_value = SimpleNamespace(base_model=base_model)
        with patch.object(pipeline, "_load_from_path") as loader:
            with pytest.raises(ValueError, match="base model"):
                await pipeline.load_async({"civitai_model_id": 1, "civitai_version_id": 2}, client)
        client.download_model_version.assert_not_awaited()
        loader.assert_not_called()

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"pipeline_class": "StableDiffusionXLPipeline"},
            {"base_model": "SD 1.5"},
            {"base_model": "SDXL 1.0", "pipeline_class": "StableDiffusionPipeline"},
            {"base_model": "Qwen", "sequential_cpu_offload": False},
            {"base_model": "Qwen", "scheduler": "dpm++"},
            {"base_model": "Krea 2", "component_repo": "custom/raw-looking-repo"},
            {"base_model": "Krea 2", "component_repo": "krea/Krea-2-Raw", "variant": "turbo"},
        ],
    )
    async def test_local_preflight_needs_known_family(
        self, config: dict[str, Any], tmp_path: Path
    ) -> None:
        """Local sources cannot bypass family, variant or scheduler preflight."""
        pipeline = CivitaiCheckpointPipeline()
        with pytest.raises(ValueError):
            await pipeline.resolve_config(
                {"checkpoint_path": str(tmp_path / "missing"), **config}, None
            )

    async def test_metadata_preflight_retains_version_without_download(
        self, tmp_path: Path
    ) -> None:
        """Resolve once, then reuse the retained version in the asynchronous loader."""
        pipeline = CivitaiCheckpointPipeline()
        version = SimpleNamespace(base_model="Pony")
        client = AsyncMock()
        client.get_model.return_value = SimpleNamespace(latest_version=version)
        resolved = await pipeline.resolve_config({"civitai_model_id": 1}, client)
        assert resolved["family"] == pipeline.family == "sdxl"
        assert pipeline._resolved_version is version
        client.download_model_version.assert_not_awaited()
        client.download_model_version.return_value = tmp_path / "pony.safetensors"
        with patch.object(pipeline, "load") as load:
            await pipeline.load_async(resolved, client, {"embeddings": {}})
        client.get_model.assert_awaited_once()
        client.download_model_version.assert_awaited_once_with(version)
        assert load.call_args.args[0]["base_model"] == "Pony"
        assert load.call_args.args[1] == {"embeddings": {}}

    @pytest.mark.parametrize("override", ["Flux.1 D", "Other"])
    async def test_remote_metadata_cannot_be_replaced_by_another_family(
        self, override: str
    ) -> None:
        """Remote metadata remains authoritative when a local base_model override conflicts."""
        pipeline = CivitaiCheckpointPipeline()
        client = AsyncMock()
        client.get_model.return_value = SimpleNamespace(
            latest_version=SimpleNamespace(base_model="Pony")
        )
        with pytest.raises(ValueError, match="base model"):
            await pipeline.resolve_config({"civitai_model_id": 1, "base_model": override}, client)
        client.download_model_version.assert_not_awaited()

    async def test_async_local_load_runs_off_event_loop(self, tmp_path: Path) -> None:
        """Even local async conversion runs in a worker, not on the event loop."""
        loop_thread = threading.get_ident()
        threads = []
        pipeline = CivitaiCheckpointPipeline()
        with patch.object(
            pipeline, "load", side_effect=lambda *args: threads.append(threading.get_ident())
        ):
            await pipeline.load_async(
                {"checkpoint_path": str(tmp_path / "local"), "base_model": "Pony"}, AsyncMock()
            )
        assert threads and threads[0] != loop_thread

    @pytest.mark.parametrize("base_model", list(CIVITAI_BASE_MODEL_PIPELINE_MAP))
    def test_every_retained_metadata_has_explicit_family(self, base_model: str) -> None:
        """Every retained metadata name selects one supported native family."""
        assert get_pipeline_config_for_base_model(base_model).family in {
            "sdxl",
            "sd3",
            "flux1",
            "flux2",
            "flux2-klein",
            "krea2",
            "qwen",
            "zimage",
        }

    def test_pipeline_config_requires_family(self) -> None:
        """Conversion class metadata cannot replace an explicit resolved family."""
        with pytest.raises(TypeError):
            PipelineConfig(pipeline_class="StableDiffusionXLPipeline")

    def test_normalized_sd3_metadata_selects_matching_components(self) -> None:
        """Case/whitespace normalization must not fall back to SD3 medium components."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._configure_recipe({"base_model": " sd 3.5 large "})
        assert pipeline._component_repo == "stabilityai/stable-diffusion-3.5-large"


def checkpoint_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_model: str = "Pony",
    variant: str | None = None,
    omit: str | None = None,
) -> tuple[CivitaiCheckpointPipeline, dict[str, Any], dict[str, Any]]:
    """Use a real original native graph/manager with model-free component doubles."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "model_index.json").write_text("{}")
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.touch()
    pipeline = CivitaiCheckpointPipeline()
    config = {
        "checkpoint_path": str(checkpoint),
        "base_model": base_model,
        "component_repo": str(repo),
        "cpu_offload": False,
        "scheduler": "default",
    }
    if variant:
        config["variant"] = variant
    pipeline._configure_recipe(config)
    components = {
        spec.name: TinyTokenizer() if "tokenizer" in spec.name else torch.nn.Linear(2, 2)
        for spec in pipeline.blocks.expected_components
        if spec.default_creation_method == "from_pretrained"
    }
    components["scheduler"] = (
        DDIMScheduler() if pipeline.family == "sdxl" else FlowMatchEulerDiscreteScheduler()
    )
    if omit:
        components.pop(omit)
    monkeypatch.setattr(pipeline, "_load_checkpoint_components", lambda *args: components)
    return pipeline, config, components


class TestCheckpointComponents:
    """Loading, resource hooks and inference use the reviewed shared lifecycle."""

    @pytest.mark.parametrize(
        ("base", "variant", "family"),
        [
            ("Pony", None, "sdxl"),
            ("Illustrious", None, "sdxl"),
            ("SD 3.5 Large", None, "sd3"),
            ("Flux.1 D", "dev", "flux1"),
            ("Flux.1 S", "schnell", "flux1"),
            ("Flux.2", None, "flux2"),
            ("Flux.2 Klein 4B", "distilled", "flux2-klein"),
            ("Flux.2 Klein 9B-base", "base", "flux2-klein"),
            ("Qwen", "image", "qwen"),
            ("Krea 2", "raw", "krea2"),
            ("Krea 2", "turbo", "krea2"),
            ("Z-Image", "turbo", "zimage"),
        ],
    )
    def test_checkpoint_components_use_shared_workflow(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        base: str,
        variant: str | None,
        family: str,
    ) -> None:
        """All retained recipes declare actual text and image request inputs."""
        pipeline, config, components = checkpoint_wrapper(tmp_path, monkeypatch, base, variant)
        if family == "zimage":
            monkeypatch.setattr(
                "oneiro.pipelines.zimage.ZImageInpaintPipeline",
                lambda **kwargs: SimpleNamespace(**kwargs),
            )
        pipeline.load(config)
        assert isinstance(pipeline, ModularPipelineWrapper)
        assert pipeline.family == family
        assert pipeline.pipe.components.get(
            "unet", pipeline.pipe.components.get("transformer")
        ) is components.get("unet", components.get("transformer"))
        assert pipeline.validate_request(has_image=True) == (
            "image_conditioned" if family in {"flux2", "flux2-klein"} else "image2image"
        )
        assert pipeline.supports_inpaint == (family in {"sdxl", "qwen", "krea2", "zimage"})

        def inference(self: BasePipeline, values: dict[str, Any], img2img: bool) -> dict[str, Any]:
            """Catch wrapper/actual native graph mismatches at the inference boundary."""
            assert values.keys() <= set(pipeline.pipe.blocks.input_names)
            return {"images": [Image.new("RGB", (32, 32))]}

        monkeypatch.setattr(BasePipeline, "run_inference", inference)
        for helper in ("sdxl", "sd3", "flux"):
            monkeypatch.setattr(
                f"oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_{helper}",
                lambda *args, _count=2 if helper == "flux" else 4, **kwargs: (
                    (torch.ones(1),) * _count
                ),
            )
        assert pipeline.generate("(cat:1.5)", width=32, height=32).workflow == "text2image"
        assert pipeline.generate("cat", init_image=image_bytes(), width=32, height=32).workflow == (
            "image_conditioned" if family in {"flux2", "flux2-klein"} else "image2image"
        )
        pipeline.unload()
        assert pipeline.pipe is pipeline.components_manager is pipeline.inpaint_pipe is None

    def test_partial_checkpoint_load_releases_components(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing required native components release the partial manager and wrapper."""
        pipeline, config, _ = checkpoint_wrapper(tmp_path, monkeypatch, omit="unet")
        monkeypatch.setattr(ComponentSpec, "load", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError, match="unet"):
            pipeline.load(config)
        assert pipeline.pipe is pipeline.components_manager is None

    def test_classic_container_is_discarded_before_placement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-file containers leave only original native graph component references."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._configure_recipe({"base_model": "Pony"})

        class Container:
            components = {"unet": object(), "watermark": object()}

        container = Container()
        reference = weakref.ref(container)
        unet = container.components["unet"]
        containers = [container]
        del container
        loader = SimpleNamespace(from_single_file=lambda *args, **kwargs: containers.pop())
        monkeypatch.setattr(
            "oneiro.pipelines.civitai_checkpoint.get_diffusers_pipeline_class", lambda name: loader
        )
        components = pipeline._load_checkpoint_components(tmp_path / "unused", {})
        assert components == {"unet": unet}
        assert reference() is None

    def test_sdxl_text_component_retry_and_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only missing SDXL text weights retry with explicitly configured components."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._configure_recipe({"base_model": "Pony"})
        container = SimpleNamespace(components={"unet": object()})
        calls = []

        def load(*args: Any, **kwargs: Any) -> Any:
            """Fail the first conversion exactly as the missing-CLIP loader does."""
            calls.append(kwargs)
            if len(calls) == 1:
                raise ValueError("Weights for this component appear to be missing CLIPTextModel")
            return container

        monkeypatch.setattr(
            "oneiro.pipelines.civitai_checkpoint.get_diffusers_pipeline_class",
            lambda name: SimpleNamespace(from_single_file=load),
        )
        with (
            patch("transformers.CLIPTextModel.from_pretrained", return_value=object()) as text,
            patch(
                "transformers.CLIPTextModelWithProjection.from_pretrained", return_value=object()
            ),
            patch("transformers.CLIPTokenizer.from_pretrained", return_value=object()),
        ):
            pipeline._load_checkpoint_components(
                tmp_path / "unused",
                {"text_encoder_repo": "custom/text", "text_encoder_subfolder": "clip"},
            )
        assert text.call_args.args == ("custom/text",)
        assert text.call_args.kwargs["subfolder"] == "clip"
        assert len(calls) == 2 and "text_encoder_2" in calls[1]
        assert not pipeline._should_retry_with_sdxl_text_components(
            ValueError("corrupt checkpoint"), {}
        )

    @pytest.mark.parametrize(
        ("base", "variant", "loader_name"),
        [
            ("Qwen", "image", "QwenImageTransformer2DModel"),
            ("Flux.2", None, "Flux2Transformer2DModel"),
            ("Flux.2 Klein 4B-base", "base", "Flux2Transformer2DModel"),
            ("Z-Image", "turbo", "ZImageTransformer2DModel"),
        ],
    )
    def test_transformer_only_conversion_injects_no_classic_pipeline(
        self,
        tmp_path: Path,
        base: str,
        variant: str | None,
        loader_name: str,
    ) -> None:
        """Non-container families inject released component converters directly."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._configure_recipe({"base_model": base})
        state = {"double_blocks.weight": torch.ones(1)}
        with (
            patch.object(pipeline, "_load_transformer_checkpoint", return_value=state),
            patch(f"diffusers.{loader_name}.from_single_file", return_value=object()) as load,
        ):
            result = pipeline._load_checkpoint_components(tmp_path / "weights.safetensors", {})
        assert load.call_args.args == (state,)
        assert "transformer" in result and set(result) <= {"transformer", "scheduler"}
        assert load.call_args.kwargs["torch_dtype"] == pipeline.policy.dtype

    def test_qwen_gguf_and_explicit_precision(self, tmp_path: Path) -> None:
        """Qwen keeps GGUF compute precision separate from requested storage precision."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._configure_recipe({"base_model": "Qwen"})
        with patch(
            "diffusers.QwenImageTransformer2DModel.from_single_file", return_value=object()
        ) as load:
            pipeline._load_checkpoint_components(
                tmp_path / "weights.gguf", {"transformer_dtype": "fp8_e4m3fn"}
            )
        assert load.call_args.kwargs["torch_dtype"] == torch.float8_e4m3fn
        assert load.call_args.kwargs["quantization_config"].compute_dtype == pipeline.policy.dtype

    def test_zimage_component_overrides_are_injected_before_gap_loading(
        self, tmp_path: Path
    ) -> None:
        """Explicit per-component sources must survive migration to hosted gap loading."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._configure_recipe({"base_model": "Z-Image"})
        with (
            patch.object(pipeline, "_load_transformer_checkpoint", return_value={}),
            patch("diffusers.ZImageTransformer2DModel.from_single_file", return_value=object()),
            patch.object(ComponentSpec, "load", return_value=object()) as load,
        ):
            components = pipeline._load_checkpoint_components(
                tmp_path / "unused",
                {"text_encoder_repo": "custom/text", "text_encoder_subfolder": "encoder"},
            )
        assert "text_encoder" in components
        assert load.call_args.kwargs["pretrained_model_name_or_path"] == "custom/text"
        assert load.call_args.kwargs["subfolder"] == "encoder"

    @pytest.mark.parametrize("variant", ["raw", "turbo"])
    def test_krea_checkpoint_uses_hosted_local_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
    ) -> None:
        """Both Krea variants share hosted graph types and the returned loader policy."""
        pipeline = CivitaiCheckpointPipeline()
        repo = "krea/Krea-2-Raw" if variant == "raw" else "krea/Krea-2-Turbo"
        pipeline._configure_recipe({"base_model": "Krea 2", "component_repo": repo})
        from oneiro.pipelines.backports.krea2 import Krea2AutoBlocks, Krea2TurboAutoBlocks

        assert type(pipeline.blocks) is (
            Krea2AutoBlocks if variant == "raw" else Krea2TurboAutoBlocks
        )
        policy = DevicePolicy(device="cpu", dtype=torch.float32, group_offload_use_stream=False)
        with (
            patch(
                "oneiro.pipelines.civitai_checkpoint.load_krea2_transformer",
                return_value=(object(), policy),
            ),
            patch("oneiro.pipelines.krea2.load_krea2_tokenizer", return_value=object()),
        ):
            result = pipeline._load_checkpoint_components(tmp_path / "unused", {})
        assert set(result) == {"tokenizer", "transformer"}
        assert pipeline.policy is policy


class TestCheckpointGeneration:
    """Native input declarations and shared rollback govern every request."""

    @pytest.mark.parametrize(
        "base,variant", [("Pony", None), ("SD 3.5", None), ("Flux.1 D", "dev")]
    )
    def test_native_state_preserves_weighted_embeddings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base: str, variant: str | None
    ) -> None:
        """The installed native embedding step preserves tensors and denoiser field tags."""
        pipeline, config, _ = checkpoint_wrapper(tmp_path, monkeypatch, base, variant)
        pipeline.load(config)
        step = pipeline.pipe.blocks.sub_blocks["text_encoder"]
        native = step.init_pipeline()
        values = {output.name: torch.ones(1, 2, 2) for output in step.intermediate_outputs}
        state = native(**values)
        assert all(state.get(name) is tensor for name, tensor in values.items())
        for output in step.intermediate_outputs:
            if output.kwargs_type is not None:
                assert state.get_by_kwargs(output.kwargs_type)[output.name] is values[output.name]
        assert {spec.name for spec in step.expected_components} >= {"text_encoder", "tokenizer"}
        pipeline.unload()

    @pytest.mark.parametrize("base,variant", [("Pony", None), ("Flux.1 D", "dev")])
    def test_checkpoint_uses_native_textual_inversion_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base: str, variant: str | None
    ) -> None:
        """Supported native loaders, not legacy family mixins, consume resolved embeddings."""
        source, embedding = local_embedding_wrapper(tmp_path)
        pipeline, config, components = checkpoint_wrapper(tmp_path, monkeypatch, base, variant)
        components.update(text_encoder=source.pipe.text_encoder, tokenizer=source.pipe.tokenizer)
        config["_resolved_embeddings"] = [embedding]
        pipeline.load(config, {"embeddings": {}})
        token_id = pipeline.pipe.tokenizer.convert_tokens_to_ids("<style>")
        torch.testing.assert_close(
            pipeline.pipe.text_encoder.get_input_embeddings().weight[token_id],
            torch.arange(8).float(),
        )
        assert pipeline.active_embeddings == ["<style>"]
        pipeline.unload_single_embedding("<style>")
        assert pipeline.active_embeddings == []
        pipeline.unload()

    @pytest.mark.parametrize("failure", [False, True])
    def test_scheduler_is_restored_after_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
    ) -> None:
        """Request-local scheduler changes restore the profile on success and failure."""
        pipeline, config, _ = checkpoint_wrapper(tmp_path, monkeypatch)
        pipeline.load(config)
        profile = pipeline.pipe.scheduler
        seen = []

        def inference(self: BasePipeline, values: dict[str, Any], img2img: bool) -> dict[str, Any]:
            """Observe the temporary scheduler and optionally fail inference."""
            seen.append(pipeline.pipe.scheduler)
            assert "prompt_embeds" in values and "prompt" not in values
            if failure:
                raise RuntimeError("inference failed")
            return {"images": [Image.new("RGB", (32, 32))]}

        monkeypatch.setattr(BasePipeline, "run_inference", inference)
        monkeypatch.setattr(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sdxl",
            lambda *args, **kwargs: (torch.ones(1),) * 4,
        )
        if failure:
            with pytest.raises(RuntimeError, match="inference failed"):
                pipeline.generate(
                    "(cat:1.5)", negative_prompt="bad", scheduler="euler", width=32, height=32
                )
        else:
            result = pipeline.generate(
                "(cat:1.5)", negative_prompt="bad", scheduler="euler", width=32, height=32
            )
            assert result.workflow == "text2image"
        assert seen[0] is not profile
        assert pipeline.pipe.scheduler is profile

    @pytest.mark.parametrize("base", ["Qwen", "Krea 2", "Flux.1 D", "Flux.2", "SD 3.5", "Z-Image"])
    def test_flow_family_rejects_discrete_scheduler(self, base: str) -> None:
        """Discrete scheduler overrides cannot be applied to flow-matching recipes."""
        with pytest.raises(ValueError, match="not compatible"):
            CivitaiCheckpointPipeline()._configure_recipe(
                {"base_model": base, "scheduler": "dpm++"}
            )

    def test_krea_checkpoint_shared_inference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tiny local Krea components run actual shared text and mask inference offline."""
        tiny = local_wrapper(tmp_path)
        pipeline = CivitaiCheckpointPipeline()
        checkpoint = tmp_path / "unused"
        checkpoint.touch()
        components = dict(tiny.pipe.components)
        # Inject only original graph components, keeping local text and transformer identities.
        monkeypatch.setattr(pipeline, "_load_checkpoint_components", lambda *args: components)
        pipeline.load(
            {
                "checkpoint_path": str(checkpoint),
                "base_model": "Krea 2",
                "component_repo": str(tmp_path / "components"),
                "variant": "turbo",
            }
        )
        pipeline.pipe.set_progress_bar_config(disable=True)
        result = pipeline.generate(
            "a cat", width=32, height=32, steps=1, seed=7, max_sequence_length=8
        )
        assert result.image.size == (32, 32)
        result = pipeline.generate(
            "a cat",
            init_image=image_bytes(),
            mask_image=image_bytes(),
            width=32,
            height=32,
            steps=2,
            strength=1.0,
            max_sequence_length=8,
        )
        assert result.workflow == "inpainting" and result.strength == 1.0

    def test_native_zimage_masks_share_components_and_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retained native mask view shares components and normalizes output size."""
        pipeline, config, _ = checkpoint_wrapper(tmp_path, monkeypatch, "Z-Image", "turbo")
        constructor = {}

        class Native:
            def __init__(self, **kwargs: Any) -> None:
                """Capture identities passed to the native mask constructor."""
                constructor.update(kwargs)

            def __call__(self, **kwargs: Any) -> SimpleNamespace:
                """Check the native mask request's normalized dimensions."""
                assert kwargs["image"].size == kwargs["mask_image"].size == (64, 32)
                return SimpleNamespace(images=[Image.new("RGB", (64, 32))])

        # workflow_inputs needs the reviewed class signature; patch only the constructor hook.
        monkeypatch.setattr(
            pipeline,
            "_initialize_native_inpaint",
            lambda: setattr(
                pipeline,
                "inpaint_pipe",
                Native(
                    **{
                        name: pipeline.pipe.components[name]
                        for name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
                    }
                ),
            ),
        )
        pipeline.load(config)
        assert all(value is pipeline.pipe.components[name] for name, value in constructor.items())
        result = pipeline.generate(
            "a cat",
            init_image=image_bytes(),
            mask_image=image_bytes(),
            width=64,
            height=32,
            steps=2,
            strength=1.0,
        )
        assert result.image.size == (64, 32) and result.workflow == "inpainting"


def tiny_krea_transformer() -> torch.nn.Module:
    """Create meta-only conversion targets, with no pretrained assets."""
    with torch.device("meta"):
        model = torch.nn.Module()
        model.img_in = torch.nn.Linear(2, 2, bias=False)
        model.final_layer = torch.nn.Module()
        model.final_layer.linear = torch.nn.Linear(2, 2, bias=False)
        model.final_layer.norm = torch.nn.LayerNorm(2, bias=False)
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(2, 2, bias=False)
        model.transformer_blocks = torch.nn.ModuleList([block])
    model._keep_in_fp32_modules = ["norm"]
    return model


def krea_weights(dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Supply the smallest valid streamed checkpoint for the meta-only target."""
    return {
        "first.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
        "last.linear.weight": torch.ones((2, 2), dtype=dtype),
        "last.norm.scale": torch.ones(2),
        "blocks.0.attn.wq.weight": torch.ones((2, 2), dtype=dtype),
    }


class TestKreaConversion:
    """Retained header/key/shape/FP8/scale gates, moved to the extracted public API."""

    @pytest.mark.parametrize(
        "dtype,precision", [(torch.bfloat16, "bf16"), (torch.float8_e4m3fn, "fp8")]
    )
    def test_get_krea2_checkpoint_precision_reads_tensor_header(
        self, tmp_path: Path, dtype: torch.dtype, precision: str
    ) -> None:
        """Precision comes from real tensor headers, not checkpoint filenames."""
        path = tmp_path / "mislabeled.safetensors"
        save_file(krea_weights(dtype), path)
        assert get_krea2_checkpoint_precision(path) == precision

    @pytest.mark.parametrize(
        "mutation,error",
        [
            ("unsupported", "Unsupported Krea 2 checkpoint tensors"),
            ("missing", "does not contain Krea 2"),
            ("orphan_scale", "missing or orphaned scales"),
            ("scale_dtype", "scalar FP32"),
            ("descriptor_dtype", "descriptors must be U8"),
            ("descriptor_shape", "descriptor shape"),
            ("metadata", "Invalid Krea 2 quantization metadata"),
            ("format", "Unsupported Krea 2 quantization format"),
            ("bool", "must be a boolean"),
            ("missing_scale", "missing weight scales"),
            ("mismatch", "does not match checkpoint weights"),
        ],
    )
    def test_header_validation_failures(self, mutation: str, error: str) -> None:
        """Retain rejection of malformed precision, scale and quantization metadata."""
        header = {
            key: {"dtype": "F8_E4M3", "shape": [2, 2]}
            for key in ("first.weight", "last.linear.weight", "blocks.0.attn.wq.weight")
        }
        if mutation == "unsupported":
            header["first.weight"]["dtype"] = "I8"
        elif mutation == "missing":
            del header["first.weight"]
        elif mutation == "orphan_scale":
            header["other.weight_scale"] = {"dtype": "F32", "shape": []}
        elif mutation in {"scale_dtype", "missing_scale"}:
            for key in list(header):
                header[key.removesuffix(".weight") + ".weight_scale"] = {
                    "dtype": "BF16" if mutation == "scale_dtype" else "F32",
                    "shape": [],
                }
            if mutation == "missing_scale":
                for key in list(header):
                    if key.endswith("_scale"):
                        del header[key]
                header["__metadata__"] = {
                    "_quantization_metadata": json.dumps(
                        {
                            "layers": {
                                key.removesuffix(".weight"): {"format": "float8_e4m3fn"}
                                for key in header
                            }
                        }
                    )
                }
        elif mutation.startswith("descriptor"):
            for key in list(header):
                header[key.removesuffix(".weight") + ".comfy_quant"] = {
                    "dtype": "F32" if mutation == "descriptor_dtype" else "U8",
                    "shape": [5000] if mutation == "descriptor_shape" else [27],
                }
        else:
            config = {"format": "mxfp8" if mutation == "format" else "float8_e4m3fn"}
            if mutation == "bool":
                config["full_precision_matrix_mult"] = "false"
            header["__metadata__"] = {
                "_quantization_metadata": "bad"
                if mutation == "metadata"
                else json.dumps({"layers": {"first": config}})
            }
        with pytest.raises(ValueError, match=error):
            get_krea2_checkpoint_precision_from_header(header)

    def test_load_krea2_transformer_uses_policy_dtype_and_keeps_norms_fp32(
        self, tmp_path: Path
    ) -> None:
        """Stream CPU weights with policy dtype while retaining FP32 normalization."""
        path = tmp_path / "krea.safetensors"
        save_file(krea_weights(torch.float32), path)
        model = tiny_krea_transformer()
        policy = DevicePolicy(device="cpu", dtype=torch.bfloat16, offload=OffloadMode.NEVER)
        with (
            patch.object(BackportedKrea2Transformer2DModel, "load_config", return_value={}),
            patch.object(BackportedKrea2Transformer2DModel, "from_config", return_value=model),
        ):
            result, returned_policy = load_krea2_transformer(path, "repo", "transformer", policy)
        assert result is model and returned_policy is policy
        assert result.img_in.weight.dtype == torch.bfloat16
        assert result.final_layer.norm.weight.dtype == torch.float32
        assert result.img_in.weight.device.type == "cpu"

    @pytest.mark.parametrize("descriptors", [False, True])
    def test_load_krea2_transformer_runs_scaled_fp8_with_offloadable_storage(
        self,
        tmp_path: Path,
        descriptors: bool,
    ) -> None:
        """FP8 metadata and descriptors preserve scale, storage and full-precision matmul."""
        path = tmp_path / "fp8.safetensors"
        weights = krea_weights(torch.float8_e4m3fn)
        layers = {}
        for key in list(weights):
            if key.endswith(".weight"):
                layer = key.removesuffix(".weight")
                weights[layer + ".weight_scale"] = torch.tensor(0.5)
                layers[layer] = {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": layer == "last.linear",
                }
                if descriptors:
                    weights[layer + ".comfy_quant"] = torch.tensor(
                        list(json.dumps(layers[layer]).encode()), dtype=torch.uint8
                    )
        save_file(
            weights,
            path,
            metadata={}
            if descriptors
            else {"_quantization_metadata": json.dumps({"layers": layers})},
        )
        model = tiny_krea_transformer()
        policy = DevicePolicy(device="cpu", dtype=torch.bfloat16)
        with (
            patch.object(BackportedKrea2Transformer2DModel, "load_config", return_value={}),
            patch.object(BackportedKrea2Transformer2DModel, "from_config", return_value=model),
        ):
            result, returned_policy = load_krea2_transformer(path, "repo", "transformer", policy)
        assert result.img_in.weight.dtype == torch.float8_e4m3fn
        assert (
            result.img_in._oneiro_weight_scale
            is dict(result.img_in.named_buffers())["_oneiro_weight_scale"]
        )
        assert (
            returned_policy.group_offload_use_stream is False
            and policy.group_offload_use_stream is True
        )
        from comfy_kitchen.tensor import TensorCoreFP8Layout

        with patch.object(
            TensorCoreFP8Layout, "quantize", wraps=TensorCoreFP8Layout.quantize
        ) as quantize:
            output = result.img_in(torch.tensor([[[2.0, 1.0]]], dtype=torch.bfloat16))
            quantize.assert_called_once()
            quantize.reset_mock()
            result.final_layer.linear(torch.tensor([[[2.0, 1.0]]], dtype=torch.bfloat16))
            quantize.assert_not_called()
        assert torch.equal(output, torch.tensor([[[2.0, 5.0]]], dtype=torch.bfloat16))

    @pytest.mark.parametrize(
        "mutation,error",
        [
            ("shape", "Invalid shape"),
            ("unexpected", "Unexpected Krea 2 checkpoint tensor"),
            ("missing", "missing tensors"),
            ("scale", "Invalid Krea 2 FP8 weight scale"),
            ("descriptor", "Invalid Krea 2 quantization descriptor"),
            ("conflict", "Conflicting Krea 2 quantization metadata"),
        ],
    )
    def test_streamed_conversion_failures(self, tmp_path: Path, mutation: str, error: str) -> None:
        """Conversion still rejects corrupt shapes, keys, scales and descriptors."""
        weights = krea_weights(
            torch.float8_e4m3fn
            if mutation in {"scale", "descriptor", "conflict"}
            else torch.bfloat16
        )
        metadata = {}
        if mutation == "shape":
            weights["first.weight"] = torch.ones((3, 2))
        elif mutation == "unexpected":
            weights["alien"] = torch.ones(1)
        elif mutation == "missing":
            del weights["last.norm.scale"]
        elif mutation == "scale":
            for key in list(weights):
                if key.endswith(".weight"):
                    weights[key.removesuffix(".weight") + ".weight_scale"] = torch.tensor(
                        float("nan")
                    )
        else:
            layers = {}
            for key in list(weights):
                if key.endswith(".weight"):
                    layer = key.removesuffix(".weight")
                    layers[layer] = {"format": "float8_e4m3fn"}
                    weights[layer + ".weight_scale"] = torch.tensor(1.0)
                    descriptor = (
                        b"not json"
                        if mutation == "descriptor"
                        else b'{"format":"float8_e4m3fn","full_precision_matrix_mult":true}'
                    )
                    weights[layer + ".comfy_quant"] = torch.tensor(
                        list(descriptor), dtype=torch.uint8
                    )
            metadata = {"_quantization_metadata": json.dumps({"layers": layers})}
        path = tmp_path / "broken.safetensors"
        save_file(weights, path, metadata=metadata)
        with (
            patch.object(BackportedKrea2Transformer2DModel, "load_config", return_value={}),
            patch.object(
                BackportedKrea2Transformer2DModel,
                "from_config",
                return_value=tiny_krea_transformer(),
            ),
            pytest.raises(ValueError, match=error),
        ):
            load_krea2_transformer(
                path, "repo", "transformer", DevicePolicy(device="cpu", dtype=torch.float32)
            )

    def test_load_krea2_transformer_explains_gated_component_repo(self, tmp_path: Path) -> None:
        """Gated Krea sources give actionable license and token diagnostics."""
        with (
            patch.object(
                BackportedKrea2Transformer2DModel, "load_config", side_effect=OSError("gated")
            ),
            pytest.raises(RuntimeError, match="accept its license.*HF_TOKEN"),
        ):
            load_krea2_transformer(
                tmp_path / "unused",
                "repo",
                "transformer",
                DevicePolicy(device="cpu", dtype=torch.float32),
            )

    @pytest.mark.parametrize(
        "source,shape,target,expected",
        [
            ("first.weight", (4, 2), "img_in.weight", (4, 2)),
            ("model.diffusion_model.first.weight", (4, 2), "img_in.weight", (4, 2)),
            ("tmlp.0.weight", (4, 2), "time_embed.linear_1.weight", (4, 2)),
            ("txtmlp.0.scale", (4,), "txt_in.norm.weight", (4,)),
            ("blocks.0.attn.wq.weight", (4, 2), "transformer_blocks.0.attn.to_q.weight", (4, 2)),
            ("blocks.0.mod.lin", (24,), "transformer_blocks.0.scale_shift_table", (6, 4)),
            ("last.modulation.lin", (8,), "final_layer.scale_shift_table", (2, 4)),
            ("last.norm.scale", (4,), "final_layer.norm.weight", (4,)),
        ],
    )
    def test_krea2_checkpoint_keys_convert_to_diffusers(
        self, source: str, shape: tuple[int, ...], target: str, expected: tuple[int, ...]
    ) -> None:
        """Published Comfy keys preserve converted names and reshaped dimensions."""
        key, tensor = convert_krea2_checkpoint_tensor(source, torch.zeros(shape))
        assert key == target and tensor.shape == expected

    def test_transformer_checkpoint_strips_comfy_diffusion_model_prefix(
        self, tmp_path: Path
    ) -> None:
        """Transformer-only loaders accept the known Comfy checkpoint wrapper prefix."""
        path = tmp_path / "weights.safetensors"
        save_file({"model.diffusion_model.double_blocks.weight": torch.ones(1)}, path)
        result = CivitaiCheckpointPipeline()._load_transformer_checkpoint(path)
        assert set(result) == {"double_blocks.weight"}

    def test_all_krea2_checkpoint_keys_exist_in_diffusers_model(self) -> None:
        """Every published Comfy key still maps to the complete compatible 430-key model."""
        from accelerate import init_empty_weights

        source_keys = [
            "first.bias",
            "first.weight",
            "last.linear.bias",
            "last.linear.weight",
            "last.modulation.lin",
            "last.norm.scale",
            "tmlp.0.bias",
            "tmlp.0.weight",
            "tmlp.2.bias",
            "tmlp.2.weight",
            "tproj.1.bias",
            "tproj.1.weight",
            "txtmlp.0.scale",
            "txtmlp.1.bias",
            "txtmlp.1.weight",
            "txtmlp.3.bias",
            "txtmlp.3.weight",
            "txtfusion.projector.weight",
        ]
        suffixes = [
            "attn.qknorm.knorm.scale",
            "attn.qknorm.qnorm.scale",
            "mod.lin",
            "postnorm.scale",
            "prenorm.scale",
            "attn.gate.weight",
            "attn.wk.weight",
            "attn.wo.weight",
            "attn.wq.weight",
            "attn.wv.weight",
            "mlp.down.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
        ]
        source_keys.extend(f"blocks.{index}.{suffix}" for index in range(28) for suffix in suffixes)
        source_keys.extend(
            f"txtfusion.{group}.{index}.{suffix}"
            for group in ("layerwise_blocks", "refiner_blocks")
            for index in range(2)
            for suffix in suffixes
            if suffix != "mod.lin"
        )
        with init_empty_weights():
            model_keys = BackportedKrea2Transformer2DModel().state_dict().keys()
        converted_keys = set()
        for source in source_keys:
            key, _ = convert_krea2_checkpoint_tensor(
                source, torch.empty(6 if source.endswith(".mod.lin") else 2)
            )
            assert key in model_keys
            converted_keys.add(key)
        assert len(source_keys) == len(converted_keys) == len(model_keys) == 430
