"""Tests for pipelines.base module."""

import asyncio
import io
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode, OffloadType
from oneiro.pipelines import PipelineManager
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.flux2 import Flux2PipelineWrapper
from oneiro.pipelines.flux2_klein import Flux2KleinPipelineWrapper
from oneiro.pipelines.krea2 import Krea2PipelineWrapper
from oneiro.pipelines.lora import LoraConfig, LoraSource
from oneiro.pipelines.qwen import QwenPipelineWrapper


async def test_manager_uses_resolved_family_for_loras(tmp_path: Path) -> None:
    """CivitAI's resolved Pony family validates explicit local resources before unload."""
    from oneiro.pipelines.lora import LoraIncompatibleError

    checkpoint = tmp_path / "pony.safetensors"
    checkpoint.touch()
    lora = tmp_path / "flux.safetensors"
    lora.touch()
    model_config = {
        "type": "civitai",
        "checkpoint_path": str(checkpoint),
        "base_model": "Pony",
        "inline_loras": [
            {"name": "flux", "source": "local", "path": str(lora), "base_model": "Flux.1 D"}
        ],
    }
    config = Mock(data={"models": {"pony": model_config}})
    config.get.return_value = model_config
    manager = PipelineManager(config)
    previous = Mock()
    manager.pipeline, manager.current_model = previous, "old"
    with pytest.raises(LoraIncompatibleError):
        await manager.load_model("pony")
    previous.unload.assert_not_called()
    assert manager.pipeline is previous


@pytest.mark.parametrize("source", [LoraSource.LOCAL, LoraSource.HUGGINGFACE])
async def test_incompatible_request_lora_never_reaches_inference(source: LoraSource) -> None:
    """Explicit request resources fail on actual family metadata, for every local source."""
    from oneiro.pipelines.lora import LoraIncompatibleError

    manager = PipelineManager(Mock(data={}))
    manager.pipeline, manager.current_model = Mock(family="sdxl"), "pony"
    lora = LoraConfig(
        name="flux",
        source=source,
        path="unused",
        repo="unused",
        base_model="Flux.1 D",
    )
    with pytest.raises(LoraIncompatibleError):
        await manager.generate("a cat", loras=[lora])
    manager.pipeline.generate.assert_not_called()


async def assert_model_switch_ownership(cancel: bool) -> None:
    """Even repeated cancellation retains the lock until the worker is really done."""
    started, release = threading.Event(), threading.Event()
    old = Mock(family="sdxl")

    def generate(*args: Any, **kwargs: Any) -> Mock:
        """Hold the inference worker until the test releases it."""
        started.set()
        assert release.wait(5)
        return Mock()

    old.generate.side_effect = generate
    config = Mock(data={})
    config.get.return_value = {"type": "qwen"}
    manager = PipelineManager(config)
    manager.pipeline, manager.current_model = old, "old"
    task = asyncio.create_task(manager.generate("a cat"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        with patch.object(QwenPipelineWrapper, "load"):
            switch_started = asyncio.Event()

            async def switch_model() -> None:
                """Signal the attempted switch before it waits on the manager lock."""
                switch_started.set()
                await manager.load_model("new")

            switch = asyncio.create_task(switch_model())
            await switch_started.wait()
            old.unload.assert_not_called()
            assert not switch.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result.model_name == "old"
            await switch
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_model_switch_waits_for_inference() -> None:
    """A queued model switch cannot unload a running inference."""
    await assert_model_switch_ownership(False)


async def test_cancelled_inference_retains_model_ownership() -> None:
    """Repeated caller cancellation does not orphan the inference thread."""
    await assert_model_switch_ownership(True)


async def test_queued_request_revalidates_new_family() -> None:
    """A strength request queued under SDXL must fail after switching to conditioned FLUX.2."""
    from diffusers import Flux2AutoBlocks

    started, release = threading.Event(), threading.Event()
    old = Mock(family="sdxl")

    def hold_generation(*args: Any, **kwargs: Any) -> Mock:
        """Keep the old model owned while the switch and request queue."""
        started.set()
        assert release.wait(5)
        return Mock()

    old.generate.side_effect = hold_generation
    config = Mock(data={})
    config.get.return_value = {"type": "flux2"}
    manager = PipelineManager(config)
    manager.pipeline, manager.current_model = old, "old"
    manager.validate_request(has_image=True, strength=0.75)

    def load(
        pipeline: Flux2PipelineWrapper, model_config: dict[str, Any], full_config: dict[str, Any]
    ) -> None:
        """Install real conditioned workflow declarations without external components."""
        pipeline.blocks = Flux2AutoBlocks()
        pipeline.pipe = MagicMock()

    first = asyncio.create_task(manager.generate("first"))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        with patch.object(Flux2PipelineWrapper, "load", autospec=True, side_effect=load):
            switch = asyncio.create_task(manager.load_model("new"))
            await asyncio.sleep(0)
            queued = asyncio.create_task(
                manager.generate("queued", init_image=b"unused", strength=0.75)
            )
            release.set()
            await first
            await switch
            with pytest.raises(ValueError, match="Denoising strength"):
                await queued
        assert manager.family == "flux2"
        assert old.generate.call_count == 1
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)


async def test_preflight_failure_keeps_old_model() -> None:
    """Invalid CivitAI metadata never unloads the usable previous model."""
    manager = PipelineManager(Mock(data={}))
    manager.config.get.return_value = {"type": "civitai", "civitai_model_id": 1}
    client = AsyncMock()
    client.get_model.return_value = Mock(latest_version=Mock(base_model="SD 1.5"))
    manager.set_civitai_client(client)
    old = Mock(family="sdxl")
    old.generate.return_value = Mock()
    manager.pipeline, manager.current_model = old, "old"
    with pytest.raises(ValueError, match="base model"):
        await manager.load_model("bad")
    old.unload.assert_not_called()
    client.download_model_version.assert_not_awaited()
    result = await manager.generate("still works")
    assert result.model_name == "old" and manager.pipeline is old


@pytest.mark.parametrize(
    "pipeline_type,repo,variant",
    [
        ("flux1", "black-forest-labs/FLUX.1-dev", "schnell"),
        ("flux2", "black-forest-labs/FLUX.2-dev", "schnell"),
        ("flux2-klein", "black-forest-labs/FLUX.2-klein-9B", "base"),
        ("krea2", "krea/Krea-2-Turbo", "raw"),
        ("qwen", "Qwen/Qwen-Image", "edit"),
        ("zimage", "Tongyi-MAI/Z-Image-Turbo", "base"),
    ],
)
@pytest.mark.parametrize(
    "error", ["custom-variant", "official-contradiction", "placement", "group-type", "group-count"]
)
async def test_hosted_recipe_preflight_preserves_current_model(
    pipeline_type: str, repo: str, variant: str, error: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All hosted loaders must share their metadata-only checks with manager preflight."""
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

    monkeypatch.setattr(BasePipeline, "_configure_cpu_threads", lambda *args: 1)

    profile = {"type": pipeline_type, "repo": repo}
    if error == "custom-variant":
        profile["repo"] = "custom/model"
    elif error == "official-contradiction":
        profile["variant"] = variant
    elif error == "placement":
        profile["offload_type"] = "not-an-offload-mode"
    elif error == "group-type":
        profile["group_offload_type"] = "invalid"
    else:
        profile["group_offload_num_blocks_per_group"] = 0
    config = Mock(data={})
    config.get.return_value = profile
    manager = PipelineManager(config)
    old = Mock(family="sdxl")
    old.generate.return_value = Mock()
    manager.pipeline, manager.current_model = old, "old"
    with (
        patch.object(ModularPipeline, "_load_pipeline_config") as assets,
        patch.object(ComponentSpec, "load") as component,
        pytest.raises(ValueError),
    ):
        await manager.load_model("bad")
    assets.assert_not_called()
    component.assert_not_called()
    old.unload.assert_not_called()
    assert manager.pipeline is old and manager.current_model == "old"
    assert (await manager.generate("still usable")).model_name == "old"


async def test_checkpoint_placement_preflight_preserves_current_model() -> None:
    """Checkpoint recipe resolution must also reject placement before assets or unload."""
    from oneiro.pipelines.civitai_checkpoint import CivitaiCheckpointPipeline

    config = Mock(data={})
    config.get.return_value = {
        "type": "civitai",
        "checkpoint_path": "unused",
        "base_model": "Pony",
        "offload_type": "invalid",
    }
    manager = PipelineManager(config)
    old = Mock(family="sdxl")
    manager.pipeline, manager.current_model = old, "old"
    with patch.object(CivitaiCheckpointPipeline, "load") as assets:
        with pytest.raises(ValueError):
            await manager.load_model("bad")
    assets.assert_not_called()
    old.unload.assert_not_called()
    assert manager.pipeline is old and manager.current_model == "old"


@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_load_retains_model_ownership(fail: bool) -> None:
    """Cancellation drains native loading and cleanup before allowing another owner."""
    started, release, unloaded = threading.Event(), threading.Event(), threading.Event()
    manager = PipelineManager(Mock(data={}))
    manager.config.get.return_value = {"type": "qwen"}
    partial = []

    def load(
        pipeline: QwenPipelineWrapper, model_config: dict[str, Any], full_config: dict[str, Any]
    ) -> None:
        """Hold a partially initialized wrapper until cancellation is observed."""
        partial.append(pipeline)
        pipeline.pipe = MagicMock()
        started.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError("native load failed")

    def unload(pipeline: QwenPipelineWrapper) -> None:
        """Record cleanup only after the loader thread finishes."""
        assert release.is_set()
        pipeline.pipe = None
        unloaded.set()

    with (
        patch.object(QwenPipelineWrapper, "load", autospec=True, side_effect=load),
        patch.object(QwenPipelineWrapper, "unload", autospec=True, side_effect=unload),
    ):
        task = asyncio.create_task(manager.load_model("qwen"))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert manager._lock.locked() and not task.done() and not unloaded.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert manager._lock.locked() is False
            assert unloaded.is_set() is fail
            assert (manager.pipeline is None) is fail
            if not fail:
                assert manager.current_model == "qwen"
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_lazy_load_does_not_reacquire_lock() -> None:
    """Generation initializes the default via the private owned loader, without deadlock."""
    manager = PipelineManager(Mock(data={}))
    manager.config.get.side_effect = ["qwen", {"type": "qwen"}]
    with (
        patch.object(QwenPipelineWrapper, "load"),
        patch.object(QwenPipelineWrapper, "validate_request", return_value="text2image"),
        patch.object(QwenPipelineWrapper, "generate", return_value=Mock()),
    ):
        result = await asyncio.wait_for(manager.generate("a cat"), timeout=5)
    assert result.model_name == "qwen"


@pytest.mark.parametrize("resource", ["loras", "embeddings"])
async def test_missing_explicit_resource_preserves_old_model(resource: str) -> None:
    """Required named references cannot silently disappear during preflight."""
    manager = PipelineManager(Mock(data={}))
    manager.config.get.return_value = {
        "type": "civitai",
        "checkpoint_path": "unused",
        "base_model": "Pony",
        resource: ["missing"],
    }
    old = Mock(family="sdxl")
    manager.pipeline, manager.current_model = old, "old"
    with pytest.raises(ValueError, match="not found"):
        await manager.load_model("bad")
    old.unload.assert_not_called()
    assert manager.pipeline is old


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_creation(self):
        """GenerationResult can be created with all fields."""
        img = Image.new("RGB", (64, 64), color="red")
        result = GenerationResult(
            image=img,
            seed=12345,
            prompt="a cat",
            negative_prompt="blurry",
            width=64,
            height=64,
            steps=20,
            guidance_scale=7.5,
        )
        assert result.image is img
        assert result.seed == 12345
        assert result.prompt == "a cat"
        assert result.negative_prompt == "blurry"
        assert result.width == 64
        assert result.height == 64
        assert result.steps == 20
        assert result.guidance_scale == 7.5
        assert result.workflow == "text2image"
        assert result.strength is None
        assert result.model_name is None

    def test_negative_prompt_optional(self):
        """GenerationResult accepts None for negative_prompt."""
        img = Image.new("RGB", (64, 64))
        result = GenerationResult(
            image=img,
            seed=0,
            prompt="test",
            negative_prompt=None,
            width=64,
            height=64,
            steps=1,
            guidance_scale=0.0,
        )
        assert result.negative_prompt is None


class ConcretePipeline(BasePipeline):
    """Concrete implementation for testing abstract base class."""

    supports_inpaint = True

    def load(self, model_config):
        pass

    def build_generation_kwargs(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        guidance_scale,
        generator,
        init_image,
        strength,
        **kwargs,
    ):
        """Build generation kwargs for testing."""
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if "mask_image" in kwargs:
            gen_kwargs["mask_image"] = kwargs["mask_image"]
        return gen_kwargs


class TestBasePipelineInit:
    """Tests for BasePipeline initialization."""

    def test_pipe_starts_none(self):
        """Pipeline.pipe is None initially."""
        pipeline = ConcretePipeline()
        assert pipeline.pipe is None

    def test_device_cuda_when_available(self):
        """Device is 'cuda' when CUDA is available."""
        mock_policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ConcretePipeline()
        assert pipeline.policy.device == "cuda"

    def test_device_cpu_when_no_cuda(self):
        """Device is 'cpu' when CUDA is not available."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ConcretePipeline()
        assert pipeline.policy.device == "cpu"


class TestPipelineManagerRegistry:
    """Registered names load the intended family and expose its image workflow."""

    @pytest.mark.parametrize(
        "family,wrapper,workflow",
        [
            ("krea2", Krea2PipelineWrapper, "image2image"),
            ("flux2-klein", Flux2KleinPipelineWrapper, "image_conditioned"),
        ],
    )
    async def test_loads_registered_family(
        self, family: str, wrapper: type[BasePipeline], workflow: str
    ) -> None:
        config = Mock(data={})
        config.get.return_value = {"type": family}
        manager = PipelineManager(config)
        with patch.object(wrapper, "load"):
            await manager.load_model("selected")
        assert manager.current_model == "selected"
        assert manager.family == family
        assert manager.validate_request(has_image=True) == workflow


class TestPipelineManagerLoad:
    """Tests for model configuration flow during pipeline loading."""

    @patch.object(Krea2PipelineWrapper, "load", autospec=True)
    async def test_resolves_named_local_lora_before_krea_load(
        self, mock_load: Mock, tmp_path: Path
    ) -> None:
        """Krea loading resolves local named LoRAs before synchronous model setup."""
        lora_path = tmp_path / "portrait.safetensors"
        lora_path.write_bytes(b"test")
        model_config = {
            "type": "krea2",
            "repo": "krea/Krea-2-Turbo",
            "cpu_offload": False,
            "loras": ["portrait"],
        }
        full_config = {
            "models": {"krea2-turbo": model_config},
            "loras": {"portrait": {"source": "local", "path": str(lora_path)}},
        }
        config = Mock()
        config.get.return_value = model_config
        config.data = full_config
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free component container without external loading."""
            pipeline.pipe = MagicMock()

        mock_load.side_effect = load_without_external_model

        await manager.load_model("krea2-turbo")

        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == ["portrait"]

    async def test_passes_full_config_without_activating_unrequested_embeddings(self) -> None:
        """Only selected resources activate; all wrappers receive the full config contract."""
        model_config = {"type": "flux2", "repo": "example/flux2", "variant": "dev"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {"embeddings": {"example": {"source": "local", "path": "/tmp/x"}}}
        manager = PipelineManager(config)

        with patch.object(Flux2PipelineWrapper, "load") as mock_load:
            await manager.load_model("flux2")

        mock_load.assert_called_once_with(model_config, config.data)

    async def test_resolves_named_local_lora_for_any_lora_pipeline(self, tmp_path):
        """Named LoRA resolution is based on wrapper capability, not model type."""
        lora_path = tmp_path / "portrait.safetensors"
        lora_path.write_bytes(b"test")
        model_config = {
            "type": "qwen",
            "repo": "example/qwen",
            "variant": "image",
            "loras": ["portrait"],
        }
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {"portrait": {"source": "local", "path": str(lora_path)}},
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free component container without external loading."""
            pipeline.pipe = MagicMock()

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == ["portrait"]

    async def test_loads_legacy_huggingface_lora_through_shared_path(self):
        """Legacy model LoRA fields remain supported by centralized loading."""
        model_config = {
            "type": "qwen",
            "repo": "example/qwen",
            "variant": "image",
            "lora": "example/qwen-lightning",
            "lora_weights": "lightning.safetensors",
        }
        config = Mock()
        config.get.return_value = model_config
        config.data = {"models": {"qwen": model_config}}
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free component container without external loading."""
            pipeline.pipe = MagicMock()

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == ["legacy_lora"]
        manager.pipeline.pipe.load_lora_weights.assert_called_once_with(
            "example/qwen-lightning",
            weight_name="lightning.safetensors",
            adapter_name="legacy_lora",
        )

    async def test_failed_auto_load_lora_does_not_block_model(self, capsys):
        """A broken global auto-load LoRA is skipped without blocking the model."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["missing"],
                "missing": {"source": "local", "path": "/missing/lora.safetensors"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free component container without external loading."""
            pipeline.pipe = MagicMock()

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.current_model == "qwen"
        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == []
        assert "Warning: Failed to resolve auto-load LoRA missing" in capsys.readouterr().out

    async def test_malformed_auto_load_lora_does_not_block_model(self, capsys):
        """A malformed global auto-load LoRA is skipped without blocking the model."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["malformed"],
                "malformed": {"source": "unsupported"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free component container without external loading."""
            pipeline.pipe = MagicMock()

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.current_model == "qwen"
        assert manager.pipeline is not None
        assert "Warning: Failed to parse auto-load LoRA malformed" in capsys.readouterr().out

    async def test_auto_load_adapter_failure_does_not_block_model(self, capsys):
        """A global adapter load failure is skipped without blocking the model."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["broken"],
                "broken": {"source": "huggingface", "repo": "missing/lora"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free component container with a failing adapter loader."""
            pipeline.pipe = MagicMock()
            pipeline.pipe.load_lora_weights.side_effect = RuntimeError("adapter load failed")

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.current_model == "qwen"
        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == []
        assert "Warning: Failed to load auto-load LoRA broken" in capsys.readouterr().out

    async def test_failed_auto_duplicate_falls_back_to_explicit_lora(self):
        """A failed auto adapter does not suppress the model's explicit reference."""
        model_config = {
            "type": "qwen",
            "repo": "example/qwen",
            "variant": "image",
            "loras": ["shared"],
        }
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["shared"],
                "shared": {"source": "huggingface", "repo": "example/shared"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free container with an optional first adapter failure."""
            pipeline.pipe = MagicMock()
            pipeline.pipe.load_lora_weights.side_effect = [RuntimeError("auto failed"), None]

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.pipeline is not None
        assert manager.pipeline.pipe.load_lora_weights.call_count == 2
        assert manager.pipeline.active_loras == ["shared"]

    async def test_failed_auto_adapter_rolls_back_partial_external_state(self):
        """A failed auto adapter removes weights mutated before the loader raised."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["broken"],
                "broken": {"source": "huggingface", "repo": "example/broken"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Model partial external adapter state without loading real model weights."""
            pipeline.pipe = MagicMock()
            pipeline.pipe.adapter_resident = False

            def fail_after_mutation(*args, **kwargs):
                pipeline.pipe.adapter_resident = True
                raise RuntimeError("adapter load failed")

            def clear_external_adapters():
                pipeline.pipe.adapter_resident = False

            pipeline.pipe.load_lora_weights.side_effect = fail_after_mutation
            pipeline.pipe.unload_lora_weights.side_effect = clear_external_adapters

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.pipeline is not None
        assert manager.pipeline.pipe.adapter_resident is False
        manager.pipeline.pipe.unload_lora_weights.assert_called_once()

    async def test_auto_adapter_activation_failure_does_not_block_model(self, capsys):
        """A global adapter activation failure is skipped without blocking the model."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["broken"],
                "broken": {"source": "huggingface", "repo": "example/broken"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free container with a failing adapter activation."""
            pipeline.pipe = MagicMock()
            pipeline.pipe.set_adapters.side_effect = RuntimeError("activation failed")

        with patch.object(
            QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
        ):
            await manager.load_model("qwen")

        assert manager.current_model == "qwen"
        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == []
        assert "Warning: Failed to activate auto-load LoRAs" in capsys.readouterr().out

    async def test_failed_auto_rollback_aborts_contaminated_pipeline(self):
        """An adapter that cannot be rolled back prevents the pipeline from becoming active."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {
                "auto_load": ["broken"],
                "broken": {"source": "huggingface", "repo": "example/broken"},
            },
        }
        manager = PipelineManager(config)

        def load_without_external_model(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Install a model-free container with failing adapter rollback."""
            pipeline.pipe = MagicMock()
            pipeline.pipe.load_lora_weights.side_effect = RuntimeError("adapter load failed")
            pipeline.pipe.unload_lora_weights.side_effect = RuntimeError("rollback failed")

        with (
            patch.object(
                QwenPipelineWrapper, "load", autospec=True, side_effect=load_without_external_model
            ),
            pytest.raises(RuntimeError, match="rollback failed"),
        ):
            await manager.load_model("qwen")

        assert manager.pipeline is None
        assert manager.current_model is None

    async def test_failed_validation_preserves_current_model(self):
        """Invalid target resources do not unload the current model."""
        model_config = {"type": "krea2", "loras": ["missing"]}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"krea2": model_config},
            "loras": {"missing": {"source": "local", "path": "/missing/lora.safetensors"}},
        }
        manager = PipelineManager(config)
        previous_pipeline = Mock()
        manager.pipeline = previous_pipeline
        manager.current_model = "previous"

        with pytest.raises(FileNotFoundError, match="not found"):
            await manager.load_model("krea2")

        previous_pipeline.unload.assert_not_called()
        assert manager.pipeline is previous_pipeline
        assert manager.current_model == "previous"

    async def test_failed_load_unloads_partial_pipeline(self):
        """A failed load releases a pipeline created before the error."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {"models": {"qwen": model_config}}
        manager = PipelineManager(config)
        partial_pipeline = None

        def fail_after_pipe_creation(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Leave a partial component container for manager cleanup."""
            nonlocal partial_pipeline
            pipeline.pipe = MagicMock()
            partial_pipeline = pipeline
            raise RuntimeError("load failed")

        with (
            patch.object(
                QwenPipelineWrapper,
                "load",
                autospec=True,
                side_effect=fail_after_pipe_creation,
            ),
            pytest.raises(RuntimeError, match="load failed"),
        ):
            await manager.load_model("qwen")

        assert partial_pipeline is not None
        assert partial_pipeline.pipe is None

    async def test_cleanup_error_does_not_mask_load_error(self, capsys):
        """Cleanup failures preserve the original model-load exception."""
        model_config = {"type": "qwen", "repo": "example/qwen", "variant": "image"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {"models": {"qwen": model_config}}
        manager = PipelineManager(config)

        def fail_after_pipe_creation(
            pipeline: BasePipeline, config: dict[str, Any], full_config: dict[str, Any]
        ) -> None:
            """Fail loading after component creation, before a failing cleanup."""
            pipeline.pipe = MagicMock()
            raise RuntimeError("load failed")

        with (
            patch.object(
                QwenPipelineWrapper,
                "load",
                autospec=True,
                side_effect=fail_after_pipe_creation,
            ),
            patch.object(QwenPipelineWrapper, "unload", side_effect=RuntimeError("cleanup failed")),
            pytest.raises(RuntimeError, match="load failed"),
        ):
            await manager.load_model("qwen")

        assert manager.pipeline is None
        assert manager.current_model is None
        assert "Warning: Failed to unload partial pipeline" in capsys.readouterr().out


class TestBasePipelineUnload:
    """Tests for BasePipeline.unload()."""

    def test_unload_clears_pipe(self):
        """Unload sets pipe to None."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ConcretePipeline()
            pipeline.pipe = Mock()
            pipeline.unload()
            assert pipeline.pipe is None

    def test_unload_handles_none_pipe(self):
        """Unload handles pipe being None."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ConcretePipeline()
            pipeline.pipe = None
            # Should not raise
            pipeline.unload()
            assert pipeline.pipe is None

    def test_unload_calls_clear_cache(self):
        """Unload calls DevicePolicy.clear_cache()."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ConcretePipeline()
            pipeline.pipe = Mock()
            with patch.object(DevicePolicy, "clear_cache") as mock_clear:
                pipeline.unload()
                mock_clear.assert_called_once()


class TestBasePipelinePrepareSeed:
    """Tests for BasePipeline._prepare_seed()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_prepare_seed_uses_provided(self, mock_cuda):
        """_prepare_seed uses provided seed when >= 0."""
        pipeline = ConcretePipeline()
        seed, generator = pipeline._prepare_seed(42)
        assert seed == 42

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_prepare_seed_generates_random(self, mock_cuda):
        """_prepare_seed generates random seed when < 0."""
        pipeline = ConcretePipeline()
        seed, generator = pipeline._prepare_seed(-1)
        assert 0 <= seed < 2**32

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_prepare_seed_returns_generator(self, mock_cuda):
        """_prepare_seed returns a torch Generator."""
        import torch

        pipeline = ConcretePipeline()
        seed, generator = pipeline._prepare_seed(42)
        assert isinstance(generator, torch.Generator)


class TestBasePipelineLoadInitImage:
    """Tests for BasePipeline._load_init_image()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_load_init_image_none(self, mock_cuda):
        """_load_init_image returns None for None input."""
        pipeline = ConcretePipeline()
        result = pipeline._load_init_image(None)
        assert result is None

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_load_init_image_from_bytes(self, mock_cuda):
        """_load_init_image loads image from bytes."""
        pipeline = ConcretePipeline()
        # Create a simple PNG in bytes
        img = Image.new("RGB", (32, 32), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = pipeline._load_init_image(img_bytes)
        assert isinstance(result, Image.Image)
        assert result.size == (32, 32)
        assert result.mode == "RGB"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_load_init_image_converts_to_rgb(self, mock_cuda):
        """_load_init_image converts RGBA to RGB."""
        pipeline = ConcretePipeline()
        # Create RGBA image
        img = Image.new("RGBA", (32, 32), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = pipeline._load_init_image(img_bytes)
        assert result.mode == "RGB"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_decodes_mask_image_from_bytes(self, mock_cuda):
        """generate decodes mask_image bytes before building generation kwargs."""
        pipeline = ConcretePipeline()
        pipeline.pipe = Mock()
        output = Image.new("RGB", (32, 32))
        pipeline.pipe.return_value.images = [output]

        mask = Image.new("L", (16, 16), color=255)
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")

        pipeline.generate("test", mask_image=buffer.getvalue())

        call_kwargs = pipeline.pipe.call_args.kwargs
        assert isinstance(call_kwargs["mask_image"], Image.Image)
        assert call_kwargs["mask_image"].size == (16, 16)
        assert call_kwargs["mask_image"].mode == "RGB"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_rejects_mask_when_pipeline_does_not_support_inpaint(self, mock_cuda):
        """generate rejects mask_image for pipelines without inpaint support."""

        class NoInpaintPipeline(ConcretePipeline):
            supports_inpaint = False

        pipeline = NoInpaintPipeline()
        pipeline.pipe = Mock()

        with pytest.raises(ValueError, match="does not support inpainting masks"):
            pipeline.generate("test", mask_image=b"not-used")

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_load_init_image_rejects_invalid_image_bytes(self, mock_cuda):
        """_load_init_image reports invalid bytes as a user-facing ValueError."""
        pipeline = ConcretePipeline()

        with pytest.raises(ValueError, match="Invalid image attachment"):
            pipeline._load_init_image(b"not an image")

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_load_init_image_rejects_oversized_images(self, mock_cuda):
        """_load_init_image rejects decoded images above the pixel limit."""
        pipeline = ConcretePipeline()
        image = Image.new("RGB", (11, 10), color="blue")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        with patch("oneiro.pipelines.base.MAX_INPUT_IMAGE_PIXELS", 100):
            with pytest.raises(ValueError, match="Input image is too large"):
                pipeline._load_init_image(buffer.getvalue())

    def test_load_init_image_rejects_oversized_attachment(self) -> None:
        """Execution repeats the attachment byte limit before decoding."""
        pipeline = ConcretePipeline()
        with patch("oneiro.pipelines.base.MAX_INPUT_IMAGE_BYTES", 3):
            with pytest.raises(ValueError, match="25 MiB"):
                pipeline._load_init_image(b"1234")

    def test_load_init_image_rejects_unapproved_format(self) -> None:
        """Pillow support alone does not make GIF an accepted attachment."""
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="GIF")
        with pytest.raises(ValueError, match="PNG, JPEG, or WebP"):
            ConcretePipeline()._load_init_image(buffer.getvalue())


class TestBasePipelineLifecycle:
    """Validation precedes setup and setup failures still reach cleanup."""

    def test_invalid_attachment_precedes_pre_generate(self) -> None:
        pipeline = ConcretePipeline()
        pipeline.pipe = Mock()
        pipeline.pre_generate = Mock()
        with pytest.raises(ValueError, match="Invalid image"):
            pipeline.generate("test", init_image=b"bad")
        pipeline.pre_generate.assert_not_called()

    def test_pre_generate_failure_runs_post_generate(self) -> None:
        pipeline = ConcretePipeline()
        pipeline.pipe = Mock()
        pipeline.pre_generate = Mock(side_effect=RuntimeError("setup failed"))
        pipeline.post_generate = Mock()
        with pytest.raises(RuntimeError, match="setup failed"):
            pipeline.generate("test")
        pipeline.post_generate.assert_called_once()

    def test_request_controls_do_not_reach_generation_kwargs(self) -> None:
        pipeline = ConcretePipeline()
        pipeline.pipe = Mock()
        pipeline.pipe.return_value.images = [Image.new("RGB", (8, 8))]
        pipeline.pre_generate = Mock()
        original = pipeline.build_generation_kwargs
        pipeline.build_generation_kwargs = Mock(wraps=original)
        pipeline.generate("test", loras=["adapter"], scheduler="default")
        pipeline.pre_generate.assert_called_once_with(loras=["adapter"], scheduler="default")
        assert "loras" not in pipeline.build_generation_kwargs.call_args.kwargs
        assert "scheduler" not in pipeline.build_generation_kwargs.call_args.kwargs


class TestBasePipelineConfigureCpuThreads:
    """Tests for BasePipeline._configure_cpu_threads()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("os.cpu_count", return_value=8)
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    def test_configure_default_utilization(self, mock_interop, mock_threads, mock_cpu, mock_cuda):
        """_configure_cpu_threads uses 75% by default."""
        pipeline = ConcretePipeline()
        result = pipeline._configure_cpu_threads()
        assert result == 6  # 75% of 8
        mock_threads.assert_called_with(6)
        mock_interop.assert_called_with(3)  # half of 6

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("os.cpu_count", return_value=8)
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    def test_configure_custom_utilization(self, mock_interop, mock_threads, mock_cpu, mock_cuda):
        """_configure_cpu_threads accepts custom utilization."""
        pipeline = ConcretePipeline()
        result = pipeline._configure_cpu_threads(utilization=0.5)
        assert result == 4  # 50% of 8
        mock_threads.assert_called_with(4)

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("os.cpu_count", return_value=None)
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    def test_configure_handles_none_cpu_count(
        self, mock_interop, mock_threads, mock_cpu, mock_cuda
    ):
        """_configure_cpu_threads handles cpu_count returning None."""
        pipeline = ConcretePipeline()
        result = pipeline._configure_cpu_threads()
        assert result >= 1  # Should at least be 1


class TestBasePipelinePostGenerate:
    """Tests for BasePipeline.post_generate() and _reset_model_state()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_post_generate_calls_reset_model_state(self, mock_cuda):
        """post_generate() calls _reset_model_state()."""
        pipeline = ConcretePipeline()
        pipeline._reset_model_state = Mock()
        pipeline.post_generate()
        pipeline._reset_model_state.assert_called_once()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_reset_model_state_calls_maybe_free_model_hooks(self, mock_cuda):
        """_reset_model_state() calls pipe.maybe_free_model_hooks()."""
        pipeline = ConcretePipeline()
        mock_pipe = Mock()
        pipeline.pipe = mock_pipe
        pipeline._reset_model_state()
        mock_pipe.maybe_free_model_hooks.assert_called_once()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_reset_model_state_skips_group_offload_pipelines(self, mock_cuda):
        """_reset_model_state() skips model hook reset for group offload."""
        pipeline = ConcretePipeline()
        mock_pipe = Mock()
        mock_pipe._oneiro_offload_type = OffloadType.GROUP.value
        pipeline.pipe = mock_pipe
        pipeline._reset_model_state()
        mock_pipe.maybe_free_model_hooks.assert_not_called()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_reset_model_state_handles_none_pipe(self, mock_cuda):
        """_reset_model_state() handles pipe being None."""
        pipeline = ConcretePipeline()
        pipeline.pipe = None
        # Should not raise
        pipeline._reset_model_state()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_post_generate_accepts_kwargs(self, mock_cuda):
        """post_generate() accepts arbitrary kwargs."""
        pipeline = ConcretePipeline()
        pipeline._reset_model_state = Mock()
        # Should not raise
        pipeline.post_generate(some_kwarg="value", another=123)
        pipeline._reset_model_state.assert_called_once()


class TestPipelineManagerLoraResolution:
    """Tests for PipelineManager.generate() LoRA path resolution."""

    def _create_manager_with_mocks(self):
        """Create a PipelineManager with mocked config and pipeline."""
        mock_config = Mock()
        mock_config.get = Mock(return_value={})
        manager = PipelineManager(mock_config)
        manager.pipeline = Mock(family="sdxl")
        manager.pipeline.generate = Mock(return_value=Mock())
        return manager

    async def test_generate_resolves_lora_paths(self):
        """generate() resolves LoRA paths before passing to pipeline."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = Mock()

        lora = LoraConfig(name="test-lora", source=LoraSource.LOCAL, path="/fake.safetensors")

        with patch("oneiro.pipelines.resolve_lora_path", new_callable=AsyncMock) as mock_resolve:
            await manager.generate("test prompt", loras=[lora])

        mock_resolve.assert_called_once()
        call_args = mock_resolve.call_args
        assert call_args.args[0] is lora

    async def test_generate_passes_resolved_loras_to_pipeline(self):
        """generate() passes resolved LoRAs to the underlying pipeline."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = Mock()

        lora = LoraConfig(name="test-lora", source=LoraSource.LOCAL, path="/fake.safetensors")

        with patch("oneiro.pipelines.resolve_lora_path", new_callable=AsyncMock):
            await manager.generate("test prompt", loras=[lora])

        call_kwargs = manager.pipeline.generate.call_args.kwargs
        assert "loras" in call_kwargs
        assert call_kwargs["loras"] == [lora]

    async def test_generate_resolves_loras_without_civitai_client(self):
        """generate() resolves local/HF LoRAs even without civitai_client."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = None

        lora = LoraConfig(name="local-lora", source=LoraSource.LOCAL, path="/local.safetensors")

        with patch("oneiro.pipelines.resolve_lora_path", new_callable=AsyncMock) as mock_resolve:
            await manager.generate("test prompt", loras=[lora])

        mock_resolve.assert_called_once()

    async def test_generate_handles_lora_resolution_failure(self):
        """Explicit resource failures abort generation, rather than silently skipping."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = None

        lora = LoraConfig(name="bad-lora", source=LoraSource.LOCAL, path="/nonexistent.safetensors")

        with patch(
            "oneiro.pipelines.resolve_lora_path",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("Not found"),
        ):
            with pytest.raises(FileNotFoundError, match="Not found"):
                await manager.generate("test prompt", loras=[lora])

        manager.pipeline.generate.assert_not_called()

    async def test_generate_resolves_multiple_loras(self):
        """Every explicitly requested LoRA must resolve before inference."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = None

        good_lora = LoraConfig(name="good", source=LoraSource.LOCAL, path="/good.safetensors")
        bad_lora = LoraConfig(name="bad", source=LoraSource.LOCAL, path="/bad.safetensors")

        async def resolve_side_effect(lora, **kwargs):
            if lora.name == "bad":
                raise FileNotFoundError("Not found")
            return Path("/good.safetensors")

        with patch(
            "oneiro.pipelines.resolve_lora_path",
            new_callable=AsyncMock,
            side_effect=resolve_side_effect,
        ):
            with pytest.raises(FileNotFoundError, match="Not found"):
                await manager.generate("test prompt", loras=[good_lora, bad_lora])

        manager.pipeline.generate.assert_not_called()

    async def test_generate_skips_lora_resolution_when_no_loras(self):
        """generate() skips LoRA resolution when no LoRAs provided."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = Mock()

        with patch("oneiro.pipelines.resolve_lora_path", new_callable=AsyncMock) as mock_resolve:
            await manager.generate("test prompt")

        mock_resolve.assert_not_called()
