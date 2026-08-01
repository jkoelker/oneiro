"""Tests for pipelines.base module."""

import io
from pathlib import Path
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
    """Tests for pipeline manager registration."""

    def test_registers_krea2_pipeline_type(self):
        """PipelineManager exposes the dedicated Krea 2 wrapper."""
        assert PipelineManager.PIPELINE_TYPES["krea2"] is Krea2PipelineWrapper

    def test_registers_flux2_klein_pipeline_type(self):
        """PipelineManager exposes the dedicated FLUX.2 Klein wrapper."""
        assert PipelineManager.PIPELINE_TYPES["flux2-klein"] is Flux2KleinPipelineWrapper


class TestPipelineManagerLoad:
    """Tests for model configuration flow during pipeline loading."""

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("oneiro.pipelines.krea2.load_krea2_tokenizer")
    @patch("diffusers.Krea2Pipeline", create=True)
    async def test_resolves_named_local_lora_before_krea_load(
        self, mock_krea2_pipeline, mock_tokenizer, mock_threads, mock_interop, tmp_path
    ):
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
        mock_krea2_pipeline.from_pretrained.return_value = MagicMock()

        await manager.load_model("krea2-turbo")

        assert manager.pipeline is not None
        assert manager.pipeline.active_loras == ["portrait"]

    async def test_does_not_pass_full_config_to_other_pipelines(self):
        """Shared Krea resource config does not activate sibling embedding loading."""
        model_config = {"type": "flux2", "repo": "example/flux2"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {"embeddings": {"example": {"source": "local", "path": "/tmp/x"}}}
        manager = PipelineManager(config)

        with patch.object(Flux2PipelineWrapper, "load") as mock_load:
            await manager.load_model("flux2")

        mock_load.assert_called_once_with(model_config)

    async def test_resolves_named_local_lora_for_any_lora_pipeline(self, tmp_path):
        """Named LoRA resolution is based on wrapper capability, not model type."""
        lora_path = tmp_path / "portrait.safetensors"
        lora_path.write_bytes(b"test")
        model_config = {"type": "qwen", "repo": "example/qwen", "loras": ["portrait"]}
        config = Mock()
        config.get.return_value = model_config
        config.data = {
            "models": {"qwen": model_config},
            "loras": {"portrait": {"source": "local", "path": str(lora_path)}},
        }
        manager = PipelineManager(config)

        def load_without_external_model(pipeline, config):
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
            "lora": "example/qwen-lightning",
            "lora_weights": "lightning.safetensors",
        }
        config = Mock()
        config.get.return_value = model_config
        config.data = {"models": {"qwen": model_config}}
        manager = PipelineManager(config)

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
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

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
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

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
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

        def load_without_external_model(pipeline, config):
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

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
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

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
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

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
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

        def load_without_external_model(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {"models": {"qwen": model_config}}
        manager = PipelineManager(config)
        partial_pipeline = None

        def fail_after_pipe_creation(pipeline, config):
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
        model_config = {"type": "qwen", "repo": "example/qwen"}
        config = Mock()
        config.get.return_value = model_config
        config.data = {"models": {"qwen": model_config}}
        manager = PipelineManager(config)

        def fail_after_pipe_creation(pipeline, config):
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
        manager.pipeline = Mock()
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
        """generate() skips LoRAs that fail resolution with warning."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = None

        lora = LoraConfig(name="bad-lora", source=LoraSource.LOCAL, path="/nonexistent.safetensors")

        with patch(
            "oneiro.pipelines.resolve_lora_path",
            new_callable=AsyncMock,
            side_effect=FileNotFoundError("Not found"),
        ):
            await manager.generate("test prompt", loras=[lora])

        call_kwargs = manager.pipeline.generate.call_args.kwargs
        assert "loras" not in call_kwargs or call_kwargs.get("loras") is None

    async def test_generate_resolves_multiple_loras(self):
        """generate() resolves multiple LoRAs, skipping failed ones."""
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
            await manager.generate("test prompt", loras=[good_lora, bad_lora])

        call_kwargs = manager.pipeline.generate.call_args.kwargs
        assert call_kwargs["loras"] == [good_lora]

    async def test_generate_skips_lora_resolution_when_no_loras(self):
        """generate() skips LoRA resolution when no LoRAs provided."""
        manager = self._create_manager_with_mocks()
        manager._civitai_client = Mock()

        with patch("oneiro.pipelines.resolve_lora_path", new_callable=AsyncMock) as mock_resolve:
            await manager.generate("test prompt")

        mock_resolve.assert_not_called()
