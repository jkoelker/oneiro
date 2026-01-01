"""Tests for pipelines.base module."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines import PipelineManager
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.lora import LoraConfig, LoraSource


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
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }


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
