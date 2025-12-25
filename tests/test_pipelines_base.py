"""Tests for pipelines.base module."""

import io
from unittest.mock import Mock, patch

from PIL import Image

from oneiro.pipelines.base import BasePipeline, GenerationResult


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

    def generate(
        self,
        prompt,
        negative_prompt=None,
        width=1024,
        height=1024,
        seed=-1,
        steps=9,
        guidance_scale=0.0,
        **kwargs,
    ):
        img = Image.new("RGB", (width, height))
        actual_seed, _ = self._prepare_seed(seed)
        return GenerationResult(
            image=img,
            seed=actual_seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
        )


class TestBasePipelineInit:
    """Tests for BasePipeline initialization."""

    def test_pipe_starts_none(self):
        """Pipeline.pipe is None initially."""
        pipeline = ConcretePipeline()
        assert pipeline.pipe is None

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=True)
    def test_device_cuda_when_available(self, mock_cuda):
        """Device is 'cuda' when CUDA is available."""
        pipeline = ConcretePipeline()
        assert pipeline._device == "cuda"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_device_cpu_when_no_cuda(self, mock_cuda):
        """Device is 'cpu' when CUDA is not available."""
        pipeline = ConcretePipeline()
        assert pipeline._device == "cpu"


class TestBasePipelineUnload:
    """Tests for BasePipeline.unload()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_unload_clears_pipe(self, mock_cuda):
        """Unload sets pipe to None."""
        pipeline = ConcretePipeline()
        pipeline.pipe = Mock()
        pipeline.unload()
        assert pipeline.pipe is None

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_unload_handles_none_pipe(self, mock_cuda):
        """Unload handles pipe being None."""
        pipeline = ConcretePipeline()
        pipeline.pipe = None
        # Should not raise
        pipeline.unload()
        assert pipeline.pipe is None

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=True)
    @patch("oneiro.pipelines.base.torch.cuda.empty_cache")
    @patch("oneiro.pipelines.base.torch.cuda.synchronize")
    def test_unload_clears_cuda_cache(self, mock_sync, mock_empty, mock_avail):
        """Unload clears CUDA cache when available."""
        pipeline = ConcretePipeline()
        pipeline.pipe = Mock()
        pipeline.unload()
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()


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
