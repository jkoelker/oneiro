"""Tests for pipelines.flux1 module."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from oneiro.pipelines.flux1 import Flux1PipelineWrapper


class TestFlux1PipelineWrapperInit:
    """Tests for Flux1PipelineWrapper initialization."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_init_creates_instance(self, mock_cuda):
        """Flux1PipelineWrapper can be instantiated."""
        pipeline = Flux1PipelineWrapper()
        assert pipeline.pipe is None
        assert pipeline._device == "cpu"


class TestFlux1PipelineWrapperLoad:
    """Tests for Flux1PipelineWrapper.load()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_with_default_repo(
        self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda
    ):
        """Load uses default repo when not specified."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({})

        mock_flux_pipeline.from_pretrained.assert_called_once()
        call_args = mock_flux_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "black-forest-labs/FLUX.1-dev"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_with_custom_repo(self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda):
        """Load uses custom repo from config."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({"repo": "black-forest-labs/FLUX.1-schnell"})

        call_args = mock_flux_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "black-forest-labs/FLUX.1-schnell"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_enables_cpu_offload_by_default(
        self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda
    ):
        """Load enables CPU offload by default."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({})

        mock_pipe.enable_model_cpu_offload.assert_called_once()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_disables_cpu_offload_when_configured(
        self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda
    ):
        """Load respects cpu_offload=False config."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({"cpu_offload": False})

        mock_pipe.enable_model_cpu_offload.assert_not_called()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_enables_vae_optimizations(
        self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda
    ):
        """Load enables VAE tiling and slicing for memory optimization."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({})

        mock_pipe.vae.enable_tiling.assert_called_once()
        mock_pipe.vae.enable_slicing.assert_called_once()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_with_lora(self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda):
        """Load applies LoRA weights when lora and lora_weights are specified."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load(
            {
                "lora": "example/flux-lora-repo",
                "lora_weights": "flux_lora.safetensors",
            }
        )

        mock_pipe.load_lora_weights.assert_called_once_with(
            "example/flux-lora-repo", weight_name="flux_lora.safetensors"
        )

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_without_lora(self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda):
        """Load does not call load_lora_weights when lora config is not specified."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({})

        mock_pipe.load_lora_weights.assert_not_called()

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.FluxPipeline")
    def test_load_with_lora_repo_only_no_weights(
        self, mock_flux_pipeline, mock_threads, mock_interop, mock_cuda
    ):
        """Load does not call load_lora_weights when only lora repo is specified (no weights)."""
        mock_pipe = MagicMock()
        mock_flux_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux1PipelineWrapper()
        pipeline.load({"lora": "example/flux-lora-repo"})

        mock_pipe.load_lora_weights.assert_not_called()


class TestFlux1PipelineWrapperGenerate:
    """Tests for Flux1PipelineWrapper.generate()."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_raises_when_not_loaded(self, mock_cuda):
        """Generate raises RuntimeError when pipeline not loaded."""
        pipeline = Flux1PipelineWrapper()
        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            pipeline.generate("test prompt")

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_returns_result(self, mock_cuda):
        """Generate returns GenerationResult with correct fields."""
        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (1024, 1024), color="blue")
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        result = pipeline.generate("a beautiful landscape", seed=42)

        assert result.image is mock_image
        assert result.seed == 42
        assert result.prompt == "a beautiful landscape"
        assert result.width == 1024
        assert result.height == 1024

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_uses_default_parameters(self, mock_cuda):
        """Generate uses correct default parameters for FLUX.1-dev."""
        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (1024, 1024))
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        pipeline.generate("test prompt", seed=42)

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 28
        assert call_kwargs["guidance_scale"] == 3.5
        assert call_kwargs["max_sequence_length"] == 512

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_with_custom_parameters(self, mock_cuda):
        """Generate respects custom parameters."""
        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (512, 512))
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        result = pipeline.generate(
            "test prompt",
            width=512,
            height=512,
            seed=123,
            steps=4,
            guidance_scale=0.0,
        )

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["width"] == 512
        assert call_kwargs["height"] == 512
        assert call_kwargs["num_inference_steps"] == 4
        assert call_kwargs["guidance_scale"] == 0.0
        assert result.steps == 4

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_accepts_negative_prompt_for_api_compatibility(self, mock_cuda):
        """Generate accepts negative_prompt even though FLUX.1 doesn't use it."""
        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (1024, 1024))
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        result = pipeline.generate("test prompt", negative_prompt="blurry", seed=42)

        # negative_prompt is stored in result for compatibility
        assert result.negative_prompt == "blurry"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_random_seed_when_negative(self, mock_cuda):
        """Generate uses random seed when seed < 0."""
        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_image = Image.new("RGB", (1024, 1024))
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        result = pipeline.generate("test prompt", seed=-1)

        assert result.seed >= 0
        assert result.seed < 2**32


class TestFlux1PipelineWrapperImg2Img:
    """Tests for Flux1PipelineWrapper img2img functionality."""

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_with_init_image(self, mock_cuda):
        """Generate supports img2img with init_image."""
        import io

        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_output = Image.new("RGB", (1024, 1024))
        mock_pipe.return_value.images = [mock_output]
        pipeline.pipe = mock_pipe

        # Create init image bytes
        init_img = Image.new("RGB", (512, 512), color="red")
        buffer = io.BytesIO()
        init_img.save(buffer, format="PNG")
        init_bytes = buffer.getvalue()

        result = pipeline.generate("test prompt", seed=42, init_image=init_bytes)

        call_kwargs = mock_pipe.call_args[1]
        assert "image" in call_kwargs
        assert call_kwargs["strength"] == 0.75  # default strength
        assert result.prompt == "test prompt"

    @patch("oneiro.pipelines.base.torch.cuda.is_available", return_value=False)
    def test_generate_with_custom_strength(self, mock_cuda):
        """Generate respects custom strength for img2img."""
        import io

        pipeline = Flux1PipelineWrapper()
        mock_pipe = MagicMock()
        mock_output = Image.new("RGB", (1024, 1024))
        mock_pipe.return_value.images = [mock_output]
        pipeline.pipe = mock_pipe

        # Create init image bytes
        init_img = Image.new("RGB", (512, 512), color="red")
        buffer = io.BytesIO()
        init_img.save(buffer, format="PNG")
        init_bytes = buffer.getvalue()

        pipeline.generate("test prompt", seed=42, init_image=init_bytes, strength=0.5)

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["strength"] == 0.5
