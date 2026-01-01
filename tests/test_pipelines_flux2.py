"""Tests for pipelines.flux2 module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.flux2 import Flux2PipelineWrapper


class TestFlux2PipelineWrapperInit:
    """Tests for Flux2PipelineWrapper initialization."""

    def test_init_creates_instance(self):
        """Flux2PipelineWrapper can be instantiated."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
        assert pipeline.pipe is None
        assert pipeline.policy.device == "cpu"


class TestFlux2PipelineWrapperLoad:
    """Tests for Flux2PipelineWrapper.load()."""

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("transformers.Mistral3ForConditionalGeneration")
    @patch("diffusers.Flux2Transformer2DModel")
    @patch("diffusers.Flux2Pipeline")
    def test_load_with_default_repo(
        self, mock_flux2_pipeline, mock_transformer, mock_text_encoder, mock_threads, mock_interop
    ):
        """Load uses default repo when not specified."""
        mock_pipe = MagicMock()
        mock_flux2_pipeline.from_pretrained.return_value = mock_pipe
        mock_transformer.from_pretrained.return_value = MagicMock()
        mock_text_encoder.from_pretrained.return_value = MagicMock()

        pipeline = Flux2PipelineWrapper()
        pipeline.load({})

        mock_flux2_pipeline.from_pretrained.assert_called_once()
        call_args = mock_flux2_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "diffusers/FLUX.2-dev-bnb-4bit"

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("transformers.Mistral3ForConditionalGeneration")
    @patch("diffusers.Flux2Transformer2DModel")
    @patch("diffusers.Flux2Pipeline")
    def test_load_with_custom_repo(
        self, mock_flux2_pipeline, mock_transformer, mock_text_encoder, mock_threads, mock_interop
    ):
        """Load uses custom repo from config."""
        mock_pipe = MagicMock()
        mock_flux2_pipeline.from_pretrained.return_value = mock_pipe
        mock_transformer.from_pretrained.return_value = MagicMock()
        mock_text_encoder.from_pretrained.return_value = MagicMock()

        pipeline = Flux2PipelineWrapper()
        pipeline.load({"repo": "custom/flux2-model"})

        call_args = mock_flux2_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "custom/flux2-model"

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("transformers.Mistral3ForConditionalGeneration")
    @patch("diffusers.Flux2Transformer2DModel")
    @patch("diffusers.Flux2Pipeline")
    def test_load_enables_cpu_offload_on_cuda(
        self, mock_flux2_pipeline, mock_transformer, mock_text_encoder, mock_threads, mock_interop
    ):
        """Load enables CPU offload on CUDA by default."""
        mock_pipe = MagicMock()
        mock_flux2_pipeline.from_pretrained.return_value = mock_pipe
        mock_transformer.from_pretrained.return_value = MagicMock()
        mock_text_encoder.from_pretrained.return_value = MagicMock()

        # Mock CUDA being available
        mock_policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
            pipeline.load({})

        mock_pipe.enable_model_cpu_offload.assert_called_once()

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("transformers.Mistral3ForConditionalGeneration")
    @patch("diffusers.Flux2Transformer2DModel")
    @patch("diffusers.Flux2Pipeline")
    def test_load_disables_cpu_offload_when_configured(
        self, mock_flux2_pipeline, mock_transformer, mock_text_encoder, mock_threads, mock_interop
    ):
        """Load respects cpu_offload=False config."""
        mock_pipe = MagicMock()
        mock_flux2_pipeline.from_pretrained.return_value = mock_pipe
        mock_transformer.from_pretrained.return_value = MagicMock()
        mock_text_encoder.from_pretrained.return_value = MagicMock()

        pipeline = Flux2PipelineWrapper()
        pipeline.load({"cpu_offload": False})

        mock_pipe.enable_model_cpu_offload.assert_not_called()


class TestFlux2PipelineWrapperGenerate:
    """Tests for Flux2PipelineWrapper.generate()."""

    def test_generate_raises_when_not_loaded(self):
        """Generate raises RuntimeError when pipeline not loaded."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
            with pytest.raises(RuntimeError, match="Pipeline not loaded"):
                pipeline.generate("test prompt")

    def test_generate_returns_result(self):
        """Generate returns GenerationResult with correct fields."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
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

    def test_generate_uses_default_parameters(self):
        """Generate uses correct default parameters for FLUX.2."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["num_inference_steps"] == 28
            assert call_kwargs["guidance_scale"] == 4.0

    def test_generate_with_custom_parameters(self):
        """Generate respects custom parameters."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (512, 512))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate(
                "test prompt",
                width=512,
                height=512,
                seed=123,
                steps=20,
                guidance_scale=5.0,
            )

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["width"] == 512
            assert call_kwargs["height"] == 512
            assert call_kwargs["num_inference_steps"] == 20
            assert call_kwargs["guidance_scale"] == 5.0
            assert result.steps == 20
            assert result.guidance_scale == 5.0

    def test_generate_stores_negative_prompt_in_result(self):
        """Generate stores negative_prompt in result but does NOT pass it to pipeline."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", negative_prompt="blurry", seed=42)

            assert result.negative_prompt == "blurry"
            call_kwargs = mock_pipe.call_args[1]
            assert "negative_prompt" not in call_kwargs

    def test_generate_random_seed_when_negative(self):
        """Generate uses random seed when seed < 0."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", seed=-1)

            assert result.seed >= 0
            assert result.seed < 2**32


class TestFlux2PipelineWrapperImg2Img:
    """Tests for Flux2PipelineWrapper img2img functionality."""

    def test_generate_with_init_image(self):
        """Generate supports img2img with init_image."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
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

    def test_generate_with_custom_strength(self):
        """Generate respects custom strength for img2img."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2PipelineWrapper()
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
