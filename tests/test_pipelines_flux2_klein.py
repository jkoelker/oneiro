"""Tests for pipelines.flux2_klein module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.flux2_klein import Flux2KleinPipelineWrapper


class TestFlux2KleinPipelineWrapperInit:
    """Tests for Flux2KleinPipelineWrapper initialization."""

    def test_init_creates_instance(self):
        """Flux2KleinPipelineWrapper can be instantiated."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
        assert pipeline.pipe is None
        assert pipeline.policy.device == "cpu"


class TestFlux2KleinPipelineWrapperLoad:
    """Tests for Flux2KleinPipelineWrapper.load()."""

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.Flux2KleinPipeline")
    def test_load_with_default_repo(self, mock_flux2_klein_pipeline, mock_threads, mock_interop):
        """Load uses the 9B distilled Klein repo when not specified."""
        mock_pipe = MagicMock()
        mock_flux2_klein_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux2KleinPipelineWrapper()
        pipeline.load({})

        mock_flux2_klein_pipeline.from_pretrained.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=pipeline.policy.dtype,
        )

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.Flux2KleinPipeline")
    def test_load_with_custom_repo(self, mock_flux2_klein_pipeline, mock_threads, mock_interop):
        """Load uses custom repo from config."""
        mock_pipe = MagicMock()
        mock_flux2_klein_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux2KleinPipelineWrapper()
        pipeline.load({"repo": "black-forest-labs/FLUX.2-klein-4B"})

        call_args = mock_flux2_klein_pipeline.from_pretrained.call_args
        assert call_args.args[0] == "black-forest-labs/FLUX.2-klein-4B"

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.Flux2KleinPipeline")
    def test_load_enables_group_offload_on_cuda(
        self, mock_flux2_klein_pipeline, mock_threads, mock_interop
    ):
        """Load enables group offload on CUDA by default."""
        mock_pipe = MagicMock()
        mock_flux2_klein_pipeline.from_pretrained.return_value = mock_pipe

        mock_policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            pipeline.load({})

        mock_pipe.enable_group_offload.assert_called_once()

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.Flux2KleinPipeline")
    def test_load_disables_cpu_offload_when_configured(
        self, mock_flux2_klein_pipeline, mock_threads, mock_interop
    ):
        """Load respects cpu_offload=False config."""
        mock_pipe = MagicMock()
        mock_flux2_klein_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = Flux2KleinPipelineWrapper()
        pipeline.load({"cpu_offload": False})

        mock_pipe.enable_group_offload.assert_not_called()
        mock_pipe.enable_model_cpu_offload.assert_not_called()


class TestFlux2KleinPipelineWrapperGenerate:
    """Tests for Flux2KleinPipelineWrapper.generate()."""

    def test_generate_raises_when_not_loaded(self):
        """Generate raises RuntimeError when pipeline not loaded."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            with pytest.raises(RuntimeError, match="Pipeline not loaded"):
                pipeline.generate("test prompt")

    def test_generate_returns_result(self):
        """Generate returns GenerationResult with correct fields."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024), color="blue")
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("a fast preview", seed=42)

            assert result.image is mock_image
            assert result.seed == 42
            assert result.prompt == "a fast preview"
            assert result.width == 1024
            assert result.height == 1024

    def test_generate_uses_distilled_defaults(self):
        """Generate uses distilled FLUX.2 Klein defaults."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            mock_pipe = MagicMock()
            mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42)

            call_kwargs = mock_pipe.call_args.kwargs
            assert call_kwargs["num_inference_steps"] == 4
            assert call_kwargs["guidance_scale"] == 1.0

    def test_generate_with_custom_parameters(self):
        """Generate respects custom parameters."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            mock_pipe = MagicMock()
            mock_pipe.return_value.images = [Image.new("RGB", (512, 512))]
            pipeline.pipe = mock_pipe

            result = pipeline.generate(
                "test prompt",
                width=512,
                height=512,
                seed=123,
                steps=20,
                guidance_scale=5.0,
            )

            call_kwargs = mock_pipe.call_args.kwargs
            assert call_kwargs["width"] == 512
            assert call_kwargs["height"] == 512
            assert call_kwargs["num_inference_steps"] == 20
            assert call_kwargs["guidance_scale"] == 5.0
            assert result.steps == 20
            assert result.guidance_scale == 5.0

    def test_generate_stores_negative_prompt_in_result(self):
        """Generate stores negative_prompt in result but does not pass it to pipeline."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            mock_pipe = MagicMock()
            mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", negative_prompt="blurry", seed=42)

            assert result.negative_prompt == "blurry"
            call_kwargs = mock_pipe.call_args.kwargs
            assert "negative_prompt" not in call_kwargs

    def test_generate_with_init_image(self):
        """Generate passes init image to FLUX.2 Klein without unsupported strength."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Flux2KleinPipelineWrapper()
            mock_pipe = MagicMock()
            mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]
            pipeline.pipe = mock_pipe

            init_img = Image.new("RGB", (512, 512), color="red")
            buffer = io.BytesIO()
            init_img.save(buffer, format="PNG")

            pipeline.generate("test prompt", seed=42, init_image=buffer.getvalue(), strength=0.5)

            call_kwargs = mock_pipe.call_args.kwargs
            assert "image" in call_kwargs
            assert "strength" not in call_kwargs
