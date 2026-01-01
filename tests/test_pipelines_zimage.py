"""Tests for pipelines.zimage module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.zimage import ZImagePipelineWrapper


class TestZImagePipelineWrapperInit:
    """Tests for ZImagePipelineWrapper initialization."""

    def test_init_creates_instance(self):
        """ZImagePipelineWrapper can be instantiated."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
        assert pipeline.pipe is None
        assert pipeline.policy.device == "cpu"


class TestZImagePipelineWrapperLoad:
    """Tests for ZImagePipelineWrapper.load()."""

    @patch("diffusers.ZImagePipeline")
    def test_load_with_default_repo(self, mock_zimage_pipeline):
        """Load uses default repo when not specified."""
        mock_pipe = MagicMock()
        mock_zimage_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = ZImagePipelineWrapper()
        pipeline.load({})

        mock_zimage_pipeline.from_pretrained.assert_called_once()
        call_args = mock_zimage_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "Tongyi-MAI/Z-Image-Turbo"

    @patch("diffusers.ZImagePipeline")
    def test_load_with_custom_repo(self, mock_zimage_pipeline):
        """Load uses custom repo from config."""
        mock_pipe = MagicMock()
        mock_zimage_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = ZImagePipelineWrapper()
        pipeline.load({"repo": "custom/z-image"})

        call_args = mock_zimage_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "custom/z-image"

    @patch("diffusers.ZImagePipeline")
    def test_load_enables_cpu_offload_on_cuda(self, mock_zimage_pipeline):
        """Load enables CPU offload on CUDA by default."""
        mock_pipe = MagicMock()
        mock_zimage_pipeline.from_pretrained.return_value = mock_pipe

        # Mock CUDA being available
        mock_policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
            pipeline.load({})

        mock_pipe.enable_model_cpu_offload.assert_called_once()

    @patch("diffusers.ZImagePipeline")
    def test_load_disables_cpu_offload_when_configured(self, mock_zimage_pipeline):
        """Load respects cpu_offload=False config."""
        mock_pipe = MagicMock()
        mock_zimage_pipeline.from_pretrained.return_value = mock_pipe

        pipeline = ZImagePipelineWrapper()
        pipeline.load({"cpu_offload": False})

        mock_pipe.enable_model_cpu_offload.assert_not_called()


class TestZImagePipelineWrapperGenerate:
    """Tests for ZImagePipelineWrapper.generate()."""

    def test_generate_raises_when_not_loaded(self):
        """Generate raises RuntimeError when pipeline not loaded."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
            with pytest.raises(RuntimeError, match="Pipeline not loaded"):
                pipeline.generate("test prompt")

    def test_generate_returns_result(self):
        """Generate returns GenerationResult with correct fields."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
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

    def test_generate_forces_guidance_scale_zero(self):
        """Generate always uses guidance_scale=0.0 for Turbo model."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            # Try to pass guidance_scale=7.5, it should be forced to 0.0
            result = pipeline.generate("test prompt", seed=42, guidance_scale=7.5)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["guidance_scale"] == 0.0
            assert result.guidance_scale == 0.0

    def test_generate_uses_default_parameters(self):
        """Generate uses correct default parameters for Z-Image-Turbo."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["num_inference_steps"] == 9
            assert call_kwargs["guidance_scale"] == 0.0

    def test_generate_with_custom_parameters(self):
        """Generate respects custom parameters (except guidance_scale)."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
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
            )

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["width"] == 512
            assert call_kwargs["height"] == 512
            assert call_kwargs["num_inference_steps"] == 4
            assert result.steps == 4

    def test_generate_accepts_negative_prompt(self):
        """Generate accepts and passes negative_prompt."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", negative_prompt="blurry", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["negative_prompt"] == "blurry"
            assert result.negative_prompt == "blurry"

    def test_generate_random_seed_when_negative(self):
        """Generate uses random seed when seed < 0."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", seed=-1)

            assert result.seed >= 0
            assert result.seed < 2**32


class TestZImagePipelineWrapperImg2Img:
    """Tests for ZImagePipelineWrapper img2img functionality."""

    def test_generate_with_init_image(self):
        """Generate supports img2img with init_image."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
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
            assert call_kwargs["guidance_scale"] == 0.0  # still forced to 0.0
            assert result.prompt == "test prompt"

    def test_generate_with_custom_strength(self):
        """Generate respects custom strength for img2img."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = ZImagePipelineWrapper()
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
