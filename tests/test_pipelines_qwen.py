"""Tests for pipelines.qwen module."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.qwen import QwenPipelineWrapper


class TestQwenPipelineWrapperInit:
    """Tests for QwenPipelineWrapper initialization."""

    def test_init_creates_instance(self):
        """QwenPipelineWrapper can be instantiated."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
        assert pipeline.pipe is None
        assert pipeline.policy.device == "cpu"


class TestQwenPipelineWrapperParseTransformerPath:
    """Tests for QwenPipelineWrapper._parse_transformer_path()."""

    def test_local_path_gguf(self, tmp_path):
        """Local GGUF path is detected correctly."""
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()

        pipeline = QwenPipelineWrapper()
        path, is_gguf = pipeline._parse_transformer_path(str(gguf_file))

        assert path == str(gguf_file)
        assert is_gguf is True

    def test_local_path_non_gguf(self, tmp_path):
        """Local non-GGUF path is detected correctly."""
        safetensors_file = tmp_path / "model.safetensors"
        safetensors_file.touch()

        pipeline = QwenPipelineWrapper()
        path, is_gguf = pipeline._parse_transformer_path(str(safetensors_file))

        assert path == str(safetensors_file)
        assert is_gguf is False

    def test_hf_hub_format_gguf(self):
        """HF Hub repo:file format with GGUF is parsed correctly."""
        pipeline = QwenPipelineWrapper()

        with patch("huggingface_hub.hf_hub_download") as mock_download:
            mock_download.return_value = "/cache/model.gguf"
            path, is_gguf = pipeline._parse_transformer_path(
                "unsloth/Qwen-Image-GGUF:qwen-image-Q4_K_S.gguf"
            )

        mock_download.assert_called_once_with(
            repo_id="unsloth/Qwen-Image-GGUF", filename="qwen-image-Q4_K_S.gguf"
        )
        assert path == "/cache/model.gguf"
        assert is_gguf is True

    def test_invalid_format_raises(self):
        """Invalid format raises ValueError."""
        pipeline = QwenPipelineWrapper()

        with pytest.raises(ValueError, match="must be 'repo_id:filename' or a local path"):
            pipeline._parse_transformer_path("invalid_format")


class TestQwenPipelineWrapperLoad:
    """Tests for QwenPipelineWrapper.load()."""

    @patch("diffusers.FlowMatchEulerDiscreteScheduler")
    @patch("diffusers.DiffusionPipeline")
    def test_load_with_default_repo(self, mock_diffusion_pipeline, mock_scheduler):
        """Load uses default repo when not specified."""
        mock_pipe = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipe
        mock_scheduler.from_config.return_value = MagicMock()

        pipeline = QwenPipelineWrapper()
        pipeline.load({})

        mock_diffusion_pipeline.from_pretrained.assert_called_once()
        call_args = mock_diffusion_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "Qwen/Qwen-Image"

    @patch("diffusers.FlowMatchEulerDiscreteScheduler")
    @patch("diffusers.DiffusionPipeline")
    def test_load_with_custom_repo(self, mock_diffusion_pipeline, mock_scheduler):
        """Load uses custom repo from config."""
        mock_pipe = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipe
        mock_scheduler.from_config.return_value = MagicMock()

        pipeline = QwenPipelineWrapper()
        pipeline.load({"repo": "custom/qwen-model"})

        call_args = mock_diffusion_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "custom/qwen-model"

    @patch("diffusers.FlowMatchEulerDiscreteScheduler")
    @patch("diffusers.DiffusionPipeline")
    def test_load_enables_cpu_offload_on_cuda(self, mock_diffusion_pipeline, mock_scheduler):
        """Load enables CPU offload on CUDA by default."""
        mock_pipe = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipe
        mock_scheduler.from_config.return_value = MagicMock()

        # Mock CUDA being available
        mock_policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            pipeline.load({})

        mock_pipe.enable_model_cpu_offload.assert_called_once()

    @patch("diffusers.FlowMatchEulerDiscreteScheduler")
    @patch("diffusers.DiffusionPipeline")
    def test_load_disables_cpu_offload_when_configured(
        self, mock_diffusion_pipeline, mock_scheduler
    ):
        """Load respects cpu_offload=False config."""
        mock_pipe = MagicMock()
        mock_diffusion_pipeline.from_pretrained.return_value = mock_pipe
        mock_scheduler.from_config.return_value = MagicMock()

        pipeline = QwenPipelineWrapper()
        pipeline.load({"cpu_offload": False})

        mock_pipe.enable_model_cpu_offload.assert_not_called()


class TestQwenPipelineWrapperGenerate:
    """Tests for QwenPipelineWrapper.generate()."""

    def test_generate_raises_when_not_loaded(self):
        """Generate raises RuntimeError when pipeline not loaded."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            with pytest.raises(RuntimeError, match="Pipeline not loaded"):
                pipeline.generate("test prompt")

    def test_generate_returns_result(self):
        """Generate returns GenerationResult with correct fields."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
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

    def test_generate_uses_true_cfg_scale(self):
        """Generate uses true_cfg_scale instead of guidance_scale."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42, guidance_scale=5.0)

            call_kwargs = mock_pipe.call_args[1]
            assert "true_cfg_scale" in call_kwargs
            assert call_kwargs["true_cfg_scale"] == 5.0
            assert "guidance_scale" not in call_kwargs

    def test_generate_accepts_true_cfg_scale_kwarg(self):
        """Generate accepts true_cfg_scale as explicit kwarg."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            # true_cfg_scale should override guidance_scale
            pipeline.generate("test prompt", seed=42, guidance_scale=4.0, true_cfg_scale=7.0)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["true_cfg_scale"] == 7.0

    def test_generate_requires_negative_prompt(self):
        """Generate defaults negative_prompt to single space when not provided."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["negative_prompt"] == " "

    def test_generate_passes_negative_prompt(self):
        """Generate passes user-provided negative_prompt."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", negative_prompt="blurry", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["negative_prompt"] == "blurry"
            assert result.negative_prompt == "blurry"

    def test_generate_uses_num_images_per_prompt(self):
        """Generate passes num_images_per_prompt=1."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["num_images_per_prompt"] == 1

    def test_generate_uses_default_parameters(self):
        """Generate uses correct default parameters for Qwen-Image."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            pipeline.generate("test prompt", seed=42)

            call_kwargs = mock_pipe.call_args[1]
            assert call_kwargs["num_inference_steps"] == 8
            assert call_kwargs["true_cfg_scale"] == 4.0

    def test_generate_with_custom_parameters(self):
        """Generate respects custom parameters."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
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
            assert call_kwargs["true_cfg_scale"] == 5.0
            assert result.steps == 20

    def test_generate_random_seed_when_negative(self):
        """Generate uses random seed when seed < 0."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
            mock_pipe = MagicMock()
            mock_image = Image.new("RGB", (1024, 1024))
            mock_pipe.return_value.images = [mock_image]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("test prompt", seed=-1)

            assert result.seed >= 0
            assert result.seed < 2**32


class TestQwenPipelineWrapperImg2Img:
    """Tests for QwenPipelineWrapper img2img functionality."""

    def test_generate_with_init_image(self):
        """Generate supports img2img with init_image."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
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
            assert call_kwargs["true_cfg_scale"] == 4.0
            assert result.prompt == "test prompt"

    def test_generate_with_custom_strength(self):
        """Generate respects custom strength for img2img."""
        import io

        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = QwenPipelineWrapper()
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
