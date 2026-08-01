"""Tests for pipelines.krea2 module."""

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

import oneiro.pipelines.krea2 as krea2
from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.krea2 import Krea2PipelineWrapper
from oneiro.pipelines.lora import LoraConfig, LoraSource


class TestLoadKrea2Tokenizer:
    """Tests for load_krea2_tokenizer()."""

    @patch("transformers.AutoTokenizer")
    def test_loads_published_fast_tokenizer(self, mock_auto_tokenizer):
        """Krea repositories load their published fast-tokenizer asset."""
        tokenizer = krea2.load_krea2_tokenizer("krea/Krea-2-Turbo")

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "krea/Krea-2-Turbo",
            subfolder="tokenizer",
            use_fast=True,
        )
        assert tokenizer is mock_auto_tokenizer.from_pretrained.return_value


class TestKrea2PipelineWrapperLoad:
    """Tests for Krea2PipelineWrapper.load()."""

    @patch("oneiro.pipelines.base.torch.set_num_interop_threads")
    @patch("oneiro.pipelines.base.torch.set_num_threads")
    @patch("diffusers.Krea2Pipeline", create=True)
    def test_load_uses_turbo_repo_and_fast_tokenizer_by_default(
        self, mock_krea2_pipeline, mock_threads, mock_interop
    ):
        """Load selects the official Turbo checkpoint and its fast tokenizer by default."""
        with patch(
            "oneiro.pipelines.krea2.load_krea2_tokenizer",
            create=True,
        ) as mock_load_tokenizer:
            pipeline = Krea2PipelineWrapper()
            pipeline.load({"cpu_offload": False})

        mock_load_tokenizer.assert_called_once_with("krea/Krea-2-Turbo")
        mock_krea2_pipeline.from_pretrained.assert_called_once_with(
            "krea/Krea-2-Turbo",
            tokenizer=mock_load_tokenizer.return_value,
            torch_dtype=pipeline.policy.dtype,
        )


class TestKrea2PipelineWrapperGenerate:
    """Tests for Krea2PipelineWrapper.generate()."""

    def test_generate_uses_turbo_defaults(self):
        """Generation uses the distilled checkpoint's recommended settings."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Krea2PipelineWrapper()
            mock_pipe = MagicMock()
            mock_pipe.return_value.images = [Image.new("RGB", (1024, 1024))]
            pipeline.pipe = mock_pipe

            result = pipeline.generate("a fox in the snow", seed=42)

        assert mock_pipe.call_args.kwargs["num_inference_steps"] == 8
        assert mock_pipe.call_args.kwargs["guidance_scale"] == 0.0
        assert result.steps == 8
        assert result.guidance_scale == 0.0

    def test_generate_passes_raw_checkpoint_parameters(self):
        """Generation forwards Raw checkpoint CFG and negative prompt settings."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Krea2PipelineWrapper()
            mock_pipe = MagicMock()
            mock_pipe.return_value.images = [Image.new("RGB", (768, 1024))]
            pipeline.pipe = mock_pipe

            result = pipeline.generate(
                "a studio portrait",
                negative_prompt="blurry",
                width=768,
                height=1024,
                seed=123,
                steps=28,
                guidance_scale=4.5,
            )

        call_kwargs = mock_pipe.call_args.kwargs
        assert call_kwargs["prompt"] == "a studio portrait"
        assert call_kwargs["negative_prompt"] == "blurry"
        assert call_kwargs["width"] == 768
        assert call_kwargs["height"] == 1024
        assert call_kwargs["num_inference_steps"] == 28
        assert call_kwargs["guidance_scale"] == 4.5
        assert call_kwargs["generator"].initial_seed() == 123
        assert result.seed == 123

    @pytest.mark.parametrize("argument", ["init_image", "mask_image"])
    def test_generate_rejects_image_inputs(self, argument):
        """Krea 2 rejects inputs unsupported by its text-to-image pipeline."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Krea2PipelineWrapper()
            pipeline.pipe = MagicMock()

            image = Image.new("RGB", (64, 64))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            with pytest.raises(ValueError, match="Krea 2 supports text-to-image only"):
                pipeline.generate("test", **{argument: buffer.getvalue()})

    def test_post_generate_restores_static_loras(self):
        """Generation cleanup replaces request-time LoRAs with the static baseline."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        static_lora = LoraConfig(
            name="static",
            source=LoraSource.HUGGINGFACE,
            repo="krea/static-lora",
            adapter_name="static",
        )
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Krea2PipelineWrapper()
            pipeline.pipe = MagicMock()
            pipeline._loaded_adapters = ["request"]
            pipeline._static_lora_configs = [static_lora]

            pipeline.post_generate()

        assert pipeline.active_loras == ["static"]

    def test_post_generate_reloads_static_lora_with_same_adapter_name(self):
        """Cleanup reloads static weights when a request reused its adapter name."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        static_lora = LoraConfig(
            name="static",
            source=LoraSource.HUGGINGFACE,
            repo="krea/static-lora",
            adapter_name="shared",
        )
        request_lora = LoraConfig(
            name="request",
            source=LoraSource.HUGGINGFACE,
            repo="krea/request-lora",
            adapter_name="shared",
        )
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Krea2PipelineWrapper()
            pipeline.pipe = MagicMock()
            pipeline._loaded_adapters = ["shared"]
            pipeline._lora_configs = [request_lora]
            pipeline._static_lora_configs = [static_lora]

            pipeline.post_generate()

        assert pipeline._lora_configs == [static_lora]

    def test_generate_applies_request_loras_during_inference(self):
        """Request-time LoRAs are active for inference and removed afterward."""
        mock_policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)
        request_lora = LoraConfig(
            name="request",
            source=LoraSource.HUGGINGFACE,
            repo="krea/request-lora",
            adapter_name="request",
        )
        with patch.object(DevicePolicy, "auto_detect", return_value=mock_policy):
            pipeline = Krea2PipelineWrapper()

            def run_inference(**kwargs):
                assert pipeline.active_loras == ["request"]
                result = MagicMock()
                result.images = [Image.new("RGB", (1024, 1024))]
                return result

            pipeline.pipe = MagicMock(side_effect=run_inference)
            pipeline.generate("test", seed=42, loras=[request_lora])

        assert pipeline.active_loras == []
