"""Tests for CivitAI checkpoint pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oneiro.pipelines.civitai_checkpoint import (
    CIVITAI_BASE_MODEL_PIPELINE_MAP,
    DEFAULT_PIPELINE_CONFIG,
    SCHEDULER_CHOICES,
    SCHEDULER_MAP,
    CivitaiCheckpointPipeline,
    PipelineConfig,
    get_diffusers_pipeline_class,
    get_pipeline_config_for_base_model,
)
from oneiro.pipelines.lora import LoraConfig, LoraSource


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """PipelineConfig has sensible defaults."""
        config = PipelineConfig(pipeline_class="StableDiffusionPipeline")
        assert config.supports_negative_prompt is True
        assert config.default_steps == 20
        assert config.default_guidance_scale == 7.5
        assert config.default_width == 512
        assert config.default_height == 512

    def test_custom_values(self):
        """PipelineConfig accepts custom values."""
        config = PipelineConfig(
            pipeline_class="FluxPipeline",
            supports_negative_prompt=False,
            default_steps=28,
            default_guidance_scale=3.5,
            default_width=1024,
            default_height=1024,
        )
        assert config.pipeline_class == "FluxPipeline"
        assert config.supports_negative_prompt is False
        assert config.default_steps == 28


class TestGetPipelineConfigForBaseModel:
    """Tests for get_pipeline_config_for_base_model function."""

    def test_exact_match_sd15(self):
        """Exact match for SD 1.5."""
        config = get_pipeline_config_for_base_model("SD 1.5")
        assert config.pipeline_class == "StableDiffusionPipeline"
        assert config.default_width == 512

    def test_exact_match_sdxl(self):
        """Exact match for SDXL 1.0."""
        config = get_pipeline_config_for_base_model("SDXL 1.0")
        assert config.pipeline_class == "StableDiffusionXLPipeline"
        assert config.default_width == 1024

    def test_exact_match_flux_dev(self):
        """Exact match for Flux.1 Dev."""
        config = get_pipeline_config_for_base_model("Flux.1 Dev")
        assert config.pipeline_class == "FluxPipeline"
        assert config.supports_negative_prompt is False
        assert config.default_steps == 28

    def test_exact_match_flux_schnell(self):
        """Exact match for Flux.1 Schnell."""
        config = get_pipeline_config_for_base_model("Flux.1 Schnell")
        assert config.pipeline_class == "FluxPipeline"
        assert config.default_steps == 4
        assert config.default_guidance_scale == 0.0

    def test_partial_match_flux(self):
        """Partial match for Flux variants."""
        config = get_pipeline_config_for_base_model("Flux.1")
        assert config.pipeline_class == "FluxPipeline"

        config = get_pipeline_config_for_base_model("flux dev")
        assert config.pipeline_class == "FluxPipeline"

    def test_partial_match_sdxl_turbo(self):
        """Partial match for SDXL Turbo."""
        config = get_pipeline_config_for_base_model("SDXL Turbo")
        assert config.pipeline_class == "StableDiffusionXLPipeline"
        assert config.default_steps == 4
        assert config.default_guidance_scale == 0.0

    def test_partial_match_pony(self):
        """Partial match for Pony models."""
        config = get_pipeline_config_for_base_model("Pony V6 XL")
        assert config.pipeline_class == "StableDiffusionXLPipeline"

        config = get_pipeline_config_for_base_model("pony")
        assert config.pipeline_class == "StableDiffusionXLPipeline"

    def test_partial_match_illustrious(self):
        """Partial match for Illustrious models."""
        config = get_pipeline_config_for_base_model("Illustrious XL v1.0")
        assert config.pipeline_class == "StableDiffusionXLPipeline"

    def test_partial_match_sd3(self):
        """Partial match for SD 3.x models."""
        config = get_pipeline_config_for_base_model("SD 3.5 Large")
        assert config.pipeline_class == "StableDiffusion3Pipeline"

        config = get_pipeline_config_for_base_model("sd3")
        assert config.pipeline_class == "StableDiffusion3Pipeline"

    def test_partial_match_sd35_turbo(self):
        """Partial match for SD 3.5 Turbo."""
        config = get_pipeline_config_for_base_model("SD 3.5 Large Turbo")
        assert config.pipeline_class == "StableDiffusion3Pipeline"
        assert config.default_steps == 4
        assert config.default_guidance_scale == 0.0

    def test_partial_match_lcm_variants(self):
        """Partial match for LCM variants."""
        config = get_pipeline_config_for_base_model("SD 1.5 LCM")
        assert config.pipeline_class == "StableDiffusionPipeline"
        assert config.default_steps == 4

        config = get_pipeline_config_for_base_model("SDXL 1.0 LCM")
        assert config.pipeline_class == "StableDiffusionXLPipeline"
        assert config.default_steps == 4

    def test_none_returns_default(self):
        """None base_model returns default config."""
        config = get_pipeline_config_for_base_model(None)
        assert config == DEFAULT_PIPELINE_CONFIG

    def test_unknown_returns_default(self):
        """Unknown base_model returns default (SDXL) config."""
        config = get_pipeline_config_for_base_model("Some Unknown Model")
        assert config == DEFAULT_PIPELINE_CONFIG

    def test_pixart_variants(self):
        """PixArt model variants."""
        config = get_pipeline_config_for_base_model("PixArt a")
        assert config.pipeline_class == "PixArtAlphaPipeline"

        config = get_pipeline_config_for_base_model("pixart sigma")
        assert config.pipeline_class == "PixArtSigmaPipeline"

    def test_kolors(self):
        """Kolors model."""
        config = get_pipeline_config_for_base_model("Kolors")
        assert config.pipeline_class == "KolorsPipeline"

    def test_hunyuan(self):
        """Hunyuan DiT model."""
        config = get_pipeline_config_for_base_model("Hunyuan DiT")
        assert config.pipeline_class == "HunyuanDiTPipeline"

    def test_auraflow(self):
        """AuraFlow model."""
        config = get_pipeline_config_for_base_model("AuraFlow")
        assert config.pipeline_class == "AuraFlowPipeline"

        config = get_pipeline_config_for_base_model("aura flow")
        assert config.pipeline_class == "AuraFlowPipeline"


class TestCivitaiBaseModelPipelineMap:
    """Tests for the CIVITAI_BASE_MODEL_PIPELINE_MAP constant."""

    def test_sd15_config(self):
        """SD 1.5 has correct configuration."""
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 1.5"]
        assert config.pipeline_class == "StableDiffusionPipeline"
        assert config.default_width == 512
        assert config.default_height == 512
        assert config.supports_negative_prompt is True

    def test_sdxl_config(self):
        """SDXL 1.0 has correct configuration."""
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL 1.0"]
        assert config.pipeline_class == "StableDiffusionXLPipeline"
        assert config.default_width == 1024
        assert config.default_height == 1024

    def test_flux_config(self):
        """Flux.1 Dev has correct configuration."""
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["Flux.1 Dev"]
        assert config.pipeline_class == "FluxPipeline"
        assert config.supports_negative_prompt is False
        assert config.default_steps == 28
        assert config.default_guidance_scale == 3.5

    def test_all_entries_have_required_fields(self):
        """All map entries have required PipelineConfig fields."""
        for base_model, config in CIVITAI_BASE_MODEL_PIPELINE_MAP.items():
            assert isinstance(config, PipelineConfig), f"Invalid config for {base_model}"
            assert config.pipeline_class, f"Missing pipeline_class for {base_model}"
            assert config.default_steps > 0, f"Invalid steps for {base_model}"
            assert config.default_width > 0, f"Invalid width for {base_model}"
            assert config.default_height > 0, f"Invalid height for {base_model}"


class TestGetDiffusersPipelineClass:
    """Tests for get_diffusers_pipeline_class function."""

    def test_valid_class_import(self):
        """Valid pipeline class can be imported."""
        # Test with actual diffusers - it should import successfully
        # We use a class that exists in diffusers
        import diffusers

        result = get_diffusers_pipeline_class("StableDiffusionPipeline")
        assert result is diffusers.StableDiffusionPipeline

    def test_invalid_class_raises(self):
        """Invalid pipeline class raises ImportError."""
        with pytest.raises(ImportError, match="not found in diffusers"):
            get_diffusers_pipeline_class("NonExistentPipeline")


class TestCivitaiCheckpointPipelineInit:
    """Tests for CivitaiCheckpointPipeline initialization."""

    def test_init(self):
        """Pipeline initializes with correct state."""
        pipeline = CivitaiCheckpointPipeline()
        assert pipeline.pipe is None
        assert pipeline._pipeline_config is None
        assert pipeline._base_model is None
        assert pipeline._lora_configs == []
        assert pipeline._loaded_adapters == []

    def test_pipeline_config_property(self):
        """pipeline_config property returns stored config."""
        pipeline = CivitaiCheckpointPipeline()
        assert pipeline.pipeline_config is None

        pipeline._pipeline_config = PipelineConfig(pipeline_class="Test")
        assert pipeline.pipeline_config.pipeline_class == "Test"

    def test_detected_base_model_property(self):
        """detected_base_model property returns stored base model."""
        pipeline = CivitaiCheckpointPipeline()
        assert pipeline.detected_base_model is None

        pipeline._base_model = "SDXL 1.0"
        assert pipeline.detected_base_model == "SDXL 1.0"


class TestCivitaiCheckpointPipelineLoad:
    """Tests for CivitaiCheckpointPipeline.load method."""

    def test_load_requires_checkpoint_path(self):
        """load() requires checkpoint_path in config."""
        pipeline = CivitaiCheckpointPipeline()
        with pytest.raises(ValueError, match="checkpoint_path required"):
            pipeline.load({})

    def test_load_file_not_found(self, tmp_path):
        """load() raises FileNotFoundError for missing file."""
        pipeline = CivitaiCheckpointPipeline()
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            pipeline.load({"checkpoint_path": str(tmp_path / "nonexistent.safetensors")})

    @patch.object(CivitaiCheckpointPipeline, "configure_scheduler")
    @patch("oneiro.pipelines.civitai_checkpoint.get_diffusers_pipeline_class")
    def test_load_with_base_model_override(self, mock_get_class, mock_config_sched, tmp_path):
        """load() uses base_model override from config."""
        # Create dummy checkpoint file
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        mock_pipeline_class = MagicMock()
        mock_pipe = MagicMock()
        mock_pipeline_class.from_single_file.return_value = mock_pipe
        mock_get_class.return_value = mock_pipeline_class

        pipeline = CivitaiCheckpointPipeline()
        pipeline.load(
            {
                "checkpoint_path": str(checkpoint),
                "base_model": "SD 1.5",
            }
        )

        assert pipeline._pipeline_config.pipeline_class == "StableDiffusionPipeline"
        mock_get_class.assert_called_once_with("StableDiffusionPipeline")

    @patch("oneiro.pipelines.civitai_checkpoint.get_diffusers_pipeline_class")
    def test_load_with_pipeline_class_override(self, mock_get_class, tmp_path):
        """load() uses pipeline_class override from config."""
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        mock_pipeline_class = MagicMock()
        mock_pipe = MagicMock()
        mock_pipeline_class.from_single_file.return_value = mock_pipe
        mock_get_class.return_value = mock_pipeline_class

        pipeline = CivitaiCheckpointPipeline()
        pipeline.load(
            {
                "checkpoint_path": str(checkpoint),
                "pipeline_class": "CustomPipeline",
                "steps": 10,
                "guidance_scale": 5.0,
            }
        )

        assert pipeline._pipeline_config.pipeline_class == "CustomPipeline"
        assert pipeline._pipeline_config.default_steps == 10
        assert pipeline._pipeline_config.default_guidance_scale == 5.0
        mock_get_class.assert_called_once_with("CustomPipeline")

    @patch.object(CivitaiCheckpointPipeline, "configure_scheduler")
    @patch("oneiro.pipelines.civitai_checkpoint.get_diffusers_pipeline_class")
    def test_load_enables_cpu_offload(self, mock_get_class, mock_config_sched, tmp_path):
        """load() enables CPU offload when configured."""
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        mock_pipeline_class = MagicMock()
        mock_pipe = MagicMock()
        mock_pipeline_class.from_single_file.return_value = mock_pipe
        mock_get_class.return_value = mock_pipeline_class

        pipeline = CivitaiCheckpointPipeline()
        pipeline._device = "cuda"  # Simulate CUDA availability
        pipeline.load(
            {
                "checkpoint_path": str(checkpoint),
                "cpu_offload": True,
            }
        )

        mock_pipe.enable_model_cpu_offload.assert_called_once()

    @patch.object(CivitaiCheckpointPipeline, "configure_scheduler")
    @patch("oneiro.pipelines.civitai_checkpoint.get_diffusers_pipeline_class")
    def test_load_enables_vae_optimizations(self, mock_get_class, mock_config_sched, tmp_path):
        """load() enables VAE tiling and slicing."""
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        mock_pipeline_class = MagicMock()
        mock_pipe = MagicMock()
        mock_vae = MagicMock()
        mock_pipe.vae = mock_vae
        mock_pipeline_class.from_single_file.return_value = mock_pipe
        mock_get_class.return_value = mock_pipeline_class

        pipeline = CivitaiCheckpointPipeline()
        pipeline.load({"checkpoint_path": str(checkpoint)})

        mock_vae.enable_tiling.assert_called_once()
        mock_vae.enable_slicing.assert_called_once()


class TestCivitaiCheckpointPipelineLoadAsync:
    """Tests for CivitaiCheckpointPipeline.load_async method."""

    @pytest.mark.asyncio
    async def test_load_async_with_checkpoint_path(self, tmp_path):
        """load_async() uses checkpoint_path if provided."""
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        pipeline = CivitaiCheckpointPipeline()

        with patch.object(pipeline, "_load_from_path") as mock_load:
            await pipeline.load_async(
                {"checkpoint_path": str(checkpoint)},
                civitai_client=MagicMock(),
            )
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_async_requires_model_id(self):
        """load_async() requires civitai_model_id when no path."""
        pipeline = CivitaiCheckpointPipeline()
        client = MagicMock()

        with pytest.raises(ValueError, match="civitai_model_id required"):
            await pipeline.load_async({}, civitai_client=client)

    @pytest.mark.asyncio
    async def test_load_async_fetches_model_info(self, tmp_path):
        """load_async() fetches model info from CivitAI."""
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        # Mock CivitAI client
        mock_version = MagicMock()
        mock_version.base_model = "SDXL 1.0"
        mock_version.name = "Test Model v1"

        mock_model = MagicMock()
        mock_model.latest_version = mock_version

        mock_client = AsyncMock()
        mock_client.get_model.return_value = mock_model
        mock_client.download_model_version.return_value = checkpoint

        pipeline = CivitaiCheckpointPipeline()

        with patch.object(pipeline, "_load_from_path") as mock_load:
            await pipeline.load_async(
                {"civitai_model_id": 12345},
                civitai_client=mock_client,
            )

            mock_client.get_model.assert_called_once_with(12345)
            mock_client.download_model_version.assert_called_once_with(mock_version)
            assert pipeline._base_model == "SDXL 1.0"
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_async_with_version_id(self, tmp_path):
        """load_async() fetches specific version when version_id provided."""
        checkpoint = tmp_path / "model.safetensors"
        checkpoint.write_bytes(b"dummy")

        mock_version = MagicMock()
        mock_version.base_model = "Flux.1 Dev"
        mock_version.name = "Test Model v2"

        mock_client = AsyncMock()
        mock_client.get_model_version.return_value = mock_version
        mock_client.download_model_version.return_value = checkpoint

        pipeline = CivitaiCheckpointPipeline()

        with patch.object(pipeline, "_load_from_path"):
            await pipeline.load_async(
                {"civitai_model_id": 12345, "civitai_version_id": 67890},
                civitai_client=mock_client,
            )

            mock_client.get_model_version.assert_called_once_with(67890)
            mock_client.get_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_async_no_versions_available(self):
        """load_async() raises when model has no versions."""
        mock_model = MagicMock()
        mock_model.latest_version = None

        mock_client = AsyncMock()
        mock_client.get_model.return_value = mock_model

        pipeline = CivitaiCheckpointPipeline()

        with pytest.raises(ValueError, match="No versions available"):
            await pipeline.load_async(
                {"civitai_model_id": 12345},
                civitai_client=mock_client,
            )


class TestCivitaiCheckpointPipelineGenerate:
    """Tests for CivitaiCheckpointPipeline.generate method."""

    def test_generate_requires_loaded_pipeline(self):
        """generate() raises if pipeline not loaded."""
        pipeline = CivitaiCheckpointPipeline()
        with pytest.raises(RuntimeError, match="Pipeline not loaded"):
            pipeline.generate("test prompt")

    def test_generate_requires_pipeline_config(self):
        """generate() raises if pipeline_config not set."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()  # Pretend pipeline is loaded
        with pytest.raises(RuntimeError, match="Pipeline config not initialized"):
            pipeline.generate("test prompt")

    def test_generate_uses_config_defaults(self):
        """generate() uses defaults from pipeline config."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            default_steps=25,
            default_guidance_scale=7.0,
            default_width=1024,
            default_height=1024,
        )

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            pipeline.generate("test prompt")

        call_kwargs = mock_pipe.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 25
        assert call_kwargs["guidance_scale"] == 7.0
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 1024

    def test_generate_respects_custom_params(self):
        """generate() respects custom parameters over defaults."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionPipeline",
            default_steps=20,
            default_guidance_scale=7.5,
        )

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 768
        mock_image.height = 768
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            pipeline.generate(
                "test prompt",
                steps=10,
                guidance_scale=5.0,
                width=768,
                height=768,
            )

        call_kwargs = mock_pipe.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 10
        assert call_kwargs["guidance_scale"] == 5.0
        assert call_kwargs["width"] == 768
        assert call_kwargs["height"] == 768

    def test_generate_handles_negative_prompt(self):
        """generate() includes negative_prompt for supporting pipelines."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            supports_negative_prompt=True,
        )

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            pipeline.generate("test prompt", negative_prompt="bad quality")

        # All pipelines using the embedding-based approach (SD 1.x, SD 2.x, SDXL, SD3)
        # handle negative prompts via embeddings (negative_prompt_embeds, and for SDXL,
        # negative_pooled_prompt_embeds) computed in _encode_prompts_to_embeddings,
        # not via direct negative_prompt kwarg. Verify the mock was called correctly.
        mock_pipe.assert_called_once()

    def test_generate_omits_negative_prompt_for_flux(self):
        """generate() omits negative_prompt for Flux pipelines."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="FluxPipeline",
            supports_negative_prompt=False,
        )

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            pipeline.generate("test prompt", negative_prompt="bad quality")

        call_kwargs = mock_pipe.call_args.kwargs
        assert "negative_prompt" not in call_kwargs

    def test_generate_returns_generation_result(self):
        """generate() returns proper GenerationResult."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            default_steps=25,
            default_guidance_scale=7.0,
        )

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            result = pipeline.generate("test prompt", seed=42)

        assert result.prompt == "test prompt"
        assert result.seed == 42
        assert result.image == mock_image
        assert result.width == 1024
        assert result.height == 1024
        assert result.steps == 25
        assert result.guidance_scale == 7.0

    def test_generate_handles_img2img(self):
        """generate() handles img2img with init_image."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
        )

        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        # Mock init image loading
        mock_init_image = MagicMock()
        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_load_init_image", return_value=mock_init_image),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            pipeline.generate("test prompt", init_image=b"dummy", strength=0.5)

        call_kwargs = mock_pipe.call_args.kwargs
        assert call_kwargs["image"] == mock_init_image
        assert call_kwargs["strength"] == 0.5
        assert "width" not in call_kwargs
        assert "height" not in call_kwargs

    def test_generate_with_scheduler_override(self):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            default_scheduler="dpm++_karras",
        )

        mock_pipe = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.config = {}
        mock_pipe.scheduler = mock_scheduler
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe

        mock_euler = MagicMock()
        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch("diffusers.EulerAncestralDiscreteScheduler", mock_euler),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
        ):
            pipeline.generate("test prompt", scheduler="euler_a")

        mock_euler.from_config.assert_called_once()


class TestSchedulerMap:
    def test_scheduler_choices_matches_map_keys(self):
        assert set(SCHEDULER_CHOICES) == set(SCHEDULER_MAP.keys())

    def test_default_entry_has_none_class(self):
        class_name, kwargs = SCHEDULER_MAP["default"]
        assert class_name is None
        assert kwargs == {}

    def test_dpm_karras_entry(self):
        class_name, kwargs = SCHEDULER_MAP["dpm++_karras"]
        assert class_name == "DPMSolverMultistepScheduler"
        assert kwargs["algorithm_type"] == "sde-dpmsolver++"
        assert kwargs["use_karras_sigmas"] is True

    def test_dpm_entry(self):
        class_name, kwargs = SCHEDULER_MAP["dpm++"]
        assert class_name == "DPMSolverMultistepScheduler"
        assert kwargs["algorithm_type"] == "sde-dpmsolver++"
        assert kwargs["use_karras_sigmas"] is False

    def test_euler_a_entry(self):
        class_name, kwargs = SCHEDULER_MAP["euler_a"]
        assert class_name == "EulerAncestralDiscreteScheduler"
        assert kwargs == {}

    def test_euler_entry(self):
        class_name, kwargs = SCHEDULER_MAP["euler"]
        assert class_name == "EulerDiscreteScheduler"
        assert kwargs == {}

    def test_heun_entry(self):
        class_name, kwargs = SCHEDULER_MAP["heun"]
        assert class_name == "HeunDiscreteScheduler"
        assert kwargs == {}

    def test_ddim_entry(self):
        class_name, kwargs = SCHEDULER_MAP["ddim"]
        assert class_name == "DDIMScheduler"
        assert kwargs == {}


class TestDefaultSchedulers:
    def test_sd15_has_dpm_karras(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 1.5"]
        assert config.default_scheduler == "dpm++_karras"

    def test_sdxl_has_dpm_karras(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL 1.0"]
        assert config.default_scheduler == "dpm++_karras"

    def test_pony_has_dpm_karras(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["Pony"]
        assert config.default_scheduler == "dpm++_karras"

    def test_illustrious_has_dpm_karras(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["Illustrious"]
        assert config.default_scheduler == "dpm++_karras"

    def test_sdxl_turbo_keeps_default(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL Turbo"]
        assert config.default_scheduler == "default"

    def test_sdxl_lightning_keeps_default(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL Lightning"]
        assert config.default_scheduler == "default"

    def test_flux_keeps_default(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["Flux.1 Dev"]
        assert config.default_scheduler == "default"

    def test_sd3_keeps_default(self):
        config = CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 3"]
        assert config.default_scheduler == "default"


class TestConfigureScheduler:
    def test_configure_scheduler_with_none_uses_pipeline_default(self):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            default_scheduler="dpm++_karras",
        )
        mock_pipe = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.config = {}
        mock_pipe.scheduler = mock_scheduler
        pipeline.pipe = mock_pipe

        mock_dpm = MagicMock()
        with patch("diffusers.DPMSolverMultistepScheduler", mock_dpm):
            pipeline.configure_scheduler(None)

        mock_dpm.from_config.assert_called_once()

    def test_configure_scheduler_default_string_no_change(self):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            default_scheduler="default",
        )
        mock_pipe = MagicMock()
        original_scheduler = MagicMock()
        mock_pipe.scheduler = original_scheduler
        pipeline.pipe = mock_pipe

        pipeline.configure_scheduler("default")

        assert mock_pipe.scheduler is original_scheduler

    def test_configure_scheduler_unknown_warns(self, capsys):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
        )
        mock_pipe = MagicMock()
        original_scheduler = MagicMock()
        mock_pipe.scheduler = original_scheduler
        pipeline.pipe = mock_pipe

        pipeline.configure_scheduler("unknown_scheduler")

        captured = capsys.readouterr()
        assert "Unknown scheduler" in captured.out
        assert mock_pipe.scheduler is original_scheduler

    def test_configure_scheduler_euler_a(self):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
        )
        mock_pipe = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.config = {}
        mock_pipe.scheduler = mock_scheduler
        pipeline.pipe = mock_pipe

        mock_euler = MagicMock()
        with patch("diffusers.EulerAncestralDiscreteScheduler", mock_euler):
            pipeline.configure_scheduler("euler_a")

        mock_euler.from_config.assert_called_once_with({})

    def test_configure_scheduler_with_kwargs(self):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
        )
        mock_pipe = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.config = {"some": "config"}
        mock_pipe.scheduler = mock_scheduler
        pipeline.pipe = mock_pipe

        mock_dpm = MagicMock()
        with patch("diffusers.DPMSolverMultistepScheduler", mock_dpm):
            pipeline.configure_scheduler("dpm++_karras")

        mock_dpm.from_config.assert_called_once_with(
            {"some": "config"},
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

    def test_configure_scheduler_skips_redundant_reconfiguration(self):
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
        )
        mock_pipe = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.config = {}
        mock_pipe.scheduler = mock_scheduler
        pipeline.pipe = mock_pipe

        mock_euler = MagicMock()
        with patch("diffusers.EulerAncestralDiscreteScheduler", mock_euler):
            pipeline.configure_scheduler("euler_a")
            pipeline.configure_scheduler("euler_a")

        mock_euler.from_config.assert_called_once()


class TestSupportsPromptEmbeddings:
    """Tests for CivitaiCheckpointPipeline._supports_prompt_embeddings method."""

    def test_returns_false_when_pipe_is_none(self):
        """Returns False when pipeline is not loaded."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(pipeline_class="StableDiffusionXLPipeline")
        # pipe is None by default
        assert pipeline._supports_prompt_embeddings() is False

    def test_returns_false_when_pipeline_config_is_none(self):
        """Returns False when pipeline config is not set."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()  # Pretend pipeline is loaded
        # _pipeline_config is None by default
        assert pipeline._supports_prompt_embeddings() is False

    def test_returns_true_for_stable_diffusion_pipeline(self):
        """Returns True for StableDiffusionPipeline (SD 1.x/2.x)."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(pipeline_class="StableDiffusionPipeline")
        assert pipeline._supports_prompt_embeddings() is True

    def test_returns_true_for_stable_diffusion_xl_pipeline(self):
        """Returns True for StableDiffusionXLPipeline."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(pipeline_class="StableDiffusionXLPipeline")
        assert pipeline._supports_prompt_embeddings() is True

    def test_returns_true_for_flux_pipeline(self):
        """Returns True for FluxPipeline."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="FluxPipeline", supports_negative_prompt=False
        )
        assert pipeline._supports_prompt_embeddings() is True

    def test_returns_true_for_stable_diffusion_3_pipeline(self):
        """Returns True for StableDiffusion3Pipeline."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(pipeline_class="StableDiffusion3Pipeline")
        assert pipeline._supports_prompt_embeddings() is True

    def test_returns_false_for_unsupported_pipeline(self):
        """Returns False for unsupported pipeline classes."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()

        unsupported_pipelines = [
            "PixArtAlphaPipeline",
            "PixArtSigmaPipeline",
            "KolorsPipeline",
            "HunyuanDiTPipeline",
            "LuminaText2ImgPipeline",
            "AuraFlowPipeline",
            "CustomPipeline",
        ]

        for pipeline_class in unsupported_pipelines:
            pipeline._pipeline_config = PipelineConfig(pipeline_class=pipeline_class)
            assert pipeline._supports_prompt_embeddings() is False, (
                f"Expected False for {pipeline_class}"
            )


class TestEncodePromptsToEmbeddings:
    """Tests for CivitaiCheckpointPipeline._encode_prompts_to_embeddings method."""

    def test_returns_empty_dict_when_pipe_is_none(self):
        """Returns empty dict when pipeline is not loaded."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(pipeline_class="StableDiffusionXLPipeline")
        # pipe is None by default
        result = pipeline._encode_prompts_to_embeddings("test prompt", None)
        assert result == {}

    def test_returns_empty_dict_when_pipeline_config_is_none(self):
        """Returns empty dict when pipeline config is not set."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        # _pipeline_config is None by default
        result = pipeline._encode_prompts_to_embeddings("test prompt", None)
        assert result == {}

    def test_flux_pipeline_encoding(self):
        """FluxPipeline returns prompt_embeds and pooled_prompt_embeds only."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="FluxPipeline", supports_negative_prompt=False
        )

        mock_prompt_embeds = MagicMock()
        mock_pooled_embeds = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_flux",
            return_value=(mock_prompt_embeds, mock_pooled_embeds),
        ) as mock_func:
            result = pipeline._encode_prompts_to_embeddings("test prompt", "negative")

        mock_func.assert_called_once_with(pipeline.pipe, prompt="test prompt")
        assert result["prompt_embeds"] is mock_prompt_embeds
        assert result["pooled_prompt_embeds"] is mock_pooled_embeds
        # Flux doesn't support negative prompts
        assert "negative_prompt_embeds" not in result

    def test_sd3_pipeline_encoding_with_negative(self):
        """SD3 Pipeline returns all embedding types including negative."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusion3Pipeline", supports_negative_prompt=True
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()
        mock_pooled = MagicMock()
        mock_neg_pooled = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sd3",
            return_value=(mock_prompt, mock_neg_prompt, mock_pooled, mock_neg_pooled),
        ) as mock_func:
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad quality")

        mock_func.assert_called_once_with(
            pipeline.pipe, prompt="test prompt", negative_prompt="bad quality"
        )
        assert result["prompt_embeds"] is mock_prompt
        assert result["pooled_prompt_embeds"] is mock_pooled
        assert result["negative_prompt_embeds"] is mock_neg_prompt
        assert result["negative_pooled_prompt_embeds"] is mock_neg_pooled

    def test_sd3_pipeline_without_negative_support(self):
        """SD3 Pipeline without negative prompt support omits negative embeddings."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusion3Pipeline", supports_negative_prompt=False
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()
        mock_pooled = MagicMock()
        mock_neg_pooled = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sd3",
            return_value=(mock_prompt, mock_neg_prompt, mock_pooled, mock_neg_pooled),
        ):
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad quality")

        assert result["prompt_embeds"] is mock_prompt
        assert result["pooled_prompt_embeds"] is mock_pooled
        assert "negative_prompt_embeds" not in result
        assert "negative_pooled_prompt_embeds" not in result

    def test_sdxl_pipeline_encoding_with_negative(self):
        """SDXL Pipeline returns all embedding types including negative."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline", supports_negative_prompt=True
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()
        mock_pooled = MagicMock()
        mock_neg_pooled = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sdxl",
            return_value=(mock_prompt, mock_neg_prompt, mock_pooled, mock_neg_pooled),
        ) as mock_func:
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad quality")

        mock_func.assert_called_once_with(
            pipeline.pipe, prompt="test prompt", negative_prompt="bad quality"
        )
        assert result["prompt_embeds"] is mock_prompt
        assert result["pooled_prompt_embeds"] is mock_pooled
        assert result["negative_prompt_embeds"] is mock_neg_prompt
        assert result["negative_pooled_prompt_embeds"] is mock_neg_pooled

    def test_sdxl_pipeline_without_negative_support(self):
        """SDXL Pipeline without negative prompt support omits negative embeddings."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline", supports_negative_prompt=False
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()
        mock_pooled = MagicMock()
        mock_neg_pooled = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sdxl",
            return_value=(mock_prompt, mock_neg_prompt, mock_pooled, mock_neg_pooled),
        ):
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad quality")

        assert result["prompt_embeds"] is mock_prompt
        assert result["pooled_prompt_embeds"] is mock_pooled
        assert "negative_prompt_embeds" not in result
        assert "negative_pooled_prompt_embeds" not in result

    def test_sd15_pipeline_encoding_with_negative(self):
        """SD 1.x/2.x Pipeline returns prompt and negative embeddings."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionPipeline", supports_negative_prompt=True
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sd15",
            return_value=(mock_prompt, mock_neg_prompt),
        ) as mock_func:
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad quality")

        mock_func.assert_called_once_with(
            pipeline.pipe, prompt="test prompt", negative_prompt="bad quality"
        )
        assert result["prompt_embeds"] is mock_prompt
        assert result["negative_prompt_embeds"] is mock_neg_prompt
        # SD 1.x/2.x don't have pooled embeddings
        assert "pooled_prompt_embeds" not in result

    def test_sd15_pipeline_without_negative_support(self):
        """SD 1.x/2.x Pipeline without negative prompt support omits negative embeddings."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionPipeline", supports_negative_prompt=False
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sd15",
            return_value=(mock_prompt, mock_neg_prompt),
        ):
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad quality")

        assert result["prompt_embeds"] is mock_prompt
        assert "negative_prompt_embeds" not in result

    def test_handles_none_negative_prompt(self):
        """Handles None negative prompt by converting to empty string."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionPipeline", supports_negative_prompt=True
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sd15",
            return_value=(mock_prompt, mock_neg_prompt),
        ) as mock_func:
            pipeline._encode_prompts_to_embeddings("test prompt", None)

        # Should convert None to empty string
        mock_func.assert_called_once_with(pipeline.pipe, prompt="test prompt", negative_prompt="")

    def test_falls_back_to_sd15_for_unknown_pipeline(self):
        """Falls back to SD 1.5 encoding for unknown pipeline classes."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline.pipe = MagicMock()
        # Use a pipeline class that's not in the supported list but would
        # still go through the else branch
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="SomeOtherDiffusionPipeline", supports_negative_prompt=True
        )

        mock_prompt = MagicMock()
        mock_neg_prompt = MagicMock()

        with patch(
            "oneiro.pipelines.civitai_checkpoint.get_weighted_text_embeddings_sd15",
            return_value=(mock_prompt, mock_neg_prompt),
        ) as mock_func:
            result = pipeline._encode_prompts_to_embeddings("test prompt", "bad")

        mock_func.assert_called_once()
        assert result["prompt_embeds"] is mock_prompt
        assert result["negative_prompt_embeds"] is mock_neg_prompt


class TestDynamicLoraGeneration:
    """Tests for dynamic LoRA loading during generation."""

    def _create_pipeline_with_mocks(self):
        """Create a pipeline with common mocks for dynamic LoRA tests."""
        pipeline = CivitaiCheckpointPipeline()
        pipeline._pipeline_config = PipelineConfig(
            pipeline_class="StableDiffusionXLPipeline",
            default_steps=25,
            default_guidance_scale=7.0,
            default_width=1024,
            default_height=1024,
        )
        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_image.width = 1024
        mock_image.height = 1024
        mock_pipe.return_value.images = [mock_image]
        pipeline.pipe = mock_pipe
        pipeline._cpu_offload = False
        return pipeline

    def test_generate_with_dynamic_loras(self):
        """generate() loads dynamic LoRAs passed via kwargs."""
        pipeline = self._create_pipeline_with_mocks()

        lora = LoraConfig(name="test-lora", source=LoraSource.LOCAL, path="/fake/path.safetensors")
        lora._resolved_path = Path("/fake/path.safetensors")

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
            patch.object(pipeline, "_load_dynamic_loras") as mock_load,
            patch.object(pipeline, "_restore_static_loras") as mock_restore,
        ):
            pipeline.generate("test prompt", loras=[lora])

        mock_load.assert_called_once_with([lora])
        mock_restore.assert_called_once()

    def test_generate_restores_static_loras_after_dynamic(self):
        """generate() restores static LoRAs after using dynamic ones."""
        pipeline = self._create_pipeline_with_mocks()

        static_lora = LoraConfig(
            name="static-lora", source=LoraSource.LOCAL, path="/static.safetensors"
        )
        pipeline._static_lora_configs = [static_lora]

        dynamic_lora = LoraConfig(
            name="dynamic-lora", source=LoraSource.LOCAL, path="/dynamic.safetensors"
        )
        dynamic_lora._resolved_path = Path("/dynamic.safetensors")

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
            patch.object(pipeline, "unload_loras") as mock_unload,
            patch.object(pipeline, "load_single_lora", return_value="dynamic-lora"),
            patch.object(pipeline, "set_lora_adapters"),
            patch.object(pipeline, "load_loras_sync") as mock_load_sync,
        ):
            pipeline.generate("test prompt", loras=[dynamic_lora])

        assert mock_unload.call_count == 2
        mock_load_sync.assert_called_once_with([static_lora])

    def test_generate_handles_dynamic_lora_loading_failure(self):
        """generate() restores static LoRAs when dynamic loading fails."""
        pipeline = self._create_pipeline_with_mocks()

        static_lora = LoraConfig(
            name="static-lora", source=LoraSource.LOCAL, path="/static.safetensors"
        )
        pipeline._static_lora_configs = [static_lora]

        dynamic_lora = LoraConfig(name="bad-lora", source=LoraSource.LOCAL, path="/bad.safetensors")
        dynamic_lora._resolved_path = Path("/bad.safetensors")

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
            patch.object(pipeline, "_load_dynamic_loras", side_effect=RuntimeError("Load failed")),
            patch.object(pipeline, "_restore_static_loras") as mock_restore,
        ):
            with pytest.raises(RuntimeError, match="Load failed"):
                pipeline.generate("test prompt", loras=[dynamic_lora])

        mock_restore.assert_called_once()

    def test_generate_cleanup_on_generation_failure(self):
        """generate() cleans up dynamic LoRAs even if generation fails."""
        pipeline = self._create_pipeline_with_mocks()

        dynamic_lora = LoraConfig(
            name="dynamic-lora", source=LoraSource.LOCAL, path="/dynamic.safetensors"
        )
        dynamic_lora._resolved_path = Path("/dynamic.safetensors")

        pipeline.pipe.side_effect = RuntimeError("Generation failed")

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
            patch.object(pipeline, "_load_dynamic_loras"),
            patch.object(pipeline, "_restore_static_loras") as mock_restore,
        ):
            with pytest.raises(RuntimeError, match="Generation failed"):
                pipeline.generate("test prompt", loras=[dynamic_lora])

        mock_restore.assert_called_once()

    def test_generate_without_dynamic_loras_skips_lora_handling(self):
        """generate() skips LoRA handling when no dynamic LoRAs provided."""
        pipeline = self._create_pipeline_with_mocks()

        with (
            patch("oneiro.pipelines.civitai_checkpoint.torch"),
            patch.object(pipeline, "_encode_prompts_to_embeddings"),
            patch.object(pipeline, "_load_dynamic_loras") as mock_load,
            patch.object(pipeline, "_restore_static_loras") as mock_restore,
        ):
            pipeline.generate("test prompt")

        mock_load.assert_not_called()
        mock_restore.assert_not_called()

    def test_load_dynamic_loras_respects_cpu_offload(self):
        """_load_dynamic_loras() skips .to(device) when cpu_offload enabled."""
        pipeline = self._create_pipeline_with_mocks()
        pipeline._cpu_offload = True

        lora = LoraConfig(name="test-lora", source=LoraSource.LOCAL, path="/fake.safetensors")
        lora._resolved_path = Path("/fake.safetensors")

        with (
            patch.object(pipeline, "unload_loras"),
            patch.object(pipeline, "load_single_lora", return_value="test-lora"),
            patch.object(pipeline, "set_lora_adapters"),
        ):
            pipeline._load_dynamic_loras([lora])

        pipeline.pipe.to.assert_not_called()

    def test_load_dynamic_loras_moves_to_device_without_cpu_offload(self):
        """_load_dynamic_loras() calls .to(device) when cpu_offload disabled."""
        pipeline = self._create_pipeline_with_mocks()
        pipeline._cpu_offload = False
        pipeline._device = "cuda"

        lora = LoraConfig(name="test-lora", source=LoraSource.LOCAL, path="/fake.safetensors")
        lora._resolved_path = Path("/fake.safetensors")

        with (
            patch.object(pipeline, "unload_loras"),
            patch.object(pipeline, "load_single_lora", return_value="test-lora"),
            patch.object(pipeline, "set_lora_adapters"),
        ):
            pipeline._load_dynamic_loras([lora])

        pipeline.pipe.to.assert_called_once_with("cuda")
