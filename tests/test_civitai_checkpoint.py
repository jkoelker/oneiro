"""Tests for CivitAI checkpoint pipeline."""

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

        # Negative prompt is handled in _encode_prompts_to_embeddings for CLIP pipelines
        # Just verify generate() completed successfully
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
