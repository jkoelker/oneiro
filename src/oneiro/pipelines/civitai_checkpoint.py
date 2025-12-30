"""CivitAI checkpoint pipeline wrapper with multi-model-type support.

Supports loading full checkpoints from CivitAI in single-file format using
diffusers' from_single_file() method. Automatically detects the appropriate
pipeline class based on CivitAI's baseModel metadata.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config

if TYPE_CHECKING:
    from oneiro.civitai import CivitaiClient, ModelVersion


class CivitaiBaseModel(str, Enum):
    """Known CivitAI base model types mapped to diffusers pipelines."""

    # Stable Diffusion 1.x
    SD_1_4 = "SD 1.4"
    SD_1_5 = "SD 1.5"
    SD_1_5_LCM = "SD 1.5 LCM"
    SD_1_5_HYPER = "SD 1.5 Hyper"

    # Stable Diffusion 2.x
    SD_2_0 = "SD 2.0"
    SD_2_0_768 = "SD 2.0 768"
    SD_2_1 = "SD 2.1"
    SD_2_1_768 = "SD 2.1 768"
    SD_2_1_UNCLIP = "SD 2.1 Unclip"

    # SDXL
    SDXL_0_9 = "SDXL 0.9"
    SDXL_1_0 = "SDXL 1.0"
    SDXL_1_0_LCM = "SDXL 1.0 LCM"
    SDXL_DISTILLED = "SDXL Distilled"
    SDXL_TURBO = "SDXL Turbo"
    SDXL_LIGHTNING = "SDXL Lightning"
    SDXL_HYPER = "SDXL Hyper"

    # Pony (SDXL-based)
    PONY = "Pony"
    PONY_V6 = "Pony V6"
    PONY_V6_XL = "Pony V6 XL"

    # Illustrious (SDXL-based)
    ILLUSTRIOUS = "Illustrious"
    ILLUSTRIOUS_XL = "Illustrious XL"

    # Flux
    FLUX_1_D = "Flux.1 D"
    FLUX_1_S = "Flux.1 S"
    FLUX_1_DEV = "Flux.1 Dev"
    FLUX_1_SCHNELL = "Flux.1 Schnell"

    # SD 3.x
    SD_3 = "SD 3"
    SD_3_MEDIUM = "SD 3 Medium"
    SD_3_5 = "SD 3.5"
    SD_3_5_MEDIUM = "SD 3.5 Medium"
    SD_3_5_LARGE = "SD 3.5 Large"
    SD_3_5_LARGE_TURBO = "SD 3.5 Large Turbo"

    # Other architectures
    PIXART_A = "PixArt a"
    PIXART_SIGMA = "PixArt Sigma"
    KOLORS = "Kolors"
    HUNYUAN_DIT = "Hunyuan DiT"
    LUMINA = "Lumina"
    AURA_FLOW = "AuraFlow"

    # Unknown/Other
    OTHER = "Other"


@dataclass
class PipelineConfig:
    """Configuration for a diffusers pipeline class."""

    pipeline_class: str
    supports_negative_prompt: bool = True
    default_steps: int = 20
    default_guidance_scale: float = 7.5
    default_width: int = 512
    default_height: int = 512
    requires_safety_checker: bool = False


# Mapping from CivitAI base model strings to pipeline configurations
# Using partial string matching to handle variations
CIVITAI_BASE_MODEL_PIPELINE_MAP: dict[str, PipelineConfig] = {
    # SD 1.x family
    "SD 1.4": PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=512,
        default_height=512,
    ),
    "SD 1.5": PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=512,
        default_height=512,
    ),
    "SD 1.5 LCM": PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=512,
        default_height=512,
    ),
    "SD 1.5 Hyper": PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=512,
        default_height=512,
    ),
    # SD 2.x family
    "SD 2.0": PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=768,
        default_height=768,
    ),
    "SD 2.1": PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=768,
        default_height=768,
    ),
    # SDXL family
    "SDXL 0.9": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    "SDXL 1.0": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    "SDXL 1.0 LCM": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=1024,
        default_height=1024,
    ),
    "SDXL Turbo": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
    ),
    "SDXL Lightning": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
    ),
    "SDXL Hyper": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
    ),
    # Pony (SDXL-based)
    "Pony": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    # Illustrious (SDXL-based)
    "Illustrious": PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    # Flux family
    "Flux.1 D": PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=28,
        default_guidance_scale=3.5,
        default_width=1024,
        default_height=1024,
    ),
    "Flux.1 S": PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
    ),
    "Flux.1 Dev": PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=28,
        default_guidance_scale=3.5,
        default_width=1024,
        default_height=1024,
    ),
    "Flux.1 Schnell": PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
    ),
    # SD 3.x family
    "SD 3": PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    "SD 3 Medium": PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    "SD 3.5": PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
    ),
    "SD 3.5 Medium": PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
    ),
    "SD 3.5 Large": PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
    ),
    "SD 3.5 Large Turbo": PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
    ),
    # Other architectures
    "PixArt a": PipelineConfig(
        pipeline_class="PixArtAlphaPipeline",
        default_steps=20,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
    ),
    "PixArt Sigma": PipelineConfig(
        pipeline_class="PixArtSigmaPipeline",
        default_steps=20,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
    ),
    "Kolors": PipelineConfig(
        pipeline_class="KolorsPipeline",
        default_steps=25,
        default_guidance_scale=5.0,
        default_width=1024,
        default_height=1024,
    ),
    "Hunyuan DiT": PipelineConfig(
        pipeline_class="HunyuanDiTPipeline",
        default_steps=50,
        default_guidance_scale=5.0,
        default_width=1024,
        default_height=1024,
    ),
    "Lumina": PipelineConfig(
        pipeline_class="LuminaText2ImgPipeline",
        default_steps=30,
        default_guidance_scale=4.0,
        default_width=1024,
        default_height=1024,
    ),
    "AuraFlow": PipelineConfig(
        pipeline_class="AuraFlowPipeline",
        default_steps=50,
        default_guidance_scale=3.5,
        default_width=1024,
        default_height=1024,
    ),
}

# Default fallback configuration
DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    pipeline_class="StableDiffusionXLPipeline",
    default_steps=25,
    default_guidance_scale=7.0,
    default_width=1024,
    default_height=1024,
)


def get_pipeline_config_for_base_model(base_model: str | None) -> PipelineConfig:
    """Get pipeline configuration for a CivitAI base model string.

    Uses partial matching to handle variations in base model naming.

    Args:
        base_model: CivitAI baseModel string (e.g., "SDXL 1.0", "Flux.1 Dev")

    Returns:
        PipelineConfig for the appropriate diffusers pipeline
    """
    if base_model is None:
        return DEFAULT_PIPELINE_CONFIG

    # Exact match first
    if base_model in CIVITAI_BASE_MODEL_PIPELINE_MAP:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[base_model]

    # Partial matching (case-insensitive)
    base_lower = base_model.lower()

    # Check for specific patterns
    if "flux" in base_lower:
        if "schnell" in base_lower or " s" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["Flux.1 Schnell"]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["Flux.1 Dev"]

    if "sd 3.5" in base_lower or "sd3.5" in base_lower:
        if "turbo" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 3.5 Large Turbo"]
        if "large" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 3.5 Large"]
        if "medium" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 3.5 Medium"]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 3.5"]

    if "sd 3" in base_lower or "sd3" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 3"]

    if "pony" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["Pony"]

    if "illustrious" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["Illustrious"]

    if "sdxl" in base_lower or "xl" in base_lower:
        if "turbo" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL Turbo"]
        if "lightning" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL Lightning"]
        if "lcm" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL 1.0 LCM"]
        if "hyper" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL Hyper"]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["SDXL 1.0"]

    if "sd 2" in base_lower or "sd2" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 2.1"]

    if "sd 1.5" in base_lower or "sd1.5" in base_lower or "sd 1" in base_lower:
        if "lcm" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 1.5 LCM"]
        if "hyper" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 1.5 Hyper"]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["SD 1.5"]

    if "pixart" in base_lower:
        if "sigma" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP["PixArt Sigma"]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["PixArt a"]

    if "kolors" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["Kolors"]

    if "hunyuan" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["Hunyuan DiT"]

    if "lumina" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["Lumina"]

    if "auraflow" in base_lower or "aura" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP["AuraFlow"]

    # Default to SDXL as it's most common on CivitAI
    return DEFAULT_PIPELINE_CONFIG


def get_diffusers_pipeline_class(class_name: str) -> type:
    """Dynamically import and return a diffusers pipeline class.

    Args:
        class_name: Name of the pipeline class (e.g., "StableDiffusionXLPipeline")

    Returns:
        The pipeline class

    Raises:
        ImportError: If the pipeline class cannot be imported
    """
    import diffusers

    if not hasattr(diffusers, class_name):
        raise ImportError(f"Pipeline class '{class_name}' not found in diffusers")

    return getattr(diffusers, class_name)


class CivitaiCheckpointPipeline(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Pipeline wrapper for CivitAI checkpoints with auto-detection.

    Supports loading single-file checkpoints from CivitAI with automatic
    pipeline class selection based on the model's baseModel metadata.

    Config options:
        civitai_model_id: CivitAI model ID (required unless checkpoint_path provided)
        civitai_version_id: Specific version ID (optional, defaults to latest)
        checkpoint_path: Local path to downloaded checkpoint (if already cached)
        base_model: Override base model detection (e.g., "SDXL 1.0", "SD 1.5")
        pipeline_class: Override pipeline class (e.g., "StableDiffusionXLPipeline")
        cpu_offload: Enable CPU offloading (default: True)
        steps: Default inference steps (auto-detected from base model)
        guidance_scale: Default guidance scale (auto-detected from base model)
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_lora_state()
        self._init_embedding_state()
        self._pipeline_config: PipelineConfig | None = None
        self._base_model: str | None = None
        self._full_config: dict[str, Any] | None = None

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load checkpoint from config (synchronous, requires checkpoint_path).

        For CivitAI downloads, use load_async() instead.

        Args:
            model_config: Configuration with checkpoint_path and optional overrides
            full_config: Full configuration dict (for accessing global sections like embeddings)
        """
        self._full_config = full_config
        checkpoint_path = model_config.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError(
                "checkpoint_path required for synchronous load. "
                "Use load_async() for CivitAI downloads."
            )

        self._load_from_path(Path(checkpoint_path), model_config)

    async def load_async(
        self,
        model_config: dict[str, Any],
        civitai_client: "CivitaiClient",
        full_config: dict[str, Any] | None = None,
    ) -> None:
        """Load checkpoint from CivitAI, downloading if needed.

        Args:
            model_config: Configuration with civitai_model_id or checkpoint_path
            civitai_client: CivitaiClient for API access and downloads
            full_config: Full configuration dict (for accessing global sections like embeddings)
        """
        self._full_config = full_config
        checkpoint_path = model_config.get("checkpoint_path")
        civitai_model_id = model_config.get("civitai_model_id")
        civitai_version_id = model_config.get("civitai_version_id")

        if checkpoint_path:
            # Use provided path
            self._load_from_path(Path(checkpoint_path), model_config)
            return

        if not civitai_model_id:
            raise ValueError("civitai_model_id required when checkpoint_path not provided")

        # Fetch model info from CivitAI
        print(f"Fetching CivitAI model info for ID {civitai_model_id}...")

        version: ModelVersion
        if civitai_version_id:
            version = await civitai_client.get_model_version(civitai_version_id)
        else:
            model = await civitai_client.get_model(civitai_model_id)
            if model.latest_version is None:
                raise ValueError(f"No versions available for model {civitai_model_id}")
            version = model.latest_version

        # Store base model for auto-detection
        self._base_model = version.base_model

        # Download checkpoint
        print(f"Downloading checkpoint: {version.name} (base: {version.base_model})")
        path = await civitai_client.download_model_version(version)

        self._load_from_path(path, model_config)

    def _load_from_path(self, checkpoint_path: Path, model_config: dict[str, Any]) -> None:
        """Load checkpoint from local path.

        Args:
            checkpoint_path: Path to checkpoint file
            model_config: Configuration with optional overrides
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine base model (from config override or stored from API)
        base_model = model_config.get("base_model", self._base_model)

        # Get pipeline configuration
        pipeline_class_override = model_config.get("pipeline_class")
        if pipeline_class_override:
            # Manual override
            self._pipeline_config = PipelineConfig(
                pipeline_class=pipeline_class_override,
                default_steps=model_config.get("steps", 25),
                default_guidance_scale=model_config.get("guidance_scale", 7.0),
                default_width=model_config.get("width", 1024),
                default_height=model_config.get("height", 1024),
                supports_negative_prompt=model_config.get("supports_negative_prompt", True),
            )
        else:
            self._pipeline_config = get_pipeline_config_for_base_model(base_model)

        print(f"Loading checkpoint from {checkpoint_path}")
        print(f"  Base model: {base_model or 'unknown'}")
        print(f"  Pipeline: {self._pipeline_config.pipeline_class}")

        # Get the pipeline class
        pipeline_class = get_diffusers_pipeline_class(self._pipeline_config.pipeline_class)

        # Load from single file
        self.pipe = pipeline_class.from_single_file(
            str(checkpoint_path),
            torch_dtype=self._dtype,
        )

        # Apply optimizations
        cpu_offload = model_config.get("cpu_offload", True)
        if cpu_offload and self._device == "cuda":
            self.pipe.enable_model_cpu_offload()
        elif self._device == "cuda":
            self.pipe.to("cuda")

        # Enable memory optimizations for VAE if available
        if hasattr(self.pipe, "vae"):
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()

        # Load LoRAs if configured
        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)

        # Load embeddings if full_config provided
        if self._full_config:
            embeddings = parse_embeddings_from_config(self._full_config, model_config)
            if embeddings:
                print(f"  Loading {len(embeddings)} embedding(s)...")
                self.load_embeddings_sync(embeddings)

        print(f"Checkpoint loaded: {checkpoint_path.name}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int = -1,
        steps: int | None = None,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate image using the loaded checkpoint.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (ignored for Flux models)
            width: Image width (defaults to pipeline config)
            height: Image height (defaults to pipeline config)
            seed: Random seed (-1 for random)
            steps: Inference steps (defaults to pipeline config)
            guidance_scale: Guidance scale (defaults to pipeline config)
            **kwargs: Additional pipeline-specific parameters

        Returns:
            GenerationResult with generated image and metadata
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        if self._pipeline_config is None:
            raise RuntimeError("Pipeline config not initialized")

        # Use defaults from pipeline config
        width = width or self._pipeline_config.default_width
        height = height or self._pipeline_config.default_height
        steps = steps or self._pipeline_config.default_steps
        guidance_scale = (
            guidance_scale
            if guidance_scale is not None
            else self._pipeline_config.default_guidance_scale
        )

        actual_seed, generator = self._prepare_seed(seed)

        # Handle img2img
        init_image = self._load_init_image(kwargs.get("init_image"))
        strength = kwargs.get("strength", 0.75)

        # Build generation kwargs
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # Add negative prompt only for pipelines that support it
        if self._pipeline_config.supports_negative_prompt and negative_prompt:
            gen_kwargs["negative_prompt"] = negative_prompt

        if init_image:
            print(f"CivitAI img2img: '{prompt[:50]}...' seed={actual_seed} strength={strength}")
            gen_kwargs["image"] = init_image
            gen_kwargs["strength"] = strength
        else:
            print(f"CivitAI generating: '{prompt[:50]}...' seed={actual_seed}")
            gen_kwargs["height"] = height
            gen_kwargs["width"] = width

        result = self.pipe(**gen_kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_image = result.images[0]
        return GenerationResult(
            image=output_image,
            seed=actual_seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output_image.width,
            height=output_image.height,
            steps=steps,
            guidance_scale=guidance_scale,
        )

    @property
    def pipeline_config(self) -> PipelineConfig | None:
        """Get the current pipeline configuration."""
        return self._pipeline_config

    @property
    def detected_base_model(self) -> str | None:
        """Get the detected/configured base model."""
        return self._base_model
