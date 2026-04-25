"""CivitAI checkpoint pipeline wrapper with multi-model-type support.

Supports loading full checkpoints from CivitAI in single-file format using
diffusers' from_single_file() method. Automatically detects the appropriate
pipeline class based on CivitAI's baseModel metadata.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy, OffloadMode
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.long_prompt import (
    get_weighted_text_embeddings_flux,
    get_weighted_text_embeddings_sd3,
    get_weighted_text_embeddings_sd15,
    get_weighted_text_embeddings_sdxl,
)
from oneiro.pipelines.lora import LoraConfig, LoraLoaderMixin, parse_loras_from_model_config

if TYPE_CHECKING:
    from oneiro.civitai import CivitaiClient, ModelVersion


class CivitaiBaseModel(StrEnum):
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
    FLUX_2_KLEIN_9B = "Flux.2 Klein 9B"
    FLUX_2_KLEIN_9B_BASE = "Flux.2 Klein 9B-base"
    FLUX_2_KLEIN_4B = "Flux.2 Klein 4B"
    FLUX_2_KLEIN_4B_BASE = "Flux.2 Klein 4B-base"

    # Qwen
    QWEN = "Qwen"

    # SD 3.x
    SD_3 = "SD 3"
    SD_3_MEDIUM = "SD 3 Medium"
    SD_3_5 = "SD 3.5"
    SD_3_5_MEDIUM = "SD 3.5 Medium"
    SD_3_5_LARGE = "SD 3.5 Large"
    SD_3_5_LARGE_TURBO = "SD 3.5 Large Turbo"

    # Other architectures
    Z_IMAGE = "Z-Image"
    Z_IMAGE_TURBO = "Z-Image Turbo"
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
    default_scheduler: str | None = None  # None = auto-detect based on model type


# Mapping from CivitAI base model strings to pipeline configurations
# Using partial string matching to handle variations
CIVITAI_BASE_MODEL_PIPELINE_MAP: dict[str, PipelineConfig] = {
    # SD 1.x family
    CivitaiBaseModel.SD_1_4: PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=512,
        default_height=512,
        default_scheduler="dpm++_karras",
    ),
    CivitaiBaseModel.SD_1_5: PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=512,
        default_height=512,
        default_scheduler="dpm++_karras",
    ),
    CivitaiBaseModel.SD_1_5_LCM: PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=512,
        default_height=512,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SD_1_5_HYPER: PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=512,
        default_height=512,
        default_scheduler="default",
    ),
    # SD 2.x family
    CivitaiBaseModel.SD_2_0: PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=768,
        default_height=768,
        default_scheduler="dpm++_karras",
    ),
    CivitaiBaseModel.SD_2_1: PipelineConfig(
        pipeline_class="StableDiffusionPipeline",
        default_steps=20,
        default_guidance_scale=7.5,
        default_width=768,
        default_height=768,
        default_scheduler="dpm++_karras",
    ),
    # SDXL family
    CivitaiBaseModel.SDXL_0_9: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="dpm++_karras",
    ),
    CivitaiBaseModel.SDXL_1_0: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="dpm++_karras",
    ),
    CivitaiBaseModel.SDXL_1_0_LCM: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SDXL_TURBO: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SDXL_LIGHTNING: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SDXL_HYPER: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    # Pony (SDXL-based)
    CivitaiBaseModel.PONY: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="dpm++_karras",
    ),
    # Illustrious (SDXL-based)
    CivitaiBaseModel.ILLUSTRIOUS: PipelineConfig(
        pipeline_class="StableDiffusionXLPipeline",
        default_steps=25,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="dpm++_karras",
    ),
    # Flux family (flow-based, incompatible with DPM schedulers)
    CivitaiBaseModel.FLUX_1_D: PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=28,
        default_guidance_scale=3.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_1_S: PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_1_DEV: PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=28,
        default_guidance_scale=3.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_1_SCHNELL: PipelineConfig(
        pipeline_class="FluxPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_2_KLEIN_9B: PipelineConfig(
        pipeline_class="Flux2KleinPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_2_KLEIN_9B_BASE: PipelineConfig(
        pipeline_class="Flux2KleinPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_2_KLEIN_4B: PipelineConfig(
        pipeline_class="Flux2KleinPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.FLUX_2_KLEIN_4B_BASE: PipelineConfig(
        pipeline_class="Flux2KleinPipeline",
        supports_negative_prompt=False,
        default_steps=4,
        default_guidance_scale=1.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    # SD 3.x family (flow-based, incompatible with DPM schedulers)
    CivitaiBaseModel.SD_3: PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SD_3_MEDIUM: PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SD_3_5: PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=7.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SD_3_5_MEDIUM: PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SD_3_5_LARGE: PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=28,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.SD_3_5_LARGE_TURBO: PipelineConfig(
        pipeline_class="StableDiffusion3Pipeline",
        default_steps=4,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    # Other architectures
    CivitaiBaseModel.PIXART_A: PipelineConfig(
        pipeline_class="PixArtAlphaPipeline",
        default_steps=20,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.PIXART_SIGMA: PipelineConfig(
        pipeline_class="PixArtSigmaPipeline",
        default_steps=20,
        default_guidance_scale=4.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.KOLORS: PipelineConfig(
        pipeline_class="KolorsPipeline",
        default_steps=25,
        default_guidance_scale=5.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.HUNYUAN_DIT: PipelineConfig(
        pipeline_class="HunyuanDiTPipeline",
        default_steps=50,
        default_guidance_scale=5.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.LUMINA: PipelineConfig(
        pipeline_class="LuminaText2ImgPipeline",
        default_steps=30,
        default_guidance_scale=4.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.AURA_FLOW: PipelineConfig(
        pipeline_class="AuraFlowPipeline",
        default_steps=50,
        default_guidance_scale=3.5,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.Z_IMAGE: PipelineConfig(
        pipeline_class="ZImagePipeline",
        default_steps=9,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.Z_IMAGE_TURBO: PipelineConfig(
        pipeline_class="ZImagePipeline",
        default_steps=9,
        default_guidance_scale=0.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
    CivitaiBaseModel.QWEN: PipelineConfig(
        pipeline_class="QwenImagePipeline",
        default_steps=8,
        default_guidance_scale=4.0,
        default_width=1024,
        default_height=1024,
        default_scheduler="default",
    ),
}

SCHEDULER_MAP: dict[str, tuple[str | None, dict[str, Any]]] = {
    "dpm++_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "dpm++": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": False},
    ),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "default": (None, {}),
}

SCHEDULER_CHOICES: list[str] = list(SCHEDULER_MAP.keys())
DEFAULT_SDXL_COMPONENT_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_ZIMAGE_COMPONENT_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_QWEN_COMPONENT_REPO = "Qwen/Qwen-Image"
DEFAULT_FLUX2_KLEIN_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_FLUX2_KLEIN_BASE_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_FLUX2_KLEIN_4B_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_FLUX2_KLEIN_4B_BASE_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
COMFY_DIFFUSION_MODEL_PREFIX = "model.diffusion_model."

# Default fallback configuration
DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    pipeline_class="StableDiffusionXLPipeline",
    default_steps=25,
    default_guidance_scale=7.0,
    default_width=1024,
    default_height=1024,
    default_scheduler="dpm++_karras",
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
        if "flux.2" in base_lower or "flux2" in base_lower:
            if "4b" in base_lower and "base" in base_lower:
                return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.FLUX_2_KLEIN_4B_BASE]
            if "4b" in base_lower:
                return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.FLUX_2_KLEIN_4B]
            if "base" in base_lower:
                return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.FLUX_2_KLEIN_9B_BASE]
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.FLUX_2_KLEIN_9B]
        if "schnell" in base_lower or " s" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.FLUX_1_SCHNELL]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.FLUX_1_DEV]

    if "qwen" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.QWEN]

    if "sd 3.5" in base_lower or "sd3.5" in base_lower:
        if "turbo" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_3_5_LARGE_TURBO]
        if "large" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_3_5_LARGE]
        if "medium" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_3_5_MEDIUM]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_3_5]

    if "sd 3" in base_lower or "sd3" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_3]

    if "pony" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.PONY]

    if "illustrious" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.ILLUSTRIOUS]

    if "sdxl" in base_lower or "xl" in base_lower:
        if "turbo" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SDXL_TURBO]
        if "lightning" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SDXL_LIGHTNING]
        if "lcm" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SDXL_1_0_LCM]
        if "hyper" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SDXL_HYPER]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SDXL_1_0]

    if "sd 2" in base_lower or "sd2" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_2_1]

    if "sd 1.5" in base_lower or "sd1.5" in base_lower or "sd 1" in base_lower:
        if "lcm" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_1_5_LCM]
        if "hyper" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_1_5_HYPER]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.SD_1_5]

    if "pixart" in base_lower:
        if "sigma" in base_lower:
            return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.PIXART_SIGMA]
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.PIXART_A]

    if "kolors" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.KOLORS]

    if "hunyuan" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.HUNYUAN_DIT]

    if "lumina" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.LUMINA]

    if "auraflow" in base_lower or "aura" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.AURA_FLOW]

    if "z-image" in base_lower or "zimage" in base_lower or "z image" in base_lower:
        return CIVITAI_BASE_MODEL_PIPELINE_MAP[CivitaiBaseModel.Z_IMAGE_TURBO]

    raise ValueError(
        f"Unsupported CivitAI base model '{base_model}'. "
        "Set pipeline_class explicitly or add a CivitAI base-model mapping before loading."
    )


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
        self._pipeline_config: PipelineConfig | None = None
        self._base_model: str | None = None
        self._full_config: dict[str, Any] | None = None
        self._current_scheduler: str | None = None
        self._static_lora_configs: list[LoraConfig] = []
        self._cpu_offload: bool = False
        self._has_dynamic_loras: bool = False

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
        if base_model is not None:
            self._base_model = base_model

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

        cpu_offload = model_config.get("cpu_offload", True)
        self.policy = DevicePolicy.auto_detect(cpu_offload=cpu_offload)

        # Get the pipeline class unless this wrapper has a custom assembly path.
        pipeline_class: type | None = None
        if self._pipeline_config.pipeline_class not in {
            "QwenImagePipeline",
            "Flux2KleinPipeline",
        }:
            pipeline_class = get_diffusers_pipeline_class(self._pipeline_config.pipeline_class)

        # Load from single file. Some architectures publish single-file checkpoints
        # without every pipeline component; preload those components when needed.
        single_file_kwargs = self._build_single_file_kwargs(model_config)
        self.pipe = self._load_pipeline_from_single_file(
            pipeline_class,
            checkpoint_path,
            model_config,
            single_file_kwargs,
        )

        scheduler_override = model_config.get("scheduler")
        self.configure_scheduler(scheduler_override)

        self.policy.apply_to_pipeline(self.pipe)
        # Track whether offload was applied (for dynamic LoRA handling)
        self._cpu_offload = (
            self.policy.offload != OffloadMode.NEVER and self.policy.device == "cuda"
        )

        # Enable memory optimizations for VAE if available
        if hasattr(self.pipe, "vae"):
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()

        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)
            self._static_lora_configs = list(loras)
        else:
            self._static_lora_configs = []

        # Load embeddings if full_config provided
        if self._full_config:
            embeddings = parse_embeddings_from_config(self._full_config, model_config)
            if embeddings:
                print(f"  Loading {len(embeddings)} embedding(s)...")
                self.load_embeddings_sync(embeddings)

        print(f"Checkpoint loaded: {checkpoint_path.name}")

    def _load_pipeline_from_single_file(
        self,
        pipeline_class: type | None,
        checkpoint_path: Path,
        model_config: dict[str, Any],
        single_file_kwargs: dict[str, Any],
    ) -> Any:
        """Load a diffusers pipeline from one checkpoint file with component fallbacks."""
        if self._pipeline_config is not None:
            if self._pipeline_config.pipeline_class == "QwenImagePipeline":
                return self._load_qwen_image_pipeline_from_single_file(
                    checkpoint_path, model_config
                )
            if self._pipeline_config.pipeline_class == "Flux2KleinPipeline":
                return self._load_flux2_klein_pipeline_from_single_file(
                    checkpoint_path, model_config
                )

        assert pipeline_class is not None
        try:
            return pipeline_class.from_single_file(str(checkpoint_path), **single_file_kwargs)
        except Exception as error:
            if not self._should_retry_with_sdxl_text_components(error, single_file_kwargs):
                raise

            print("  Text encoder weights missing; retrying with SDXL text components")
            retry_kwargs = dict(single_file_kwargs)
            retry_kwargs.update(self._load_sdxl_text_components(model_config))
            return pipeline_class.from_single_file(str(checkpoint_path), **retry_kwargs)

    def _load_qwen_image_pipeline_from_single_file(
        self,
        checkpoint_path: Path,
        model_config: dict[str, Any],
    ) -> Any:
        """Load a Qwen CivitAI transformer checkpoint into the Qwen-Image pipeline."""
        from diffusers import (
            DiffusionPipeline,
            FlowMatchEulerDiscreteScheduler,
            QwenImageTransformer2DModel,
        )

        component_repo = (
            model_config.get("qwen_component_repo")
            or model_config.get("component_repo")
            or model_config.get("repo")
            or DEFAULT_QWEN_COMPONENT_REPO
        )
        transformer_config = model_config.get("single_file_config_repo") or component_repo
        transformer_subfolder = model_config.get("transformer_subfolder", "transformer")

        print(f"  Loading Qwen transformer from {checkpoint_path}")
        checkpoint = self._load_transformer_checkpoint(checkpoint_path)
        transformer = QwenImageTransformer2DModel.from_single_file(
            checkpoint,
            torch_dtype=self.policy.dtype,
            config=transformer_config,
            subfolder=transformer_subfolder,
        )

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self._qwen_scheduler_config())

        print(f"  Assembling Qwen-Image pipeline from {component_repo}")
        return DiffusionPipeline.from_pretrained(
            component_repo,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=self.policy.dtype,
        )

    def _load_flux2_klein_pipeline_from_single_file(
        self,
        checkpoint_path: Path,
        model_config: dict[str, Any],
    ) -> Any:
        """Load a FLUX.2 Klein transformer checkpoint into the Klein pipeline."""
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        component_repo = (
            model_config.get("flux2_component_repo")
            or model_config.get("component_repo")
            or model_config.get("repo")
            or self._default_flux2_component_repo()
        )
        transformer_config = model_config.get("single_file_config_repo") or component_repo
        transformer_subfolder = model_config.get("transformer_subfolder", "transformer")

        print(f"  Loading FLUX.2 Klein transformer from {checkpoint_path}")
        checkpoint = self._load_transformer_checkpoint(checkpoint_path)
        transformer = Flux2Transformer2DModel.from_single_file(
            checkpoint,
            torch_dtype=self.policy.dtype,
            config=transformer_config,
            subfolder=transformer_subfolder,
        )

        print(f"  Assembling FLUX.2 Klein pipeline from {component_repo}")
        return Flux2KleinPipeline.from_pretrained(
            component_repo,
            transformer=transformer,
            torch_dtype=self.policy.dtype,
        )

    def _load_transformer_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load and normalize CivitAI/Comfy transformer checkpoint keys for Diffusers.

        CivitAI/ComfyUI transformer checkpoints commonly prefix model keys with
        ``model.diffusion_model.``. Some Diffusers component loaders already
        strip that prefix, but Qwen Image uses an identity converter and FLUX.2
        expects the keyspace to start at ``double_blocks`` / ``single_blocks``.
        Strip only that known wrapper prefix before handing the state dict to
        those component loaders.
        """
        from safetensors.torch import load_file

        checkpoint = load_file(checkpoint_path, device="cpu")
        if not checkpoint:
            return checkpoint

        if not any(key.startswith(COMFY_DIFFUSION_MODEL_PREFIX) for key in checkpoint):
            return checkpoint

        normalized_checkpoint: dict[str, Any] = {}
        for key, tensor in checkpoint.items():
            normalized_key = key.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX)
            normalized_checkpoint[normalized_key] = tensor

        return normalized_checkpoint

    def _default_flux2_component_repo(self) -> str:
        """Return the default FLUX.2 Klein repo for the detected CivitAI base model."""
        base_model = (self._base_model or "").lower()
        if "4b" in base_model and "base" in base_model:
            return DEFAULT_FLUX2_KLEIN_4B_BASE_COMPONENT_REPO
        if "4b" in base_model:
            return DEFAULT_FLUX2_KLEIN_4B_COMPONENT_REPO
        if "base" in base_model:
            return DEFAULT_FLUX2_KLEIN_BASE_COMPONENT_REPO
        return DEFAULT_FLUX2_KLEIN_COMPONENT_REPO

    def _qwen_scheduler_config(self) -> dict[str, Any]:
        """Return the scheduler config used by Qwen-Image Diffusers pipelines."""
        return {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

    def _should_retry_with_sdxl_text_components(
        self,
        error: Exception,
        single_file_kwargs: dict[str, Any],
    ) -> bool:
        """Return whether an SDXL checkpoint should retry with explicit text components."""
        if self._pipeline_config is None:
            return False
        if self._pipeline_config.pipeline_class != "StableDiffusionXLPipeline":
            return False
        if "text_encoder" in single_file_kwargs or "text_encoder_2" in single_file_kwargs:
            return False

        message = str(error)
        return "Weights for this component appear to be missing" in message and (
            "CLIPTextModel" in message or "CLIPTextModelWithProjection" in message
        )

    def _build_single_file_kwargs(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Build keyword arguments for diffusers ``from_single_file`` loading.

        Z-Image checkpoints commonly store the transformer weights in the single
        file while the Qwen3 text encoder lives in the base Hugging Face repo.
        Diffusers requires callers to preload and inject that component instead
        of expecting it to be reconstructed from the checkpoint.

        Args:
            model_config: Model configuration from TOML/state.

        Returns:
            Keyword arguments to pass to ``pipeline_class.from_single_file``.
        """
        kwargs: dict[str, Any] = {"torch_dtype": self.policy.dtype}

        if (
            self._pipeline_config is None
            or self._pipeline_config.pipeline_class != "ZImagePipeline"
        ):
            return kwargs

        kwargs.update(self._load_zimage_text_components(model_config))
        return kwargs

    def _load_sdxl_text_components(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Load SDXL text components when a single-file checkpoint omits them."""
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        component_repo = model_config.get("sdxl_component_repo") or model_config.get(
            "component_repo"
        )
        if component_repo is None:
            component_repo = DEFAULT_SDXL_COMPONENT_REPO

        text_encoder_repo = model_config.get("text_encoder_repo") or component_repo
        text_encoder_2_repo = model_config.get("text_encoder_2_repo") or component_repo
        tokenizer_repo = model_config.get("tokenizer_repo") or component_repo
        tokenizer_2_repo = model_config.get("tokenizer_2_repo") or component_repo

        text_encoder_subfolder = model_config.get("text_encoder_subfolder", "text_encoder")
        text_encoder_2_subfolder = model_config.get("text_encoder_2_subfolder", "text_encoder_2")
        tokenizer_subfolder = model_config.get("tokenizer_subfolder", "tokenizer")
        tokenizer_2_subfolder = model_config.get("tokenizer_2_subfolder", "tokenizer_2")
        single_file_config = model_config.get("single_file_config_repo") or component_repo

        print(f"  Loading SDXL text encoder from {text_encoder_repo}/{text_encoder_subfolder}")
        text_encoder = CLIPTextModel.from_pretrained(
            text_encoder_repo,
            subfolder=text_encoder_subfolder,
            torch_dtype=self.policy.dtype,
        )

        print(
            f"  Loading SDXL text encoder 2 from {text_encoder_2_repo}/{text_encoder_2_subfolder}"
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            text_encoder_2_repo,
            subfolder=text_encoder_2_subfolder,
            torch_dtype=self.policy.dtype,
        )

        print(f"  Loading SDXL tokenizer from {tokenizer_repo}/{tokenizer_subfolder}")
        tokenizer = CLIPTokenizer.from_pretrained(
            tokenizer_repo,
            subfolder=tokenizer_subfolder,
        )

        print(f"  Loading SDXL tokenizer 2 from {tokenizer_2_repo}/{tokenizer_2_subfolder}")
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            tokenizer_2_repo,
            subfolder=tokenizer_2_subfolder,
        )

        return {
            "config": single_file_config,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
        }

    def _load_zimage_text_components(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Load Z-Image text components that are not stored in single-file checkpoints."""
        from transformers import AutoTokenizer, Qwen3Model

        component_repo = model_config.get("component_repo") or model_config.get("repo")
        text_encoder_repo = model_config.get("text_encoder_repo") or component_repo
        tokenizer_repo = model_config.get("tokenizer_repo") or component_repo

        if text_encoder_repo is None:
            text_encoder_repo = DEFAULT_ZIMAGE_COMPONENT_REPO
        if tokenizer_repo is None:
            tokenizer_repo = text_encoder_repo

        text_encoder_subfolder = model_config.get("text_encoder_subfolder", "text_encoder")
        tokenizer_subfolder = model_config.get("tokenizer_subfolder", "tokenizer")

        print(f"  Loading Z-Image text encoder from {text_encoder_repo}/{text_encoder_subfolder}")
        text_encoder = Qwen3Model.from_pretrained(
            text_encoder_repo,
            subfolder=text_encoder_subfolder,
            torch_dtype=self.policy.dtype,
        )

        print(f"  Loading Z-Image tokenizer from {tokenizer_repo}/{tokenizer_subfolder}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repo,
            subfolder=tokenizer_subfolder,
        )

        return {"text_encoder": text_encoder, "tokenizer": tokenizer}

    def configure_scheduler(self, scheduler_name: str | None) -> None:
        if self.pipe is None or self._pipeline_config is None:
            return

        if scheduler_name is None or scheduler_name == "default":
            scheduler_name = self._pipeline_config.default_scheduler

        if scheduler_name is None or scheduler_name == "default":
            return

        if scheduler_name == self._current_scheduler:
            return

        if scheduler_name not in SCHEDULER_MAP:
            print(f"  Warning: Unknown scheduler '{scheduler_name}', keeping default")
            return

        class_name, kwargs = SCHEDULER_MAP[scheduler_name]
        if class_name is None:
            self._current_scheduler = scheduler_name
            return

        import diffusers

        scheduler_class = getattr(diffusers, class_name)
        self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config, **kwargs)
        self._current_scheduler = scheduler_name
        print(f"  Scheduler: {scheduler_name}")

    def validate_pipeline(self) -> None:
        """Validate pipeline and config are ready for generation."""
        super().validate_pipeline()
        if self._pipeline_config is None:
            raise RuntimeError("Pipeline config not initialized")

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
        """
        Run a text-to-image or image-to-image generation using the loaded CivitAI
        checkpoint-backed diffusers pipeline.

        This method uses defaults from the resolved pipeline configuration for
        image size, number of inference steps, and guidance scale when those
        values are not provided explicitly.

        Parameters
        ----------
        prompt:
            The primary prompt describing the desired image.
        negative_prompt:
            An optional negative prompt used to steer the model away from
            unwanted content. This is only applied if the underlying pipeline
            supports negative prompts.
        width:
            Target image width in pixels. If omitted, the pipeline's
            ``default_width`` is used for text-to-image generation.
        height:
            Target image height in pixels. If omitted, the pipeline's
            ``default_height`` is used for text-to-image generation.
        seed:
            Random seed used to create the diffusion generator. A value of
            ``-1`` indicates that a random seed should be chosen.
        steps:
            Number of diffusion inference steps. If omitted, the pipeline's
            ``default_steps`` is used.
        guidance_scale:
            Classifier-free guidance scale. If ``None``, the pipeline's
            ``default_guidance_scale`` is used.
        **kwargs:
            Additional options forwarded to the underlying diffusers pipeline.
            Recognized options in this wrapper include:

            - ``scheduler``: Optional scheduler name to override the pipeline's
              default scheduler. Must be a key in ``SCHEDULER_MAP``; if
              provided, :meth:`configure_scheduler` is called before
              generation.
            - ``init_image``: Optional initial image (path-like, PIL image, or
              other supported type) to enable image-to-image (img2img) mode.
              When provided, the method will call :meth:`_load_init_image` and
              pass the resulting image to the pipeline.
            - ``strength``: Strength parameter for img2img generation, used
              when ``init_image`` is provided. Defaults to ``0.75``.
            - ``loras``: List of LoraConfig objects for dynamic LoRA loading.
              These are loaded before generation and unloaded after.

            Any other keyword arguments are passed through unchanged to the
            underlying diffusers pipeline call and may be used to control
            advanced generation options.

        Returns
        -------
        GenerationResult
            An object containing the generated image along with metadata such
            as the actual seed used, prompts, final image size, number of
            steps, and guidance scale.
        """
        # Apply defaults from pipeline config (validation happens in super().generate())
        # Note: We need to check _pipeline_config here before applying defaults,
        # but full validation happens in validate_pipeline() called by super()
        if self._pipeline_config is not None:
            width = width if width is not None else self._pipeline_config.default_width
            height = height if height is not None else self._pipeline_config.default_height
            steps = steps if steps is not None else self._pipeline_config.default_steps
            guidance_scale = (
                guidance_scale
                if guidance_scale is not None
                else self._pipeline_config.default_guidance_scale
            )

        return super().generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width or 1024,
            height=height or 1024,
            seed=seed,
            steps=steps or 20,
            guidance_scale=guidance_scale if guidance_scale is not None else 7.0,
            **kwargs,
        )

    def pre_generate(self, **kwargs: Any) -> None:
        """Pre-generation setup: scheduler override and dynamic LoRA loading."""
        scheduler_override = kwargs.pop("scheduler", None)
        if scheduler_override:
            self.configure_scheduler(scheduler_override)

        dynamic_loras = kwargs.pop("loras", None)
        self._has_dynamic_loras = False
        if dynamic_loras:
            self._has_dynamic_loras = True
            try:
                self._load_dynamic_loras(dynamic_loras)
            except Exception:
                self._restore_static_loras()
                self._has_dynamic_loras = False
                raise

    def build_generation_kwargs(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator: "torch.Generator",
        init_image: "Image.Image | None",
        strength: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build generation kwargs with embedding support."""
        assert self._pipeline_config is not None

        if self._pipeline_config.pipeline_class == "QwenImagePipeline":
            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if negative_prompt else " ",
                "num_inference_steps": steps,
                "true_cfg_scale": kwargs.get("true_cfg_scale", guidance_scale),
                "generator": generator,
                "num_images_per_prompt": 1,
            }

            if init_image:
                print(f"CivitAI Qwen img2img: '{prompt[:50]}...' strength={strength}")
                gen_kwargs["image"] = init_image
                gen_kwargs["strength"] = strength
            else:
                print(f"CivitAI Qwen generating: '{prompt[:50]}...'")
                gen_kwargs["height"] = height
                gen_kwargs["width"] = width

            return gen_kwargs

        gen_kwargs: dict[str, Any] = {
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # Use embedding-based prompt handling for pipelines that support it
        if self._supports_prompt_embeddings():
            gen_kwargs.update(self._encode_prompts_to_embeddings(prompt, negative_prompt))
        else:
            # Fallback for unsupported pipelines
            gen_kwargs["prompt"] = prompt
            if self._pipeline_config.supports_negative_prompt and negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt

        if init_image:
            print(f"CivitAI img2img: '{prompt[:50]}...' strength={strength}")
            gen_kwargs["image"] = init_image
            gen_kwargs["strength"] = strength
        else:
            print(f"CivitAI generating: '{prompt[:50]}...'")
            gen_kwargs["height"] = height
            gen_kwargs["width"] = width

        return gen_kwargs

    def post_generate(self, **kwargs: Any) -> None:
        """Post-generation cleanup: reset model state and restore static LoRAs."""
        super().post_generate(**kwargs)
        if self._has_dynamic_loras:
            self._restore_static_loras()
            self._has_dynamic_loras = False

    def _load_dynamic_loras(self, loras: list[LoraConfig]) -> None:
        if self.pipe is None or not loras:
            return

        lora_configs: list[LoraConfig] = [cfg for cfg in loras if isinstance(cfg, LoraConfig)]
        if not lora_configs:
            return

        self.unload_loras()

        # Only move pipeline to device manually when CPU offload is not enabled.
        # With CPU offload, diffusers manages device placement automatically.
        if not self._cpu_offload:
            self.pipe.to(self.policy.device)

        loaded_names: list[str] = []
        loaded_weights: list[float] = []

        for lora in lora_configs:
            try:
                name = self.load_single_lora(lora)
                loaded_names.append(name)
                loaded_weights.append(lora.weight)
                print(f"Loaded dynamic LoRA: {lora.name} (weight={lora.weight})")
            except Exception as e:
                print(f"Warning: Failed to load LoRA {lora.name}: {e}")

        if loaded_names:
            self.set_lora_adapters(loaded_names, loaded_weights)

    def _restore_static_loras(self) -> None:
        self.unload_loras()

        if not self._static_lora_configs:
            return

        if not self._cpu_offload:
            self.pipe.to(self.policy.device)
        self.load_loras_sync(self._static_lora_configs)
        print(f"Restored {len(self._static_lora_configs)} static LoRA(s)")

    @property
    def pipeline_config(self) -> PipelineConfig | None:
        """Get the current pipeline configuration."""
        return self._pipeline_config

    @property
    def detected_base_model(self) -> str | None:
        """Get the detected/configured base model."""
        return self._base_model

    def _supports_prompt_embeddings(self) -> bool:
        """Check if this pipeline supports embedding-based prompt handling.

        Returns True for pipelines that support pre-computed embeddings with
        weight handling (CLIP-based: SD 1.x, SD 2.x, SDXL; flow-based: Flux;
        and MMDiT-based: SD3).

        Returns:
            True if the pipeline supports prompt embeddings
        """
        if self.pipe is None or self._pipeline_config is None:
            return False

        # Pipelines that support embedding-based prompt handling
        pipeline_class = self._pipeline_config.pipeline_class
        supported_pipelines = {
            "StableDiffusionPipeline",
            "StableDiffusionXLPipeline",
            "FluxPipeline",
            "StableDiffusion3Pipeline",
        }

        return pipeline_class in supported_pipelines

    def _encode_prompts_to_embeddings(
        self,
        prompt: str,
        negative_prompt: str | None,
    ) -> dict[str, Any]:
        """Encode prompts to embeddings with weight and chunking support.

        Converts text prompts to pre-computed embeddings, supporting:
        - A1111-style weight syntax like (word:1.5) and [word]
        - Prompts longer than CLIP's 77-token limit via chunking
        - BREAK keyword for forcing chunk boundaries

        This method handles all supported pipelines (SD 1.x/2.x, SDXL, Flux, SD3)
        with appropriate embedding generation for each architecture.

        Args:
            prompt: The positive prompt
            negative_prompt: The negative prompt (may be None)

        Returns:
            Dict of embedding kwargs to pass to the pipeline
        """
        if self.pipe is None or self._pipeline_config is None:
            return {}

        neg_prompt = negative_prompt or ""
        pipeline_class = self._pipeline_config.pipeline_class
        result: dict[str, Any] = {}

        if pipeline_class == "FluxPipeline":
            # Flux uses T5 for main embeddings + CLIP for pooled
            # Note: Flux does not support negative prompts
            prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux(
                self.pipe,
                prompt=prompt,
            )

            result["prompt_embeds"] = prompt_embeds
            result["pooled_prompt_embeds"] = pooled_prompt_embeds

        elif pipeline_class == "StableDiffusion3Pipeline":
            # SD3 uses dual CLIP + T5 encoders
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sd3(
                self.pipe,
                prompt=prompt,
                negative_prompt=neg_prompt,
            )

            result["prompt_embeds"] = prompt_embeds
            result["pooled_prompt_embeds"] = pooled_prompt_embeds

            if self._pipeline_config.supports_negative_prompt:
                result["negative_prompt_embeds"] = negative_prompt_embeds
                result["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        elif pipeline_class == "StableDiffusionXLPipeline":
            # SDXL uses dual text encoders
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sdxl(
                self.pipe,
                prompt=prompt,
                negative_prompt=neg_prompt,
            )

            result["prompt_embeds"] = prompt_embeds
            result["pooled_prompt_embeds"] = pooled_prompt_embeds

            if self._pipeline_config.supports_negative_prompt:
                result["negative_prompt_embeds"] = negative_prompt_embeds
                result["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        else:
            # SD 1.x / 2.x use single text encoder
            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings_sd15(
                self.pipe,
                prompt=prompt,
                negative_prompt=neg_prompt,
            )

            result["prompt_embeds"] = prompt_embeds

            if self._pipeline_config.supports_negative_prompt:
                result["negative_prompt_embeds"] = negative_prompt_embeds

        return result
