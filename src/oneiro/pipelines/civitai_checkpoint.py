"""Convert CivitAI checkpoints to components for the shared native lifecycle."""

import asyncio
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)
from PIL import Image

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import GenerationResult
from oneiro.pipelines.embedding import parse_embeddings_from_config
from oneiro.pipelines.krea2_checkpoint import (
    COMFY_DIFFUSION_MODEL_PREFIX,
    get_krea2_checkpoint_precision,
    get_krea2_checkpoint_precision_from_header,
    load_krea2_transformer,
)
from oneiro.pipelines.long_prompt import (
    get_weighted_text_embeddings_flux,
    get_weighted_text_embeddings_sd3,
    get_weighted_text_embeddings_sdxl,
)
from oneiro.pipelines.modular import ModularPipelineWrapper
from oneiro.pipelines.zimage import ZImagePipelineWrapper

if TYPE_CHECKING:
    from oneiro.civitai import CivitaiClient, ModelVersion

# Re-export precision readers until command imports are migrated in Task 6.
__all__ = ["get_krea2_checkpoint_precision", "get_krea2_checkpoint_precision_from_header"]


class CivitaiBaseModel(StrEnum):
    """Retained CivitAI checkpoint architectures."""

    SDXL_0_9 = "SDXL 0.9"
    SDXL_1_0 = "SDXL 1.0"
    SDXL_1_0_LCM = "SDXL 1.0 LCM"
    SDXL_DISTILLED = "SDXL Distilled"
    SDXL_TURBO = "SDXL Turbo"
    SDXL_LIGHTNING = "SDXL Lightning"
    SDXL_HYPER = "SDXL Hyper"
    PONY = "Pony"
    PONY_V6 = "Pony V6"
    PONY_V6_XL = "Pony V6 XL"
    ILLUSTRIOUS = "Illustrious"
    ILLUSTRIOUS_XL = "Illustrious XL"
    FLUX_1_D = "Flux.1 D"
    FLUX_1_S = "Flux.1 S"
    FLUX_1_DEV = "Flux.1 Dev"
    FLUX_1_SCHNELL = "Flux.1 Schnell"
    FLUX_2 = "Flux.2"
    FLUX_2_KLEIN_9B = "Flux.2 Klein 9B"
    FLUX_2_KLEIN_9B_BASE = "Flux.2 Klein 9B-base"
    FLUX_2_KLEIN_4B = "Flux.2 Klein 4B"
    FLUX_2_KLEIN_4B_BASE = "Flux.2 Klein 4B-base"
    QWEN = "Qwen"
    SD_3 = "SD 3"
    SD_3_MEDIUM = "SD 3 Medium"
    SD_3_5 = "SD 3.5"
    SD_3_5_MEDIUM = "SD 3.5 Medium"
    SD_3_5_LARGE = "SD 3.5 Large"
    SD_3_5_LARGE_TURBO = "SD 3.5 Large Turbo"
    Z_IMAGE = "Z-Image"
    Z_IMAGE_TURBO = "Z-Image Turbo"
    KREA_2 = "Krea 2"


@dataclass
class PipelineConfig:
    """Resolved family and defaults; classic class names are conversion metadata only."""

    family: str
    pipeline_class: str
    supports_negative_prompt: bool = True
    default_steps: int = 20
    default_guidance_scale: float = 7.5
    default_width: int = 1024
    default_height: int = 1024
    requires_safety_checker: bool = False
    default_scheduler: str | None = None


CIVITAI_BASE_MODEL_PIPELINE_MAP: dict[str, PipelineConfig] = {
    **{
        name: PipelineConfig(
            "sdxl",
            "StableDiffusionXLPipeline",
            default_steps=25,
            default_guidance_scale=7.0,
            default_scheduler="dpm++_karras",
        )
        for name in (
            "SDXL 0.9",
            "SDXL 1.0",
            "Pony",
            "Pony V6",
            "Pony V6 XL",
            "Illustrious",
            "Illustrious XL",
        )
    },
    **{
        name: PipelineConfig(
            "sdxl",
            "StableDiffusionXLPipeline",
            default_steps=4,
            default_guidance_scale=guidance,
            default_scheduler="default",
        )
        for name, guidance in (
            ("SDXL 1.0 LCM", 1.0),
            ("SDXL Distilled", 1.0),
            ("SDXL Turbo", 0.0),
            ("SDXL Lightning", 0.0),
            ("SDXL Hyper", 0.0),
        )
    },
    **{
        name: PipelineConfig(
            "flux1",
            "FluxPipeline",
            supports_negative_prompt=False,
            default_steps=steps,
            default_guidance_scale=guidance,
        )
        for name, steps, guidance in (
            ("Flux.1 D", 28, 3.5),
            ("Flux.1 Dev", 28, 3.5),
            ("Flux.1 S", 4, 0.0),
            ("Flux.1 Schnell", 4, 0.0),
        )
    },
    "Flux.2": PipelineConfig(
        "flux2",
        "Flux2Pipeline",
        supports_negative_prompt=False,
        default_steps=50,
        default_guidance_scale=4.0,
    ),
    **{
        name: PipelineConfig(
            "flux2-klein",
            "Flux2KleinPipeline",
            supports_negative_prompt=False,
            default_steps=steps,
            default_guidance_scale=guidance,
        )
        for name, steps, guidance in (
            ("Flux.2 Klein 9B", 4, 1.0),
            ("Flux.2 Klein 4B", 4, 1.0),
            ("Flux.2 Klein 9B-base", 50, 4.0),
            ("Flux.2 Klein 4B-base", 50, 4.0),
        )
    },
    **{
        name: PipelineConfig(
            "sd3", "StableDiffusion3Pipeline", default_steps=steps, default_guidance_scale=guidance
        )
        for name, steps, guidance in (
            ("SD 3", 28, 7.0),
            ("SD 3 Medium", 28, 7.0),
            ("SD 3.5", 28, 7.0),
            ("SD 3.5 Medium", 28, 4.5),
            ("SD 3.5 Large", 28, 4.5),
            ("SD 3.5 Large Turbo", 4, 0.0),
        )
    },
    "Qwen": PipelineConfig(
        "qwen", "QwenImagePipeline", default_steps=8, default_guidance_scale=4.0
    ),
    "Krea 2": PipelineConfig("krea2", "Krea2Pipeline", default_steps=8, default_guidance_scale=0.0),
    **{
        name: PipelineConfig(
            "zimage", "ZImagePipeline", default_steps=9, default_guidance_scale=0.0
        )
        for name in ("Z-Image", "Z-Image Turbo")
    },
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
SCHEDULER_CHOICES = list(SCHEDULER_MAP)
DEFAULT_SDXL_COMPONENT_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_ZIMAGE_COMPONENT_REPO = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_QWEN_COMPONENT_REPO = "Qwen/Qwen-Image"
DEFAULT_KREA2_COMPONENT_REPO = "krea/Krea-2-Turbo"
DEFAULT_KREA2_RAW_COMPONENT_REPO = "krea/Krea-2-Raw"
DEFAULT_FLUX2_KLEIN_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-9B"
DEFAULT_FLUX2_KLEIN_BASE_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_FLUX2_KLEIN_4B_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_FLUX2_KLEIN_4B_BASE_COMPONENT_REPO = "black-forest-labs/FLUX.2-klein-base-4B"


def get_pipeline_config_for_base_model(base_model: str | None) -> PipelineConfig:
    """Reject missing/unknown/removed metadata instead of guessing an architecture."""
    if isinstance(base_model, str):
        normalized = base_model.strip().casefold()
        aliases = {
            "krea-2": "krea 2",
            "qwen-image": "qwen",
            "zimage": "z-image",
            "zimageturbo": "z-image turbo",
            "zimage turbo": "z-image turbo",
            "z image turbo": "z-image turbo",
            "flux.2 d": "flux.2",
            "flux.1 krea": "flux.1 dev",
        }
        normalized = aliases.get(normalized, normalized)
        for name, config in CIVITAI_BASE_MODEL_PIPELINE_MAP.items():
            if name.casefold() == normalized:
                return config
    raise ValueError(
        f"Unsupported or missing CivitAI base model '{base_model}'; set a known base_model"
    )


def get_krea2_generation_defaults(component_repo: str) -> tuple[int, float]:
    """Return official Krea defaults, never infer custom-source variants from substrings."""
    if component_repo == DEFAULT_KREA2_RAW_COMPONENT_REPO:
        return 28, 4.5
    if component_repo == DEFAULT_KREA2_COMPONENT_REPO:
        return 8, 0.0
    raise ValueError("Custom Krea sources require explicit variant metadata")


def get_diffusers_pipeline_class(class_name: str) -> type:
    """Import a released single-file conversion container, not a generation backend."""
    import diffusers

    if not hasattr(diffusers, class_name):
        raise ImportError(f"Pipeline class '{class_name}' not found in diffusers")
    return getattr(diffusers, class_name)


class _CheckpointEmbeddingsStep(ModularPipelineBlocks):
    """Keep native text-component specs while accepting the weighted helper's outputs."""

    def __init__(self, text_encoder: ModularPipelineBlocks) -> None:
        """Retain original specs without retaining a second encoder execution path."""
        super().__init__()
        self.text_encoder = text_encoder

    @property
    def expected_components(self) -> list[ComponentSpec]:
        """Own the same encoders, tokenizers and guider as the original native step."""
        return self.text_encoder.expected_components

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        """Preserve the original native text configuration."""
        return self.text_encoder.expected_configs

    @property
    def inputs(self) -> list[InputParam]:
        """Declare embeddings with their original downstream field tags."""
        return [
            InputParam(output.name, type_hint=output.type_hint, kwargs_type=output.kwargs_type)
            for output in self.intermediate_outputs
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        """Pass precomputed tensors through to the unchanged native downstream blocks."""
        return self.text_encoder.intermediate_outputs

    def __call__(self, pipeline: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        """Leave declared embeddings in the native state without encoding them twice."""
        return pipeline, state


class CivitaiCheckpointPipeline(ModularPipelineWrapper):
    """Resolve metadata first, then inject converted components into native family graphs."""

    def __init__(self) -> None:
        """Initialize shared resource state and metadata-only checkpoint ownership."""
        super().__init__()
        self._pipeline_config: PipelineConfig | None = None
        self._base_model: str | None = None
        self._resolved_version: ModelVersion | None = None
        self.inpaint_pipe: Any = None
        self.variant: str | None = None

    async def resolve_config(
        self,
        model_config: dict[str, Any],
        civitai_client: "CivitaiClient | None",
    ) -> dict[str, Any]:
        """Preflight only metadata and recipe controls; retain the version for download."""
        resolved = dict(model_config)
        if not resolved.get("checkpoint_path") and self._resolved_version is None:
            if not resolved.get("civitai_model_id"):
                raise ValueError("civitai_model_id required when checkpoint_path not provided")
            if civitai_client is None:
                raise ValueError("CivitaiClient required when checkpoint_path not provided")
            if resolved.get("civitai_version_id"):
                self._resolved_version = await civitai_client.get_model_version(
                    resolved["civitai_version_id"]
                )
            else:
                model = await civitai_client.get_model(resolved["civitai_model_id"])
                if model.latest_version is None:
                    raise ValueError(
                        f"No versions available for model {resolved['civitai_model_id']}"
                    )
                self._resolved_version = model.latest_version
        if self._resolved_version is not None:
            # Invalid remote metadata is not bypassed by an unrestricted local override.
            actual = get_pipeline_config_for_base_model(self._resolved_version.base_model)
            if "base_model" in resolved:
                configured = get_pipeline_config_for_base_model(resolved["base_model"])
                if configured.family != actual.family:
                    raise ValueError("Configured base model conflicts with CivitAI metadata")
            resolved["base_model"] = self._resolved_version.base_model
        self._configure_recipe(resolved)
        resolved.update(
            base_model=self._base_model,
            family=self.family,
            component_repo=self._component_repo,
            variant=self.variant,
        )
        return resolved

    def _configure_recipe(self, model_config: dict[str, Any]) -> None:
        """Select original native blocks from known metadata and explicit source variants."""
        from diffusers import (
            Flux2AutoBlocks,
            Flux2KleinAutoBlocks,
            Flux2KleinBaseAutoBlocks,
            FluxAutoBlocks,
            QwenImageAutoBlocks,
            StableDiffusion3AutoBlocks,
            StableDiffusionXLAutoBlocks,
            ZImageAutoBlocks,
        )

        from oneiro.pipelines.backports.krea2 import Krea2AutoBlocks, Krea2TurboAutoBlocks

        if "pipeline_class" in model_config or "sequential_cpu_offload" in model_config:
            raise ValueError(
                "Use base_model and offload_type, not legacy pipeline/offload overrides"
            )
        self._base_model = model_config.get("base_model")
        config = get_pipeline_config_for_base_model(self._base_model)
        self.family = config.family
        default_variant = None
        known: dict[str, str] = {}
        if self.family == "sdxl":
            repo, blocks = DEFAULT_SDXL_COMPONENT_REPO, StableDiffusionXLAutoBlocks()
        elif self.family == "sd3":
            repo = {
                "sd 3.5 medium": "stabilityai/stable-diffusion-3.5-medium",
                "sd 3.5 large": "stabilityai/stable-diffusion-3.5-large",
                "sd 3.5 large turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
                "sd 3.5": "stabilityai/stable-diffusion-3.5-large",
            }.get(
                self._base_model.strip().casefold(),
                "stabilityai/stable-diffusion-3-medium-diffusers",
            )
            blocks = StableDiffusion3AutoBlocks()
        elif self.family == "flux1":
            default_variant = "schnell" if config.default_steps == 4 else "dev"
            repo = f"black-forest-labs/FLUX.1-{default_variant}"
            known = {
                "black-forest-labs/FLUX.1-dev": "dev",
                "black-forest-labs/FLUX.1-schnell": "schnell",
            }
            blocks = FluxAutoBlocks()
        elif self.family == "flux2":
            repo, blocks = "black-forest-labs/FLUX.2-dev", Flux2AutoBlocks()
        elif self.family == "flux2-klein":
            repo = self._default_flux2_component_repo()
            known = {
                DEFAULT_FLUX2_KLEIN_COMPONENT_REPO: "distilled",
                DEFAULT_FLUX2_KLEIN_4B_COMPONENT_REPO: "distilled",
                DEFAULT_FLUX2_KLEIN_BASE_COMPONENT_REPO: "base",
                DEFAULT_FLUX2_KLEIN_4B_BASE_COMPONENT_REPO: "base",
            }
            default_variant = known[repo]
            blocks = None
        elif self.family == "krea2":
            repo = DEFAULT_KREA2_COMPONENT_REPO
            known = {repo: "turbo", DEFAULT_KREA2_RAW_COMPONENT_REPO: "raw"}
            default_variant, blocks = "turbo", None
        elif self.family == "qwen":
            repo, blocks = DEFAULT_QWEN_COMPONENT_REPO, QwenImageAutoBlocks()
            known = {repo: "image", "Qwen/Qwen-Image-2512": "image"}
            default_variant = "image"
        else:
            repo, blocks = DEFAULT_ZIMAGE_COMPONENT_REPO, ZImageAutoBlocks()
            known, default_variant = {repo: "turbo"}, "turbo"
        component_repo = (
            model_config.get(f"{self.family.replace('-', '_')}_component_repo")
            or (model_config.get("flux2_component_repo") if self.family == "flux2-klein" else None)
            or model_config.get("component_repo")
            or model_config.get("repo")
            or repo
        )
        variant = model_config.get(
            "variant", known.get(component_repo) if known else default_variant
        )
        if known and (
            variant not in set(known.values())
            or (component_repo in known and variant != known[component_repo])
        ):
            raise ValueError(
                f"{self.family} requires an explicit matching variant for custom sources"
            )
        if self.family == "krea2":
            blocks = Krea2TurboAutoBlocks() if variant == "turbo" else Krea2AutoBlocks()
            config = replace(
                config,
                default_steps=8 if variant == "turbo" else 28,
                default_guidance_scale=0.0 if variant == "turbo" else 4.5,
            )
        if self.family == "flux2-klein":
            blocks = (
                Flux2KleinAutoBlocks() if variant == "distilled" else Flux2KleinBaseAutoBlocks()
            )
            config = replace(
                config,
                default_steps=4 if variant == "distilled" else 50,
                default_guidance_scale=1.0 if variant == "distilled" else 4.0,
            )
        if self.family == "flux1":
            config = replace(
                config,
                default_steps=4 if variant == "schnell" else 28,
                default_guidance_scale=0.0 if variant == "schnell" else 3.5,
            )
        self._pipeline_config = replace(
            config,
            default_steps=model_config.get("steps", config.default_steps),
            default_guidance_scale=model_config.get(
                "guidance_scale", config.default_guidance_scale
            ),
            default_width=model_config.get("width", config.default_width),
            default_height=model_config.get("height", config.default_height),
        )
        self.default_steps = self._pipeline_config.default_steps
        self.default_guidance_scale = self._pipeline_config.default_guidance_scale
        self._component_repo, self.variant, self.blocks = component_repo, variant, blocks
        self._validate_scheduler(model_config.get("scheduler"))

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load a local checkpoint through the same lifecycle as downloaded checkpoints."""
        if not model_config.get("checkpoint_path"):
            raise ValueError("checkpoint_path required for synchronous load. Use load_async()")
        self._load_from_path(Path(model_config["checkpoint_path"]), model_config, full_config)

    async def load_async(
        self,
        model_config: dict[str, Any],
        civitai_client: "CivitaiClient",
        full_config: dict[str, Any] | None = None,
    ) -> None:
        """Download only after preflight; keep conversion and native loading in a worker."""
        resolved = await self.resolve_config(model_config, civitai_client)
        if not resolved.get("checkpoint_path"):
            assert self._resolved_version is not None
            resolved["checkpoint_path"] = await civitai_client.download_model_version(
                self._resolved_version
            )
        await asyncio.to_thread(self.load, resolved, full_config)

    def _load_from_path(
        self,
        checkpoint_path: Path,
        model_config: dict[str, Any],
        full_config: dict[str, Any] | None = None,
    ) -> None:
        """Discard conversion containers before shared placement and resource loading."""
        self._configure_recipe(model_config)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.policy = DevicePolicy.auto_detect(
            cpu_offload=model_config.get("cpu_offload", True),
            offload_type=model_config.get("offload_type", "group"),
            group_offload_type=model_config.get("group_offload_type", "leaf_level"),
            group_offload_use_stream=model_config.get("group_offload_use_stream", True),
            group_offload_num_blocks_per_group=model_config.get(
                "group_offload_num_blocks_per_group"
            ),
        )
        try:
            components = self._load_checkpoint_components(checkpoint_path, model_config)
            if self.family in {"sdxl", "sd3", "flux1"}:
                self.blocks.sub_blocks["text_encoder"] = _CheckpointEmbeddingsStep(
                    self.blocks.sub_blocks["text_encoder"]
                )
            self.initialize_pipeline(self._component_repo, self.blocks, components)
            if self.family == "flux2-klein":
                self.pipe.register_to_config(is_distilled=self.variant == "distilled")
            self.configure_scheduler(
                model_config.get("scheduler") or self._pipeline_config.default_scheduler
            )
            vae = self.pipe.components.get("vae")
            for optimization in ("enable_tiling", "enable_slicing"):
                if callable(getattr(vae, optimization, None)):
                    getattr(vae, optimization)()
            embeddings = model_config.get("_resolved_embeddings")
            if embeddings is None:
                embeddings = parse_embeddings_from_config(full_config or {}, model_config)
            if embeddings:
                self.load_embeddings_sync(embeddings)
        except Exception:
            self.unload()
            raise

    def _load_checkpoint_components(
        self,
        checkpoint_path: Path,
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Use single-file containers only where released component converters need them."""
        if self.family in {"sdxl", "sd3", "flux1"}:
            loader = get_diffusers_pipeline_class(self._pipeline_config.pipeline_class)
            kwargs: dict[str, Any] = {"torch_dtype": self.policy.dtype}
            if model_config.get("single_file_config_repo"):
                kwargs["config"] = model_config["single_file_config_repo"]
            try:
                container = loader.from_single_file(str(checkpoint_path), **kwargs)
            except Exception as error:
                if not self._should_retry_with_sdxl_text_components(error, kwargs):
                    raise
                kwargs.update(self._load_sdxl_text_components(model_config))
                container = loader.from_single_file(str(checkpoint_path), **kwargs)
            names = {spec.name for spec in self.blocks.expected_components}
            components = {
                name: value
                for name, value in container.components.items()
                if name in names and value is not None
            }
            del container
            return components
        if self.family == "krea2":
            from oneiro.pipelines.krea2 import load_krea2_tokenizer

            transformer, self.policy = load_krea2_transformer(
                checkpoint_path,
                self._component_repo,
                model_config.get("transformer_subfolder", "transformer"),
                self.policy,
            )
            return {
                "transformer": transformer,
                "tokenizer": load_krea2_tokenizer(self._component_repo),
            }
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            Flux2Transformer2DModel,
            QwenImageTransformer2DModel,
            ZImageTransformer2DModel,
        )

        loader = {
            "qwen": QwenImageTransformer2DModel,
            "flux2": Flux2Transformer2DModel,
            "flux2-klein": Flux2Transformer2DModel,
            "zimage": ZImageTransformer2DModel,
        }[self.family]
        kwargs = {
            "torch_dtype": self.policy.dtype,
            "config": model_config.get("single_file_config_repo") or self._component_repo,
            "subfolder": model_config.get("transformer_subfolder", "transformer"),
        }
        if checkpoint_path.suffix.lower() == ".gguf":
            from diffusers import GGUFQuantizationConfig

            checkpoint: Any = str(checkpoint_path)
            kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=self.policy.dtype)
        else:
            checkpoint = self._load_transformer_checkpoint(checkpoint_path)
        if self.family == "qwen":
            kwargs["torch_dtype"] = self._resolve_qwen_transformer_dtype(model_config)
        components = {"transformer": loader.from_single_file(checkpoint, **kwargs)}
        if self.family == "zimage":
            for spec in self.blocks.expected_components:
                if spec.name not in {"text_encoder", "tokenizer", "vae"}:
                    continue
                repo = model_config.get(f"{spec.name}_repo")
                subfolder = model_config.get(f"{spec.name}_subfolder")
                if repo is not None or subfolder is not None:
                    components[spec.name] = spec.load(
                        pretrained_model_name_or_path=repo or self._component_repo,
                        subfolder=subfolder or spec.name,
                        torch_dtype=self.policy.dtype,
                    )
        if self.family == "qwen":
            components["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(
                {
                    "base_image_seq_len": 256,
                    "base_shift": math.log(3),
                    "max_image_seq_len": 8192,
                    "max_shift": math.log(3),
                    "use_dynamic_shifting": True,
                }
            )
        return components

    def _load_transformer_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Normalize only the known Comfy wrapper prefix before released conversion."""
        from safetensors.torch import load_file

        checkpoint = load_file(checkpoint_path, device="cpu")
        if not any(key.startswith(COMFY_DIFFUSION_MODEL_PREFIX) for key in checkpoint):
            return checkpoint
        return {
            key.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX): value
            for key, value in checkpoint.items()
        }

    def _resolve_qwen_transformer_dtype(self, model_config: dict[str, Any]) -> torch.dtype:
        """Default FP8 Qwen conversion to the activation dtype unless explicitly configured."""
        value = model_config.get("qwen_transformer_dtype") or model_config.get("transformer_dtype")
        if value is None:
            return self.policy.dtype
        if isinstance(value, torch.dtype):
            return value
        if not isinstance(value, str):
            raise TypeError("transformer_dtype must be a string or torch.dtype")
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "float8_e5m2": torch.float8_e5m2,
        }
        dtype = aliases.get(value.removeprefix("torch.").replace("-", "_").lower())
        if dtype is None:
            raise ValueError(f"Unsupported transformer_dtype '{value}'")
        return dtype

    def _default_flux2_component_repo(self) -> str:
        """Choose Klein size/base from validated base metadata, never from custom repo names."""
        base = (self._base_model or "").lower()
        if "4b" in base:
            return (
                DEFAULT_FLUX2_KLEIN_4B_BASE_COMPONENT_REPO
                if "base" in base
                else DEFAULT_FLUX2_KLEIN_4B_COMPONENT_REPO
            )
        return (
            DEFAULT_FLUX2_KLEIN_BASE_COMPONENT_REPO
            if "base" in base
            else DEFAULT_FLUX2_KLEIN_COMPONENT_REPO
        )

    def _should_retry_with_sdxl_text_components(
        self,
        error: Exception,
        single_file_kwargs: dict[str, Any],
    ) -> bool:
        """Retry only SDXL's missing CLIP components, not corrupt checkpoint errors."""
        return (
            self.family == "sdxl"
            and "text_encoder" not in single_file_kwargs
            and "text_encoder_2" not in single_file_kwargs
            and "Weights for this component appear to be missing" in str(error)
            and ("CLIPTextModel" in str(error) or "CLIPTextModelWithProjection" in str(error))
        )

    def _load_sdxl_text_components(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Preserve independent SDXL text-component repository/subfolder overrides."""
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        repo = model_config.get("sdxl_component_repo") or self._component_repo
        components: dict[str, Any] = {"config": model_config.get("single_file_config_repo") or repo}
        for name, loader in (
            ("text_encoder", CLIPTextModel),
            ("text_encoder_2", CLIPTextModelWithProjection),
            ("tokenizer", CLIPTokenizer),
            ("tokenizer_2", CLIPTokenizer),
        ):
            kwargs = {"subfolder": model_config.get(f"{name}_subfolder", name)}
            if name.startswith("text_encoder"):
                kwargs["torch_dtype"] = self.policy.dtype
            components[name] = loader.from_pretrained(
                model_config.get(f"{name}_repo") or repo, **kwargs
            )
        return components

    def _initialize_native_inpaint(self) -> None:
        """Reuse the reviewed Z-Image native view before shared placement."""
        if self.family == "zimage":
            ZImagePipelineWrapper._initialize_native_inpaint(self)

    @property
    def supports_inpaint(self) -> bool:
        """Include Z-Image's retained native exception in capability reporting."""
        return self.family == "zimage" or super().supports_inpaint

    def validate_request(
        self,
        *,
        has_image: bool = False,
        has_mask: bool = False,
        has_reference: bool = False,
        strength: float | None = None,
    ) -> str:
        """Apply shared validation, plus only the existing native Z-Image mask exception."""
        if self.family == "zimage" and has_mask:
            if not has_image or has_reference:
                raise ValueError("Inpainting requires an image and mask, without reference images")
            super().validate_request(has_image=True, strength=strength)
            return "inpainting"
        return super().validate_request(
            has_image=has_image, has_mask=has_mask, has_reference=has_reference, strength=strength
        )

    def workflow_inputs(self, workflow: str) -> set[str]:
        """Use native declarations, including the reviewed mask and Schnell exceptions."""
        if self.family == "zimage" and workflow == "inpainting":
            return ZImagePipelineWrapper.workflow_inputs(self, workflow)
        inputs = super().workflow_inputs(workflow)
        if "prompt_embeds" in inputs and self.family in {"sdxl", "sd3", "flux1"}:
            inputs.add("prompt")
            if "negative_prompt_embeds" in inputs:
                inputs.add("negative_prompt")
        if self.family == "flux1" and self.variant == "schnell":
            inputs.discard("guidance_scale")
        return inputs

    def run_inference(self, gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        """Retain the existing normalized native Z-Image mask path; otherwise shared call."""
        if self.family == "zimage" and "mask_image" in gen_kwargs:
            return ZImagePipelineWrapper.run_inference(self, gen_kwargs, is_img2img)
        return super().run_inference(gen_kwargs, is_img2img)

    def _validate_scheduler(self, name: str | None) -> None:
        """Discrete scheduler recipes belong to SDXL, not flow-matching families."""
        if name is None or name == "default":
            return
        if name not in SCHEDULER_MAP:
            raise ValueError(f"Unknown scheduler '{name}'")
        if self.family != "sdxl":
            raise ValueError(f"Scheduler '{name}' is not compatible with {self.family}")

    def configure_scheduler(self, scheduler_name: str | None) -> None:
        """Replace native scheduler components rather than bypassing manager ownership."""
        self._validate_scheduler(scheduler_name)
        if scheduler_name in (None, "default"):
            return
        import diffusers

        class_name, kwargs = SCHEDULER_MAP[scheduler_name]
        scheduler = getattr(diffusers, class_name).from_config(self.pipe.scheduler.config, **kwargs)
        self.pipe.update_components(scheduler=scheduler)

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
        """Use the shared resource/generation lifecycle with a request-local scheduler."""
        self.validate_pipeline()
        scheduler_name = kwargs.pop("scheduler", None)
        self._validate_scheduler(scheduler_name)
        original = self.pipe.components["scheduler"]
        try:
            self.configure_scheduler(scheduler_name)
            return super().generate(
                prompt,
                negative_prompt,
                self._pipeline_config.default_width if width is None else width,
                self._pipeline_config.default_height if height is None else height,
                seed,
                steps,
                guidance_scale,
                **kwargs,
            )
        finally:
            if self.pipe.components["scheduler"] is not original:
                self.pipe.update_components(scheduler=original)

    def build_generation_kwargs(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        init_image: Image.Image | None,
        strength: float | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject weighted embeddings only when the selected native graph declares them."""
        if self.family in {"sdxl", "sd3"} and guidance_scale <= 1.0:
            # Classic <=1 means positive-only; native CFG uses 1, not 0, for that path.
            # LCM's native loop disables CFG and needs its original embedding scale.
            if (
                self.family != "sdxl"
                or getattr(getattr(self.pipe.unet, "config", None), "time_cond_proj_dim", None)
                is None
            ):
                guidance_scale = 1.0
        values = super().build_generation_kwargs(
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance_scale,
            generator,
            init_image,
            strength,
            **kwargs,
        )
        workflow = self.validate_request(
            has_image=init_image is not None,
            has_mask=kwargs.get("mask_image") is not None,
            has_reference=kwargs.get("reference_image") is not None,
            strength=strength,
        )
        allowed = self.workflow_inputs(workflow)
        if self.family == "flux1" and {"prompt_embeds", "pooled_prompt_embeds"} <= allowed:
            prompt_embeds, pooled = get_weighted_text_embeddings_flux(self.pipe, prompt=prompt)
            values.update(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled)
        elif (
            self.family in {"sdxl", "sd3"}
            and {
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            }
            <= allowed
        ):
            helper = (
                get_weighted_text_embeddings_sdxl
                if self.family == "sdxl"
                else get_weighted_text_embeddings_sd3
            )
            embeds = helper(self.pipe, prompt=prompt, negative_prompt=negative_prompt or "")
            values.update(
                zip(
                    (
                        "prompt_embeds",
                        "negative_prompt_embeds",
                        "pooled_prompt_embeds",
                        "negative_pooled_prompt_embeds",
                    ),
                    embeds,
                    strict=True,
                )
            )
        if "prompt_embeds" in values:
            values.pop("prompt", None)
            values.pop("negative_prompt", None)
        return values

    @property
    def pipeline_config(self) -> PipelineConfig | None:
        """Expose resolved defaults to existing command callers."""
        return self._pipeline_config

    @property
    def detected_base_model(self) -> str | None:
        """Expose actual checkpoint metadata, not the source type 'civitai'."""
        return self._base_model

    def unload(self) -> None:
        """Drop the native mask view before shared component cleanup."""
        self.inpaint_pipe = None
        super().unload()
