"""Base classes and types for pipeline implementations."""

import gc
import io
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from oneiro.device import DevicePolicy, OffloadType

MAX_INPUT_IMAGE_PIXELS = 4096 * 4096
MAX_INPUT_IMAGE_BYTES = 25 * 1024 * 1024


@dataclass
class GenerationResult:
    """Result of an image generation."""

    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    workflow: str = "text2image"
    strength: float | None = None
    model_name: str | None = None


class BasePipeline(ABC):
    """Base class for all pipeline types."""

    supports_inpaint: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.pipe: Any = None
        self.policy: DevicePolicy = DevicePolicy.auto_detect()

    @abstractmethod
    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load the model from config.

        Args:
            model_config: Model-specific configuration dict
            full_config: Full configuration dict (for accessing global sections like embeddings)
        """

    def validate_config(  # noqa: B027
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Preflight deterministic loader controls without loading model assets."""

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 9,
        guidance_scale: float = 0.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate an image using Template Method pattern.

        Subclasses should override hooks (validate_pipeline, pre_generate,
        build_generation_kwargs, run_inference, build_result, post_generate),
        not this method.
        """
        self.validate_pipeline()
        init_bytes = kwargs.pop("init_image", None)
        mask_bytes = kwargs.pop("mask_image", None)
        reference_bytes = kwargs.pop("reference_image", None)
        strength = kwargs.pop("strength", None)
        workflow = self.validate_request(
            has_image=init_bytes is not None,
            has_mask=mask_bytes is not None,
            has_reference=reference_bytes is not None,
            strength=strength,
        )
        init_image = self._load_init_image(init_bytes)
        mask_image = self._load_init_image(mask_bytes)
        if reference_bytes is not None:
            if isinstance(reference_bytes, list):
                if not reference_bytes:
                    raise ValueError("Reference image list must not be empty")
                if any(not isinstance(image, bytes) for image in reference_bytes):
                    raise ValueError("Reference image attachments must be bytes")
                reference_image = [self._load_init_image(image) for image in reference_bytes]
            else:
                reference_image = self._load_init_image(reference_bytes)
            kwargs["reference_image"] = reference_image
        if strength is None and workflow in {"image2image", "inpainting"}:
            strength = 0.75
        controls = {
            name: kwargs.pop(name)
            for name in ("loras", "embeddings", "scheduler")
            if name in kwargs
        }
        actual_seed, generator = self._prepare_seed(seed)
        try:
            self.pre_generate(**controls)

            gen_kwargs = self.build_generation_kwargs(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                init_image=init_image,
                strength=strength,
                mask_image=mask_image,
                **kwargs,
            )
            is_img2img = init_image is not None
            result = self.run_inference(gen_kwargs, is_img2img)

            DevicePolicy.clear_cache()
            generation_result = self.build_result(
                result=result,
                seed=actual_seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
            )
            generation_result.workflow = workflow
            generation_result.strength = (
                strength if workflow in {"image2image", "inpainting"} else None
            )
            return generation_result
        finally:
            self.post_generate(**controls)

    def validate_request(
        self,
        *,
        has_image: bool = False,
        has_mask: bool = False,
        has_reference: bool = False,
        strength: float | None = None,
    ) -> str:
        """Keep classic capabilities while allowing native wrappers to validate workflows."""
        if has_mask and not self.supports_inpaint:
            raise ValueError("This pipeline does not support inpainting masks")
        return "inpainting" if has_mask else "image2image" if has_image else "text2image"

    def validate_pipeline(self) -> None:
        """Validate pipeline is ready for generation.

        This is called before pre_generate() and before any kwargs are consumed.
        Override for additional validation checks (e.g., config state validation).
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

    def pre_generate(self, **kwargs: Any) -> None:  # noqa: B027
        """Pre-generation hook called before building kwargs.

        Override for scheduler/LoRA setup or other pre-processing.
        This is an optional hook with a no-op default; it is intentionally
        not abstract so subclasses can choose whether to implement it.

        Request-only controls are consumed by generate() and passed to this hook
        and post_generate(), never to the inference kwargs builder. Mutating this
        hook's kwargs does not mutate generate()'s dictionary.
        """
        pass

    @abstractmethod
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
        """Build pipeline-specific generation kwargs.

        This is the REQUIRED hook that each subclass must implement.
        Return a dict to be passed to self.pipe().
        """

    def run_inference(self, gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        """Run the diffusion pipeline.

        Args:
            gen_kwargs: Keyword arguments to pass to the underlying pipeline.
            is_img2img: Whether this is an image-to-image generation. This flag
                is not used by the base implementation but is provided for
                subclasses that need to branch on img2img vs txt2img behavior.

        Returns:
            Pipeline output (typically has .images attribute).

        Override if the pipeline call signature or behavior differs.
        """
        return self.pipe(**gen_kwargs)

    def build_result(
        self,
        result: Any,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
        steps: int,
        guidance_scale: float,
    ) -> GenerationResult:
        """Build GenerationResult from pipeline output.

        Override if result format differs.
        """
        output_image = result.images[0]
        return GenerationResult(
            image=output_image,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output_image.width,
            height=output_image.height,
            steps=steps,
            guidance_scale=guidance_scale,
        )

    def post_generate(self, **kwargs: Any) -> None:
        """Post-generation cleanup hook called after generation completes.

        This base implementation resets stateful model caches using the diffusers
        `maybe_free_model_hooks()` API. This prevents state leakage between
        generations (e.g., KV cache, attention state, hook state).

        Subclasses should call super().post_generate(**kwargs) first, then perform
        any additional cleanup (e.g., LoRA restore).

        Only request-time resource controls are passed here. This hook also runs
        when pre_generate() raises, but not when input validation fails.
        """
        self._reset_model_state()

    def _reset_model_state(self) -> None:
        """Reset stateful model caches between generations.

        Uses the diffusers `maybe_free_model_hooks()` API to reset:
        - Stateful caches (KV cache, attention state)
        - CPU offload hooks (if model offloading is enabled)

        This is the canonical way to reset diffusers pipeline state.
        """
        if self.pipe is None:
            return
        if getattr(self.pipe, "_oneiro_offload_type", None) == OffloadType.GROUP.value:
            return
        self.pipe.maybe_free_model_hooks()

    def unload(self) -> None:
        """Free GPU memory."""
        if self.pipe is not None:
            # Move to CPU first to free VRAM
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
            del self.pipe
            self.pipe = None

        gc.collect()
        DevicePolicy.clear_cache()

    def _prepare_seed(self, seed: int) -> tuple[int, torch.Generator]:
        """Prepare seed and generator for generation."""
        actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(actual_seed)
        return actual_seed, generator

    def _load_init_image(self, init_image: bytes | None) -> Image.Image | None:
        """Load init_image from bytes if provided."""
        if init_image is None:
            return None
        if not isinstance(init_image, bytes):
            raise ValueError("Image attachment must be bytes")
        if len(init_image) > MAX_INPUT_IMAGE_BYTES:
            raise ValueError("Image attachment exceeds the 25 MiB limit")
        try:
            with Image.open(io.BytesIO(init_image)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Image attachment must be PNG, JPEG, or WebP")
                width, height = image.size
                if width * height > MAX_INPUT_IMAGE_PIXELS:
                    raise ValueError(
                        f"Input image is too large ({width}×{height}); "
                        "maximum supported size is 4096×4096"
                    )
                return image.convert("RGB")
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as e:
            raise ValueError("Invalid image attachment; upload a valid image file") from e

    def _configure_cpu_threads(self, utilization: float = 0.75) -> int:
        """Configure PyTorch CPU threading for optimal performance.

        Args:
            utilization: Fraction of CPU cores to use (0.0-1.0). Default 75%.

        Returns:
            Number of threads configured.
        """
        cpu_count = os.cpu_count() or 1
        num_threads = max(1, int(cpu_count * utilization))

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))

        print(f"CPU threading: {num_threads} threads ({cpu_count} cores @ {utilization:.0%})")
        return num_threads
