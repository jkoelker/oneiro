"""Base classes and types for pipeline implementations."""

import gc
import io
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy


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


class BasePipeline(ABC):
    """Base class for all pipeline types."""

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
        self.pre_generate(**kwargs)
        try:
            actual_seed, generator = self._prepare_seed(seed)
            # Pop init_image and strength from kwargs to avoid passing twice
            init_image = self._load_init_image(kwargs.pop("init_image", None))
            strength = kwargs.pop("strength", 0.75)

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
                **kwargs,
            )
            is_img2img = init_image is not None
            result = self.run_inference(gen_kwargs, is_img2img)

            DevicePolicy.clear_cache()
            return self.build_result(
                result=result,
                seed=actual_seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
            )
        finally:
            self.post_generate(**kwargs)

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

        Note: This method may pop keys from kwargs to consume them before
        build_generation_kwargs() is called. The modified kwargs are then
        passed through to build_generation_kwargs() and post_generate().
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
        strength: float,
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

    def post_generate(self, **kwargs: Any) -> None:  # noqa: B027
        """Post-generation cleanup hook called after generation completes.

        Override for LoRA restore or other cleanup. This is an optional hook
        with a no-op default; it is intentionally not abstract so subclasses
        can choose whether to implement it.

        Note: The kwargs passed here have already had 'init_image' and 'strength'
        removed by generate(). If a subclass needs access to these values,
        it should save them in pre_generate() before they are consumed.
        """
        pass

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
        return Image.open(io.BytesIO(init_image)).convert("RGB")

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
