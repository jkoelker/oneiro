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

    def __init__(self):
        self.pipe: Any = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

    @abstractmethod
    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load the model from config.

        Args:
            model_config: Model-specific configuration dict
            full_config: Full configuration dict (for accessing global sections like embeddings)
        """

    @abstractmethod
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
        """Generate an image."""

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

        # Aggressive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
