"""Pipeline implementations for different model types."""

import asyncio
import io
from typing import TYPE_CHECKING, Any

from PIL import Image

from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.flux2 import Flux2PipelineWrapper
from oneiro.pipelines.qwen import QwenPipelineWrapper
from oneiro.pipelines.zimage import ZImagePipelineWrapper

if TYPE_CHECKING:
    from oneiro.config import Config

__all__ = [
    "BasePipeline",
    "GenerationResult",
    "PipelineManager",
    "Flux2PipelineWrapper",
    "QwenPipelineWrapper",
    "ZImagePipelineWrapper",
]


class PipelineManager:
    """Manages pipeline loading and switching based on config."""

    PIPELINE_TYPES: dict[str, type[BasePipeline]] = {
        "zimage": ZImagePipelineWrapper,
        "flux2": Flux2PipelineWrapper,
        "qwen": QwenPipelineWrapper,
    }

    def __init__(self, config: "Config"):
        self.config = config
        self.current_model: str | None = None
        self.pipeline: BasePipeline | None = None

    async def load_model(self, model_name: str | None = None) -> None:
        """Load a model by name from config.

        Args:
            model_name: Name of model to load. If None, loads default from config.
        """
        # Get model name from config if not specified
        if model_name is None:
            model_name = self.config.get("defaults", "model", default="zimage-turbo")

        # Already loaded this model
        if self.current_model == model_name and self.pipeline is not None:
            return

        # Get model config
        model_config = self.config.get("models", model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline_type = model_config.get("type")
        if pipeline_type not in self.PIPELINE_TYPES:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        # Unload current pipeline if different
        if self.pipeline and self.current_model != model_name:
            await asyncio.to_thread(self.pipeline.unload)

        # Load new pipeline
        wrapper_class = self.PIPELINE_TYPES[pipeline_type]
        self.pipeline = wrapper_class()
        await asyncio.to_thread(self.pipeline.load, model_config)
        self.current_model = model_name

    async def generate(
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
        """Generate an image using the current pipeline."""
        if self.pipeline is None:
            await self.load_model()

        if self.pipeline is None:
            raise RuntimeError("No pipeline loaded")

        return await asyncio.to_thread(
            self.pipeline.generate,
            prompt,
            negative_prompt,
            width,
            height,
            seed,
            steps,
            guidance_scale,
            **kwargs,
        )

    def get_available_models(self) -> list[str]:
        """List available model names from config."""
        models = self.config.get("models", default={})
        return list(models.keys()) if isinstance(models, dict) else []

    def image_to_bytes(self, image: Image.Image, format: str = "PNG") -> io.BytesIO:
        """Convert a PIL Image to bytes for Discord upload."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer
