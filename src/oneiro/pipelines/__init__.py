"""Pipeline implementations for different model types."""

import asyncio
import io
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.civitai_checkpoint import (
    CIVITAI_BASE_MODEL_PIPELINE_MAP,
    SCHEDULER_CHOICES,
    SCHEDULER_MAP,
    CivitaiCheckpointPipeline,
    PipelineConfig,
    get_pipeline_config_for_base_model,
)
from oneiro.pipelines.flux1 import Flux1PipelineWrapper
from oneiro.pipelines.flux2 import Flux2PipelineWrapper
from oneiro.pipelines.flux2_klein import Flux2KleinPipelineWrapper
from oneiro.pipelines.krea2 import Krea2PipelineWrapper
from oneiro.pipelines.lora import (
    LoraConfig,
    LoraIncompatibleError,
    LoraLoaderMixin,
    LoraSource,
    parse_lora_config,
    parse_loras_from_config,
    parse_loras_from_model_config,
    resolve_lora_path,
)
from oneiro.pipelines.qwen import QwenPipelineWrapper
from oneiro.pipelines.zimage import ZImagePipelineWrapper

if TYPE_CHECKING:
    from oneiro.civitai import CivitaiClient
    from oneiro.config import Config

__all__ = [
    "BasePipeline",
    "GenerationResult",
    "PipelineManager",
    "Flux1PipelineWrapper",
    "Flux2PipelineWrapper",
    "Flux2KleinPipelineWrapper",
    "Krea2PipelineWrapper",
    "QwenPipelineWrapper",
    "ZImagePipelineWrapper",
    "CivitaiCheckpointPipeline",
    "PipelineConfig",
    "CIVITAI_BASE_MODEL_PIPELINE_MAP",
    "SCHEDULER_CHOICES",
    "SCHEDULER_MAP",
    "get_pipeline_config_for_base_model",
    "LoraConfig",
    "LoraSource",
    "LoraLoaderMixin",
    "LoraIncompatibleError",
    "parse_lora_config",
    "parse_loras_from_model_config",
    "resolve_lora_path",
]


class PipelineManager:
    """Manages pipeline loading and switching based on config."""

    PIPELINE_TYPES: dict[str, type[BasePipeline]] = {
        "zimage": ZImagePipelineWrapper,
        "flux1": Flux1PipelineWrapper,
        "flux2": Flux2PipelineWrapper,
        "flux2-klein": Flux2KleinPipelineWrapper,
        "krea2": Krea2PipelineWrapper,
        "qwen": QwenPipelineWrapper,
        "civitai": CivitaiCheckpointPipeline,
    }

    def __init__(self, config: "Config"):
        self.config = config
        self.current_model: str | None = None
        self.pipeline: BasePipeline | None = None
        self._civitai_client: CivitaiClient | None = None

    def set_civitai_client(self, client: "CivitaiClient") -> None:
        """Set the CivitAI client for checkpoint downloads.

        Args:
            client: CivitaiClient instance for API access and downloads
        """
        self._civitai_client = client

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

        # Get model config - model_name is guaranteed to be str at this point
        assert model_name is not None
        model_config = self.config.get("models", model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        pipeline_type = model_config.get("type")
        if pipeline_type not in self.PIPELINE_TYPES:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        wrapper_class = self.PIPELINE_TYPES[pipeline_type]
        new_pipeline = wrapper_class()

        auto_loras: list[LoraConfig] = []
        model_loras: list[LoraConfig] = []
        if isinstance(new_pipeline, LoraLoaderMixin):
            full_config = self.config.data
            parsed_auto_loras = parse_loras_from_config(
                full_config,
                {},
                ignore_auto_load_errors=True,
            )
            for lora in parsed_auto_loras:
                try:
                    await resolve_lora_path(
                        lora,
                        civitai_client=self._civitai_client,
                        pipeline_type=pipeline_type,
                        validate_compatibility=True,
                    )
                except Exception as error:
                    print(f"Warning: Failed to resolve auto-load LoRA {lora.name}: {error}")
                else:
                    auto_loras.append(lora)

            parsed_model_loras = parse_loras_from_config(
                full_config,
                model_config,
                include_auto_load=False,
            )
            for lora in parsed_model_loras:
                await resolve_lora_path(
                    lora,
                    civitai_client=self._civitai_client,
                    pipeline_type=pipeline_type,
                    validate_compatibility=True,
                )
                model_loras.append(lora)

        # Unload the current model only after target resources validate.
        if self.pipeline and self.current_model != model_name:
            await asyncio.to_thread(self.pipeline.unload)

        self.pipeline = new_pipeline

        try:
            # Special handling for CivitAI checkpoints (async loading)
            if pipeline_type == "civitai":
                civitai_pipeline = cast(CivitaiCheckpointPipeline, self.pipeline)
                if self._civitai_client is None:
                    # Check if checkpoint_path is provided (can load without client)
                    if not model_config.get("checkpoint_path"):
                        raise ValueError(
                            "CivitAI pipeline requires either checkpoint_path in config "
                            "or a CivitaiClient set via set_civitai_client()"
                        )
                    # Load synchronously from path
                    await asyncio.to_thread(civitai_pipeline.load, model_config)
                else:
                    # Load asynchronously with CivitAI client
                    await civitai_pipeline.load_async(model_config, self._civitai_client)
            else:
                await asyncio.to_thread(self.pipeline.load, model_config)

            if auto_loras or model_loras:
                lora_pipeline = cast(LoraLoaderMixin, self.pipeline)
                loaded_loras: list[LoraConfig] = []
                loaded_names: list[str] = []
                loaded_auto_names: set[str] = set()
                for lora in auto_loras:
                    try:
                        name = await asyncio.to_thread(lora_pipeline.load_single_lora, lora)
                    except Exception as error:
                        print(f"Warning: Failed to load auto-load LoRA {lora.name}: {error}")
                        previous_loras = list(loaded_loras)
                        await asyncio.to_thread(lora_pipeline.unload_loras, True)
                        loaded_loras.clear()
                        loaded_names.clear()
                        loaded_auto_names.clear()
                        if previous_loras:
                            try:
                                restored_names = await asyncio.to_thread(
                                    lora_pipeline.load_loras_sync,
                                    previous_loras,
                                )
                            except Exception as restore_error:
                                print(
                                    "Warning: Failed to restore auto-load LoRAs after "
                                    f"rollback: {restore_error}"
                                )
                                await asyncio.to_thread(lora_pipeline.unload_loras, True)
                            else:
                                loaded_loras.extend(previous_loras)
                                loaded_names.extend(restored_names)
                                loaded_auto_names.update(restored_names)
                    else:
                        loaded_loras.append(lora)
                        loaded_names.append(name)
                        loaded_auto_names.add(name)

                if loaded_loras:
                    try:
                        lora_pipeline.set_lora_adapters(
                            loaded_names,
                            [lora.weight for lora in loaded_loras],
                        )
                    except Exception as error:
                        print(f"Warning: Failed to activate auto-load LoRAs: {error}")
                        await asyncio.to_thread(lora_pipeline.unload_loras, True)
                        loaded_loras.clear()
                        loaded_names.clear()
                        loaded_auto_names.clear()

                loaded_model_loras = False
                for lora in model_loras:
                    adapter_name = lora.adapter_name or lora.name
                    if adapter_name in loaded_auto_names:
                        continue
                    name = await asyncio.to_thread(lora_pipeline.load_single_lora, lora)
                    loaded_loras.append(lora)
                    loaded_names.append(name)
                    loaded_model_loras = True

                if loaded_model_loras:
                    weights = [lora.weight for lora in loaded_loras]
                    lora_pipeline.set_lora_adapters(loaded_names, weights)
                if loaded_loras:
                    lora_pipeline.set_static_loras(loaded_loras)
        except Exception:
            failed_pipeline = self.pipeline
            if failed_pipeline is not None:
                try:
                    await asyncio.to_thread(failed_pipeline.unload)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to unload partial pipeline: {cleanup_error}")
            self.pipeline = None
            self.current_model = None
            raise

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

        loras: list[LoraConfig] | None = kwargs.pop("loras", None)
        if loras:
            pipeline_type = None
            if self.current_model:
                model_config = self.config.get("models", self.current_model)
                if model_config:
                    pipeline_type = model_config.get("type")

            resolved_loras: list[LoraConfig] = []
            for lora in loras:
                try:
                    await resolve_lora_path(
                        lora,
                        civitai_client=self._civitai_client,
                        pipeline_type=pipeline_type,
                        validate_compatibility=True,
                    )
                    resolved_loras.append(lora)
                except Exception as e:
                    print(f"Warning: Failed to resolve LoRA {lora.name}: {e}")

            if resolved_loras:
                kwargs["loras"] = resolved_loras

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

    async def load_civitai_loras(
        self,
        loras: list[LoraConfig],
        civitai_client: "CivitaiClient",
        validate_compatibility: bool = True,
    ) -> list[str]:
        """Load LoRAs from Civitai, downloading as needed.

        This method should be called after load_model() when using Civitai LoRAs.
        Local and HuggingFace LoRAs are loaded automatically in load_model().

        Args:
            loras: List of LoRA configurations
            civitai_client: CivitaiClient for downloads
            validate_compatibility: Whether to check base model compatibility

        Returns:
            List of loaded adapter names
        """
        if self.pipeline is None:
            raise RuntimeError("No pipeline loaded")

        if not hasattr(self.pipeline, "load_loras_async"):
            raise RuntimeError(f"Pipeline {type(self.pipeline)} does not support LoRAs")

        pipeline_type = None
        if self.current_model:
            model_config = self.config.get("models", self.current_model)
            if model_config:
                pipeline_type = model_config.get("type")

        lora_pipeline = cast(LoraLoaderMixin, self.pipeline)
        return await lora_pipeline.load_loras_async(
            loras,
            civitai_client=civitai_client,
            pipeline_type=pipeline_type,
            validate_compatibility=validate_compatibility,
        )
