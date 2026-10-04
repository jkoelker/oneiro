"""Pipeline implementations for different model types."""

import asyncio
import io
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from oneiro.pipelines.base import (
    BasePipeline,
    GenerationResult,
    get_generation_defaults,
    validate_request_inputs,
)
from oneiro.pipelines.civitai_checkpoint import (
    CIVITAI_BASE_MODEL_PIPELINE_MAP,
    SCHEDULER_CHOICES,
    SCHEDULER_MAP,
    CivitaiCheckpointPipeline,
    PipelineConfig,
    get_pipeline_config_for_base_model,
)
from oneiro.pipelines.embedding import (
    EmbeddingIncompatibleError,
    parse_embeddings_from_config,
    resolve_embedding_path,
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

    def __init__(self, config: "Config") -> None:
        """Initialize model identity and the single load/generation ownership lock."""
        self.config = config
        self.current_model: str | None = None
        self.pipeline: BasePipeline | None = None
        self._civitai_client: CivitaiClient | None = None
        self._lock = asyncio.Lock()

    @property
    def family(self) -> str | None:
        """Return the loaded checkpoint's actual architecture, not its source type."""
        return getattr(self.pipeline, "family", None) if self.pipeline is not None else None

    def validate_request(
        self,
        *,
        has_image: bool = False,
        has_mask: bool = False,
        has_reference: bool = False,
        strength: float | None = None,
        **controls: Any,
    ) -> str:
        """Validate against the currently owned model; generation repeats this under lock."""
        if self.pipeline is None:
            raise RuntimeError("No pipeline loaded; use preflight_request for recovery")
        return self.pipeline.validate_request(
            has_image=has_image,
            has_mask=has_mask,
            has_reference=has_reference,
            strength=strength,
            **controls,
        )

    async def preflight_request(self, **controls: Any) -> tuple[str, str]:
        """Admit controls and expose the recovery family without downloading components."""
        validate_request_inputs(**controls)
        pipeline, model_name = self.pipeline, self.current_model
        if pipeline is None:
            model_name = self.config.get("defaults", "model", default="zimage-turbo")
            pipeline, _ = await self._resolve_model(model_name)
        steps, guidance = self._sampling_values(
            pipeline, model_name, controls.pop("steps", None), controls.pop("guidance_scale", None)
        )
        workflow = pipeline.validate_request(steps=steps, guidance_scale=guidance, **controls)
        return workflow, pipeline.family

    def _sampling_values(
        self,
        pipeline: BasePipeline,
        model_name: str | None,
        steps: int | None,
        guidance_scale: float | None,
    ) -> tuple[int, float]:
        """Apply one precedence rule in admission, target preflight and owned execution."""
        default_steps, default_guidance = get_generation_defaults(pipeline)
        defaults = {"steps": default_steps, "guidance_scale": default_guidance}
        if model_name is not None:
            profile = self.config.get("models", model_name, default={})
            overrides = self.config.get("model_overrides", model_name, default={})
            if isinstance(profile, dict):
                defaults.update({key: profile[key] for key in defaults if key in profile})
                if profile.get("true_cfg_scale") is not None:
                    defaults["guidance_scale"] = profile["true_cfg_scale"]
            if isinstance(overrides, dict):
                defaults.update({key: overrides[key] for key in defaults if key in overrides})
        return (
            defaults["steps"] if steps is None else steps,
            defaults["guidance_scale"] if guidance_scale is None else guidance_scale,
        )

    async def _resolve_model(self, model_name: str) -> tuple[BasePipeline, dict[str, Any]]:
        """Resolve the real target's recipe and config without loading resources or weights."""
        model_config = self.config.get("models", model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        pipeline_type = model_config.get("type")
        if pipeline_type not in self.PIPELINE_TYPES:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        pipeline = self.PIPELINE_TYPES[pipeline_type]()
        if isinstance(pipeline, CivitaiCheckpointPipeline):
            model_config = await pipeline.resolve_config(model_config, self._civitai_client)
        else:
            pipeline.validate_config(model_config, self.config.data)
        return pipeline, model_config

    @staticmethod
    async def _await_owned(operation: Coroutine[Any, Any, Any]) -> Any:
        """Drain shielded work before propagating even repeated caller cancellation."""
        task = asyncio.create_task(operation)
        cancelled: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception:
                # Retrieve/propagate the worker error below, including after cancellation.
                break
        if cancelled is not None:
            try:
                task.result()
            except BaseException as error:
                raise cancelled from error
            raise cancelled
        return task.result()

    def set_civitai_client(self, client: "CivitaiClient") -> None:
        """Set the CivitAI client for checkpoint downloads.

        Args:
            client: CivitaiClient instance for API access and downloads
        """
        self._civitai_client = client

    async def load_model(
        self,
        model_name: str | None = None,
        *,
        scheduler: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> None:
        """Load a model by name from config.

        Args:
            model_name: Name of model to load. If None, loads default from config.
            scheduler: Persistent checkpoint override applied to that target under ownership.
        """

        async def load_owned() -> None:
            await self._load_model(
                model_name, scheduler=scheduler, steps=steps, guidance_scale=guidance_scale
            )
            if scheduler is not None:
                target = cast(CivitaiCheckpointPipeline, self.pipeline)
                await asyncio.to_thread(target.configure_scheduler, scheduler)

        async with self._lock:
            await self._await_owned(load_owned())

    @staticmethod
    def _validate_scheduler_override(pipeline: BasePipeline, scheduler: str | None) -> None:
        """Admit the override with the actual target's policy, without component mutation."""
        if scheduler is not None:
            if not isinstance(pipeline, CivitaiCheckpointPipeline):
                raise ValueError("Scheduler override is not supported for this pipeline type")
            pipeline._validate_scheduler(scheduler)

    async def _load_model(
        self,
        model_name: str | None = None,
        *,
        scheduler: str | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        request_controls: dict[str, Any] | None = None,
    ) -> None:
        """Load while already owning the lock, including lazy load during generation."""
        # Get model name from config if not specified
        if model_name is None:
            model_name = self.config.get("defaults", "model", default="zimage-turbo")
        controls = dict(request_controls or {})

        # Already loaded this model
        if self.current_model == model_name and self.pipeline is not None:
            self._validate_scheduler_override(self.pipeline, scheduler)
            steps, guidance_scale = self._sampling_values(
                self.pipeline, model_name, steps, guidance_scale
            )
            controls.update(steps=steps, guidance_scale=guidance_scale)
            self.pipeline.validate_request(**controls)
            return

        # Get model config - model_name is guaranteed to be str at this point
        assert model_name is not None
        new_pipeline, model_config = await self._resolve_model(model_name)
        pipeline_type = model_config.get("type")
        self._validate_scheduler_override(new_pipeline, scheduler)
        steps, guidance_scale = self._sampling_values(
            new_pipeline, model_name, steps, guidance_scale
        )
        controls.update(steps=steps, guidance_scale=guidance_scale)
        new_pipeline.validate_request(**controls)
        family = new_pipeline.family
        full_config = self.config.data
        embeddings = parse_embeddings_from_config(
            full_config, model_config, include_auto_load=False
        )
        supports_embeddings = isinstance(new_pipeline, CivitaiCheckpointPipeline) and family in {
            "sdxl",
            "flux1",
        }
        if embeddings and not supports_embeddings:
            raise ValueError(f"{family} does not support textual inversion embeddings")
        for embedding in embeddings:
            await resolve_embedding_path(
                embedding,
                civitai_client=self._civitai_client,
                pipeline_type=family,
                validate_compatibility=True,
            )
        required_names = {embedding.name for embedding in embeddings}
        for embedding in parse_embeddings_from_config(full_config, {}):
            if embedding.name in required_names:
                continue
            if not supports_embeddings:
                print(
                    f"Warning: Skipping auto-load embedding {embedding.name}: "
                    f"{family} does not support textual inversion embeddings"
                )
                continue
            try:
                await resolve_embedding_path(
                    embedding,
                    civitai_client=self._civitai_client,
                    pipeline_type=family,
                    validate_compatibility=True,
                )
            except EmbeddingIncompatibleError as error:
                print(f"Warning: Skipping auto-load embedding {embedding.name}: {error}")
            else:
                embeddings.append(embedding)
        # An empty resolved list must suppress checkpoint fallback to unfiltered globals.
        if isinstance(new_pipeline, CivitaiCheckpointPipeline):
            model_config = {**model_config, "_resolved_embeddings": embeddings}

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
                        pipeline_type=family,
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
                    pipeline_type=family,
                    validate_compatibility=True,
                )
                model_loras.append(lora)

        # Unload the current model only after target resources validate.
        if self.pipeline and self.current_model != model_name:
            await asyncio.to_thread(self.pipeline.unload)

        self.pipeline = new_pipeline
        self.current_model = None

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
                    await asyncio.to_thread(civitai_pipeline.load, model_config, full_config)
                else:
                    # Load asynchronously with CivitAI client
                    await civitai_pipeline.load_async(
                        model_config, self._civitai_client, full_config
                    )
            else:
                await asyncio.to_thread(self.pipeline.load, model_config, full_config)

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
                            except Exception:
                                await asyncio.to_thread(lora_pipeline.unload_loras, True)
                                raise
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
                        await asyncio.to_thread(
                            lora_pipeline.set_lora_adapters,
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
                    await asyncio.to_thread(lora_pipeline.set_lora_adapters, loaded_names, weights)
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
        steps: int | None = None,
        guidance_scale: float | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate an image using the current pipeline."""
        async with self._lock:
            return await self._await_owned(
                self._generate(
                    prompt, negative_prompt, width, height, seed, steps, guidance_scale, **kwargs
                )
            )

    async def _generate(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        seed: int,
        steps: int | None,
        guidance_scale: float | None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Revalidate queued controls and stamp model identity while ownership is held."""
        controls = {
            "has_image": kwargs.get("init_image") is not None,
            "has_mask": kwargs.get("mask_image") is not None,
            "has_reference": kwargs.get("reference_image") is not None,
            "strength": kwargs.get("strength"),
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"init_image", "mask_image", "reference_image", "strength"}
            },
        }
        if self.pipeline is None:
            await self.preflight_request(**controls)
            await self._load_model(
                steps=steps, guidance_scale=guidance_scale, request_controls=controls
            )

        if self.pipeline is None:
            raise RuntimeError("No pipeline loaded")

        steps, guidance_scale = self._sampling_values(
            self.pipeline, self.current_model, steps, guidance_scale
        )
        controls.update(steps=steps, guidance_scale=guidance_scale)
        self.validate_request(**controls)

        loras: list[LoraConfig] | None = kwargs.pop("loras", None)
        if loras:
            for lora in loras:
                await resolve_lora_path(
                    lora,
                    civitai_client=self._civitai_client,
                    pipeline_type=self.family,
                    validate_compatibility=True,
                )
            kwargs["loras"] = loras

        result = await asyncio.to_thread(
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
        result.model_name = self.current_model
        return result

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

        async def load_owned() -> list[str]:
            if self.pipeline is None:
                raise RuntimeError("No pipeline loaded")
            if not isinstance(self.pipeline, LoraLoaderMixin):
                raise RuntimeError(f"Pipeline {type(self.pipeline)} does not support LoRAs")
            for lora in loras:
                await resolve_lora_path(
                    lora,
                    civitai_client=civitai_client,
                    pipeline_type=self.family,
                    validate_compatibility=validate_compatibility,
                )
            return await asyncio.to_thread(self.pipeline.load_loras_sync, loras)

        async with self._lock:
            return await self._await_owned(load_owned())
