"""Shared lifecycle around Diffusers' native modular workflows."""

import gc
import math
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from diffusers.modular_pipelines.components_manager import ComponentsManager
from PIL import Image

from oneiro.device import DevicePolicy, OffloadType
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingConfig, EmbeddingLoaderMixin
from oneiro.pipelines.lora import LoraConfig, LoraLoaderMixin


class ModularPipelineWrapper(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Own one native pipeline and manager; recipes supply the original block graph."""

    family: str = ""
    default_steps: int = 9
    default_guidance_scale: float = 0.0
    blocks: Any = None
    _product_workflows = {
        "text2image",
        "image2image",
        "inpainting",
        "image_conditioned",
        "reference",
    }

    def __init__(self) -> None:
        super().__init__()
        self.components_manager: ComponentsManager | None = None
        self._original_guider: Any = None

    @property
    def supports_inpaint(self) -> bool:
        """Derive mask support from declarations, without requiring a loaded manager."""
        return self.blocks is not None and "inpainting" in self.blocks.available_workflows

    def initialize_pipeline(
        self,
        component_repo: str,
        blocks: Any,
        components: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the original graph, inject components, then load required gaps."""
        if self.pipe is not None or self.components_manager is not None:
            raise RuntimeError("Unload the current pipeline before initializing another")
        self.blocks = blocks
        self.components_manager = ComponentsManager()
        try:
            # Turbo's init override must run before any workflow extraction.
            self.pipe = blocks.init_pipeline(
                component_repo, components_manager=self.components_manager, collection=self.family
            )
            components = components or {}
            unknown = components.keys() - self.pipe.components.keys()
            if unknown:
                raise ValueError(f"Unknown pipeline components: {sorted(unknown)}")
            # Native specs retain locally initialized Transformers processor identities.
            self.pipe.register_components(**components)
            required = {
                spec.name
                for workflow in blocks.available_workflows
                if workflow in self._product_workflows
                for spec in blocks.get_workflow(workflow).expected_components
            }
            missing = sorted(name for name in required if self.pipe.components.get(name) is None)
            if missing:
                self.pipe.load_components(names=missing, torch_dtype=self.policy.dtype)
            missing = sorted(name for name in required if self.pipe.components.get(name) is None)
            if missing:
                raise RuntimeError(f"Missing required pipeline components: {', '.join(missing)}")
            self._initialize_native_inpaint()
            self.policy.apply_to_modular_pipeline(self.pipe, self.components_manager)
        except Exception:
            self.unload()
            raise

    def _initialize_native_inpaint(self) -> None:
        """Let Z-Image attach its native mask view before the single shared placement."""

    def workflow_inputs(self, workflow: str) -> set[str]:
        """Read real input declarations, with a narrow native-inpaint override seam."""
        return set(self.blocks.get_workflow(workflow).input_names)

    def validate_request(
        self,
        *,
        has_image: bool = False,
        has_mask: bool = False,
        has_reference: bool = False,
        strength: float | None = None,
    ) -> str:
        """Select a declared workflow and reject incompatible denoising controls."""
        if self.blocks is None:
            raise RuntimeError("Pipeline workflow declarations are not initialized")
        workflows = self.blocks.available_workflows
        if has_mask and (not has_image or has_reference):
            raise ValueError("Inpainting requires an image and mask, without reference images")
        if has_reference and has_image:
            raise ValueError("Reference and initial images cannot be combined")
        if has_mask:
            workflow = "inpainting"
        elif has_reference:
            workflow = "reference" if "reference" in workflows else "image_conditioned"
        elif has_image:
            workflow = "image2image" if "image2image" in workflows else "image_conditioned"
        else:
            workflow = "text2image"
        if workflow not in workflows:
            raise ValueError(f"{self.family} does not support the requested {workflow} workflow")
        if strength is not None:
            if workflow not in {"image2image", "inpainting"}:
                raise ValueError(f"Denoising strength is not supported for {workflow}")
            if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
                raise ValueError("Strength must be finite, greater than 0, and at most 1")
        return workflow

    def generate(
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
        """Validate native controls before BasePipeline decodes and sets up resources."""
        self.validate_pipeline()
        steps = self.default_steps if steps is None else steps
        guidance_scale = self.default_guidance_scale if guidance_scale is None else guidance_scale
        workflow = self.validate_request(
            has_image=kwargs.get("init_image") is not None,
            has_mask=kwargs.get("mask_image") is not None,
            has_reference=kwargs.get("reference_image") is not None,
            strength=kwargs.get("strength"),
        )
        allowed = self.workflow_inputs(workflow)
        guider = self.pipe.components.get("guider")
        if "true_cfg_scale" in kwargs:
            if guider is None:
                raise ValueError("This recipe does not support classifier-free guidance")
            guidance_scale = kwargs.pop("true_cfg_scale")
        unknown = (
            kwargs.keys()
            - allowed
            - {"init_image", "mask_image", "reference_image", "strength", "loras"}
        )
        if unknown:
            raise ValueError(f"Unsupported generation controls: {sorted(unknown)}")
        if negative_prompt is not None and "negative_prompt" not in allowed:
            raise ValueError(f"Negative prompts are not supported for {workflow}")
        recipe_controlled = guider is not None and not guider.config.enabled
        if recipe_controlled or ("guidance_scale" not in allowed and guider is None):
            if guidance_scale != self.default_guidance_scale:
                raise ValueError("Guidance is controlled by the distilled recipe")
        if kwargs.get("output_type", "pil") != "pil":
            raise ValueError("Image generation requires output_type='pil'")
        if kwargs.get("loras") and not callable(getattr(self.pipe, "load_lora_weights", None)):
            raise ValueError(f"{self.family} does not support LoRA adapters")
        if width <= 0 or height <= 0 or steps <= 0 or not math.isfinite(guidance_scale):
            raise ValueError("Dimensions and steps must be positive and guidance must be finite")
        return super().generate(
            prompt, negative_prompt, width, height, seed, steps, guidance_scale, **kwargs
        )

    def pre_generate(self, **kwargs: Any) -> None:
        """Apply explicitly requested native resources; post_generate rolls them back."""
        if kwargs.get("loras"):
            self.unload_loras()
            self.load_loras_sync(kwargs["loras"])

    @contextmanager
    def _resource_update(self, *, refresh_group: bool = False) -> Iterator[None]:
        """Suspend sequential hooks and refresh native groups after embedding mutations."""
        from accelerate import cpu_offload
        from accelerate.hooks import remove_hook_from_module

        sequential = (
            self.pipe is not None
            and getattr(self.pipe, "_oneiro_offload_type", None) == OffloadType.SEQUENTIAL.value
        )
        if not sequential:
            try:
                yield
            finally:
                if refresh_group and self.pipe is not None:
                    from diffusers.hooks.group_offloading import (
                        _maybe_remove_and_reapply_group_offloading,
                    )

                    encoder = self.pipe.components.get("text_encoder")
                    if isinstance(encoder, torch.nn.Module):
                        _maybe_remove_and_reapply_group_offloading(encoder)
            return
        modules = {
            id(module): module
            for module in self.pipe.components.values()
            if isinstance(module, torch.nn.Module)
        }
        active = any(
            hasattr(child, "_hf_hook") for module in modules.values() for child in module.modules()
        )
        if active:
            for module in modules.values():
                remove_hook_from_module(module, recurse=True)
        try:
            yield
        finally:
            if active and self.pipe is not None:
                for module in modules.values():
                    cpu_offload(module, execution_device=torch.device(self.policy.device))

    def load_single_lora(self, lora: LoraConfig) -> str:
        """Load native adapters while preserving this wrapper's sequential placement."""
        with self._resource_update():
            return super().load_single_lora(lora)

    def load_loras_sync(self, loras: list[LoraConfig]) -> list[str]:
        """Update a batch under one sequential hook suspension."""
        with self._resource_update():
            return super().load_loras_sync(loras)

    def unload_loras(self, force: bool = False) -> None:
        """Restore real CPU weights before removing native PEFT layers."""
        with self._resource_update():
            super().unload_loras(force=force)

    def load_single_embedding(self, embedding: EmbeddingConfig) -> str:
        """Load supported native textual inversions without classic offload callbacks."""
        with self._resource_update(refresh_group=True):
            return super().load_single_embedding(embedding)

    def unload_single_embedding(self, token: str) -> None:
        """Remove a native embedding with real weights and refresh its owned placement."""
        with self._resource_update(refresh_group=True):
            super().unload_single_embedding(token)

    def unload_embeddings(self) -> None:
        """Remove all native embeddings under the same lifecycle as individual updates."""
        with self._resource_update(refresh_group=True):
            super().unload_embeddings()

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
        """Use selected native input names, keeping CFG changes request-local."""
        mask = kwargs.pop("mask_image", None)
        reference = kwargs.pop("reference_image", None)
        workflow = self.validate_request(
            has_image=init_image is not None,
            has_mask=mask is not None,
            has_reference=reference is not None,
            strength=strength,
        )
        allowed = self.workflow_inputs(workflow)
        unknown = kwargs.keys() - allowed
        if unknown:
            raise ValueError(f"Unsupported generation controls: {sorted(unknown)}")
        values = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "generator": generator,
        }
        if negative_prompt is not None:
            values["negative_prompt"] = negative_prompt
        if init_image is not None:
            values["image"] = init_image
        if reference is not None:
            values["reference_image" if workflow == "reference" else "image"] = reference
        if mask is not None:
            values["mask_image"] = mask
        if workflow in {"image2image", "inpainting"}:
            values["strength"] = 0.75 if strength is None else strength
        guider = self.pipe.components.get("guider")
        if guider is not None and guider.config.enabled:
            self._original_guider = guider
            self.pipe.register_components(guider=guider.new(guidance_scale=guidance_scale))
        elif "guidance_scale" in allowed:
            values["guidance_scale"] = guidance_scale
        values.update(kwargs)
        return {name: value for name, value in values.items() if name in allowed}

    def build_result(
        self,
        result: Any,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
        steps: int,
        guidance_scale: float,
    ) -> GenerationResult:
        """Read the native PipelineState's image output without retaining request state."""
        images = result.get("images")
        return super().build_result(
            SimpleNamespace(images=images),
            seed,
            prompt,
            negative_prompt,
            steps,
            guidance_scale,
        )

    def _reset_model_state(self) -> None:
        """Reset native stateful hooks, without freeing persistent group placement hooks."""
        if self.pipe is None:
            return
        modules = {
            id(module): module
            for module in self.pipe.components.values()
            if isinstance(module, torch.nn.Module)
        }
        for module in modules.values():
            registry = getattr(module, "_diffusers_hook", None)
            if registry is not None:
                registry.reset_stateful_hooks()

    def post_generate(self, **kwargs: Any) -> None:
        """Restore CFG and the static adapter baseline, even after setup failure."""
        try:
            if self._original_guider is not None:
                self.pipe.register_components(guider=self._original_guider)
                self._original_guider = None
            self._reset_model_state()
        finally:
            self.restore_static_loras()

    def unload(self) -> None:
        """Release this wrapper's native hooks, manager ownership, and component refs."""
        if self.components_manager is not None:
            self.components_manager.disable_auto_cpu_offload()
        if self.pipe is not None:
            from accelerate.hooks import remove_hook_from_module

            modules = {
                id(module): module
                for module in self.pipe.components.values()
                if isinstance(module, torch.nn.Module)
            }
            for module in modules.values():
                remove_hook_from_module(module, recurse=True)
                for child in module.modules():
                    registry = getattr(child, "_diffusers_hook", None)
                    if registry is not None:
                        for name in list(registry.hooks):
                            registry.remove_hook(name, recurse=False)
            del modules
            self.pipe.unload_components(list(self.pipe.components))
        if self.components_manager is not None:
            for component_id in list(self.components_manager.components):
                self.components_manager.remove(component_id)
        self.pipe = None
        self.components_manager = None
        self._original_guider = None
        self._loaded_adapters.clear()
        self._lora_configs.clear()
        self._static_lora_configs.clear()
        self._lora_load_failed = False
        self._loaded_tokens.clear()
        self._embedding_configs.clear()
        gc.collect()
        DevicePolicy.clear_cache()
