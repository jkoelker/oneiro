"""Krea 2 pipeline wrapper with CPU offloading and LoRA support."""

from typing import Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.lora import LoraLoaderMixin


class Krea2PipelineWrapper(LoraLoaderMixin, BasePipeline):
    """Wrapper for Krea 2 Raw and Turbo models."""

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load Krea 2 from a hosted Diffusers repository."""
        from diffusers import Krea2Pipeline

        repo = model_config.get("repo", "krea/Krea-2-Turbo")
        cpu_offload = model_config.get("cpu_offload", True)
        offload_type = model_config.get("offload_type", "group")
        group_offload_type = model_config.get("group_offload_type", "leaf_level")
        group_offload_use_stream = model_config.get("group_offload_use_stream", True)
        group_offload_num_blocks_per_group = model_config.get("group_offload_num_blocks_per_group")
        cpu_utilization = model_config.get("cpu_utilization", 0.75)

        print(f"Loading Krea 2 from {repo}")
        self._configure_cpu_threads(cpu_utilization)

        self.policy = DevicePolicy.auto_detect(
            cpu_offload=cpu_offload,
            offload_type=offload_type,
            group_offload_type=group_offload_type,
            group_offload_use_stream=group_offload_use_stream,
            group_offload_num_blocks_per_group=group_offload_num_blocks_per_group,
        )
        self.pipe = Krea2Pipeline.from_pretrained(repo, torch_dtype=self.policy.dtype)

        self.policy.apply_to_pipeline(self.pipe)
        print(f"Krea 2 loaded from {repo}")

    def pre_generate(self, **kwargs: Any) -> None:
        """Replace static LoRAs with any request-time adapters."""
        dynamic_loras = kwargs.get("loras")
        if not dynamic_loras:
            return
        self.unload_loras()
        try:
            self.load_loras_sync(dynamic_loras)
        except Exception:
            self.restore_static_loras()
            raise

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 8,
        guidance_scale: float = 0.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate an image using Krea 2 Turbo defaults."""
        if kwargs.get("init_image") is not None or kwargs.get("mask_image") is not None:
            raise ValueError("Krea 2 supports text-to-image only")
        return super().generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )

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
        """Build Krea 2 generation arguments."""
        print(f"Krea 2 generating: '{prompt[:50]}...'")
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

    def post_generate(self, **kwargs: Any) -> None:
        """Reset LoRA state after generation to prevent state leakage."""
        super().post_generate(**kwargs)
        self.restore_static_loras()
