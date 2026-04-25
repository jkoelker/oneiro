"""FLUX.2 Klein pipeline wrapper with CPU offloading, LoRA, and embedding support."""

from typing import Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config


class Flux2KleinPipelineWrapper(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Wrapper for lightweight FLUX.2 Klein models."""

    def __init__(self) -> None:
        super().__init__()

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load FLUX.2 Klein from a hosted Diffusers repository."""
        from diffusers import Flux2KleinPipeline

        repo = model_config.get("repo", "black-forest-labs/FLUX.2-klein-9B")
        cpu_offload = model_config.get("cpu_offload", True)
        offload_type = model_config.get("offload_type", "group")
        group_offload_type = model_config.get("group_offload_type", "leaf_level")
        group_offload_use_stream = model_config.get("group_offload_use_stream", True)
        group_offload_num_blocks_per_group = model_config.get("group_offload_num_blocks_per_group")
        cpu_utilization = model_config.get("cpu_utilization", 0.75)

        print(f"Loading FLUX.2 Klein from {repo}")

        self._configure_cpu_threads(cpu_utilization)

        self.policy = DevicePolicy.auto_detect(
            cpu_offload=cpu_offload,
            offload_type=offload_type,
            group_offload_type=group_offload_type,
            group_offload_use_stream=group_offload_use_stream,
            group_offload_num_blocks_per_group=group_offload_num_blocks_per_group,
        )

        self.pipe = Flux2KleinPipeline.from_pretrained(
            repo,
            torch_dtype=self.policy.dtype,
        )

        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)
            self.set_static_loras(loras)

        if full_config:
            embeddings = parse_embeddings_from_config(full_config, model_config)
            if embeddings:
                print(f"  Loading {len(embeddings)} embedding(s)...")
                self.load_embeddings_sync(embeddings)

        self.policy.apply_to_pipeline(self.pipe)

        print(f"FLUX.2 Klein loaded from {repo}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 4,
        guidance_scale: float = 1.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate image with current FLUX.2 Klein distilled model-card defaults."""
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
        """Build FLUX.2 Klein generation kwargs."""
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if init_image is not None:
            print(f"FLUX.2 Klein img2img: '{prompt[:50]}...'")
            gen_kwargs["image"] = init_image
        else:
            print(f"FLUX.2 Klein generating: '{prompt[:50]}...'")
        return gen_kwargs

    def post_generate(self, **kwargs: Any) -> None:
        """Reset LoRA state after generation to prevent state leakage."""
        super().post_generate(**kwargs)
        self.restore_static_loras()
