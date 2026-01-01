"""Z-Image-Turbo pipeline wrapper with LoRA and embedding support."""

from typing import Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config


class ZImagePipelineWrapper(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Wrapper for Z-Image-Turbo pipeline with multi-LoRA and embedding support."""

    def __init__(self) -> None:
        super().__init__()

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load Z-Image-Turbo model."""
        from diffusers import ZImagePipeline

        repo = model_config.get("repo", "Tongyi-MAI/Z-Image-Turbo")
        cpu_offload = model_config.get("cpu_offload", True)

        self.policy = DevicePolicy.auto_detect(cpu_offload=cpu_offload)

        print(f"Loading Z-Image from {repo}")

        self.pipe = ZImagePipeline.from_pretrained(
            repo,
            torch_dtype=self.policy.dtype,
        )

        self.policy.apply_to_pipeline(self.pipe)

        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)
            self.set_static_loras(loras)

        # Load embeddings if full_config provided
        if full_config:
            embeddings = parse_embeddings_from_config(full_config, model_config)
            if embeddings:
                print(f"  Loading {len(embeddings)} embedding(s)...")
                self.load_embeddings_sync(embeddings)

        print(f"Z-Image loaded from {repo}")

    def build_generation_kwargs(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,  # Ignored - always 0.0 for Turbo
        generator: torch.Generator,
        init_image: Image.Image | None,
        strength: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build Z-Image generation kwargs. Forces guidance_scale=0.0 for Turbo."""
        if init_image:
            print(f"Z-Image img2img: '{prompt[:50]}...' strength={strength}")
            return {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": init_image,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": 0.0,  # Always 0.0 for Turbo
                "generator": generator,
            }
        else:
            print(f"Z-Image generating: '{prompt[:50]}...'")
            return {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": 0.0,  # Always 0.0 for Turbo
                "generator": generator,
            }

    def build_result(
        self,
        result: Any,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
        steps: int,
        guidance_scale: float,  # Ignored - always 0.0 for Turbo
    ) -> GenerationResult:
        """Build result with forced guidance_scale=0.0."""
        output_image = result.images[0]
        return GenerationResult(
            image=output_image,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output_image.width,
            height=output_image.height,
            steps=steps,
            guidance_scale=0.0,  # Always 0.0 for Turbo
        )

    def post_generate(self, **kwargs: Any) -> None:
        """Reset LoRA state after generation to prevent state leakage."""
        super().post_generate(**kwargs)
        self.restore_static_loras()
