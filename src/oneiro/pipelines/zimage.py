"""Z-Image-Turbo pipeline wrapper with LoRA support."""

from typing import Any

import torch

from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config


class ZImagePipelineWrapper(LoraLoaderMixin, BasePipeline):
    """Wrapper for Z-Image-Turbo pipeline with multi-LoRA support."""

    def __init__(self) -> None:
        super().__init__()
        self._init_lora_state()

    def load(self, model_config: dict[str, Any]) -> None:
        """Load Z-Image-Turbo model."""
        from diffusers import ZImagePipeline

        repo = model_config.get("repo", "Tongyi-MAI/Z-Image-Turbo")
        print(f"Loading Z-Image from {repo}")

        self.pipe = ZImagePipeline.from_pretrained(
            repo,
            torch_dtype=self._dtype,
        )

        if self._device == "cuda":
            self.pipe.enable_model_cpu_offload()

        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)

        print(f"Z-Image loaded from {repo}")

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
        """Generate image with Z-Image-Turbo."""
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        actual_seed, generator = self._prepare_seed(seed)

        # Handle img2img
        init_image = self._load_init_image(kwargs.get("init_image"))
        strength = kwargs.get("strength", 0.75)

        if init_image:
            print(f"Z-Image img2img: '{prompt[:50]}...' seed={actual_seed} strength={strength}")
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=0.0,  # Always 0.0 for Turbo
                generator=generator,
            )
        else:
            print(f"Z-Image generating: '{prompt[:50]}...' seed={actual_seed}")
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=0.0,  # Always 0.0 for Turbo
                generator=generator,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_image = result.images[0]
        return GenerationResult(
            image=output_image,
            seed=actual_seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output_image.width,
            height=output_image.height,
            steps=steps,
            guidance_scale=0.0,
        )
