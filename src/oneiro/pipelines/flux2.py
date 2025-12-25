"""FLUX.2 pipeline wrapper with CPU offloading."""

from typing import Any

import torch

from oneiro.pipelines.base import BasePipeline, GenerationResult


class Flux2PipelineWrapper(BasePipeline):
    """Wrapper for FLUX.2 with CPU offloading for text encoder."""

    def load(self, model_config: dict[str, Any]) -> None:
        """Load FLUX.2 model with components on CPU for memory efficiency."""
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import Mistral3ForConditionalGeneration

        repo = model_config.get("repo", "diffusers/FLUX.2-dev-bnb-4bit")
        cpu_offload = model_config.get("cpu_offload", True)
        cpu_utilization = model_config.get("cpu_utilization", 0.75)

        print(f"Loading FLUX.2 from {repo}")

        # Configure CPU threading for text encoder
        self._configure_cpu_threads(cpu_utilization)

        # Load transformer and text encoder on CPU first
        print("  Loading transformer on CPU...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            torch_dtype=self._dtype,
            device_map="cpu",
        )

        print("  Loading text encoder on CPU...")
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo,
            subfolder="text_encoder",
            torch_dtype=self._dtype,
            device_map="cpu",
        )

        print("  Creating pipeline...")
        self.pipe = Flux2Pipeline.from_pretrained(
            repo,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=self._dtype,
        )

        if cpu_offload:
            self.pipe.enable_model_cpu_offload()

        print(f"FLUX.2 loaded from {repo}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 28,
        guidance_scale: float = 4.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate image with FLUX.2."""
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        actual_seed, generator = self._prepare_seed(seed)

        # Handle img2img
        init_image = self._load_init_image(kwargs.get("init_image"))
        strength = kwargs.get("strength", 0.75)

        if init_image:
            print(f"FLUX.2 img2img: '{prompt[:50]}...' seed={actual_seed} strength={strength}")
            result = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            print(f"FLUX.2 generating: '{prompt[:50]}...' seed={actual_seed}")
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
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
            guidance_scale=guidance_scale,
        )
