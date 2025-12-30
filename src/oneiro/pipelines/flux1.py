"""FLUX.1 pipeline wrapper with CPU offloading."""

from typing import Any

import torch

from oneiro.pipelines.base import BasePipeline, GenerationResult


class Flux1PipelineWrapper(BasePipeline):
    """Wrapper for FLUX.1 (dev/schnell variants).

    FLUX.1 uses:
    - FluxPipeline from diffusers
    - FluxTransformer2DModel (MMDiT architecture)
    - T5-v1.1-XXL + CLIP ViT-L/14 text encoders
    - AutoencoderKL VAE
    - FlowMatchEulerDiscreteScheduler

    Variants:
    - FLUX.1-dev: High-quality distilled (steps=28, guidance_scale=3.5)
    - FLUX.1-schnell: Fast 4-step generation (steps=4, guidance_scale=0.0)
    """

    def load(self, model_config: dict[str, Any]) -> None:
        """Load FLUX.1 model with optional CPU offloading and LoRA support.

        Args:
            model_config: Configuration dict with keys:
                - repo: HuggingFace repo ID (default: "black-forest-labs/FLUX.1-dev")
                - cpu_offload: Enable CPU offloading (default: True)
                - cpu_utilization: Fraction of CPU cores to use (default: 0.75)
                - lora: LoRA repository ID (optional)
                - lora_weights: LoRA weights filename (optional, required if lora is set)
        """
        from diffusers import FluxPipeline

        repo = model_config.get("repo", "black-forest-labs/FLUX.1-dev")
        cpu_offload = model_config.get("cpu_offload", True)
        cpu_utilization = model_config.get("cpu_utilization", 0.75)

        print(f"Loading FLUX.1 from {repo}")

        # Configure CPU threading for text encoder
        self._configure_cpu_threads(cpu_utilization)

        print("  Creating pipeline...")
        self.pipe = FluxPipeline.from_pretrained(
            repo,
            torch_dtype=self._dtype,
        )

        if cpu_offload:
            self.pipe.enable_model_cpu_offload()

        # Memory optimization for large T5 encoder and high-res VAE decoding
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        # Load LoRA if specified
        lora_repo = model_config.get("lora")
        lora_weights = model_config.get("lora_weights")

        if lora_repo and lora_weights:
            print(f"Loading LoRA from {lora_repo}")
            self.pipe.load_lora_weights(lora_repo, weight_name=lora_weights)

        print(f"FLUX.1 loaded from {repo}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 28,
        guidance_scale: float = 3.5,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate image with FLUX.1.

        Args:
            prompt: Text prompt for generation.
            negative_prompt: Not used by FLUX.1 but accepted for API compatibility.
            width: Output image width (default: 1024).
            height: Output image height (default: 1024).
            seed: Random seed (-1 for random).
            steps: Number of inference steps (default: 28 for dev, 4 for schnell).
            guidance_scale: Guidance scale (default: 3.5 for dev, 0.0 for schnell).
            **kwargs: Additional parameters (init_image, strength for img2img).

        Returns:
            GenerationResult with generated image and metadata.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        actual_seed, generator = self._prepare_seed(seed)

        # Handle img2img
        init_image = self._load_init_image(kwargs.get("init_image"))
        strength = kwargs.get("strength", 0.75)

        if init_image:
            print(f"FLUX.1 img2img: '{prompt[:50]}...' seed={actual_seed} strength={strength}")
            result = self.pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=512,
            )
        else:
            print(f"FLUX.1 generating: '{prompt[:50]}...' seed={actual_seed}")
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                max_sequence_length=512,
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
