"""Z-Image-Turbo pipeline wrapper with LoRA and embedding support."""

from typing import Any

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config


class ZImagePipelineWrapper(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Wrapper for Z-Image-Turbo pipeline with multi-LoRA and embedding support."""

    def __init__(self) -> None:
        super().__init__()
        self._init_lora_state()
        self._init_embedding_state()

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

        # Load embeddings if full_config provided
        if full_config:
            embeddings = parse_embeddings_from_config(full_config, model_config)
            if embeddings:
                print(f"  Loading {len(embeddings)} embedding(s)...")
                self.load_embeddings_sync(embeddings)

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

        DevicePolicy.clear_cache()

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
