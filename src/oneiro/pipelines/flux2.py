"""FLUX.2 pipeline wrapper with CPU offloading, LoRA, and embedding support."""

from typing import Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config


class Flux2PipelineWrapper(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Wrapper for FLUX.2 with CPU offloading, multi-LoRA, and embedding support."""

    def __init__(self) -> None:
        super().__init__()

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load FLUX.2 model with components on CPU for memory efficiency."""
        from diffusers import Flux2Pipeline, Flux2Transformer2DModel
        from transformers import Mistral3ForConditionalGeneration

        repo = model_config.get("repo", "diffusers/FLUX.2-dev-bnb-4bit")
        cpu_offload = model_config.get("cpu_offload", True)
        cpu_utilization = model_config.get("cpu_utilization", 0.75)

        print(f"Loading FLUX.2 from {repo}")

        # Configure CPU threading for text encoder
        self._configure_cpu_threads(cpu_utilization)

        self.policy = DevicePolicy.auto_detect(cpu_offload=cpu_offload)

        # Load transformer and text encoder on CPU first
        print("  Loading transformer on CPU...")
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            torch_dtype=self.policy.dtype,
            device_map="cpu",
        )

        print("  Loading text encoder on CPU...")
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo,
            subfolder="text_encoder",
            torch_dtype=self.policy.dtype,
            device_map="cpu",
        )

        print("  Creating pipeline...")
        self.pipe = Flux2Pipeline.from_pretrained(
            repo,
            transformer=transformer,
            text_encoder=text_encoder,
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
        negative_prompt: str | None,  # Not used by FLUX.2 but stored in result
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        init_image: Image.Image | None,
        strength: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build FLUX.2 generation kwargs."""
        if init_image:
            print(f"FLUX.2 img2img: '{prompt[:50]}...' strength={strength}")
            return {
                "prompt": prompt,
                "image": init_image,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
        else:
            print(f"FLUX.2 generating: '{prompt[:50]}...'")
            return {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
