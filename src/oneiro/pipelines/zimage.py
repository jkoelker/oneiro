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

    supports_inpaint = True

    def __init__(self) -> None:
        super().__init__()
        self.img2img_pipe: Any = None
        self.inpaint_pipe: Any = None
        self._active_pipe: Any = None

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load Z-Image-Turbo model."""
        from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline, ZImagePipeline

        repo = model_config.get("repo", "Tongyi-MAI/Z-Image-Turbo")
        cpu_offload = model_config.get("cpu_offload", True)

        self.policy = DevicePolicy.auto_detect(cpu_offload=cpu_offload)

        print(f"Loading Z-Image from {repo}")

        self.pipe = ZImagePipeline.from_pretrained(
            repo,
            torch_dtype=self.policy.dtype,
        )
        self.img2img_pipe = ZImageImg2ImgPipeline(**self.pipe.components)
        self.inpaint_pipe = ZImageInpaintPipeline(**self.pipe.components)
        self._active_pipe = self.pipe

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
        mask_image = kwargs.pop("mask_image", None)
        if mask_image is not None and init_image is None:
            raise ValueError("Z-Image inpainting requires both image and mask_image")

        if init_image:
            if mask_image is not None:
                print(f"Z-Image inpaint: '{prompt[:50]}...' strength={strength}")
            else:
                print(f"Z-Image img2img: '{prompt[:50]}...' strength={strength}")

            gen_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": init_image,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": 0.0,  # Always 0.0 for Turbo
                "generator": generator,
            }
            if mask_image is not None:
                gen_kwargs["mask_image"] = mask_image
            return gen_kwargs
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

    def run_inference(self, gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        """Run the correct Z-Image pipeline for text, image, or inpaint generation."""
        if "mask_image" in gen_kwargs:
            if self.inpaint_pipe is None:
                raise RuntimeError("Z-Image inpaint pipeline not loaded")
            self._active_pipe = self.inpaint_pipe
            return self._active_pipe(**gen_kwargs)

        if is_img2img:
            if self.img2img_pipe is None:
                raise RuntimeError("Z-Image img2img pipeline not loaded")
            self._active_pipe = self.img2img_pipe
            return self._active_pipe(**gen_kwargs)

        self._active_pipe = self.pipe
        return super().run_inference(gen_kwargs, is_img2img)

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

    def _reset_model_state(self) -> None:
        """Reset hooks on the Z-Image pipeline variant used for this generation."""
        if self._active_pipe is None:
            return
        self._active_pipe.maybe_free_model_hooks()

    def unload(self) -> None:
        """Free all Z-Image pipeline wrappers and shared components."""
        self._active_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        super().unload()
