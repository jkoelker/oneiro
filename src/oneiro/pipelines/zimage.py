"""Z-Image Turbo native modular workflows, retaining classic native inpainting."""

import inspect
from typing import Any

from diffusers import ZImageAutoBlocks, ZImageInpaintPipeline
from PIL import Image

from oneiro.pipelines.modular import ModularPipelineWrapper


class ZImagePipelineWrapper(ModularPipelineWrapper):
    """Share every loaded resource and placement hook with the native mask exception."""

    family = "zimage"
    default_steps = 9
    default_guidance_scale = 0.0
    supports_inpaint = True

    def __init__(self) -> None:
        super().__init__()
        self.inpaint_pipe: Any = None

    def validate_config(
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Validate Turbo variant and placement without loading components."""
        repo = model_config.get("repo", "Tongyi-MAI/Z-Image-Turbo")
        variant = model_config.get(
            "variant", "turbo" if repo == "Tongyi-MAI/Z-Image-Turbo" else None
        )
        if variant != "turbo":
            raise ValueError("Z-Image requires variant='turbo' for custom model sources")
        if model_config.get("embeddings") or model_config.get("inline_embeddings"):
            raise ValueError("Z-Image does not support textual inversion embeddings")
        self._component_repo, self.blocks = repo, ZImageAutoBlocks()
        super().validate_config(model_config, full_config)

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load Turbo components once and construct native inpaint before shared placement."""
        self.validate_config(model_config, full_config)
        self.initialize_pipeline(self._component_repo, self.blocks)

    def _initialize_native_inpaint(self) -> None:
        """Pass only classic constructor components, never modular processors or guiders."""
        self.inpaint_pipe = ZImageInpaintPipeline(
            **{
                name: self.pipe.components[name]
                for name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler")
            }
        )

    def workflow_inputs(self, workflow: str) -> set[str]:
        """Declare mask inputs only for the actual native exception, not modular blocks."""
        if workflow == "inpainting":
            return set(inspect.signature(ZImageInpaintPipeline.__call__).parameters) - {
                "self",
                "return_dict",
            }
        return super().workflow_inputs(workflow)

    def validate_request(
        self,
        *,
        has_image: bool = False,
        has_mask: bool = False,
        has_reference: bool = False,
        strength: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Add only the retained native mask workflow to shared capability validation."""
        if has_mask:
            if not has_image or has_reference:
                raise ValueError("Inpainting requires an image and mask, without reference images")
            super().validate_request(has_image=True, strength=strength, **kwargs)
            return "inpainting"
        return super().validate_request(
            has_image=has_image, has_reference=has_reference, strength=strength, **kwargs
        )

    def run_inference(self, gen_kwargs: dict[str, Any], is_img2img: bool) -> Any:
        """Normalize the native mask output into the shared image-result lifecycle."""
        if "mask_image" in gen_kwargs:
            if self.inpaint_pipe is None:
                raise RuntimeError("Z-Image native inpaint pipeline not loaded")
            # Native inpaint preprocesses the source without the requested dimensions.
            size = (gen_kwargs["width"], gen_kwargs["height"])
            gen_kwargs["image"] = gen_kwargs["image"].resize(size, Image.Resampling.LANCZOS)
            gen_kwargs["mask_image"] = gen_kwargs["mask_image"].resize(
                size, Image.Resampling.NEAREST
            )
            return {"images": self.inpaint_pipe(**gen_kwargs).images}
        return super().run_inference(gen_kwargs, is_img2img)

    def unload(self) -> None:
        """Drop the native view before releasing its shared resources and hooks once."""
        self.inpaint_pipe = None
        super().unload()
