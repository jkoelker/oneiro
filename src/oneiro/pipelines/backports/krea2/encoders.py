# Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for Oneiro: isolated released-stack backport; see PROVENANCE.md.

"""PR #14370's image-grounded text encoding and normalized image VAE encoding."""

import PIL.Image
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import InpaintProcessor, VaeImageProcessor
from diffusers.models import AutoencoderKLQwenImage
from diffusers.modular_pipelines.krea2.encoders import (
    _PROMPT_TEMPLATE_ENCODE_START_IDX,
    KREA2_TEXT_ENCODER_SELECT_LAYERS,
    Krea2TextEncoderStep,
)
from diffusers.modular_pipelines.krea2.modular_pipeline import Krea2ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from transformers import Qwen2VLImageProcessor

_REFERENCE_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    "{}{}<|im_end|>\n<|im_start|>assistant\n"
)


class Krea2ReferenceImageProcessor(Qwen2VLImageProcessor):
    """Construct the PR's Qwen3-VL preprocessing defaults without remote assets."""

    def __init__(
        self,
        size: dict[str, int] | None = None,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        """Match the training-time image normalization and patch sizes."""
        super().__init__(
            size=size or {"longest_edge": 16777216, "shortest_edge": 65536},
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    @property
    def device(self) -> torch.device:
        """Expose only an explicitly assigned processor device, as upstream does."""
        if self._processor_device is None:
            raise AttributeError("Krea2ReferenceImageProcessor is device-independent")
        return self._processor_device

    @device.setter
    def device(self, value: torch.device) -> None:
        self._processor_device = value


class Krea2ReferenceTextEncoderStep(Krea2TextEncoderStep):
    """Encode ordered vision tokens and text, including Raw's negative branch."""

    @property
    def description(self) -> str:
        return "Encode prompts with ordered reference images through Qwen3-VL."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return super().expected_components + [
            ComponentSpec(
                "reference_image_processor",
                Krea2ReferenceImageProcessor,
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", required=True),
            InputParam("negative_prompt", type_hint=str),
            InputParam(
                "reference_image", required=True, type_hint=PIL.Image.Image | list[PIL.Image.Image]
            ),
            InputParam("reference_image_encoder_resolution", type_hint=int, default=768),
        ]

    def _encode_prompt(
        self,
        components: Krea2ModularPipeline,
        prompts: list[str],
        reference_images: PIL.Image.Image | list[PIL.Image.Image],
        encoder_resolution: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the PR's image-token counts, multimodal prompt, and tapped features."""
        if isinstance(reference_images, PIL.Image.Image):
            reference_images = [reference_images]
        if not isinstance(reference_images, list) or not reference_images:
            raise ValueError("`reference_image` must be an image or a non-empty list of images.")
        if not all(isinstance(image, PIL.Image.Image) for image in reference_images):
            raise ValueError("Every item in `reference_image` must be a PIL image.")
        processed_images = []
        for _ in prompts:
            for image in reference_images:
                image = image.convert("RGB")
                if encoder_resolution and max(image.size) > encoder_resolution:
                    scale = encoder_resolution / max(image.size)
                    image = image.resize(
                        (max(16, round(image.width * scale)), max(16, round(image.height * scale))),
                        PIL.Image.Resampling.LANCZOS,
                    )
                processed_images.append(image)
        image_inputs = components.reference_image_processor(
            images=processed_images, return_tensors="pt"
        )
        image_token = "<|image_pad|>"
        image_token_counts = (
            image_inputs.image_grid_thw.prod(dim=1)
            // components.reference_image_processor.merge_size**2
        ).tolist()
        num_references = len(reference_images)
        vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
        texts = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_vision_blocks = ""
            for reference_index in range(num_references):
                count = image_token_counts[prompt_index * num_references + reference_index]
                prompt_vision_blocks += vision_block.replace(image_token, image_token * count)
            texts.append(_REFERENCE_PROMPT_TEMPLATE.format(prompt_vision_blocks, prompt))
        text_inputs = components.tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        image_token_id = components.tokenizer.convert_tokens_to_ids(image_token)
        outputs = components.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image_inputs.pixel_values.to(device),
            image_grid_thw=image_inputs.image_grid_thw.to(device),
            mm_token_type_ids=input_ids.eq(image_token_id).long(),
            output_hidden_states=True,
        )
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in KREA2_TEXT_ENCODER_SELECT_LAYERS], dim=2
        )
        return (
            hidden_states[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:],
            attention_mask[:, _PROMPT_TEMPLATE_ENCODE_START_IDX:].bool(),
        )

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Align positive/negative feature lengths before Raw's CFG attention."""
        block_state = self.get_block_state(state)
        device = components._execution_device
        prompts = (
            [block_state.prompt]
            if isinstance(block_state.prompt, str)
            else list(block_state.prompt)
        )
        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components,
            prompts,
            block_state.reference_image,
            block_state.reference_image_encoder_resolution,
            device,
        )
        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            negative_prompts = block_state.negative_prompt
            if negative_prompts is None:
                negative_prompts = ""
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts] * len(prompts)
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = (
                self._encode_prompt(
                    components,
                    negative_prompts,
                    block_state.reference_image,
                    block_state.reference_image_encoder_resolution,
                    device,
                )
            )
            prompt_length = block_state.prompt_embeds.shape[1]
            negative_length = block_state.negative_prompt_embeds.shape[1]
            if prompt_length < negative_length:
                padding = negative_length - prompt_length
                block_state.prompt_embeds = torch.nn.functional.pad(
                    block_state.prompt_embeds, (0, 0, 0, 0, 0, padding)
                )
                block_state.prompt_embeds_mask = torch.nn.functional.pad(
                    block_state.prompt_embeds_mask, (0, padding), value=False
                )
            elif negative_length < prompt_length:
                padding = prompt_length - negative_length
                block_state.negative_prompt_embeds = torch.nn.functional.pad(
                    block_state.negative_prompt_embeds, (0, 0, 0, 0, 0, padding)
                )
                block_state.negative_prompt_embeds_mask = torch.nn.functional.pad(
                    block_state.negative_prompt_embeds_mask, (0, padding), value=False
                )
        self.set_block_state(state, block_state)
        return components, state


class Krea2TurboReferenceTextEncoderStep(Krea2ReferenceTextEncoderStep):
    """Use the same image-grounded text features without Raw's negative/CFG branch."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [spec for spec in super().expected_components if spec.name != "guider"]

    @property
    def inputs(self) -> list[InputParam]:
        return [param for param in super().inputs if param.name != "negative_prompt"]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("prompt_embeds"), OutputParam.template("prompt_embeds_mask")]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Encode only the conditional prompt for the distilled checkpoint."""
        block_state = self.get_block_state(state)
        prompts = (
            [block_state.prompt]
            if isinstance(block_state.prompt, str)
            else list(block_state.prompt)
        )
        block_state.prompt_embeds, block_state.prompt_embeds_mask = self._encode_prompt(
            components,
            prompts,
            block_state.reference_image,
            block_state.reference_image_encoder_resolution,
            components._execution_device,
        )
        self.set_block_state(state, block_state)
        return components, state


class Krea2ProcessImagesInputStep(ModularPipelineBlocks):
    """Resize and normalize source images using the released VAE processor."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Preprocess an input image for Krea 2 image-to-image generation."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_processor",
                VaeImageProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image", required=True),
            InputParam.template("height"),
            InputParam.template("width"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("processed_image", type_hint=torch.Tensor)]

    @staticmethod
    def check_inputs(height: int | None, width: int | None, multiple: int) -> None:
        """Reject sizes incompatible with the processor's packed-latent grid."""
        if height is not None and height % multiple != 0:
            raise ValueError(f"`height` must be divisible by {multiple}, but is {height}")
        if width is not None and width % multiple != 0:
            raise ValueError(f"`width` must be divisible by {multiple}, but is {width}")

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Use native preprocessing, respecting the pipeline's default output size."""
        block_state = self.get_block_state(state)
        self.check_inputs(
            block_state.height,
            block_state.width,
            components.image_processor.config.vae_scale_factor,
        )
        block_state.processed_image = components.image_processor.preprocess(
            image=block_state.image,
            height=block_state.height or components.default_height,
            width=block_state.width or components.default_width,
        )
        self.set_block_state(state, block_state)
        return components, state


class Krea2InpaintProcessImagesInputStep(Krea2ProcessImagesInputStep):
    """Resize/crop source images and masks together using the native inpaint processor."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            )
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam.template("mask_image", required=True),
            InputParam.template("padding_mask_crop"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return super().intermediate_outputs + [
            OutputParam("processed_mask_image", type_hint=torch.Tensor),
            OutputParam("mask_overlay_kwargs", type_hint=dict),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Carry crop coordinates and original pixels through to the inpaint decoder."""
        block_state = self.get_block_state(state)
        self.check_inputs(
            block_state.height,
            block_state.width,
            components.image_mask_processor.config.vae_scale_factor,
        )
        (
            block_state.processed_image,
            block_state.processed_mask_image,
            block_state.mask_overlay_kwargs,
        ) = components.image_mask_processor.preprocess(
            image=block_state.image,
            mask=block_state.mask_image,
            height=block_state.height or components.default_height,
            width=block_state.width or components.default_width,
            padding_mask_crop=block_state.padding_mask_crop,
        )
        self.set_block_state(state, block_state)
        return components, state


class Krea2VaeEncoderStep(ModularPipelineBlocks):
    """Encode the deterministic VAE mode and apply native per-channel normalization."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Encode a preprocessed image into normalized Krea 2 image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLQwenImage)]

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("processed_image", required=True, type_hint=torch.Tensor)]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("image_latents")]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Add the single-frame dimension, then normalize clean latent modes."""
        block_state = self.get_block_state(state)
        image = block_state.processed_image
        if image.ndim == 4:
            image = image.unsqueeze(2)
        elif image.ndim != 5:
            raise ValueError(f"`processed_image` must have 4 or 5 dimensions, but got {image.ndim}")
        image = image.to(device=components._execution_device, dtype=components.vae.dtype)
        image_latents = components.vae.encode(image).latent_dist.mode()
        latents_mean = torch.tensor(components.vae.config.latents_mean).view(
            1, components.vae.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1
        )
        block_state.image_latents = (
            image_latents - latents_mean.to(image_latents)
        ) / latents_std.to(image_latents)
        self.set_block_state(state, block_state)
        return components, state


class Krea2ReferenceProcessImagesInputStep(Krea2ProcessImagesInputStep):
    """Resize ordered reference images to the target grid for clean VAE conditioning."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "reference_image", required=True, type_hint=PIL.Image.Image | list[PIL.Image.Image]
            ),
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("processed_reference_images", type_hint=list[torch.Tensor])]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Validate the image boundary and preserve conditioning list order."""
        block_state = self.get_block_state(state)
        multiple = components.image_processor.config.vae_scale_factor
        if block_state.height % multiple != 0 or block_state.width % multiple != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {multiple} for reference conditioning."
            )
        reference_images = block_state.reference_image
        if isinstance(reference_images, PIL.Image.Image):
            reference_images = [reference_images]
        if not isinstance(reference_images, list) or not reference_images:
            raise ValueError("`reference_image` must be an image or a non-empty list of images.")
        if not all(isinstance(image, PIL.Image.Image) for image in reference_images):
            raise ValueError("Every item in `reference_image` must be a PIL image.")
        block_state.processed_reference_images = [
            components.image_processor.preprocess(
                image=image, height=block_state.height, width=block_state.width
            )
            for image in reference_images
        ]
        self.set_block_state(state, block_state)
        return components, state


class Krea2ReferenceVaeEncoderStep(Krea2VaeEncoderStep):
    """Encode normalized clean references individually in their conditioning order."""

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("processed_reference_images", required=True, type_hint=list[torch.Tensor])
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("reference_image_latents", type_hint=list[torch.Tensor])]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Use deterministic VAE modes, never sampled or noise-mixed reference latents."""
        block_state = self.get_block_state(state)
        latents_mean = torch.tensor(components.vae.config.latents_mean).view(
            1, components.vae.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1
        )
        block_state.reference_image_latents = []
        for processed in block_state.processed_reference_images:
            image = processed.unsqueeze(2).to(
                device=components._execution_device, dtype=components.vae.dtype
            )
            latents = components.vae.encode(image).latent_dist.mode()
            block_state.reference_image_latents.append(
                (latents - latents_mean.to(latents)) / latents_std.to(latents)
            )
        self.set_block_state(state, block_state)
        return components, state
