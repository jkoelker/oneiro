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

"""PR #14370's image, mask, strength, and reference preparation additions."""

import torch
from diffusers.modular_pipelines.krea2.modular_pipeline import Krea2ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class Krea2ImageInputsStep(ModularPipelineBlocks):
    """Pack source latents and expand images/masks to the effective prompt batch."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Pack source latents and expand images/masks to the effective prompt batch."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("image_latents"),
            InputParam("processed_mask_image", type_hint=torch.Tensor),
            InputParam.template("height"),
            InputParam.template("width"),
            InputParam.template("num_images_per_prompt", default=1),
            InputParam("batch_size", required=True, type_hint=int),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template("image_latents"),
            OutputParam("processed_mask_image", type_hint=torch.Tensor),
            OutputParam("height", type_hint=int),
            OutputParam("width", type_hint=int),
        ]

    @staticmethod
    def repeat_to_batch_size(
        input_name: str,
        input_tensor: torch.Tensor,
        prompt_batch_size: int,
        num_images_per_prompt: int,
    ) -> torch.Tensor:
        """Repeat one shared image or one image per prompt, rejecting ambiguous batches."""
        if input_tensor.shape[0] == 1:
            repeat_by = prompt_batch_size * num_images_per_prompt
        elif input_tensor.shape[0] == prompt_batch_size:
            repeat_by = num_images_per_prompt
        else:
            raise ValueError(
                f"`{input_name}` must have batch size 1 or {prompt_batch_size}, "
                f"but got {input_tensor.shape[0]}"
            )
        return input_tensor.repeat_interleave(repeat_by, dim=0)

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Pack spatial VAE output without changing its channel/patch ordering."""
        block_state = self.get_block_state(state)
        image_latents = block_state.image_latents
        if image_latents.ndim != 5 or image_latents.shape[2] != 1:
            raise ValueError(
                "`image_latents` must have shape (batch, channels, 1, height, width), "
                f"got {image_latents.shape}"
            )
        image_height = image_latents.shape[-2] * components.vae_scale_factor
        image_width = image_latents.shape[-1] * components.vae_scale_factor
        block_state.height = block_state.height or image_height
        block_state.width = block_state.width or image_width
        if block_state.height != image_height or block_state.width != image_width:
            raise ValueError(
                f"The encoded image is {image_height}x{image_width}, but the requested output is "
                f"{block_state.height}x{block_state.width}."
            )
        p = components.patch_size
        batch_size, channels, _, latent_height, latent_width = image_latents.shape
        image_latents = image_latents[:, :, 0].view(
            batch_size, channels, latent_height // p, p, latent_width // p, p
        )
        image_latents = image_latents.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, (latent_height // p) * (latent_width // p), channels * p * p
        )
        prompt_batch_size = block_state.batch_size // block_state.num_images_per_prompt
        block_state.image_latents = self.repeat_to_batch_size(
            "image_latents", image_latents, prompt_batch_size, block_state.num_images_per_prompt
        )
        if block_state.processed_mask_image is not None:
            block_state.processed_mask_image = self.repeat_to_batch_size(
                "processed_mask_image",
                block_state.processed_mask_image,
                prompt_batch_size,
                block_state.num_images_per_prompt,
            )
        self.set_block_state(state, block_state)
        return components, state


class Krea2ReferenceInputsStep(ModularPipelineBlocks):
    """Pack ordered references and expand them to the effective prompt batch."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Pack ordered references and expand them to the effective prompt batch."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("reference_image_latents", required=True, type_hint=list[torch.Tensor]),
            InputParam.template("num_images_per_prompt", default=1),
            InputParam("batch_size", required=True, type_hint=int),
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
        """Keep reference list order and the PR's channel-major patch layout."""
        block_state = self.get_block_state(state)
        p = components.patch_size
        prompt_batch_size = block_state.batch_size // block_state.num_images_per_prompt
        packed = []
        for reference in block_state.reference_image_latents:
            if reference.ndim != 5 or reference.shape[2] != 1:
                raise ValueError(
                    "Each `reference_image_latents` tensor must have shape "
                    f"(batch, channels, 1, height, width), but got {reference.shape}."
                )
            batch_size, channels, _, latent_height, latent_width = reference.shape
            reference = reference[:, :, 0].view(
                batch_size, channels, latent_height // p, p, latent_width // p, p
            )
            reference = reference.permute(0, 2, 4, 1, 3, 5).reshape(
                batch_size, (latent_height // p) * (latent_width // p), channels * p * p
            )
            packed.append(
                Krea2ImageInputsStep.repeat_to_batch_size(
                    "reference_image_latents",
                    reference,
                    prompt_batch_size,
                    block_state.num_images_per_prompt,
                )
            )
        block_state.reference_image_latents = packed
        self.set_block_state(state, block_state)
        return components, state


class Krea2ApplyStrengthStep(ModularPipelineBlocks):
    """Truncate the native schedule using the PR's fractional-step rounding."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Truncate the native schedule for image-to-image or inpainting strength."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("strength", default=0.9),
            InputParam.template("num_inference_steps", required=True),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("timesteps", type_hint=torch.Tensor),
            OutputParam("num_inference_steps", type_hint=int),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Select remaining timesteps and set the scheduler's matching begin index."""
        block_state = self.get_block_state(state)
        if block_state.strength < 0 or block_state.strength > 1:
            raise ValueError(f"`strength` must be in [0.0, 1.0], but is {block_state.strength}")
        init_timestep = min(
            block_state.num_inference_steps * block_state.strength, block_state.num_inference_steps
        )
        t_start = int(max(block_state.num_inference_steps - init_timestep, 0))
        begin_index = t_start * components.scheduler.order
        block_state.timesteps = block_state.timesteps[begin_index:]
        block_state.num_inference_steps -= t_start
        if block_state.num_inference_steps < 1:
            raise ValueError(
                f"After applying `strength={block_state.strength}`, the number of denoising steps "
                f"is {block_state.num_inference_steps}, but it must be at least 1."
            )
        components.scheduler.set_begin_index(begin_index)
        self.set_block_state(state, block_state)
        return components, state


class Krea2PrepareImageLatentsStep(ModularPipelineBlocks):
    """Mix sampled noise with normalized packed source latents at the selected time."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Add noise at the first selected timestep to packed Krea 2 image latents."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("latents", required=True),
            InputParam.template("image_latents", required=True),
            InputParam("timesteps", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("initial_noise", type_hint=torch.Tensor),
            OutputParam.template("latents"),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Preserve the original noise for subsequent unmasked inpainting blends."""
        block_state = self.get_block_state(state)
        if block_state.image_latents.shape != block_state.latents.shape:
            raise ValueError(
                "`image_latents` and `latents` must have the same shape, got "
                f"{block_state.image_latents.shape} and {block_state.latents.shape}"
            )
        latent_timestep = block_state.timesteps[:1].repeat(block_state.latents.shape[0])
        block_state.initial_noise = block_state.latents
        block_state.latents = components.scheduler.scale_noise(
            block_state.image_latents, latent_timestep, block_state.initial_noise
        )
        self.set_block_state(state, block_state)
        return components, state


class Krea2PrepareMaskLatentsStep(ModularPipelineBlocks):
    """Resize and pack masks with black preservation and white repaint semantics."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Resize and pack a preprocessed inpainting mask into Krea 2 image-token space."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("processed_mask_image", required=True, type_hint=torch.Tensor),
            InputParam.template("height", required=True),
            InputParam.template("width", required=True),
            InputParam.template("dtype"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("mask", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Repeat the nearest-neighbor latent mask across channels before packing."""
        block_state = self.get_block_state(state)
        p = components.patch_size
        latent_height = block_state.height // components.vae_scale_factor
        latent_width = block_state.width // components.vae_scale_factor
        mask = torch.nn.functional.interpolate(
            block_state.processed_mask_image, size=(latent_height, latent_width), mode="nearest"
        )
        channels = components.transformer.config.in_channels // (p**2)
        mask = mask.repeat(1, channels, 1, 1).to(
            device=components._execution_device, dtype=block_state.dtype
        )
        batch_size = mask.shape[0]
        mask = mask.view(batch_size, channels, latent_height // p, p, latent_width // p, p)
        mask = mask.permute(0, 2, 4, 1, 3, 5)
        block_state.mask = mask.reshape(
            batch_size, (latent_height // p) * (latent_width // p), channels * p * p
        )
        self.set_block_state(state, block_state)
        return components, state


class Krea2PrepareReferencePositionIdsStep(ModularPipelineBlocks):
    """Build the PR's [text | ordered reference frames | target frame 0] positions."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Build rotary position ids for a [text | reference | target] Krea 2 sequence."

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("height", default=1024),
            InputParam.template("width", default=1024),
            InputParam("prompt_embeds", required=True, type_hint=torch.Tensor),
            InputParam("reference_image_latents", required=True, type_hint=list[torch.Tensor]),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("position_ids", type_hint=torch.Tensor)]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Require references resized to the target grid and assign frames in list order."""
        block_state = self.get_block_state(state)
        device = components._execution_device
        p = components.patch_size
        grid_height = block_state.height // (components.vae_scale_factor * p)
        grid_width = block_state.width // (components.vae_scale_factor * p)
        image_seq_len = grid_height * grid_width
        if any(
            reference.shape[1] != image_seq_len for reference in block_state.reference_image_latents
        ):
            reference_lengths = [
                reference.shape[1] for reference in block_state.reference_image_latents
            ]
            raise ValueError(
                "Each packed reference image and the target must have the same token count, "
                f"but got reference lengths {reference_lengths} and target length {image_seq_len}."
            )
        text_ids = torch.zeros(block_state.prompt_embeds.shape[1], 3, device=device)
        image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
        image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        reference_ids = []
        for frame in range(1, len(block_state.reference_image_latents) + 1):
            ids = image_ids.clone()
            ids[..., 0] = frame
            reference_ids.append(ids.reshape(image_seq_len, 3))
        target_ids = image_ids.clone()
        target_ids[..., 0] = 0
        block_state.position_ids = torch.cat(
            [text_ids, *reference_ids, target_ids.reshape(image_seq_len, 3)], dim=0
        )
        self.set_block_state(state, block_state)
        return components, state
