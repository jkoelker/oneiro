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

"""Reference forwarding and masked blending additions to released denoising loops."""

import torch
from diffusers.modular_pipelines.krea2.denoise import (
    Krea2DenoiseLoopWrapper,
    Krea2LoopAfterDenoiser,
    Krea2LoopBeforeDenoiser,
    Krea2LoopDenoiser,
    Krea2TurboLoopDenoiser,
)
from diffusers.modular_pipelines.krea2.modular_pipeline import Krea2ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import BlockState, ModularPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class Krea2ReferenceLoopDenoiser(Krea2LoopDenoiser):
    """Forward ordered clean reference tokens through Raw's released CFG guider."""

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam("reference_image_latents", required=True, type_hint=list[torch.Tensor]),
            InputParam("reference_attention_scale", type_hint=float | list[float], default=1.0),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[Krea2ModularPipeline, BlockState]:
        """Apply reference conditioning to both CFG branches without noising the references."""
        transformer = components.transformer
        latents = block_state.latents.to(transformer.dtype)
        timestep = block_state.timestep.to(transformer.dtype)
        references = [
            reference.to(transformer.dtype) for reference in block_state.reference_image_latents
        ]
        guider_inputs = {
            "encoder_hidden_states": (
                block_state.prompt_embeds.to(transformer.dtype),
                block_state.negative_prompt_embeds.to(transformer.dtype)
                if block_state.negative_prompt_embeds is not None
                else None,
            ),
            "encoder_attention_mask": (
                block_state.prompt_embeds_mask,
                block_state.negative_prompt_embeds_mask,
            ),
        }
        components.guider.set_state(
            step=i, num_inference_steps=block_state.num_inference_steps, timestep=t
        )
        guider_state = components.guider.prepare_inputs(guider_inputs)
        for batch in guider_state:
            components.guider.prepare_models(transformer)
            cond_kwargs = {name: getattr(batch, name) for name in guider_inputs}
            batch.noise_pred = transformer(
                hidden_states=latents,
                reference_hidden_states=references,
                reference_attention_scale=block_state.reference_attention_scale,
                timestep=timestep,
                position_ids=block_state.position_ids,
                attention_kwargs=block_state.attention_kwargs,
                return_dict=False,
                **cond_kwargs,
            )[0]
            components.guider.cleanup_models(transformer)
        block_state.noise_pred = components.guider(guider_state).pred
        return components, block_state


class Krea2TurboReferenceLoopDenoiser(Krea2TurboLoopDenoiser):
    """Forward ordered clean references without CFG for the distilled checkpoint."""

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [
            InputParam("reference_image_latents", required=True, type_hint=list[torch.Tensor]),
            InputParam("reference_attention_scale", type_hint=float | list[float], default=1.0),
        ]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[Krea2ModularPipeline, BlockState]:
        """Run the native model layers with the PR's additional reference arguments."""
        transformer = components.transformer
        block_state.noise_pred = transformer(
            hidden_states=block_state.latents.to(transformer.dtype),
            reference_hidden_states=[
                reference.to(transformer.dtype) for reference in block_state.reference_image_latents
            ],
            reference_attention_scale=block_state.reference_attention_scale,
            timestep=block_state.timestep.to(transformer.dtype),
            position_ids=block_state.position_ids,
            attention_kwargs=block_state.attention_kwargs,
            encoder_hidden_states=block_state.prompt_embeds.to(transformer.dtype),
            encoder_attention_mask=block_state.prompt_embeds_mask,
            return_dict=False,
        )[0]
        return components, block_state


class Krea2LoopAfterDenoiserInpaint(ModularPipelineBlocks):
    """Preserve unmasked source latents at the next noise level after each native step."""

    model_name = "krea2"

    @property
    def description(self) -> str:
        return "Preserve unmasked source latents at the next noise level."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("mask", required=True, type_hint=torch.Tensor),
            InputParam.template("image_latents", required=True),
            InputParam("initial_noise", required=True, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        block_state: BlockState,
        i: int,
        t: torch.Tensor,
    ) -> tuple[Krea2ModularPipeline, BlockState]:
        """Blend black-mask source and white-mask generated latents, ending with clean source."""
        image_latents = block_state.image_latents
        if i < len(block_state.timesteps) - 1:
            next_timestep = block_state.timesteps[i + 1]
            image_latents = components.scheduler.scale_noise(
                image_latents, next_timestep.reshape(1), block_state.initial_noise
            )
        block_state.latents = (
            1 - block_state.mask
        ) * image_latents + block_state.mask * block_state.latents
        return components, block_state


class Krea2ReferenceDenoiseStep(Krea2DenoiseLoopWrapper):
    """Compose released schedule steps around Raw's reference-aware model call."""

    block_classes = [Krea2LoopBeforeDenoiser, Krea2ReferenceLoopDenoiser, Krea2LoopAfterDenoiser]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]


class Krea2TurboReferenceDenoiseStep(Krea2DenoiseLoopWrapper):
    """Compose released schedule steps around Turbo's reference-aware model call."""

    block_classes = [
        Krea2LoopBeforeDenoiser,
        Krea2TurboReferenceLoopDenoiser,
        Krea2LoopAfterDenoiser,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser"]


class Krea2InpaintDenoiseStep(Krea2DenoiseLoopWrapper):
    """Use released Raw denoising followed by the PR's unmasked-source blend."""

    block_classes = [
        Krea2LoopBeforeDenoiser,
        Krea2LoopDenoiser,
        Krea2LoopAfterDenoiser,
        Krea2LoopAfterDenoiserInpaint,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser", "inpaint"]


class Krea2TurboInpaintDenoiseStep(Krea2DenoiseLoopWrapper):
    """Use released Turbo denoising followed by the PR's unmasked-source blend."""

    block_classes = [
        Krea2LoopBeforeDenoiser,
        Krea2TurboLoopDenoiser,
        Krea2LoopAfterDenoiser,
        Krea2LoopAfterDenoiserInpaint,
    ]
    block_names = ["before_denoiser", "denoiser", "after_denoiser", "inpaint"]
