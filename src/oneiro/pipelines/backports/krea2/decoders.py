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

"""Inpaint decoding addition; standard decoding remains the released block."""

import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import InpaintProcessor
from diffusers.models import AutoencoderKLQwenImage
from diffusers.modular_pipelines.krea2.decoders import Krea2DecodeStep
from diffusers.modular_pipelines.krea2.modular_pipeline import Krea2ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam


def _decode_latents(
    components: Krea2ModularPipeline,
    latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unpack and denormalize exactly as the native decoder and PR helper do."""
    vae = components.vae
    p = components.patch_size
    batch_size, _, channels = latents.shape
    latent_height = p * (height // (components.vae_scale_factor * p))
    latent_width = p * (width // (components.vae_scale_factor * p))
    latents = latents.view(
        batch_size, latent_height // p, latent_width // p, channels // (p * p), p, p
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (p * p), 1, latent_height, latent_width).to(
        vae.dtype
    )
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents * latents_std.to(latents) + latents_mean.to(latents)
    return vae.decode(latents, return_dict=False)[0][:, :, 0]


class Krea2InpaintDecodeStep(Krea2DecodeStep):
    """Decode normalized target latents and composite cropped results on original pixels."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLQwenImage),
            ComponentSpec(
                "image_mask_processor",
                InpaintProcessor,
                config=FrozenDict({"vae_scale_factor": 16}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return super().inputs + [InputParam("mask_overlay_kwargs", type_hint=dict)]

    @torch.no_grad()
    def __call__(
        self,
        components: Krea2ModularPipeline,
        state: PipelineState,
    ) -> tuple[Krea2ModularPipeline, PipelineState]:
        """Use the released inpaint processor's resize/crop overlay, not a custom composite."""
        block_state = self.get_block_state(state)
        image = _decode_latents(
            components, block_state.latents, int(block_state.height), int(block_state.width)
        )
        block_state.images = components.image_mask_processor.postprocess(
            image, output_type=block_state.output_type, **(block_state.mask_overlay_kwargs or {})
        )
        self.set_block_state(state, block_state)
        return components, state
