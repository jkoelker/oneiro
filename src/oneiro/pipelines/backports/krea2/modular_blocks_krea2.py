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

"""Compose the PR's Raw workflows from local additions and released blocks."""

import torch
from diffusers.modular_pipelines.krea2.before_denoise import (
    Krea2PrepareLatentsStep,
    Krea2PreparePositionIdsStep,
    Krea2SetTimestepsStep,
    Krea2TextInputsStep,
)
from diffusers.modular_pipelines.krea2.decoders import Krea2DecodeStep
from diffusers.modular_pipelines.krea2.denoise import Krea2DenoiseStep
from diffusers.modular_pipelines.krea2.encoders import Krea2TextEncoderStep
from diffusers.modular_pipelines.krea2.modular_blocks_krea2 import (
    Krea2AutoBlocks as ReleasedKrea2AutoBlocks,
)
from diffusers.modular_pipelines.krea2.modular_blocks_krea2 import Krea2CoreDenoiseStep
from diffusers.modular_pipelines.modular_pipeline import (
    AutoPipelineBlocks,
    ConditionalPipelineBlocks,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import OutputParam

from .before_denoise import (
    Krea2ApplyStrengthStep,
    Krea2ImageInputsStep,
    Krea2PrepareImageLatentsStep,
    Krea2PrepareMaskLatentsStep,
    Krea2PrepareReferencePositionIdsStep,
    Krea2ReferenceInputsStep,
)
from .decoders import Krea2InpaintDecodeStep
from .denoise import Krea2InpaintDenoiseStep, Krea2ReferenceDenoiseStep
from .encoders import (
    Krea2InpaintProcessImagesInputStep,
    Krea2ProcessImagesInputStep,
    Krea2ReferenceProcessImagesInputStep,
    Krea2ReferenceTextEncoderStep,
    Krea2ReferenceVaeEncoderStep,
    Krea2VaeEncoderStep,
)


class Krea2AutoTextEncoderStep(AutoPipelineBlocks):
    """Select image-grounded or released text-only encoding."""

    model_name = "krea2"
    block_classes = [Krea2ReferenceTextEncoderStep, Krea2TextEncoderStep]
    block_names = ["reference", "text"]
    block_trigger_inputs = ["reference_image", None]


class Krea2Img2ImgVaeEncoderStep(SequentialPipelineBlocks):
    """Preprocess and VAE-encode a source image."""

    model_name = "krea2"
    block_classes = [Krea2ProcessImagesInputStep, Krea2VaeEncoderStep]
    block_names = ["preprocess", "encode"]


class Krea2InpaintVaeEncoderStep(SequentialPipelineBlocks):
    """Preprocess source and mask together and VAE-encode the source."""

    model_name = "krea2"
    block_classes = [Krea2InpaintProcessImagesInputStep, Krea2VaeEncoderStep]
    block_names = ["preprocess", "encode"]


class Krea2ReferenceVaeEncoderBlocks(SequentialPipelineBlocks):
    """Preprocess and VAE-encode ordered clean references."""

    model_name = "krea2"
    block_classes = [Krea2ReferenceProcessImagesInputStep, Krea2ReferenceVaeEncoderStep]
    block_names = ["preprocess", "encode"]


class Krea2AutoVaeEncoderStep(AutoPipelineBlocks):
    """Prefer references, then masks, then ordinary source images, as upstream does."""

    model_name = "krea2"
    block_classes = [
        Krea2ReferenceVaeEncoderBlocks,
        Krea2InpaintVaeEncoderStep,
        Krea2Img2ImgVaeEncoderStep,
    ]
    block_names = ["reference", "inpaint", "img2img"]
    block_trigger_inputs = ["reference_image", "mask_image", "image"]


class Krea2Img2ImgInputStep(SequentialPipelineBlocks):
    """Expand released text features before packing source image latents."""

    model_name = "krea2"
    block_classes = [Krea2TextInputsStep, Krea2ImageInputsStep]
    block_names = ["text_inputs", "image_inputs"]


class Krea2InpaintPrepareLatentsStep(SequentialPipelineBlocks):
    """Mix initial source noise and pack its mask."""

    model_name = "krea2"
    block_classes = [Krea2PrepareImageLatentsStep, Krea2PrepareMaskLatentsStep]
    block_names = ["add_noise", "prepare_mask"]


class Krea2ReferenceInputStep(SequentialPipelineBlocks):
    """Expand released text features before packing ordered references."""

    model_name = "krea2"
    block_classes = [Krea2TextInputsStep, Krea2ReferenceInputsStep]
    block_names = ["text_inputs", "reference_inputs"]


class Krea2ReferenceCoreDenoiseStep(Krea2CoreDenoiseStep):
    """Generate noisy targets conditioned on clean reference tokens, without strength."""

    block_classes = [
        Krea2ReferenceInputStep,
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2PrepareReferencePositionIdsStep,
        Krea2ReferenceDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "prepare_position_ids", "denoise"]


class Krea2Img2ImgCoreDenoiseStep(Krea2CoreDenoiseStep):
    """Denoise source latents using the strength-selected native schedule."""

    block_classes = [
        Krea2Img2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2PrepareImageLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2DenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "apply_strength",
        "prepare_image_latents",
        "prepare_position_ids",
        "denoise",
    ]


class Krea2InpaintCoreDenoiseStep(Krea2CoreDenoiseStep):
    """Denoise masked source latents with post-step source preservation."""

    block_classes = [
        Krea2Img2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2InpaintPrepareLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2InpaintDenoiseStep,
    ]
    block_names = [
        "input",
        "prepare_latents",
        "set_timesteps",
        "apply_strength",
        "prepare_inpaint_latents",
        "prepare_position_ids",
        "denoise",
    ]


class Krea2AutoCoreDenoiseStep(ConditionalPipelineBlocks):
    """Choose the PR's denoise branch from encoded conditioning inputs."""

    model_name = "krea2"
    block_classes = [
        Krea2CoreDenoiseStep,
        Krea2ReferenceCoreDenoiseStep,
        Krea2InpaintCoreDenoiseStep,
        Krea2Img2ImgCoreDenoiseStep,
    ]
    block_names = ["text2image", "reference", "inpaint", "img2img"]
    block_trigger_inputs = ["reference_image_latents", "processed_mask_image", "image_latents"]
    default_block_name = "text2image"

    def select_block(
        self,
        reference_image_latents: list[torch.Tensor] | None = None,
        processed_mask_image: torch.Tensor | None = None,
        image_latents: torch.Tensor | None = None,
    ) -> str:
        """Match upstream precedence; reference conditioning is not img2img."""
        if reference_image_latents is not None:
            return "reference"
        if processed_mask_image is not None:
            return "inpaint"
        if image_latents is not None:
            return "img2img"
        return "text2image"

    @property
    def outputs(self) -> list[OutputParam]:
        return [OutputParam.template("latents")]


class Krea2AutoDecodeStep(AutoPipelineBlocks):
    """Select cropped inpaint compositing or the unchanged released decoder."""

    model_name = "krea2"
    block_classes = [Krea2InpaintDecodeStep, Krea2DecodeStep]
    block_names = ["inpaint", "default"]
    block_trigger_inputs = ["mask", None]


class Krea2AutoBlocks(ReleasedKrea2AutoBlocks):
    """Expose the PR's four Raw workflows through the native modular pipeline."""

    block_classes = [
        Krea2AutoTextEncoderStep,
        Krea2AutoVaeEncoderStep,
        Krea2AutoCoreDenoiseStep,
        Krea2AutoDecodeStep,
    ]
    block_names = ["text_encoder", "vae_encoder", "denoise", "decode"]
    _workflow_map = {
        "text2image": {"prompt": True},
        "image2image": {"prompt": True, "image": True},
        "inpainting": {"prompt": True, "image": True, "mask_image": True},
        "reference": {"prompt": True, "reference_image": True},
    }
