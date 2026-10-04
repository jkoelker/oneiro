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

"""Compose Turbo workflows, retaining released Turbo text/schedule/denoising."""

import os
from copy import deepcopy

from diffusers.modular_pipelines.components_manager import ComponentsManager
from diffusers.modular_pipelines.krea2.before_denoise import (
    Krea2PrepareLatentsStep,
    Krea2PreparePositionIdsStep,
    Krea2TurboSetTimestepsStep,
    Krea2TurboTextInputsStep,
)
from diffusers.modular_pipelines.krea2.denoise import Krea2TurboDenoiseStep
from diffusers.modular_pipelines.krea2.encoders import Krea2TurboTextEncoderStep
from diffusers.modular_pipelines.krea2.modular_blocks_krea2_turbo import (
    Krea2TurboCoreDenoiseStep,
)
from diffusers.modular_pipelines.krea2.modular_pipeline import Krea2TurboModularPipeline
from diffusers.modular_pipelines.modular_pipeline import (
    AutoPipelineBlocks,
    SequentialPipelineBlocks,
)

from .before_denoise import (
    Krea2ApplyStrengthStep,
    Krea2ImageInputsStep,
    Krea2PrepareImageLatentsStep,
    Krea2PrepareReferencePositionIdsStep,
    Krea2ReferenceInputsStep,
)
from .denoise import Krea2TurboInpaintDenoiseStep, Krea2TurboReferenceDenoiseStep
from .encoders import Krea2TurboReferenceTextEncoderStep
from .modular_blocks_krea2 import (
    Krea2AutoBlocks,
    Krea2AutoCoreDenoiseStep,
    Krea2AutoDecodeStep,
    Krea2AutoVaeEncoderStep,
    Krea2InpaintPrepareLatentsStep,
)


class Krea2TurboAutoTextEncoderStep(AutoPipelineBlocks):
    """Select image-grounded or unchanged released Turbo text encoding."""

    model_name = "krea2"
    block_classes = [Krea2TurboReferenceTextEncoderStep, Krea2TurboTextEncoderStep]
    block_names = ["reference", "text"]
    block_trigger_inputs = ["reference_image", None]


class Krea2TurboImg2ImgInputStep(SequentialPipelineBlocks):
    """Expand conditional-only text features and source images."""

    model_name = "krea2"
    block_classes = [Krea2TurboTextInputsStep, Krea2ImageInputsStep]
    block_names = ["text_inputs", "image_inputs"]


class Krea2TurboReferenceInputStep(SequentialPipelineBlocks):
    """Expand conditional-only text features and ordered references."""

    model_name = "krea2"
    block_classes = [Krea2TurboTextInputsStep, Krea2ReferenceInputsStep]
    block_names = ["text_inputs", "reference_inputs"]


class Krea2TurboReferenceCoreDenoiseStep(Krea2TurboCoreDenoiseStep):
    """Attend to clean reference tokens on Turbo's fixed time-shift schedule."""

    block_classes = [
        Krea2TurboReferenceInputStep,
        Krea2PrepareLatentsStep,
        Krea2TurboSetTimestepsStep,
        Krea2PrepareReferencePositionIdsStep,
        Krea2TurboReferenceDenoiseStep,
    ]
    block_names = ["input", "prepare_latents", "set_timesteps", "prepare_position_ids", "denoise"]


class Krea2TurboImg2ImgCoreDenoiseStep(Krea2TurboCoreDenoiseStep):
    """Denoise source latents with strength-selected Turbo timesteps, without CFG."""

    block_classes = [
        Krea2TurboImg2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2TurboSetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2PrepareImageLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2TurboDenoiseStep,
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


class Krea2TurboInpaintCoreDenoiseStep(Krea2TurboCoreDenoiseStep):
    """Denoise masked source latents without CFG and preserve their unmasked regions."""

    block_classes = [
        Krea2TurboImg2ImgInputStep,
        Krea2PrepareLatentsStep,
        Krea2TurboSetTimestepsStep,
        Krea2ApplyStrengthStep,
        Krea2InpaintPrepareLatentsStep,
        Krea2PreparePositionIdsStep,
        Krea2TurboInpaintDenoiseStep,
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


class Krea2TurboAutoCoreDenoiseStep(Krea2AutoCoreDenoiseStep):
    """Use upstream workflow precedence with Turbo's native computation."""

    block_classes = [
        Krea2TurboCoreDenoiseStep,
        Krea2TurboReferenceCoreDenoiseStep,
        Krea2TurboInpaintCoreDenoiseStep,
        Krea2TurboImg2ImgCoreDenoiseStep,
    ]


class Krea2TurboAutoBlocks(Krea2AutoBlocks):
    """Expose four Turbo workflows, explicitly composing the native Turbo pipeline."""

    block_classes = [
        Krea2TurboAutoTextEncoderStep,
        Krea2AutoVaeEncoderStep,
        Krea2TurboAutoCoreDenoiseStep,
        Krea2AutoDecodeStep,
    ]

    def init_pipeline(
        self,
        pretrained_model_name_or_path: str | os.PathLike | None = None,
        components_manager: ComponentsManager | None = None,
        collection: str | None = None,
    ) -> Krea2TurboModularPipeline:
        """Avoid the released krea2 map's Raw fallback when no model config is supplied."""
        return Krea2TurboModularPipeline(
            blocks=deepcopy(self),
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            components_manager=components_manager,
            collection=collection,
        )
