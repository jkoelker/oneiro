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

"""Reference-sequence additions only; all model layers/processors remain released."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import Krea2Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale


class BackportedKrea2Transformer2DModel(Krea2Transformer2DModel):
    """Keep native checkpoint identity while adding ordered reference attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        reference_hidden_states: list[torch.Tensor] | None = None,
        reference_attention_scale: float | list[float] = 1.0,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """Predict target-token velocity, optionally attending to clean references.

        Args:
            hidden_states: Packed noisy target latents (B, L, C).
            encoder_hidden_states: Stacked text features (B, T, layers, dim).
            timestep: Flow time in [0, 1].
            position_ids: Shared (frame, row, column) positions for the full sequence.
            encoder_attention_mask: Boolean text-key padding mask.
            reference_hidden_states: Ordered packed clean reference latents.
            reference_attention_scale: Scalar or one non-negative finite scale per reference.
            attention_kwargs: Native attention options, including temporary LoRA scale.
            return_dict: Return a native output object rather than a tuple.

        Returns:
            Target-only velocity with the same shape as hidden_states.
        """
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must have shape (sequence_length, 3), got {tuple(position_ids.shape)}."
            )
        reference_lengths = (
            [] if reference_hidden_states is None else [x.shape[1] for x in reference_hidden_states]
        )
        if reference_hidden_states is None and reference_attention_scale != 1.0:
            raise ValueError("`reference_attention_scale` requires `reference_hidden_states`.")
        if reference_hidden_states is not None and not reference_hidden_states:
            raise ValueError("`reference_hidden_states` must contain at least one tensor.")
        scales = (
            reference_attention_scale
            if isinstance(reference_attention_scale, list)
            else [reference_attention_scale] * len(reference_lengths)
        )
        if len(scales) != len(reference_lengths):
            raise ValueError(
                "`reference_attention_scale` must contain one value per reference tensor, but got "
                f"{len(scales)} values for {len(reference_lengths)} references."
            )
        if any(not math.isfinite(scale) or scale < 0 for scale in scales):
            raise ValueError("`reference_attention_scale` must be finite and non-negative.")
        sequence_length = (
            encoder_hidden_states.shape[1] + sum(reference_lengths) + hidden_states.shape[1]
        )
        if position_ids.shape[0] != sequence_length:
            raise ValueError(
                f"`position_ids` has sequence length {position_ids.shape[0]}, but the combined "
                f"text, reference, and image sequence has length {sequence_length}."
            )
        if reference_hidden_states is None:
            # Delegate entirely, including the released LoRA-scale decorator.
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                position_ids=position_ids,
                encoder_attention_mask=encoder_attention_mask,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )
        return self._forward_reference(
            hidden_states,
            encoder_hidden_states,
            timestep,
            position_ids,
            encoder_attention_mask,
            reference_hidden_states,
            scales,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    @apply_lora_scale("attention_kwargs")
    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        references: list[torch.Tensor],
        scales: list[float],
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """Apply the PR's reference sequence, target-to-reference bias, and output slice."""
        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]
        reference_lengths = [x.shape[1] for x in references]
        reference_seq_len = sum(reference_lengths)
        sequence_length = text_seq_len + reference_seq_len + image_seq_len
        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = encoder_attention_mask[:, None, None, :]
            image_mask = encoder_attention_mask.new_ones(
                (batch_size, reference_seq_len + image_seq_len)
            )
            attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[
                :, None, None, :
            ]
        encoder_hidden_states = self.text_fusion(
            encoder_hidden_states, attention_mask=text_attention_mask
        )
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        hidden_states = torch.cat(
            [
                encoder_hidden_states,
                *[self.img_in(x) for x in references],
                self.img_in(hidden_states),
            ],
            dim=1,
        )

        if any(scale != 1.0 for scale in scales):
            bias = hidden_states.new_zeros((batch_size, 1, sequence_length, sequence_length))
            target_start = text_seq_len + reference_seq_len
            reference_start = text_seq_len
            for reference_length, scale in zip(reference_lengths, scales, strict=True):
                reference_end = reference_start + reference_length
                bias[:, :, target_start:, reference_start:reference_end] = math.log(
                    max(scale, 1e-4)
                )
                reference_start = reference_end
            if attention_mask is not None:
                bias.masked_fill_(~attention_mask, float("-inf"))
            attention_mask = bias

        image_rotary_emb = self.rotary_emb(position_ids)
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb_mod, image_rotary_emb, attention_mask
                )
            else:
                hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)
        output = self.final_layer(hidden_states[:, -image_seq_len:], temb)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)
