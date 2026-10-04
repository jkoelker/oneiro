# Copyright 2026 HuggingFace Inc.
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
# Modified for Oneiro: offline invariants, not the upstream test harness.

"""Offline tiny-tensor checks for the upstream-derived Krea image workflows."""

import socket
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Transformer2DModel,
)
from diffusers.image_processor import InpaintProcessor, VaeImageProcessor
from diffusers.modular_pipelines.krea2 import Krea2ModularPipeline, Krea2TurboModularPipeline
from diffusers.modular_pipelines.modular_pipeline import BlockState, PipelineState
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image
from transformers import BatchEncoding

from oneiro.pipelines.backports.krea2 import (
    BackportedKrea2Transformer2DModel,
    Krea2AutoBlocks,
    Krea2TurboAutoBlocks,
)

# From PR #14370's Krea2TransformerTesterConfig, not a downloaded model config.
TINY_CONFIG = {
    "in_channels": 16,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "timestep_embed_dim": 8,
    "text_hidden_dim": 16,
    "num_text_layers": 3,
    "text_num_attention_heads": 2,
    "text_num_key_value_heads": 1,
    "text_intermediate_size": 16,
    "num_layerwise_text_blocks": 1,
    "num_refiner_text_blocks": 1,
    "axes_dims_rope": (4, 2, 2),
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make accidental model/config/tokenizer downloads fail rather than hit the network."""

    def disallow_connection(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access is forbidden in the backport gate")

    monkeypatch.setattr(socket.socket, "connect", disallow_connection)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(autouse=True)
def cpu_threads() -> Iterator[None]:
    """Avoid thread-pool overhead on tiny local tensors."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_inputs() -> dict[str, torch.Tensor]:
    """Build the PR's 2x2 grid with one padded text token."""
    generator = torch.Generator().manual_seed(0)
    positions = torch.zeros(8, 3)
    positions[4:, 1] = torch.tensor([0, 0, 1, 1])
    positions[4:, 2] = torch.tensor([0, 1, 0, 1])
    return {
        "hidden_states": torch.randn(1, 4, 16, generator=generator),
        "encoder_hidden_states": torch.randn(1, 4, 3, 16, generator=generator),
        "timestep": torch.tensor([0.5]),
        "position_ids": positions,
        "encoder_attention_mask": torch.tensor([[True, True, True, False]]),
    }


@pytest.mark.parametrize(
    ("blocks_class", "pipeline_class"),
    [(Krea2AutoBlocks, Krea2ModularPipeline), (Krea2TurboAutoBlocks, Krea2TurboModularPipeline)],
)
def test_krea_workflow_declarations(blocks_class: type, pipeline_class: type) -> None:
    """Catch omitted image branches or composition with the wrong native pipeline."""
    blocks = blocks_class()
    assert {
        "text2image",
        "image2image",
        "inpainting",
        "reference",
    } <= set(blocks.available_workflows)
    assert type(blocks.init_pipeline()) is pipeline_class


@torch.no_grad()
def test_backport_transformer_preserves_checkpoint_keys(tmp_path: Path) -> None:
    """Catch changed checkpoint layouts, text computation, or unsliced reference output."""
    native = Krea2Transformer2DModel(**TINY_CONFIG).eval()
    local = BackportedKrea2Transformer2DModel(**TINY_CONFIG).eval()
    native_shapes = {name: tuple(value.shape) for name, value in native.state_dict().items()}
    backport_shapes = {name: tuple(value.shape) for name, value in local.state_dict().items()}
    assert native_shapes == backport_shapes
    local.load_state_dict(native.state_dict(), strict=True)
    assert dict(native.config) == dict(local.config)
    inputs = model_inputs()
    native_text_output = native(**inputs).sample
    backport_text_output = local(**inputs).sample
    torch.testing.assert_close(native_text_output, backport_text_output)
    native.save_pretrained(tmp_path)
    reloaded = BackportedKrea2Transformer2DModel.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(reloaded(**inputs).sample, native_text_output)
    reference = inputs["hidden_states"].clone()
    positions = inputs["position_ids"]
    reference_positions = positions[4:].clone()
    reference_positions[:, 0] = 1
    inputs["position_ids"] = torch.cat([positions[:4], reference_positions, positions[4:]])
    reference_output = local(**inputs, reference_hidden_states=[reference]).sample
    assert reference_output.shape == inputs["hidden_states"].shape
    # Both paths must retain temporary native LoRA scale and restore adapter state.
    for model in (native, local):
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["img_in"]))
    for name, parameter in native.named_parameters():
        if "lora_B" in name:
            parameter.fill_(0.2)
    local.load_state_dict(native.state_dict(), strict=True)
    text_inputs = model_inputs()
    for scale in (0.0, 0.5, 1.0):
        torch.testing.assert_close(
            native(**text_inputs, attention_kwargs={"scale": scale}).sample,
            local(**text_inputs, attention_kwargs={"scale": scale}).sample,
        )
    ref_zero = local(
        **inputs, reference_hidden_states=[reference], attention_kwargs={"scale": 0.0}
    ).sample
    ref_one = local(
        **inputs, reference_hidden_states=[reference], attention_kwargs={"scale": 1.0}
    ).sample
    assert not torch.allclose(ref_zero, ref_one)
    torch.testing.assert_close(local(**inputs, reference_hidden_states=[reference]).sample, ref_one)
    for module in local.modules():
        if isinstance(module, BaseTunerLayer):
            assert module.scaling["default"] == 1.0


class TinyTokenizer:
    """Deterministic text double with padding and the Qwen vision-token contract."""

    def __call__(self, texts: list[str], **kwargs: Any) -> BatchEncoding:
        """Emit a system prefix, variable prompt, padding, and assistant suffix."""
        if texts[0] == "<|im_end|>\n<|im_start|>assistant\n":
            ids = torch.full((len(texts), 5), 3)
            return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        lengths = [36 + text.count("<|image_pad|>") + len(text) % 3 for text in texts]
        length = kwargs.get("max_length", max(lengths))
        ids = torch.zeros(len(texts), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, (text, count) in enumerate(zip(texts, lengths, strict=True)):
            count = min(count, length)
            ids[row, :count] = 1
            ids[row, 34 : 34 + text.count("<|image_pad|>")] = 2
            mask[row, :count] = 1
        return BatchEncoding({"input_ids": ids, "attention_mask": mask})

    def convert_tokens_to_ids(self, token: str) -> int:
        """Return the image marker ID understood by the text double."""
        assert token == "<|image_pad|>"
        return 2


class TinyTextEncoder(torch.nn.Module):
    """Avoid tokenizer/model assets while exercising actual encoder blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(16).float(), requires_grad=False)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
        """Supply all 36 layer taps expected by the released text encoder block."""
        features = torch.sin(input_ids[..., None].float() + self.weight / 16)
        if "pixel_values" in kwargs:
            features = features + kwargs["pixel_values"].float().mean() / 10
        return SimpleNamespace(hidden_states=tuple(features + index / 100 for index in range(36)))


@pytest.fixture(params=[Krea2AutoBlocks, Krea2TurboAutoBlocks], ids=["raw", "turbo"])
def tiny_pipeline(request: pytest.FixtureRequest) -> Krea2ModularPipeline:
    """Use upstream tiny model/VAE dimensions and no downloaded assets."""
    from oneiro.pipelines.backports.krea2.encoders import Krea2ReferenceImageProcessor

    torch.manual_seed(0)
    pipe = request.param().init_pipeline()
    pipe.update_components(
        transformer=BackportedKrea2Transformer2DModel(
            **{**TINY_CONFIG, "num_text_layers": 12}
        ).eval(),
        vae=AutoencoderKLQwenImage(
            base_dim=24,
            z_dim=4,
            dim_mult=[1, 2, 4],
            num_res_blocks=1,
            temperal_downsample=[False, True],
            # Non-identity statistics catch omitted normalization.
            latents_mean=[0.1, -0.2, 0.3, -0.4],
            latents_std=[1.5, 0.5, 2.0, 0.75],
        ).eval(),
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        image_processor=VaeImageProcessor(vae_scale_factor=8),
        image_mask_processor=InpaintProcessor(vae_scale_factor=8),
        text_encoder=TinyTextEncoder(),
        tokenizer=TinyTokenizer(),
    )
    # Transformers processors are not Diffusers ConfigMixin objects: register locally,
    # rather than asking update_components() to infer a from_config loading identity.
    pipe.register_components(
        reference_image_processor=Krea2ReferenceImageProcessor(
            size={"shortest_edge": 1024, "longest_edge": 1024}
        )
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def pipeline_inputs() -> dict[str, Any]:
    """Seed each invocation independently."""
    return {
        "prompt": "a squirrel",
        "height": 32,
        "width": 32,
        "max_sequence_length": 8,
        "num_inference_steps": 5,
        "generator": torch.Generator().manual_seed(7),
        "output_type": "pt",
    }


@torch.no_grad()
def test_strength_schedule_and_noise(tiny_pipeline: Krea2ModularPipeline) -> None:
    """Catch schedule rounding, wrong noise mixing, or missing packed VAE normalization."""
    from diffusers.modular_pipelines.krea2.before_denoise import (
        Krea2PrepareLatentsStep,
        Krea2SetTimestepsStep,
        Krea2TurboSetTimestepsStep,
    )

    from oneiro.pipelines.backports.krea2.before_denoise import (
        Krea2ApplyStrengthStep,
        Krea2ImageInputsStep,
        Krea2PrepareImageLatentsStep,
    )
    from oneiro.pipelines.backports.krea2.encoders import Krea2VaeEncoderStep

    pipe = tiny_pipeline
    state = PipelineState(
        values={
            "processed_image": torch.ones(1, 3, 32, 32),
            "batch_size": 1,
            "height": 32,
            "width": 32,
            "dtype": torch.float32,
            "num_inference_steps": 5,
            "strength": 0.5,
            "generator": torch.Generator().manual_seed(7),
        }
    )
    assert Krea2VaeEncoderStep()(pipe, state) == (pipe, state)
    raw = pipe.vae.encode(torch.ones(1, 3, 1, 32, 32)).latent_dist.mode()
    normalized = (raw - torch.tensor([0.1, -0.2, 0.3, -0.4]).view(1, 4, 1, 1, 1)) / (
        torch.tensor([1.5, 0.5, 2.0, 0.75]).view(1, 4, 1, 1, 1)
    )
    torch.testing.assert_close(state.image_latents, normalized)
    Krea2ImageInputsStep()(pipe, state)
    expected_packed = torch.stack(
        [
            normalized[0, :, 0, h : h + 2, w : w + 2].flatten()
            for h in range(0, 8, 2)
            for w in range(0, 8, 2)
        ]
    )[None]
    torch.testing.assert_close(state.image_latents, expected_packed)
    Krea2PrepareLatentsStep()(pipe, state)
    initial_noise = state.latents.clone()
    repeat = PipelineState(
        values={
            "batch_size": 1,
            "height": 32,
            "width": 32,
            "dtype": torch.float32,
            "generator": torch.Generator().manual_seed(7),
        }
    )
    Krea2PrepareLatentsStep()(pipe, repeat)
    torch.testing.assert_close(initial_noise, repeat.latents, rtol=0, atol=0)
    schedule = (
        Krea2TurboSetTimestepsStep()
        if isinstance(pipe, Krea2TurboModularPipeline)
        else Krea2SetTimestepsStep()
    )
    schedule(pipe, state)
    full_schedule = state.timesteps.clone()
    Krea2ApplyStrengthStep()(pipe, state)
    assert state.num_inference_steps == 3
    assert pipe.scheduler.begin_index == 2
    torch.testing.assert_close(state.timesteps, full_schedule[2:])
    sigma = pipe.scheduler.sigmas[2]
    expected = sigma * initial_noise + (1 - sigma) * expected_packed
    Krea2PrepareImageLatentsStep()(pipe, state)
    torch.testing.assert_close(state.initial_noise, initial_noise)
    torch.testing.assert_close(state.latents, expected)
    zero = PipelineState(
        values={
            "num_inference_steps": 5,
            "strength": 0.0,
            "timesteps": full_schedule,
        }
    )
    with pytest.raises(ValueError, match="number of denoising steps is 0"):
        Krea2ApplyStrengthStep()(pipe, zero)
    output = pipe(
        **pipeline_inputs(),
        image=Image.new("RGB", (32, 32), "white"),
        strength=0.5,
        output="images",
    )
    assert output.shape == (1, 3, 32, 32)
    assert pipe(**pipeline_inputs(), output="images").shape == (1, 3, 32, 32)


@torch.no_grad()
def test_mask_preservation(tiny_pipeline: Krea2ModularPipeline) -> None:
    """Catch inverted masks, lost source latents, or omitted crop compositing."""
    from PIL import Image

    from oneiro.pipelines.backports.krea2.before_denoise import Krea2PrepareMaskLatentsStep
    from oneiro.pipelines.backports.krea2.denoise import Krea2LoopAfterDenoiserInpaint

    pipe = tiny_pipeline
    image = Image.new("RGB", (32, 32), "white")
    black = Image.new("L", (32, 32), "black")
    white = Image.new("L", (32, 32), "white")
    low = pipe(**pipeline_inputs(), image=image, mask_image=black, strength=0.5)
    full = pipe(**pipeline_inputs(), image=image, mask_image=black, strength=1.0)
    torch.testing.assert_close(low.latents, low.image_latents, atol=0, rtol=0)
    torch.testing.assert_close(low.images, full.images, atol=1e-6, rtol=0)
    repainted = pipe(**pipeline_inputs(), image=image, mask_image=white, strength=1.0)
    assert not torch.allclose(repainted.latents, repainted.image_latents)
    assert torch.count_nonzero(low.mask) == 0
    assert torch.all(repainted.mask == 1)
    # Independently enumerate six rectangular 2x2 tokens, each with two mask channels.
    mask_state = PipelineState(
        values={
            "processed_mask_image": torch.arange(24).reshape(1, 1, 4, 6).float() / 23,
            "height": 4,
            "width": 6,
            "dtype": torch.float32,
        }
    )
    mask_components = SimpleNamespace(
        patch_size=2,
        vae_scale_factor=1,
        _execution_device=torch.device("cpu"),
        transformer=SimpleNamespace(config=SimpleNamespace(in_channels=8)),
    )
    Krea2PrepareMaskLatentsStep()(mask_components, mask_state)
    expected_mask = (
        torch.tensor(
            [
                [
                    [0, 1, 6, 7, 0, 1, 6, 7],
                    [2, 3, 8, 9, 2, 3, 8, 9],
                    [4, 5, 10, 11, 4, 5, 10, 11],
                    [12, 13, 18, 19, 12, 13, 18, 19],
                    [14, 15, 20, 21, 14, 15, 20, 21],
                    [16, 17, 22, 23, 16, 17, 22, 23],
                ]
            ]
        ).float()
        / 23
    )
    torch.testing.assert_close(mask_state.mask, expected_mask, atol=0, rtol=0)
    assert repainted.images.shape == (1, 3, 32, 32)
    # At intermediate noise levels, a black mask preserves the appropriately noised source.
    pipe.scheduler.set_timesteps(3, mu=0.5)
    pipe.scheduler.set_begin_index(0)
    pipe.scheduler.step(torch.zeros_like(low.latents), pipe.scheduler.timesteps[0], low.latents)
    block_state = BlockState(
        image_latents=low.image_latents,
        initial_noise=low.initial_noise,
        latents=torch.zeros_like(low.latents),
        mask=torch.zeros_like(low.mask),
        timesteps=pipe.scheduler.timesteps,
    )
    sigma = pipe.scheduler.sigmas[1]
    expected = sigma * low.initial_noise + (1 - sigma) * low.image_latents
    Krea2LoopAfterDenoiserInpaint()(pipe, block_state, 0, pipe.scheduler.timesteps[0])
    torch.testing.assert_close(block_state.latents, expected)
    original = Image.new("RGB", (48, 40), "white")
    mask = Image.new("L", original.size, "black")
    mask.paste(255, (18, 14, 30, 26))
    cropped = pipe(
        **{**pipeline_inputs(), "output_type": "pil"},
        image=original,
        mask_image=mask,
        padding_mask_crop=2,
        output="images",
    )
    assert cropped[0].size == original.size
    actual = np.asarray(cropped[0])
    outside = np.asarray(mask) == 0
    np.testing.assert_array_equal(actual[outside], np.asarray(original)[outside])
    assert np.any(actual[~outside] != 255)


@torch.no_grad()
def test_reference_order_and_scaling(tiny_pipeline: Krea2ModularPipeline) -> None:
    """Catch reordered references, ignored per-reference bias, or invalid padding attention."""
    from PIL import Image

    pipe = tiny_pipeline
    references = [Image.new("RGB", (32, 32), color) for color in ("white", "black", "gray")]
    single = pipe(**pipeline_inputs(), reference_image=references[0], reference_attention_scale=2.0)
    as_list = pipe(
        **pipeline_inputs(), reference_image=[references[0]], reference_attention_scale=[2.0]
    )
    torch.testing.assert_close(single.images, as_list.images)
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        seen.append((args[0].detach().clone(), args[3].detach().clone()))

    hook = pipe.transformer.transformer_blocks[0].register_forward_pre_hook(capture)
    try:
        multi = pipe(
            **pipeline_inputs(),
            reference_image=references,
            reference_attention_scale=[1.0, 2.0, 0.0],
        )
    finally:
        hook.remove()
    assert multi.images.shape == (1, 3, 32, 32)
    assert multi.latents.shape == (1, 16, 16)
    assert len(multi.reference_image_latents) == 3
    assert not torch.allclose(single.images, multi.images)
    torch.testing.assert_close(multi.processed_reference_images[0], torch.ones(1, 3, 32, 32))
    torch.testing.assert_close(multi.processed_reference_images[1], -torch.ones(1, 3, 32, 32))
    text_length = multi.prompt_embeds.shape[1]
    for index, packed in enumerate(multi.reference_image_latents):
        start = text_length + index * 16
        torch.testing.assert_close(
            seen[0][0][:, start : start + 16], pipe.transformer.img_in(packed)
        )
        assert torch.all(multi.position_ids[start : start + 16, 0] == index + 1)
        normalized = pipe.vae.encode(
            multi.processed_reference_images[index].unsqueeze(2)
        ).latent_dist.mode()
        normalized = (normalized - torch.tensor([0.1, -0.2, 0.3, -0.4]).view(1, 4, 1, 1, 1)) / (
            torch.tensor([1.5, 0.5, 2.0, 0.75]).view(1, 4, 1, 1, 1)
        )
        expected_packed = torch.stack(
            [
                normalized[0, :, 0, h : h + 2, w : w + 2].flatten()
                for h in range(0, 8, 2)
                for w in range(0, 8, 2)
            ]
        )[None]
        torch.testing.assert_close(packed, expected_packed)
    assert torch.all(multi.position_ids[-16:, 0] == 0)
    bias = seen[0][1]
    assert bias.shape[-2:] == (text_length + 64, text_length + 64)
    assert torch.isfinite(bias[..., text_length:]).all()
    torch.testing.assert_close(
        bias[:, :, -16:, text_length + 16 : text_length + 32],
        torch.full((1, 1, 16, 16), np.log(2.0), dtype=bias.dtype),
    )
    torch.testing.assert_close(
        bias[:, :, -16:, text_length + 32 : text_length + 48],
        torch.full((1, 1, 16, 16), np.log(1e-4), dtype=bias.dtype),
    )
    assert torch.isfinite(bias[:, :, :-16, text_length:]).all()
    masks = [multi.prompt_embeds_mask]
    if pipe.requires_unconditional_embeds:
        assert multi.prompt_embeds_mask.shape == multi.negative_prompt_embeds_mask.shape
        masks.append(multi.negative_prompt_embeds_mask)
        assert any((~mask).any() for mask in masks)
    # Native ClassifierFreeGuidance.prepare_inputs() orders conditional before negative.
    for index, mask in enumerate(masks):
        text_bias = seen[index][1][:, :, :, :text_length]
        if (~mask).any():
            assert torch.isneginf(text_bias[..., ~mask[0]]).all()
        assert torch.isfinite(text_bias[..., mask[0]]).all()
    changed = pipe(
        **pipeline_inputs(), reference_image=references, reference_attention_scale=[1.0, 1.0, 1.0]
    )
    assert not torch.allclose(multi.latents, changed.latents)
    scalar = pipe(**pipeline_inputs(), reference_image=references, reference_attention_scale=2.0)
    repeated = pipe(
        **pipeline_inputs(), reference_image=references, reference_attention_scale=[2.0, 2.0, 2.0]
    )
    torch.testing.assert_close(scalar.latents, repeated.latents)
    batched = pipe(
        **{
            **pipeline_inputs(),
            "prompt": ["a squirrel", "a squirrel"],
            "generator": [torch.Generator().manual_seed(7), torch.Generator().manual_seed(8)],
        },
        reference_image=references,
        reference_attention_scale=[1.0, 2.0, 0.0],
    )
    assert batched.images.shape == (2, 3, 32, 32)
    torch.testing.assert_close(batched.images[:1], multi.images, atol=5e-3, rtol=0)


@torch.no_grad()
def test_reference_validation(tiny_pipeline: Krea2ModularPipeline) -> None:
    """Catch unsafe scales, malformed reference inputs, or ignored position lengths."""
    from PIL import Image

    pipe = tiny_pipeline
    with pytest.raises(ValueError, match="non-empty list"):
        pipe(**pipeline_inputs(), reference_image=[])
    with pytest.raises(ValueError, match="PIL image"):
        pipe(**pipeline_inputs(), reference_image=["not an image"])
    references = [Image.new("RGB", (32, 32), "white")]
    for scale, message in [
        ([1.0, 2.0], "one value per reference"),
        (-1.0, "non-negative"),
        (float("nan"), "finite"),
        ([float("inf")], "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            pipe(**pipeline_inputs(), reference_image=references, reference_attention_scale=scale)
    model = BackportedKrea2Transformer2DModel(**TINY_CONFIG).eval()
    inputs = model_inputs()
    with pytest.raises(ValueError, match="at least one tensor"):
        model(**inputs, reference_hidden_states=[])
    with pytest.raises(ValueError, match="sequence length"):
        model(**inputs, reference_hidden_states=[inputs["hidden_states"]])
    with pytest.raises(ValueError, match="requires `reference_hidden_states`"):
        model(**inputs, reference_attention_scale=2.0)
    with pytest.raises(ValueError, match="sequence length"):
        model(**{**inputs, "position_ids": torch.zeros(7, 3)})


@torch.no_grad()
def test_reference_padding_mask() -> None:
    """Padded text keys must not affect target output on boolean or biased attention."""
    model = BackportedKrea2Transformer2DModel(**TINY_CONFIG).eval()
    inputs = model_inputs()
    positions = inputs["position_ids"]
    inputs["position_ids"] = torch.cat([positions[:4], positions[4:], positions[4:]])
    reference = [inputs["hidden_states"].clone()]
    for scale in (1.0, 2.0, 0.0):
        baseline = model(
            **inputs, reference_hidden_states=reference, reference_attention_scale=scale
        ).sample
        perturbed = {**inputs, "encoder_hidden_states": inputs["encoder_hidden_states"].clone()}
        perturbed["encoder_hidden_states"][:, -1] += 1000
        actual = model(
            **perturbed,
            reference_hidden_states=reference,
            reference_attention_scale=scale,
            return_dict=False,
        )[0]
        torch.testing.assert_close(actual, baseline)
        assert torch.isfinite(actual).all()
