# Krea 2 modular image-workflow backport

Derived from [Hugging Face Diffusers PR #14370](https://github.com/huggingface/diffusers/pull/14370),
source repository **lucasruan1618/diffusers**, exact revision
**e67bd2caf337868c70259ad6e4edda5101af2864**. Retrieved 2026-10-04.
Copyright 2026 Krea AI and The HuggingFace Team. All rights reserved.
Upstream test sources: Copyright 2026 HuggingFace Inc.
The derived code is Apache-2.0 licensed; the complete upstream `LICENSE` is
included alongside this document and both are included in Oneiro package data.

## Source paths and SHA-256 of the retrieved, unmodified files

All paths below are relative to the pinned source repository. Files were fetched
from `https://raw.githubusercontent.com/lucasruan1618/diffusers/e67bd2caf337868c70259ad6e4edda5101af2864/`.

| Source path | SHA-256 |
| --- | --- |
| `src/diffusers/models/transformers/transformer_krea2.py` | `c6412e66547361258f068626defb2f56980cec882fc74db57c5716c85e3d7968` |
| `src/diffusers/modular_pipelines/krea2/before_denoise.py` | `b542520f295eb868232ba29981cfef18990bbd911b992757a4fe22183d8f6e2d` |
| `src/diffusers/modular_pipelines/krea2/encoders.py` | `7eaa86a9b921ab09f90acafb5dbbf34ed79ce0bfc28162e3bfa30ab89610ad41` |
| `src/diffusers/modular_pipelines/krea2/denoise.py` | `1067aaf37daf7618083756ca01079e9054c6cb94a4371d891b1b0b41e52f33f8` |
| `src/diffusers/modular_pipelines/krea2/decoders.py` | `035883af2549017f0791511d21eda098580538bb178947219021647ff775f79c` |
| `src/diffusers/modular_pipelines/krea2/modular_blocks_krea2.py` | `a38b30de3fa830e7317d69b97aa1ff887eda77c9792b76538bbbdb6a8c3b4434` |
| `src/diffusers/modular_pipelines/krea2/modular_blocks_krea2_turbo.py` | `8688adf3a6fbafd24d117e20412669a944088d1e92f0f81aa0a8e057efdbb837` |
| `tests/models/transformers/test_models_transformer_krea2.py` | `1e366da9a79cc1aafb672fe5d90b6644e994133d2d5e72657f398ce82a723605` |
| `tests/pipelines/krea2/test_krea2.py` | `34b967a8326b66f6a9aca876217a63101a62c52a58e3b128577eb1a25938bff7` |
| `tests/modular_pipelines/krea2/test_modular_pipeline_krea2.py` | `d6d39ea05207d38e1c5dbd953f42d1d4510bf5377fe83bf88dd898681baf428b` |
| `tests/modular_pipelines/krea2/test_modular_pipeline_krea2_turbo.py` | `850629be13b0f8aa0ef0956e631d95d6caf75c264662995cea7c00429bec6d71` |
| `LICENSE` | `f9e2070c247517b1ddf65f7b11b393484a18da91a958fd18a97bd0f241c3125c` |

## Local modifications and compatibility choices

- Only PR additions are local. Unchanged text encoding, noise sampling, Raw/Turbo
  timestep selection, CFG guidance, scheduler steps, transformer construction,
  normalization, rotary layers, and attention processors come from released
  Diffusers (minimum 0.40.0). No upstream modules or test harness are copied wholesale.
- `BackportedKrea2Transformer2DModel` subclasses the released model without a new
  constructor or parameters. No-reference calls delegate to the released forward,
  including its LoRA-scale handling. Reference calls retain the PR's sequence,
  mask, target-to-reference logarithmic bias with the `1e-4` floor, checkpointed
  block execution, and target-only output slicing, using the same released layers
  and LoRA-scale decorator. In particular, the released unconditional grouped-query
  key/value head expansion is preserved: the PR's older `enable_gqa` processor is
  intentionally **not** copied.
- Safety adaptation: reference scales must be **finite** as well as non-negative.
  The PR's count, empty-reference, and position-length validation is retained.
- Graph imports point either to released blocks or these local additions. Shared
  descriptors and trivial identical methods use inheritance; auto-generated
  documentation is shortened and type hints are added. Computational ordering and
  native `(components, state)` calls are unchanged.
- Raw `init_pipeline()` uses the released native mapping. Turbo explicitly
  constructs `Krea2TurboModularPipeline` with a deep copy of the blocks and the
  same arguments. Evidence: Diffusers 0.40.0's `_krea2_map_fn(None)` returns Raw,
  even for Turbo blocks. Explicit composition avoids that fallback without global
  mapping changes or monkey-patching.
- Standard decoding is imported unchanged. Only the PR's new decode helper and
  inpainting decoder are local, with native VAE denormalization and native
  `InpaintProcessor` resize/crop overlay.
- Project-owned tests use the pinned upstream tiny transformer and VAE dimensions,
  local random weights, non-identity VAE statistics, and deterministic lightweight
  tokenizer/text doubles. Pipeline tests use 12 tapped text layers to match the
  released encoder block. No model, tokenizer, or config assets are downloaded.

This module is isolated: hosted/CivitAI integration and shared lifecycle code are
not changed by this backport. Native loaders can populate the subclass using the
unchanged checkpoint keys and tensor shapes; reference workflows require that
subclass rather than the released transformer's text-only forward.
