# Oneiro

Discord bot for image generation using Hugging Face Diffusers with Civitai integration.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
export TOKEN="your-discord-bot-token"
python -m oneiro
```

Or after installation:

```bash
oneiro
```

## Configuration

Config uses layered TOML with hot-reload:

- **Base config**: Required, primary settings (`config.toml`)
- **Overlay config**: Optional, overrides base values
- **State file**: JSON, runtime-persisted values

The bundled config includes Krea 2 Turbo and Raw profiles. The official model repositories
may require accepting their Hugging Face license and setting `HF_TOKEN` before loading.

### Modular pipeline migration

Oneiro uses released `diffusers>=0.40.0` (no VCS dependency) and native family workflows.
Hosted model types are `flux1`, `flux2`, `flux2-klein`, `qwen`, `krea2`, and `zimage`.
`type = "civitai"` selects the checkpoint source, **not** its architecture: SDXL/Pony/
Illustrious and SD3/3.5 are retained through that source, alongside the hosted families.

| Actual family | Product workflows |
|---------------|-------------------|
| FLUX.1 dev/schnell | Text, denoising image-to-image |
| FLUX.2 / Klein distilled/base | Text, image-conditioned generation (not denoising) |
| Qwen-Image | Text, denoising image-to-image, masks |
| Krea 2 Raw/Turbo | Text, denoising image-to-image, masks, reference conditioning |
| Z-Image Turbo recipe | Text, denoising image-to-image, native masks |
| SDXL/Pony/Illustrious checkpoints | Text, denoising image-to-image, masks |
| SD3/3.5 checkpoints | Text, denoising image-to-image |

Before reusing an older config or persisted state:

1. Remove SD1/SD2, PixArt, Kolors, Hunyuan-DiT, Lumina, and AuraFlow profiles. Their
   backend mappings are retired. Missing/unknown CivitAI `base_model` metadata is an error,
   never an SDXL fallback; a remote family's metadata cannot be bypassed with an override.
2. Remove checkpoint `pipeline_class` and `sequential_cpu_offload`. Use a known
   `base_model` for local files and `offload_type = "sequential"` when desired. Remove
   legacy capability/default toggles such as `supports_negative_prompt`,
   `requires_negative`, and `quantized`; native inputs and component configs decide these.
3. Keep `repo` for hosted models; use `checkpoint_path` or `civitai_model_id` plus optional
   `civitai_version_id` for checkpoints. `component_repo` supplies missing native components;
   `single_file_config_repo` supplies single-file conversion config. Existing family-specific
   component overrides (for example `krea2_component_repo` and `sdxl_component_repo`) remain.
   `/fetch` persists the resolved `family`, `variant`, and `component_repo` together with
   those CivitAI IDs, not legacy pipeline-class/offload overrides.
4. For custom variant-sensitive sources, specify `variant`: FLUX.1 `dev`/`schnell`,
   hosted FLUX.2 `dev`, Klein `distilled`/`base`, Krea `raw`/`turbo`, Qwen `image`,
   Z-Image `turbo`. These sources resolve known official variants and reject explicit
   contradictions. FLUX.2 checkpoints always select the native FLUX.2 graph from
   `base_model`; their component-source `variant` is neither required nor validated.
   Repository-name substrings are not a recipe detector.

The retained profiles in `config.toml` keep their sampling defaults: Krea Turbo 8 steps/0.0
guidance, Raw 28/4.5, Klein distilled 4/1.0 (base 50/4.0). Discord uses profile `steps` and
`guidance_scale`, with `/model` persisted overrides and `/dream` per-request overrides.
Qwen's existing `true_cfg_scale` profile key maps to native CFG guidance. Negative prompts
are not synthesized. Controls are validated against the actual loaded recipe again at
execution, so a queued request cannot silently use stale capabilities after a model switch.

CFG and negative prompts are recipe-specific, not a universal switch. Krea Raw accepts
negative prompts and CFG; Turbo rejects negatives and non-recipe guidance. FLUX.1/2 and
Klein do not expose negative prompts here; Schnell and distilled Klein reject unsupported
guidance overrides. Qwen and SDXL/SD3 use native CFG; Qwen and classic SDXL/SD3 checkpoint
guidance `<=1` preserve positive-only/no-CFG sampling, with the native LCM recipe retained.
Z-Image Turbo rejects negative prompts, including masks, because its fixed guidance 0
does not apply negative conditioning. Z-Image mask generation remains its native pipeline,
sharing components, adapters, and placement with the modular owner; source and mask align
to the requested size.

### Placement and quantization

`cpu_offload = true` defaults to `offload_type = "group"` on CUDA for **all** retained
families, including Qwen checkpoints. `"model"` uses native component-manager offload;
`"sequential"` is explicit. `cpu_offload = false` disables offload. Group controls are
`group_offload_type = "leaf_level"` or `"block_level"`, `group_offload_use_stream` (default
true), and `group_offload_num_blocks_per_group` (block-level default 1). Placement is shared,
not reinstalled for each workflow. Streaming is disabled for FP8-bearing components.

Hosted FLUX.2 retains repository-native BNB quantization. Qwen's `transformer` accepts a
local single file or `repo:filename`; `.gguf` selects native GGUF loading. Transformer-only
Qwen/FLUX.2/Klein/Z-Image checkpoint loaders also recognize GGUF. Krea's Comfy checkpoint
conversion preserves FP8 weights, scales, and quantized layer identities using
`comfy-kitchen`; placement does not cast those weights to the activation dtype. Qwen
checkpoint `qwen_transformer_dtype`/`transformer_dtype` remains an explicit conversion knob,
not a promise that every FP8 file format is supported. Compatible LoRAs use the native loader;
textual inversions are limited to supported SDXL and FLUX.1 checkpoint recipes. Incompatible
global embedding auto-load entries warn and skip; model-specific embeddings remain required.
Download and native loading errors still fail rather than risk partially loaded resources.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TOKEN` | Yes | Discord bot token |
| `CONFIG_PATH` | No | Base config path (default: `config.toml`) |
| `CONFIG_OVERLAY_PATH` | No | Overlay config for overrides |
| `STATE_PATH` | No | Runtime state persistence (JSON) |
| `HF_HOME` | No | Hugging Face cache directory |
| `HF_TOKEN` | No | Hugging Face token for gated model repositories |
| `CIVITAI_API_KEY` | No | Civitai API key for downloading restricted models |
| `CIVITAI_CACHE_DIR` | No | Civitai cache directory (default: `~/.cache/civitai`) |
| `ENABLE_PROGRESS_BARS` | No | Set to `1` to enable tqdm progress bars (disabled by default) |

## Discord Commands

| Command | Description |
|---------|-------------|
| `/dream` | Generate an image from a text prompt |
| `/model` | Switch the active model |
| `/queue` | Check your queue status |
| `/config` | Show current configuration |
| `/fetch` | Fetch and auto-configure a model from Civitai URL |

### /dream Parameters

| Parameter | Description |
|-----------|-------------|
| `prompt` | Text prompt for generation (required) |
| `negative_prompt` | What to avoid in the image |
| `image` | Initial image for denoising, or conditioning image for FLUX.2/Klein |
| `mask` | Mask image for supported inpaint models (white = repaint, black = keep) |
| `reference_image` | One optional reference attachment for supported conditioning workflows |
| `strength` | Denoising only: finite `0 < strength <= 1`; omitted resolves to 0.75 at execution |
| `width` | Image width (512, 768, 1024) |
| `height` | Image height (512, 768, 1024) |
| `seed` | Random seed (-1 for random) |
| `steps` / `guidance_scale` | Per-request sampling controls, subject to the native recipe |
| `lora` | LoRA(s) to apply (see below) |
| `scheduler` | Non-default scheduler overrides for SDXL-family checkpoints only |

Masks require `image`; reference and initial/mask images cannot be combined. Reference and
`image_conditioned` workflows reject denoising strength rather than treating it as conditioning
weight. Explicit zero strength is invalid, including before attachment reads or resource work.
Discord exposes one `reference_image`; the backend accepts multiple ordered references for
native workflows that support them. Attachments can be PNG, JPEG, WebP, HEIC/HEIF, AVIF,
TIFF, or BMP, at most 25 MiB each and at most `4096 * 4096` decoded pixels. Image orientation
is preserved; multi-image files use the HEIF primary image or the first TIFF/AVIF frame.
JPEG phone photos with MPF/MPO secondary images use their primary (first) image.
Generated results and input thumbnails are PNG. Completion metadata reports the actual
execution model/workflow and shows strength only for denoising workflows.

### LoRA Usage

The `lora` parameter supports multiple formats:

```
# Named LoRA from config
/dream prompt:"a portrait" lora:my-lora

# Named LoRA with custom weight
/dream prompt:"a portrait" lora:my-lora:0.8

# Direct Civitai reference (downloads on-demand)
/dream prompt:"a portrait" lora:civitai:12345

# Civitai reference with weight
/dream prompt:"a portrait" lora:civitai:12345:0.7

# Multiple LoRAs
/dream prompt:"a portrait" lora:my-lora:0.8,civitai:12345:0.5
```

## Civitai Integration

Oneiro supports downloading and using models from [Civitai](https://civitai.com):

- **LoRAs**: Use in `/dream` via `lora:civitai:<id>` or fetch with `/fetch`
- **Checkpoints**: Fetch with `/fetch` and switch with `/model`
- **Embeddings**: Fetch with `/fetch` for textual inversions

### /fetch Command

Download and auto-configure resources from Civitai:

```
/fetch url:https://civitai.com/models/12345
/fetch url:https://civitai.com/models/12345 name:my-custom-name
```

The command automatically:
- Detects resource type (LoRA, Checkpoint, Embedding)
- Downloads to the cache directory
- Configures the resource in the state file
- Shows usage instructions

### API Key

Some Civitai models require authentication. Set `CIVITAI_API_KEY` to download restricted content:

1. Go to https://civitai.com/user/account
2. Generate an API key
3. Set the environment variable

See `config.toml` for full configuration options with examples.

## Container and acceptance checks

`Containerfile` uses these official, immutable runtime sources (no image substitution):

- `ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21`
- `docker.io/pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime@sha256:c4ab67f95221a342dff0e8ca4543a7b8885f79f7a0029c0e2e39685d5eaf1722`

The image installs into the base Python with a disposable-image PEP 668 override and constrains
Torch to the base's installed wheel. The unused `spin` developer tool is removed because its
Click cap conflicts with required Hugging Face Hub. The final Oneiro wheel install uses
`--no-deps`. No shadow virtualenv or replacement Torch is installed.

```bash
podman build -f Containerfile -t localhost/oneiro:modernization .
podman run --rm -i --network=none \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=-1 \
  --entrypoint python localhost/oneiro:modernization - < tests/runtime_container_smoke.py
```

The smoke check imports the **installed** package, native blocks, backport and quantization
modules, verifies one unchanged CUDA-enabled Torch distribution and packaged license/provenance,
checks native AVIF support and the maintained HEIF codec,
and runs `uv pip check --system`. It loads no model assets, connects no bot, mounts no production
config, and allocates no GPU. The wheel gate builds offline using installed Setuptools:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/test_packaging.py::test_backport_is_in_built_wheel -q
```

These checks and tiny local-weight workflow tests validate packaging/interfaces, **not** real
model inference, FP8/GGUF GPU kernels, VRAM/throughput, or image quality. Those remain unverified;
no deployment or production inference is implied by a successful import.

### Local Krea backport

The isolated [Krea backport](src/oneiro/pipelines/backports/krea2/PROVENANCE.md) derives from
Diffusers PR #14370 revision `e67bd2caf337868c70259ad6e4edda5101af2864`. Its eight local modules,
Apache-2.0 [license](src/oneiro/pipelines/backports/krea2/LICENSE), and provenance ship in the wheel.
It reuses released attention fixes and adds image/mask/reference workflows without patching
site-packages. The **original Turbo graph initializes before workflow pruning**, preserving its
native defaults. Remove this backport only when a released Diffusers provides all four Raw/Turbo
workflows and the existing tiny workflow, checkpoint-shape, and initialization tests pass against
the native replacements; then switch the local imports and remove its modules/package-data entry.

## License

Oneiro: MIT - see [LICENSE](LICENSE). The local Krea-derived code retains Apache-2.0 notices
and provenance as linked above. Model repository licenses are separate.

HEIF decoding uses maintained `pillow-heif` (BSD-3-Clause Python code). Its binary wheels
are distributed under GPLv2 because they bundle the x265 encoder; bundled libheif and
libde265 are LGPLv3. The runtime image includes this dependency. Bundled codec licenses
and source references ship in its installed distribution.
