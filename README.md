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

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TOKEN` | Yes | Discord bot token |
| `CONFIG_PATH` | No | Base config path (default: `config.toml`) |
| `CONFIG_OVERLAY_PATH` | No | Overlay config for overrides |
| `STATE_PATH` | No | Runtime state persistence (JSON) |
| `HF_HOME` | No | Hugging Face cache directory |
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
| `image` | Reference image for img2img |
| `strength` | img2img strength (0.0-1.0, higher = more change) |
| `width` | Image width (512, 768, 1024) |
| `height` | Image height (512, 768, 1024) |
| `seed` | Random seed (-1 for random) |
| `lora` | LoRA(s) to apply (see below) |

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

## License

MIT - see [LICENSE](LICENSE)
