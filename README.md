# Oneiro

Discord bot for image generation using Hugging Face Diffusers.

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

Environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `TOKEN` | Yes | Discord bot token |
| `CONFIG_PATH` | No | Base config path (default: `config.toml`) |
| `CONFIG_OVERLAY_PATH` | No | Overlay config for overrides |
| `STATE_PATH` | No | Runtime state persistence (JSON) |
| `HF_HOME` | No | Hugging Face cache directory |

## License

MIT - see [LICENSE](LICENSE)
