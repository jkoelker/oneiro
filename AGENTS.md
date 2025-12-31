# AGENTS.md - Oneiro

Discord bot for image generation with Huggingface Diffusers.

## Quick Reference

```bash
uv pip install -e ".[dev]"           # Install dev dependencies
uv run --extra dev pytest -v         # Run all tests
uv run --extra dev pytest tests/test_config.py -v  # Single file
ruff check src/ --fix                # Lint + auto-fix
ruff format src/                     # Format
```

## Project Structure

```
src/oneiro/
  pipelines/       # Model pipelines (base.py, flux2.py, qwen.py, zimage.py)
  bot.py           # Discord slash commands
  config.py        # Layered TOML config with hot reload
  filters.py       # Content filtering
  queue.py         # Async generation queue
tests/             # Mirrors src structure
```

## Code Style

- **Python 3.11+**, line length 100 (Ruff enforced)
- **Type hints required** on all functions
- **Google-style docstrings**
- **isort order**: stdlib → third-party → local (`from oneiro...`)

### Naming

| Item | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `GenerationQueue` |
| Functions | snake_case | `load_model` |
| Constants | UPPER_SNAKE | `PIPELINE_TYPES` |
| Private | `_prefix` | `_config` |
| Tests | `Test*`/`test_*` | `TestConfig`, `test_load` |

### Key Patterns

```python
# Config access
model = self.config.get("defaults", "model", default="zimage-turbo")

# Discord commands - always defer first
@bot.slash_command(name="dream", description="Generate an image")
async def dream(ctx: discord.ApplicationContext, prompt: str):
    await ctx.defer()  # Avoid 3-second timeout
    # ... processing ...
    await ctx.followup.send(embed=embed, file=file)

# Blocking ops in async context
result = await asyncio.to_thread(self.pipeline.generate, prompt)

# Task cancellation
async def stop(self) -> None:
    if self._task:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

# Dataclass with mutable default
@dataclass
class ModelVersion:
    files: list[ModelFile] = field(default_factory=list)  # Never default=[]

# TYPE_CHECKING for circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from oneiro.config import Config

class MyClass:
    def __init__(self, config: "Config"): ...
```

### Exception Pattern

```python
class CivitaiError(Exception):
    """Base exception for Civitai errors."""

class CivitaiAuthError(CivitaiError):
    """Auth failed."""

class CivitaiRateLimitError(CivitaiError):
    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after
```

## Testing

```python
class TestConfigInit:
    """Tests for Config initialization."""
    
    def test_accepts_string_path(self, tmp_path):
        base = tmp_path / "config.toml"
        base.write_text("[section]\nkey = 'value'\n")
        assert Config(str(base)).base_path == base

# Async tests - no decorator needed (asyncio_mode = "auto")
async def test_queue_start(self):
    queue = GenerationQueue()
    await queue.start(pipeline)
    assert queue._running is True
    await queue.stop()
```

### Fixtures

```python
@pytest.fixture
def base_config_file(tmp_path: Path) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text("[section]\nkey = 'value'\n")
    return config_file
```

## Ruff Config

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]

[tool.ruff.lint.isort]
known-first-party = ["oneiro"]
```
