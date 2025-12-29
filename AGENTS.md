# AGENTS.md - Oneiro

Guidelines for AI agents working on this Discord bot for image generation with Huggingface Diffusers.

## Quick Reference

```bash
# Install dependencies (dev mode)
uv pip install -e ".[dev]"

# Run all tests
uv run --extra dev pytest -v

# Run single test file
uv run --extra dev pytest tests/test_config.py -v

# Run single test
uv run --extra dev pytest tests/test_config.py::TestConfigLoad::test_load_base_config -v

# Lint & format
ruff check src/                  # Lint check
ruff check src/ --fix            # Auto-fix lint
ruff format src/ --check         # Format check
ruff format src/                 # Auto-format
```

## Project Structure

```
src/oneiro/           # Main package (src layout)
  pipelines/          # Model pipeline implementations (base.py, flux2.py, qwen.py, zimage.py)
  bot.py              # Discord bot setup, slash commands
  config.py           # Layered TOML config with hot reload
  filters.py          # Content filtering
  queue.py            # Async generation queue
tests/                # Test files mirror src structure
  conftest.py         # Shared fixtures
```

## Code Style

- **Python 3.11+** required
- **Line length: 100** max (Ruff handles formatting)

### Type Hints (Required)

```python
def method(self, value: str | None = None) -> bool: ...     # Union syntax
def get_models(self) -> dict[str, Any]: ...                  # Generic types
callback: Callable[[Any], Coroutine[Any, Any, None]]         # Callable types
```

### Imports (isort order)

```python
# 1. Standard library → 2. Third-party → 3. Local (from oneiro...)
from typing import TYPE_CHECKING

if TYPE_CHECKING:               # Circular import pattern
    from oneiro.config import Config

class MyClass:
    def __init__(self, config: "Config"):  # String annotation
```

### Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `GenerationQueue` |
| Functions/Methods | snake_case | `load_model` |
| Constants | UPPER_SNAKE_CASE | `PIPELINE_TYPES` |
| Private | Leading underscore | `_config`, `_worker` |
| Test classes | `Test*` | `TestConfigLoad` |
| Test methods | `test_*` | `test_load_base_config` |

### Dataclasses

```python
@dataclass
class GenerationResult:
    """Result of an image generation."""
    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str | None
```

### Async Patterns

```python
result = await asyncio.to_thread(self.pipeline.generate, prompt)  # Blocking ops
self._worker_task = asyncio.create_task(self._worker())           # Background tasks

async def stop(self) -> None:
    if self._task:
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
```

### Error Handling

```python
if not self.base_path.exists():
    raise FileNotFoundError(f"Base config not found: {self.base_path}")

try:
    await callback(self._config)
except Exception as e:
    print(f"Config callback error: {e}")  # Graceful degradation
```

### Abstract Base Classes

```python
class BasePipeline(ABC):
    @abstractmethod
    def load(self, model_config: dict[str, Any]) -> None: ...
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> GenerationResult: ...
```

## Testing

```python
"""Tests for Config."""

class TestConfigInit:
    """Tests for Config initialization."""
    
    def test_init_with_path_string(self, tmp_path):
        """Config accepts string paths."""
        base = tmp_path / "config.toml"
        base.write_text("[section]\nkey = 'value'\n")
        config = Config(str(base))
        assert config.base_path == base

# Async tests - no decorator needed (asyncio_mode = "auto")
async def test_start_sets_running(self):
    queue = GenerationQueue()
    await queue.start(pipeline)
    assert queue._running is True
    await queue.stop()  # Always cleanup
```

### Fixtures (conftest.py)

```python
@pytest.fixture
def base_config_file(tmp_path: Path, base_config_content: str) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(base_config_content)
    return config_file

@pytest.fixture
def mock_config() -> Mock:
    config = Mock()
    config.get = Mock(side_effect=lambda *keys, default=None: {...}.get(keys, default))
    return config
```

## Common Patterns

### Config Access

```python
model_name = self.config.get("defaults", "model", default="zimage-turbo")
blacklist = self.config.get("blacklist", "words", default=[])
```

### Discord Commands

```python
@bot.slash_command(name="dream", description="Generate an image")
@option("prompt", str, description="The prompt", required=True)
async def dream(ctx: discord.ApplicationContext, prompt: str):
    await ctx.defer()  # Avoid 3-second timeout
    # ... processing ...
    await ctx.followup.send(embed=embed, file=file)
```

## Ruff Config (pyproject.toml)

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["oneiro"]
```
