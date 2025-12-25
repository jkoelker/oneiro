"""Shared fixtures for oneiro tests."""

from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def base_config_content() -> str:
    """Minimal valid TOML config."""
    return """
[defaults]
model = "test-model"
steps = 20

[queue]
max_global = 100
max_per_user = 20

[blacklist]
words = ["banned", "forbidden"]
allow_in_negative = true
"""


@pytest.fixture
def base_config_file(tmp_path: Path, base_config_content: str) -> Path:
    """Create a temporary base config file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(base_config_content)
    return config_file


@pytest.fixture
def mock_config() -> Mock:
    """Mock Config object with .get() method."""
    config = Mock()
    config.get = Mock(
        side_effect=lambda *keys, default=None: {
            ("blacklist", "words"): ["banned", "forbidden"],
            ("blacklist", "allow_in_negative"): True,
        }.get(keys, default)
    )
    return config
