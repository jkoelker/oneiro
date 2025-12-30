"""Tests for Config."""

import json

import pytest

from oneiro.config import Config


class TestConfigInit:
    """Tests for Config initialization."""

    def test_init_with_path_string(self, tmp_path):
        """Config accepts string paths."""
        base = tmp_path / "config.toml"
        base.write_text("[section]\nkey = 'value'\n")
        config = Config(str(base))
        assert config.base_path == base

    def test_init_with_path_object(self, tmp_path):
        """Config accepts Path objects."""
        base = tmp_path / "config.toml"
        base.write_text("[section]\nkey = 'value'\n")
        config = Config(base)
        assert config.base_path == base

    def test_init_with_overlay_path(self, tmp_path):
        """Config accepts optional overlay path."""
        base = tmp_path / "config.toml"
        overlay = tmp_path / "overlay.toml"
        base.write_text("[section]\nkey = 'value'\n")
        overlay.write_text("[section]\nkey = 'override'\n")
        config = Config(base, overlay_path=overlay)
        assert config.overlay_path == overlay

    def test_init_with_state_path(self, tmp_path):
        """Config accepts optional state path."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text("[section]\nkey = 'value'\n")
        config = Config(base, state_path=state)
        assert config.state_path == state


class TestConfigLoad:
    """Tests for Config.load()."""

    def test_load_base_config(self, tmp_path):
        """Load reads base config correctly."""
        base = tmp_path / "config.toml"
        base.write_text('[section]\nkey = "value"\nnumber = 42\n')
        config = Config(base)
        config.load()
        assert config.get("section", "key") == "value"
        assert config.get("section", "number") == 42

    def test_load_raises_when_base_missing(self, tmp_path):
        """Load raises FileNotFoundError when base config missing."""
        base = tmp_path / "nonexistent.toml"
        config = Config(base)
        with pytest.raises(FileNotFoundError, match="Base config not found"):
            config.load()

    def test_load_merges_overlay(self, tmp_path):
        """Load merges overlay on top of base."""
        base = tmp_path / "config.toml"
        overlay = tmp_path / "overlay.toml"
        base.write_text('[section]\nkey = "base"\nother = "kept"\n')
        overlay.write_text('[section]\nkey = "overlay"\n')
        config = Config(base, overlay_path=overlay)
        config.load()
        assert config.get("section", "key") == "overlay"
        assert config.get("section", "other") == "kept"

    def test_load_ignores_missing_overlay(self, tmp_path):
        """Load works when overlay path doesn't exist."""
        base = tmp_path / "config.toml"
        overlay = tmp_path / "nonexistent.toml"
        base.write_text('[section]\nkey = "value"\n')
        config = Config(base, overlay_path=overlay)
        config.load()
        assert config.get("section", "key") == "value"

    def test_load_merges_state(self, tmp_path):
        """Load merges state on top of config."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text('[section]\nkey = "base"\n')
        state.write_text('{"section": {"key": "state"}}')
        config = Config(base, state_path=state)
        config.load()
        assert config.get("section", "key") == "state"

    def test_load_ignores_missing_state(self, tmp_path):
        """Load works when state path doesn't exist."""
        base = tmp_path / "config.toml"
        state = tmp_path / "nonexistent.json"
        base.write_text('[section]\nkey = "value"\n')
        config = Config(base, state_path=state)
        config.load()
        assert config.get("section", "key") == "value"


class TestConfigGet:
    """Tests for Config.get()."""

    def test_get_single_key(self, tmp_path):
        """Get retrieves top-level key."""
        base = tmp_path / "config.toml"
        base.write_text('key = "value"\n')
        config = Config(base)
        config.load()
        assert config.get("key") == "value"

    def test_get_nested_keys(self, tmp_path):
        """Get retrieves deeply nested keys."""
        base = tmp_path / "config.toml"
        base.write_text('[a]\n[a.b]\n[a.b.c]\nkey = "deep"\n')
        config = Config(base)
        config.load()
        assert config.get("a", "b", "c", "key") == "deep"

    def test_get_returns_default_when_missing(self, tmp_path):
        """Get returns default for missing keys."""
        base = tmp_path / "config.toml"
        base.write_text('[section]\nkey = "value"\n')
        config = Config(base)
        config.load()
        assert config.get("missing", default="fallback") == "fallback"
        assert config.get("section", "missing", default=123) == 123

    def test_get_returns_none_when_missing_no_default(self, tmp_path):
        """Get returns None when key missing and no default."""
        base = tmp_path / "config.toml"
        base.write_text('[section]\nkey = "value"\n')
        config = Config(base)
        config.load()
        assert config.get("missing") is None

    def test_get_returns_dict_section(self, tmp_path):
        """Get can return entire section as dict."""
        base = tmp_path / "config.toml"
        base.write_text("[section]\na = 1\nb = 2\n")
        config = Config(base)
        config.load()
        section = config.get("section")
        assert section == {"a": 1, "b": 2}


class TestConfigSet:
    """Tests for Config.set()."""

    def test_set_updates_config(self, tmp_path):
        """Set updates in-memory config."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text('[section]\nkey = "original"\n')
        config = Config(base, state_path=state)
        config.load()
        config.set("section", "key", value="updated")
        assert config.get("section", "key") == "updated"

    def test_set_creates_nested_path(self, tmp_path):
        """Set creates intermediate dicts for nested paths."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text("")
        config = Config(base, state_path=state)
        config.load()
        config.set("a", "b", "c", value="deep")
        assert config.get("a", "b", "c") == "deep"

    def test_set_persists_to_state_file(self, tmp_path):
        """Set writes to state file."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text("")
        config = Config(base, state_path=state)
        config.load()
        config.set("key", value="persisted")
        # Verify file was written
        with open(state) as f:
            data = json.load(f)
        assert data["key"] == "persisted"

    def test_set_raises_without_state_path(self, tmp_path):
        """Set raises when no state_path configured."""
        base = tmp_path / "config.toml"
        base.write_text("")
        config = Config(base)  # No state_path
        config.load()
        with pytest.raises(RuntimeError, match="No state_path configured"):
            config.set("key", value="value")

    def test_set_raises_without_keys(self, tmp_path):
        """Set raises when no keys provided."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text("")
        config = Config(base, state_path=state)
        config.load()
        with pytest.raises(ValueError, match="At least one key required"):
            config.set(value="value")

    def test_set_creates_parent_dirs(self, tmp_path):
        """Set creates parent directories for state file."""
        base = tmp_path / "config.toml"
        state = tmp_path / "nested" / "dir" / "state.json"
        base.write_text("")
        config = Config(base, state_path=state)
        config.load()
        config.set("key", value="value")
        assert state.exists()

    def test_set_writes_trailing_newline(self, tmp_path):
        """Set writes state file with trailing newline."""
        base = tmp_path / "config.toml"
        state = tmp_path / "state.json"
        base.write_text("")
        config = Config(base, state_path=state)
        config.load()
        config.set("key", value="value")
        content = state.read_text()
        assert content.endswith("\n"), "state.json should end with a newline"


class TestConfigDeepMerge:
    """Tests for Config._deep_merge()."""

    def test_deep_merge_simple(self, tmp_path):
        """Deep merge overwrites simple values."""
        base = tmp_path / "config.toml"
        base.write_text("")
        config = Config(base)
        result = config._deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_deep_merge_nested(self, tmp_path):
        """Deep merge recursively merges nested dicts."""
        base = tmp_path / "config.toml"
        base.write_text("")
        config = Config(base)
        result = config._deep_merge(
            {"outer": {"a": 1, "b": 2}},
            {"outer": {"b": 3, "c": 4}},
        )
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deep_merge_preserves_base(self, tmp_path):
        """Deep merge doesn't mutate original dicts."""
        base = tmp_path / "config.toml"
        base.write_text("")
        config = Config(base)
        base_dict = {"a": {"b": 1}}
        overlay_dict = {"a": {"c": 2}}
        config._deep_merge(base_dict, overlay_dict)
        assert base_dict == {"a": {"b": 1}}
        assert overlay_dict == {"a": {"c": 2}}


class TestConfigData:
    """Tests for Config.data property."""

    def test_data_returns_copy(self, tmp_path):
        """Data property returns a copy, not the original."""
        base = tmp_path / "config.toml"
        base.write_text('[section]\nkey = "value"\n')
        config = Config(base)
        config.load()
        data = config.data
        data["section"]["key"] = "modified"
        assert config.get("section", "key") == "value"


class TestConfigCallbacks:
    """Tests for Config.on_change()."""

    def test_on_change_registers_callback(self, tmp_path):
        """on_change adds callback to list."""
        base = tmp_path / "config.toml"
        base.write_text("")
        config = Config(base)

        async def callback(cfg):
            pass

        config.on_change(callback)
        assert callback in config._callbacks

    def test_multiple_callbacks(self, tmp_path):
        """Multiple callbacks can be registered."""
        base = tmp_path / "config.toml"
        base.write_text("")
        config = Config(base)

        async def cb1(cfg):
            pass

        async def cb2(cfg):
            pass

        config.on_change(cb1)
        config.on_change(cb2)
        assert len(config._callbacks) == 2
