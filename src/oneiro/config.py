"""Layered TOML configuration with hot reload support."""

import asyncio
import json
import tomllib
from collections.abc import Callable, Coroutine
from copy import deepcopy
from pathlib import Path
from typing import Any

from watchfiles import awatch


class Config:
    """Layered configuration with hot reload.

    Loads a base config and optionally merges an overlay config on top.
    Supports async watching for changes with callbacks.
    """

    def __init__(
        self,
        base_path: Path | str,
        overlay_path: Path | str | None = None,
        state_path: Path | str | None = None,
    ):
        self.base_path = Path(base_path) if isinstance(base_path, str) else base_path
        self.overlay_path = Path(overlay_path) if isinstance(overlay_path, str) else overlay_path
        self.state_path = Path(state_path) if isinstance(state_path, str) else state_path
        self._config: dict[str, Any] = {}
        self._state: dict[str, Any] = {}  # Runtime state (persisted separately)
        self._callbacks: list[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = []
        self._watch_task: asyncio.Task | None = None

    def load(self) -> None:
        """Load and merge configs from base, overlay, and state paths."""
        # Load base config (required)
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_path}")

        with open(self.base_path, "rb") as f:
            base = tomllib.load(f)

        # Load overlay if it exists
        if self.overlay_path and self.overlay_path.exists():
            try:
                with open(self.overlay_path, "rb") as f:
                    overlay = tomllib.load(f)
                config = self._deep_merge(base, overlay)
            except tomllib.TOMLDecodeError as e:
                print(f"Warning: Invalid overlay TOML, using base only: {e}")
                config = base
        else:
            config = base

        # Load runtime state if it exists (JSON, machine-generated)
        if self.state_path and self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    self._state = json.load(f)
                self._config = self._deep_merge(config, self._state)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid state JSON, ignoring: {e}")
                self._config = config
        else:
            self._config = config

    def _deep_merge(self, base: dict, overlay: dict) -> dict:
        """Deep merge overlay into base. Overlay values override base."""
        result = deepcopy(base)
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested configuration value.

        Args:
            *keys: Path of keys to traverse (e.g., 'queue', 'max_global')
            default: Value to return if path doesn't exist

        Returns:
            The config value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys: str, value: Any) -> None:
        """Set a value and persist to runtime state file.

        Args:
            *keys: Path of keys (e.g., 'defaults', 'model')
            value: Value to set (passed as keyword argument)

        Example:
            config.set("defaults", "model", value="flux2-dev")
        """
        if not keys:
            raise ValueError("At least one key required")

        if self.state_path is None:
            raise RuntimeError("No state_path configured, cannot persist state")

        # Update nested state dict
        target = self._state
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

        # Update in-memory config
        target = self._config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

        # Persist state to disk
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._state, f, indent=2)

        print(f"State updated: {'.'.join(keys)} = {value}")

    def on_change(self, callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]]) -> None:
        """Register a callback for config changes.

        Callback receives the new merged config dict.
        """
        self._callbacks.append(callback)

    async def start_watching(self) -> None:
        """Start watching config files for changes."""
        if self._watch_task is not None:
            return  # Already watching

        paths_to_watch: list[Path] = [self.base_path.parent]
        if self.overlay_path:
            # Add overlay parent if different from base parent
            if self.overlay_path.parent != self.base_path.parent:
                paths_to_watch.append(self.overlay_path.parent)

        self._watch_task = asyncio.create_task(self._watch(paths_to_watch))
        print(f"Config watcher started for: {[str(p) for p in paths_to_watch]}")

    async def stop_watching(self) -> None:
        """Stop watching config files."""
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None
            print("Config watcher stopped")

    async def _watch(self, paths: list[Path]) -> None:
        """Watch paths for changes and trigger reload."""
        try:
            async for changes in awatch(*paths, debounce=2000):
                # Check if any of our config files changed
                for _change_type, changed_path in changes:
                    changed = Path(changed_path)
                    if changed == self.base_path or changed == self.overlay_path:
                        print(f"Config changed: {changed}")
                        try:
                            self.load()
                            # Notify callbacks
                            for callback in self._callbacks:
                                try:
                                    await callback(self._config)
                                except Exception as e:
                                    print(f"Config callback error: {e}")
                        except Exception as e:
                            print(f"Config reload error: {e}")
                        break  # Only reload once per change batch
        except asyncio.CancelledError:
            pass

    @property
    def data(self) -> dict[str, Any]:
        """Get the full merged config dict (read-only copy)."""
        return deepcopy(self._config)
