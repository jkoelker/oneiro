"""LoRA configuration types and loading utilities."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from oneiro.civitai import CivitaiClient


class LoraSource(str, Enum):
    """Source type for LoRA weights."""

    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class LoraConfig:
    """Configuration for a single LoRA adapter.

    Supports three sources:
    - civitai: Download from Civitai by model ID or URL
    - huggingface: Load from HuggingFace Hub repository
    - local: Load from local file path

    Attributes:
        name: Unique name for referencing this LoRA (required)
        source: Where to load the LoRA from (civitai, huggingface, local)
        adapter_name: Name for the diffusers adapter (defaults to name if not provided)
        weight: Adapter weight for blending (default: 1.0)
        civitai_id: Civitai model ID (for civitai source)
        civitai_version: Specific version ID (optional, defaults to latest)
        civitai_url: Civitai URL (alternative to civitai_id)
        repo: HuggingFace repository (for huggingface source)
        weight_name: Filename in HF repo (for huggingface source)
        path: Local file path (for local source)
    """

    name: str
    source: LoraSource
    adapter_name: str | None = None
    weight: float = 1.0

    # Civitai-specific
    civitai_id: int | None = None
    civitai_version: int | None = None
    civitai_url: str | None = None

    # HuggingFace-specific
    repo: str | None = None
    weight_name: str | None = None

    # Local-specific
    path: str | None = None

    # Resolved path (filled after download)
    _resolved_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        if self.adapter_name is None:
            object.__setattr__(self, "adapter_name", self.name)

        if self.source == LoraSource.CIVITAI:
            if not self.civitai_id and not self.civitai_url:
                raise ValueError("civitai source requires civitai_id or civitai_url")
        elif self.source == LoraSource.HUGGINGFACE:
            if not self.repo:
                raise ValueError("huggingface source requires repo")
        elif self.source == LoraSource.LOCAL:
            if not self.path:
                raise ValueError("local source requires path")


def parse_civitai_url(url: str) -> tuple[int, int | None]:
    """Parse Civitai URL to extract model ID and optional version ID.

    Supports formats:
    - https://civitai.com/models/12345
    - https://civitai.com/models/12345/model-name
    - https://civitai.com/models/12345?modelVersionId=67890
    - https://civitai.com/models/12345/name?modelVersionId=67890

    Args:
        url: Civitai model URL

    Returns:
        Tuple of (model_id, version_id or None)

    Raises:
        ValueError: If URL format is invalid
    """
    # Match model ID in path
    model_match = re.search(r"/models/(\d+)", url)
    if not model_match:
        raise ValueError(f"Invalid Civitai URL format: {url}")

    model_id = int(model_match.group(1))

    # Check for version in query string
    version_match = re.search(r"modelVersionId=(\d+)", url)
    version_id = int(version_match.group(1)) if version_match else None

    return model_id, version_id


def parse_lora_config(config: dict[str, Any] | str, index: int = 0) -> LoraConfig:
    """Parse a LoRA configuration from TOML config format.

    Supports multiple formats:

    1. Simple Civitai URL string:
       ```toml
       civitai_lora = "https://civitai.com/models/12345"
       ```

    2. Dict with explicit source:
       ```toml
       [[models.my-model.loras]]
       source = "civitai"
       id = 12345
       weight = 0.8
       ```

    3. Dict with HuggingFace repo:
       ```toml
       [[models.my-model.loras]]
       source = "huggingface"
       repo = "XLabs-AI/flux-RealismLora"
       weight_name = "lora.safetensors"
       ```

    Args:
        config: Either a URL string or a config dict
        index: Index for auto-generating adapter name

    Returns:
        LoraConfig instance

    Raises:
        ValueError: If config format is invalid
    """
    # Simple URL string
    if isinstance(config, str):
        if "civitai.com" in config:
            model_id, version_id = parse_civitai_url(config)
            auto_name = f"civitai_{model_id}"
            return LoraConfig(
                name=auto_name,
                source=LoraSource.CIVITAI,
                civitai_id=model_id,
                civitai_version=version_id,
                civitai_url=config,
            )
        # Assume local path
        auto_name = f"local_{index}"
        return LoraConfig(
            name=auto_name,
            source=LoraSource.LOCAL,
            path=config,
        )

    # Dict configuration
    if not isinstance(config, dict):
        raise ValueError(f"Invalid LoRA config type: {type(config)}")

    # Determine source
    source_str = config.get("source", "civitai")
    try:
        source = LoraSource(source_str)
    except ValueError as err:
        raise ValueError(f"Invalid LoRA source: {source_str}") from err

    # Common fields
    lora_name = config.get("name")
    adapter_name = config.get("adapter_name")
    weight = config.get("weight", 1.0)

    if source == LoraSource.CIVITAI:
        civitai_id = config.get("id") or config.get("civitai_id")
        civitai_version = config.get("version") or config.get("civitai_version")
        civitai_url = config.get("url") or config.get("civitai_url")

        if civitai_url and not civitai_id:
            civitai_id, parsed_version = parse_civitai_url(civitai_url)
            civitai_version = civitai_version or parsed_version

        if not lora_name:
            lora_name = f"civitai_{civitai_id}" if civitai_id else f"lora_{index}"

        return LoraConfig(
            name=lora_name,
            source=source,
            adapter_name=adapter_name,
            weight=weight,
            civitai_id=civitai_id,
            civitai_version=civitai_version,
            civitai_url=civitai_url,
        )

    elif source == LoraSource.HUGGINGFACE:
        repo = config.get("repo")
        weight_name = config.get("weight_name")

        if not lora_name:
            lora_name = repo.replace("/", "_") if repo else f"hf_{index}"

        return LoraConfig(
            name=lora_name,
            source=source,
            adapter_name=adapter_name,
            weight=weight,
            repo=repo,
            weight_name=weight_name,
        )

    else:  # LOCAL
        path = config.get("path")

        if not lora_name:
            lora_name = f"local_{index}"

        return LoraConfig(
            name=lora_name,
            source=source,
            adapter_name=adapter_name,
            weight=weight,
            path=path,
        )


def _get_lora_unique_key(lora: LoraConfig) -> tuple:
    """Generate a unique key for a LoRA config to detect duplicates.

    The key is based on the source type and identifying attributes:
    - Civitai: (source, civitai_id, civitai_version)
    - HuggingFace: (source, repo, weight_name)
    - Local: (source, path)

    Args:
        lora: LoRA configuration

    Returns:
        Tuple that uniquely identifies this LoRA
    """
    if lora.source == LoraSource.CIVITAI:
        return (lora.source, lora.civitai_id, lora.civitai_version)
    elif lora.source == LoraSource.HUGGINGFACE:
        return (lora.source, lora.repo, lora.weight_name)
    else:  # LOCAL
        return (lora.source, lora.path)


def parse_loras_from_model_config(model_config: dict[str, Any]) -> list[LoraConfig]:
    """Parse all LoRA configurations from a model config section.

    Supports multiple config formats:

    1. Single LoRA via civitai_lora URL:
       ```toml
       [models.my-model]
       civitai_lora = "https://civitai.com/models/12345"
       ```

    2. Single LoRA via civitai_lora_id:
       ```toml
       [models.my-model]
       civitai_lora_id = 12345
       civitai_lora_version = 67890  # optional
       ```

    3. Multiple LoRAs via loras array:
       ```toml
       [[models.my-model.loras]]
       source = "civitai"
       id = 12345
       weight = 0.8
       ```

    4. Legacy single LoRA via lora/lora_weights (for backward compat):
       ```toml
       [models.my-model]
       lora = "user/repo"
       lora_weights = "filename.safetensors"
       ```

    Duplicate detection is performed based on source type:
    - Civitai: (source, civitai_id, civitai_version)
    - HuggingFace: (source, repo, weight_name)
    - Local: (source, path)

    Args:
        model_config: Model configuration dict from TOML

    Returns:
        List of LoraConfig instances (duplicates are skipped with a warning)
    """
    loras: list[LoraConfig] = []
    loaded_keys: set[tuple] = set()

    def _add_lora(lora: LoraConfig) -> None:
        """Add a LoRA if not already loaded, with duplicate warning."""
        key = _get_lora_unique_key(lora)
        if key in loaded_keys:
            print(f"Warning: duplicate LoRA detected, skipping: {lora.adapter_name}")
            return
        loaded_keys.add(key)
        loras.append(lora)

    # Check for loras array (preferred format)
    if "loras" in model_config:
        loras_config = model_config["loras"]
        if isinstance(loras_config, list):
            for i, lora_config in enumerate(loras_config):
                _add_lora(parse_lora_config(lora_config, index=i))
        else:
            # Single dict
            _add_lora(parse_lora_config(loras_config, index=0))

    # Check for civitai_lora URL
    elif "civitai_lora" in model_config:
        _add_lora(parse_lora_config(model_config["civitai_lora"], index=0))

    # Check for civitai_lora_id
    elif "civitai_lora_id" in model_config:
        civitai_id = model_config["civitai_lora_id"]
        _add_lora(
            LoraConfig(
                name=f"civitai_{civitai_id}",
                source=LoraSource.CIVITAI,
                civitai_id=civitai_id,
                civitai_version=model_config.get("civitai_lora_version"),
                weight=model_config.get("civitai_lora_weight", 1.0),
            )
        )

    # Check for legacy lora/lora_weights format (backward compatibility)
    elif "lora" in model_config:
        lora_repo = model_config["lora"]
        lora_weights = model_config.get("lora_weights")

        # Check if it's a local path
        if lora_repo.startswith(("/", "./", "~/")):
            _add_lora(
                LoraConfig(
                    name="legacy_lora",
                    source=LoraSource.LOCAL,
                    path=lora_repo,
                    weight=model_config.get("lora_weight", 1.0),
                )
            )
        else:
            # Assume HuggingFace repo
            _add_lora(
                LoraConfig(
                    name="legacy_lora",
                    source=LoraSource.HUGGINGFACE,
                    repo=lora_repo,
                    weight_name=lora_weights,
                    weight=model_config.get("lora_weight", 1.0),
                )
            )

    return loras


def parse_loras_from_config(
    full_config: dict[str, Any],
    model_config: dict[str, Any],
) -> list[LoraConfig]:
    """Parse all LoRA configurations for a model from full config.

    Handles three types of LoRA sources:
    1. Global auto_load: LoRAs loaded for ALL models
    2. Named references: Model references LoRAs defined in [loras.name]
    3. Inline definitions: Model-specific LoRAs defined directly in model config

    Config structure:
    ```toml
    [loras]
    auto_load = ["realism-enhancer"]  # Loaded for every model

    [loras.realism-enhancer]
    source = "civitai"
    id = 123456
    weight = 0.5

    [loras.detail-lora]
    source = "huggingface"
    repo = "user/detail-lora"
    weight = 0.8

    [models.my-model]
    loras = ["detail-lora"]  # Named reference

    # Inline definitions
    [[models.my-model.inline_loras]]
    source = "civitai"
    id = 99999
    weight = 0.7
    ```

    Args:
        full_config: The complete config dict (for accessing [loras] section)
        model_config: Model-specific config section

    Returns:
        List of LoraConfig instances (auto_load + named refs + inline)
    """
    loras: list[LoraConfig] = []
    loras_section = full_config.get("loras", {})

    # Track names to avoid duplicates
    loaded_names: set[str] = set()

    # 1. Global auto_load LoRAs
    auto_load = loras_section.get("auto_load", [])
    if isinstance(auto_load, list):
        for ref_name in auto_load:
            if ref_name in loaded_names:
                continue
            if ref_name in loras_section and isinstance(loras_section[ref_name], dict):
                lora_config = loras_section[ref_name]
                parsed = parse_lora_config(lora_config, index=len(loras))
                if not lora_config.get("name"):
                    object.__setattr__(parsed, "name", ref_name)
                    if (
                        parsed.adapter_name == parsed.name
                        or lora_config.get("adapter_name") is None
                    ):
                        object.__setattr__(parsed, "adapter_name", ref_name)
                loras.append(parsed)
                loaded_names.add(parsed.name)
            else:
                print(f"Warning: auto_load LoRA '{ref_name}' not found in [loras] section")

    # 2. Named references from model config
    model_loras = model_config.get("loras", [])
    if isinstance(model_loras, list):
        for ref in model_loras:
            if isinstance(ref, str):
                if ref in loaded_names:
                    continue
                if ref in loras_section and isinstance(loras_section[ref], dict):
                    lora_config = loras_section[ref]
                    parsed = parse_lora_config(lora_config, index=len(loras))
                    if not lora_config.get("name"):
                        object.__setattr__(parsed, "name", ref)
                        if (
                            parsed.adapter_name == parsed.name
                            or lora_config.get("adapter_name") is None
                        ):
                            object.__setattr__(parsed, "adapter_name", ref)
                    loras.append(parsed)
                    loaded_names.add(parsed.name)
                else:
                    print(f"Warning: LoRA '{ref}' not found in [loras] section")
            elif isinstance(ref, dict):
                parsed = parse_lora_config(ref, index=len(loras))
                if parsed.name not in loaded_names:
                    loras.append(parsed)
                    loaded_names.add(parsed.name)

    # 3. Inline definitions via [[models.X.inline_loras]]
    inline_loras = model_config.get("inline_loras", [])
    if isinstance(inline_loras, list):
        for inline_config in inline_loras:
            if isinstance(inline_config, dict):
                parsed = parse_lora_config(inline_config, index=len(loras))
                if parsed.name not in loaded_names:
                    loras.append(parsed)
                    loaded_names.add(parsed.name)

    return loras


# Pipeline type to Civitai base model mapping
PIPELINE_BASE_MODEL_MAP: dict[str, list[str]] = {
    "flux2": ["Flux.1 D", "Flux.1 S", "Flux.1", "Flux.2", "Flux.1 Dev", "Flux.1 Schnell"],
    "zimage": ["ZImageTurbo", "ZImageBase", "Z-Image"],
    "qwen": ["Qwen", "Qwen-Image"],
    "sdxl": ["SDXL 1.0", "SDXL Turbo", "SDXL Lightning", "Pony", "Illustrious"],
    "sd15": ["SD 1.5", "SD 1.4"],
    "sd3": ["SD 3", "SD 3.5"],
}


def is_lora_compatible(pipeline_type: str, civitai_base_model: str | None) -> bool:
    """Check if a Civitai LoRA is compatible with a pipeline type.

    Args:
        pipeline_type: Pipeline type (flux2, zimage, qwen, etc.)
        civitai_base_model: Base model string from Civitai API

    Returns:
        True if compatible, False otherwise
    """
    if civitai_base_model is None:
        # Can't verify, assume compatible
        return True

    compatible_bases = PIPELINE_BASE_MODEL_MAP.get(pipeline_type, [])
    if not compatible_bases:
        # Unknown pipeline type, assume compatible
        return True

    # Check if any compatible base model matches (case-insensitive substring)
    civitai_lower = civitai_base_model.lower()
    for base in compatible_bases:
        if base.lower() in civitai_lower or civitai_lower in base.lower():
            return True

    return False


class LoraIncompatibleError(Exception):
    """Raised when a LoRA is incompatible with the pipeline type."""

    def __init__(self, lora_name: str, pipeline_type: str, base_model: str | None):
        self.lora_name = lora_name
        self.pipeline_type = pipeline_type
        self.base_model = base_model
        super().__init__(
            f"LoRA '{lora_name}' (base: {base_model}) is not compatible with "
            f"pipeline type '{pipeline_type}'"
        )


async def resolve_lora_path(
    lora: LoraConfig,
    civitai_client: CivitaiClient | None = None,
    pipeline_type: str | None = None,
    validate_compatibility: bool = True,
) -> Path:
    """Resolve a LoRA config to a local file path, downloading if necessary.

    Args:
        lora: LoRA configuration
        civitai_client: Client for downloading from Civitai
        pipeline_type: Pipeline type for compatibility validation
        validate_compatibility: Whether to validate base model compatibility

    Returns:
        Path to the LoRA file

    Raises:
        LoraIncompatibleError: If LoRA is incompatible with pipeline type
        ValueError: If required parameters are missing
    """
    if lora.source == LoraSource.LOCAL:
        if not lora.path:
            raise ValueError("Local LoRA requires path")
        path = Path(lora.path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local LoRA not found: {path}")
        lora._resolved_path = path
        return path

    elif lora.source == LoraSource.HUGGINGFACE:
        # HuggingFace repos are loaded directly by diffusers, no pre-download needed
        # Return a sentinel path - the actual loading happens in load_lora_weights
        lora._resolved_path = None  # Signal to use repo directly
        if not lora.repo:
            raise ValueError("HuggingFace LoRA requires repo")
        return Path(lora.repo)

    elif lora.source == LoraSource.CIVITAI:
        if civitai_client is None:
            raise ValueError("CivitaiClient required for Civitai LoRA downloads")

        if not lora.civitai_id:
            raise ValueError("civitai_id required for Civitai LoRA")

        # Fetch model info
        print(f"Fetching Civitai model info for ID {lora.civitai_id}...")

        if lora.civitai_version:
            # Fetch specific version
            version = await civitai_client.get_model_version(lora.civitai_version)
        else:
            # Fetch model and get latest version
            model = await civitai_client.get_model(lora.civitai_id)
            version = model.latest_version
            if version is None:
                raise ValueError(f"No versions available for model {lora.civitai_id}")

        # Validate compatibility
        if validate_compatibility and pipeline_type:
            if not is_lora_compatible(pipeline_type, version.base_model):
                raise LoraIncompatibleError(
                    lora.adapter_name or f"civitai_{lora.civitai_id}",
                    pipeline_type,
                    version.base_model,
                )

        # Download
        print(f"Downloading LoRA: {version.name} (base: {version.base_model})")
        path = await civitai_client.download_model_version(version)
        lora._resolved_path = path
        return path

    raise ValueError(f"Unknown LoRA source: {lora.source}")


class LoraLoaderMixin:
    """Mixin for pipelines that support loading multiple LoRAs via PEFT/diffusers."""

    pipe: Any
    _lora_configs: list[LoraConfig]
    _loaded_adapters: list[str]

    def _init_lora_state(self) -> None:
        """Initialize LoRA-related state. Call this in __init__."""
        self._lora_configs = []
        self._loaded_adapters = []

    def load_single_lora(
        self,
        lora: LoraConfig,
    ) -> str:
        """Load a single LoRA into the pipeline.

        Args:
            lora: LoRA configuration with resolved path or repo

        Returns:
            Adapter name that was loaded

        Raises:
            RuntimeError: If pipeline not loaded
            ValueError: If LoRA configuration is invalid
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        adapter_name = lora.adapter_name or f"lora_{len(self._loaded_adapters)}"

        if lora.source == LoraSource.HUGGINGFACE:
            print(f"Loading LoRA from HF: {lora.repo} (adapter: {adapter_name})")
            self.pipe.load_lora_weights(
                lora.repo,
                weight_name=lora.weight_name,
                adapter_name=adapter_name,
            )
        elif lora.source in (LoraSource.CIVITAI, LoraSource.LOCAL):
            if lora._resolved_path is None:
                raise ValueError(f"LoRA path not resolved: {lora}")
            print(f"Loading LoRA from path: {lora._resolved_path} (adapter: {adapter_name})")
            self.pipe.load_lora_weights(
                str(lora._resolved_path),
                adapter_name=adapter_name,
            )
        else:
            raise ValueError(f"Unknown LoRA source: {lora.source}")

        self._loaded_adapters.append(adapter_name)
        self._lora_configs.append(lora)
        return adapter_name

    def load_loras_sync(
        self,
        loras: list[LoraConfig],
    ) -> list[str]:
        """Load multiple LoRAs synchronously (assuming paths are already resolved).

        Args:
            loras: List of LoRA configurations with resolved paths

        Returns:
            List of loaded adapter names
        """
        adapter_names = []
        adapter_weights = []

        for lora in loras:
            name = self.load_single_lora(lora)
            adapter_names.append(name)
            adapter_weights.append(lora.weight)

        if adapter_names:
            self.set_lora_adapters(adapter_names, adapter_weights)

        return adapter_names

    async def load_loras_async(
        self,
        loras: list[LoraConfig],
        civitai_client: CivitaiClient | None = None,
        pipeline_type: str | None = None,
        validate_compatibility: bool = True,
    ) -> list[str]:
        """Load multiple LoRAs, downloading from Civitai as needed.

        Args:
            loras: List of LoRA configurations
            civitai_client: Client for Civitai downloads
            pipeline_type: Pipeline type for compatibility validation
            validate_compatibility: Whether to validate base model compatibility

        Returns:
            List of loaded adapter names
        """
        for lora in loras:
            await resolve_lora_path(
                lora,
                civitai_client=civitai_client,
                pipeline_type=pipeline_type,
                validate_compatibility=validate_compatibility,
            )

        return self.load_loras_sync(loras)

    def set_lora_adapters(
        self,
        adapter_names: list[str],
        adapter_weights: list[float] | None = None,
    ) -> None:
        """Set active LoRA adapters and their weights.

        Args:
            adapter_names: List of adapter names to activate
            adapter_weights: List of weights for each adapter (default: all 1.0)
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        if adapter_weights is None:
            adapter_weights = [1.0] * len(adapter_names)

        print(f"Setting LoRA adapters: {dict(zip(adapter_names, adapter_weights, strict=True))}")
        self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    def disable_loras(self) -> None:
        """Disable all LoRA adapters without unloading them."""
        if self.pipe is None or not self._loaded_adapters:
            return

        print("Disabling LoRA adapters")
        self.pipe.set_adapters([])

    def enable_loras(self, adapter_weights: list[float] | None = None) -> None:
        """Re-enable all loaded LoRA adapters.

        Args:
            adapter_weights: Optional custom weights (default: use config weights)
        """
        if self.pipe is None or not self._loaded_adapters:
            return

        if adapter_weights is None:
            adapter_weights = [lora.weight for lora in self._lora_configs]

        self.set_lora_adapters(self._loaded_adapters, adapter_weights)

    def unload_loras(self) -> None:
        """Completely unload all LoRA adapters and free memory."""
        if self.pipe is None or not self._loaded_adapters:
            return

        print(f"Unloading {len(self._loaded_adapters)} LoRA adapter(s)")
        try:
            self.pipe.unload_lora_weights()
        except Exception as e:
            print(f"Warning: Error unloading LoRAs: {e}")

        self._loaded_adapters.clear()
        self._lora_configs.clear()

    def fuse_loras(self, lora_scale: float = 1.0) -> None:
        """Fuse LoRA weights into base model for faster inference.

        After fusing, LoRAs cannot be dynamically adjusted. Use unfuse_loras() to restore.

        Args:
            lora_scale: Scale factor for fused weights
        """
        if self.pipe is None or not self._loaded_adapters:
            return

        print(f"Fusing {len(self._loaded_adapters)} LoRA(s) with scale {lora_scale}")
        self.pipe.fuse_lora(adapter_names=self._loaded_adapters, lora_scale=lora_scale)

    def unfuse_loras(self) -> None:
        """Unfuse LoRA weights from base model."""
        if self.pipe is None:
            return

        print("Unfusing LoRAs")
        self.pipe.unfuse_lora()

    @property
    def active_loras(self) -> list[str]:
        """Get list of currently loaded adapter names."""
        return list(self._loaded_adapters)

    @property
    def lora_count(self) -> int:
        """Get number of loaded LoRAs."""
        return len(self._loaded_adapters)
