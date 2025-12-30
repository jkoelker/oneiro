"""Textual inversion / embedding configuration types and loading utilities."""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from oneiro.civitai import CivitaiClient


class EmbeddingSource(str, Enum):
    """Source type for textual inversion embeddings."""

    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class EmbeddingConfig:
    """Configuration for a single textual inversion embedding.

    Supports three sources:
    - civitai: Download from Civitai by model ID or URL
    - huggingface: Load from HuggingFace Hub repository
    - local: Load from local file path

    Attributes:
        name: Unique name for referencing this embedding
        source: Where to load the embedding from (civitai, huggingface, local)
        token: Trigger token for the embedding (auto-detected if omitted)
        civitai_id: Civitai model ID (for civitai source)
        civitai_version: Specific version ID (optional, defaults to latest)
        civitai_url: Civitai URL (alternative to civitai_id)
        repo: HuggingFace repository (for huggingface source)
        path: Local file path (for local source)
    """

    name: str
    source: EmbeddingSource
    token: str | None = None

    # Civitai-specific
    civitai_id: int | None = None
    civitai_version: int | None = None
    civitai_url: str | None = None

    # HuggingFace-specific
    repo: str | None = None

    # Local-specific
    path: str | None = None

    # Resolved path (filled after download)
    _resolved_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration based on source type."""
        if self.source == EmbeddingSource.CIVITAI:
            if not self.civitai_id and not self.civitai_url:
                raise ValueError("civitai source requires civitai_id or civitai_url")
        elif self.source == EmbeddingSource.HUGGINGFACE:
            if not self.repo:
                raise ValueError("huggingface source requires repo")
        elif self.source == EmbeddingSource.LOCAL:
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


def parse_embedding_config(
    config: dict[str, Any] | str, name: str | None = None
) -> EmbeddingConfig:
    """Parse an embedding configuration from TOML config format.

    Supports multiple formats:

    1. Simple Civitai URL string:
       ```toml
       [embeddings.my-style]
       source = "civitai"
       url = "https://civitai.com/models/12345"
       ```

    2. Dict with explicit source:
       ```toml
       [embeddings.easynegative]
       source = "civitai"
       id = 7808
       token = "easynegative"
       ```

    3. Dict with HuggingFace repo:
       ```toml
       [embeddings.cat-toy]
       source = "huggingface"
       repo = "sd-concepts-library/cat-toy"
       ```

    Args:
        config: Either a URL string or a config dict
        name: Name for the embedding (required if not in config)

    Returns:
        EmbeddingConfig instance

    Raises:
        ValueError: If config format is invalid
    """
    # Simple URL string (inline shorthand)
    if isinstance(config, str):
        if "civitai.com" in config:
            model_id, version_id = parse_civitai_url(config)
            return EmbeddingConfig(
                name=name or f"civitai_{model_id}",
                source=EmbeddingSource.CIVITAI,
                civitai_id=model_id,
                civitai_version=version_id,
                civitai_url=config,
            )
        # Assume local path
        return EmbeddingConfig(
            name=name or "local_embedding",
            source=EmbeddingSource.LOCAL,
            path=config,
        )

    # Dict configuration
    if not isinstance(config, dict):
        raise ValueError(f"Invalid embedding config type: {type(config)}")

    # Determine source
    source_str = config.get("source", "civitai")
    try:
        source = EmbeddingSource(source_str)
    except ValueError as err:
        raise ValueError(f"Invalid embedding source: {source_str}") from err

    # Common fields
    embedding_name = config.get("name") or name
    if not embedding_name:
        raise ValueError("Embedding requires a name")

    token = config.get("token")

    if source == EmbeddingSource.CIVITAI:
        # Parse Civitai config
        civitai_id = config.get("id") or config.get("civitai_id")
        civitai_version = config.get("version") or config.get("civitai_version")
        civitai_url = config.get("url") or config.get("civitai_url")

        # Parse URL if provided but no ID
        if civitai_url and not civitai_id:
            civitai_id, parsed_version = parse_civitai_url(civitai_url)
            civitai_version = civitai_version or parsed_version

        return EmbeddingConfig(
            name=embedding_name,
            source=source,
            token=token,
            civitai_id=civitai_id,
            civitai_version=civitai_version,
            civitai_url=civitai_url,
        )

    elif source == EmbeddingSource.HUGGINGFACE:
        repo = config.get("repo")

        return EmbeddingConfig(
            name=embedding_name,
            source=source,
            token=token,
            repo=repo,
        )

    else:  # LOCAL
        path = config.get("path")

        return EmbeddingConfig(
            name=embedding_name,
            source=source,
            token=token,
            path=path,
        )


def parse_embeddings_from_config(
    full_config: dict[str, Any],
    model_config: dict[str, Any],
) -> list[EmbeddingConfig]:
    """Parse all embedding configurations for a model from full config.

    Handles three types of embedding sources:
    1. Global auto_load: Embeddings loaded for ALL models
    2. Named references: Model references embeddings defined in [embeddings.name]
    3. Inline definitions: Model-specific embeddings defined directly in model config

    Config structure:
    ```toml
    [embeddings]
    auto_load = ["easynegative"]  # Loaded for every model

    [embeddings.easynegative]
    source = "civitai"
    id = 7808
    token = "easynegative"

    [embeddings.bad-hands]
    source = "civitai"
    id = 116230

    [models.my-model]
    embeddings = ["bad-hands"]  # Named reference

    # Inline definitions
    [[models.my-model.inline_embeddings]]
    source = "civitai"
    id = 99999
    token = "custom-style"
    ```

    Args:
        full_config: The complete config dict (for accessing [embeddings] section)
        model_config: Model-specific config section

    Returns:
        List of EmbeddingConfig instances (auto_load + named refs + inline)
    """
    embeddings: list[EmbeddingConfig] = []
    embeddings_section = full_config.get("embeddings", {})

    # Track names to avoid duplicates
    loaded_names: set[str] = set()

    # 1. Global auto_load embeddings
    auto_load = embeddings_section.get("auto_load", [])
    if isinstance(auto_load, list):
        for ref_name in auto_load:
            if ref_name in loaded_names:
                continue
            if ref_name in embeddings_section and isinstance(embeddings_section[ref_name], dict):
                emb_config = embeddings_section[ref_name]
                embeddings.append(parse_embedding_config(emb_config, name=ref_name))
                loaded_names.add(ref_name)
            else:
                print(
                    f"Warning: auto_load embedding '{ref_name}' not found in [embeddings] section"
                )

    # 2. Named references from model config
    model_embeddings = model_config.get("embeddings", [])
    if isinstance(model_embeddings, list):
        for ref in model_embeddings:
            if isinstance(ref, str):
                # String reference to named embedding
                if ref in loaded_names:
                    continue
                if ref in embeddings_section and isinstance(embeddings_section[ref], dict):
                    emb_config = embeddings_section[ref]
                    embeddings.append(parse_embedding_config(emb_config, name=ref))
                    loaded_names.add(ref)
                else:
                    print(f"Warning: embedding '{ref}' not found in [embeddings] section")
            elif isinstance(ref, dict):
                # Inline dict definition in the embeddings array
                emb_name = ref.get("name", f"inline_{len(embeddings)}")
                if emb_name not in loaded_names:
                    embeddings.append(parse_embedding_config(ref, name=emb_name))
                    loaded_names.add(emb_name)

    # 3. Inline definitions via [[models.X.inline_embeddings]]
    inline_embeddings = model_config.get("inline_embeddings", [])
    if isinstance(inline_embeddings, list):
        for inline_config in inline_embeddings:
            if isinstance(inline_config, dict):
                emb_name = inline_config.get("name", f"inline_{len(embeddings)}")
                if emb_name not in loaded_names:
                    embeddings.append(parse_embedding_config(inline_config, name=emb_name))
                    loaded_names.add(emb_name)

    return embeddings


# Pipeline type to Civitai base model mapping for embeddings
# Embeddings are generally more model-specific than LoRAs
PIPELINE_BASE_MODEL_MAP: dict[str, list[str]] = {
    "flux2": ["Flux.1 D", "Flux.1 S", "Flux.1", "Flux.2", "Flux.1 Dev", "Flux.1 Schnell"],
    "zimage": ["ZImageTurbo", "ZImageBase", "Z-Image"],
    "qwen": ["Qwen", "Qwen-Image"],
    "sdxl": ["SDXL 1.0", "SDXL Turbo", "SDXL Lightning", "Pony", "Illustrious"],
    "sd15": ["SD 1.5", "SD 1.4"],
    "sd3": ["SD 3", "SD 3.5"],
}


def is_embedding_compatible(pipeline_type: str, civitai_base_model: str | None) -> bool:
    """Check if a Civitai embedding is compatible with a pipeline type.

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


class EmbeddingIncompatibleError(Exception):
    """Raised when an embedding is incompatible with the pipeline type."""

    def __init__(self, embedding_name: str, pipeline_type: str, base_model: str | None):
        self.embedding_name = embedding_name
        self.pipeline_type = pipeline_type
        self.base_model = base_model
        super().__init__(
            f"Embedding '{embedding_name}' (base: {base_model}) is not compatible with "
            f"pipeline type '{pipeline_type}'"
        )


async def resolve_embedding_path(
    embedding: EmbeddingConfig,
    civitai_client: CivitaiClient | None = None,
    pipeline_type: str | None = None,
    validate_compatibility: bool = True,
) -> Path:
    """Resolve an embedding config to a local file path, downloading if necessary.

    Args:
        embedding: Embedding configuration
        civitai_client: Client for downloading from Civitai
        pipeline_type: Pipeline type for compatibility validation
        validate_compatibility: Whether to validate base model compatibility

    Returns:
        Path to the embedding file

    Raises:
        EmbeddingIncompatibleError: If embedding is incompatible with pipeline type
        ValueError: If required parameters are missing
    """
    if embedding.source == EmbeddingSource.LOCAL:
        if not embedding.path:
            raise ValueError("Local embedding requires path")
        path = Path(embedding.path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Local embedding not found: {path}")
        embedding._resolved_path = path
        return path

    elif embedding.source == EmbeddingSource.HUGGINGFACE:
        # HuggingFace repos are loaded directly by diffusers, no pre-download needed
        # Return a sentinel path - the actual loading happens in load_textual_inversion
        embedding._resolved_path = None  # Signal to use repo directly
        if not embedding.repo:
            raise ValueError("HuggingFace embedding requires repo")
        return Path(embedding.repo)

    elif embedding.source == EmbeddingSource.CIVITAI:
        if civitai_client is None:
            raise ValueError("CivitaiClient required for Civitai embedding downloads")

        if not embedding.civitai_id:
            raise ValueError("civitai_id required for Civitai embedding")

        # Fetch model info
        print(f"Fetching Civitai model info for embedding ID {embedding.civitai_id}...")

        if embedding.civitai_version:
            # Fetch specific version
            version = await civitai_client.get_model_version(embedding.civitai_version)
        else:
            # Fetch model and get latest version
            model = await civitai_client.get_model(embedding.civitai_id)
            version = model.latest_version
            if version is None:
                raise ValueError(f"No versions available for model {embedding.civitai_id}")

        # Validate compatibility
        if validate_compatibility and pipeline_type:
            if not is_embedding_compatible(pipeline_type, version.base_model):
                raise EmbeddingIncompatibleError(
                    embedding.name,
                    pipeline_type,
                    version.base_model,
                )

        # Download
        print(f"Downloading embedding: {version.name} (base: {version.base_model})")
        path = await civitai_client.download_model_version(version)
        embedding._resolved_path = path
        return path

    raise ValueError(f"Unknown embedding source: {embedding.source}")


class EmbeddingLoaderMixin:
    """Mixin for pipelines that support loading textual inversions / embeddings."""

    pipe: Any
    _embedding_configs: list[EmbeddingConfig]
    _loaded_tokens: list[str]

    def _init_embedding_state(self) -> None:
        """Initialize embedding-related state. Call this in __init__."""
        self._embedding_configs = []
        self._loaded_tokens = []

    def load_single_embedding(
        self,
        embedding: EmbeddingConfig,
    ) -> str:
        """Load a single textual inversion embedding into the pipeline.

        Args:
            embedding: Embedding configuration with resolved path or repo

        Returns:
            Token that was loaded

        Raises:
            RuntimeError: If pipeline not loaded
            ValueError: If embedding configuration is invalid
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        # Determine token - use configured token, or auto-detect
        token = embedding.token or embedding.name

        if embedding.source == EmbeddingSource.HUGGINGFACE:
            print(f"Loading embedding from HF: {embedding.repo} (token: {token})")
            self.pipe.load_textual_inversion(
                embedding.repo,
                token=token,
            )
        elif embedding.source in (EmbeddingSource.CIVITAI, EmbeddingSource.LOCAL):
            if embedding._resolved_path is None:
                raise ValueError(f"Embedding path not resolved: {embedding}")
            print(f"Loading embedding from path: {embedding._resolved_path} (token: {token})")
            self.pipe.load_textual_inversion(
                str(embedding._resolved_path),
                token=token,
            )
        else:
            raise ValueError(f"Unknown embedding source: {embedding.source}")

        self._loaded_tokens.append(token)
        self._embedding_configs.append(embedding)
        return token

    def load_embeddings_sync(
        self,
        embeddings: list[EmbeddingConfig],
    ) -> list[str]:
        """Load multiple embeddings synchronously (assuming paths are already resolved).

        Args:
            embeddings: List of embedding configurations with resolved paths

        Returns:
            List of loaded tokens
        """
        tokens = []

        for embedding in embeddings:
            token = self.load_single_embedding(embedding)
            tokens.append(token)

        return tokens

    async def load_embeddings_async(
        self,
        embeddings: list[EmbeddingConfig],
        civitai_client: CivitaiClient | None = None,
        pipeline_type: str | None = None,
        validate_compatibility: bool = True,
    ) -> list[str]:
        """Load multiple embeddings, downloading from Civitai as needed.

        Args:
            embeddings: List of embedding configurations
            civitai_client: Client for Civitai downloads
            pipeline_type: Pipeline type for compatibility validation
            validate_compatibility: Whether to validate base model compatibility

        Returns:
            List of loaded tokens
        """
        for embedding in embeddings:
            await resolve_embedding_path(
                embedding,
                civitai_client=civitai_client,
                pipeline_type=pipeline_type,
                validate_compatibility=validate_compatibility,
            )

        return self.load_embeddings_sync(embeddings)

    @property
    def active_embeddings(self) -> list[str]:
        """Get list of currently loaded embedding tokens."""
        return list(self._loaded_tokens)

    @property
    def embedding_count(self) -> int:
        """Get number of loaded embeddings."""
        return len(self._loaded_tokens)
