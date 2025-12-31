"""Civitai API client with caching and download support."""

import asyncio
import hashlib
import json
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class CivitaiError(Exception):
    """Base exception for Civitai client errors."""

    pass


class CivitaiAuthError(CivitaiError):
    """Authentication failed or API key invalid."""

    pass


class CivitaiRateLimitError(CivitaiError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class CivitaiNotFoundError(CivitaiError):
    """Resource not found."""

    pass


class ModelType(str, Enum):
    """Civitai model types."""

    CHECKPOINT = "Checkpoint"
    TEXTUAL_INVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETIC_GRADIENT = "AestheticGradient"
    LORA = "LORA"
    CONTROLNET = "Controlnet"
    POSES = "Poses"


class BaseModel(str, Enum):
    """Supported base models for filtering."""

    SD_1_5 = "SD 1.5"
    SD_2_1 = "SD 2.1"
    SDXL_1_0 = "SDXL 1.0"
    PONY = "Pony"
    FLUX_1 = "Flux.1"


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


@dataclass
class ModelFile:
    """Represents a downloadable file in a model version."""

    id: int
    name: str
    size_kb: float
    type: str
    format: str | None
    fp: str | None
    download_url: str
    sha256: str | None = None
    primary: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelFile":
        """Create ModelFile from API response dict."""
        hashes = data.get("hashes", {})
        return cls(
            id=data["id"],
            name=data["name"],
            size_kb=data.get("sizeKB", 0),
            type=data.get("type", "Model"),
            format=data.get("metadata", {}).get("format"),
            fp=data.get("metadata", {}).get("fp"),
            download_url=data["downloadUrl"],
            sha256=hashes.get("SHA256"),
            primary=data.get("primary", False),
        )


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""

    id: int
    model_id: int
    name: str
    description: str | None
    base_model: str | None
    trained_words: list[str] = field(default_factory=list)
    files: list[ModelFile] = field(default_factory=list)
    download_url: str | None = None
    created_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create ModelVersion from API response dict."""
        files = [ModelFile.from_dict(f) for f in data.get("files", [])]
        return cls(
            id=data["id"],
            model_id=data.get("modelId", 0),
            name=data["name"],
            description=data.get("description"),
            base_model=data.get("baseModel"),
            trained_words=data.get("trainedWords", []),
            files=files,
            download_url=data.get("downloadUrl"),
            created_at=data.get("createdAt"),
        )

    @property
    def primary_file(self) -> ModelFile | None:
        """Get the primary downloadable file."""
        for f in self.files:
            if f.primary:
                return f
        return self.files[0] if self.files else None


@dataclass
class Model:
    """Represents a Civitai model."""

    id: int
    name: str
    description: str | None
    type: str
    nsfw: bool
    tags: list[str] = field(default_factory=list)
    versions: list[ModelVersion] = field(default_factory=list)
    creator: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Model":
        """Create Model from API response dict."""
        versions = [ModelVersion.from_dict(v) for v in data.get("modelVersions", [])]
        creator = data.get("creator", {})
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            type=data.get("type", "Unknown"),
            nsfw=data.get("nsfw", False),
            tags=data.get("tags", []),
            versions=versions,
            creator=creator.get("username") if isinstance(creator, dict) else None,
        )

    @property
    def latest_version(self) -> ModelVersion | None:
        """Get the most recent version."""
        return self.versions[0] if self.versions else None


@dataclass
class CacheEntry:
    """Metadata for a cached file."""

    sha256: str
    model_id: int
    version_id: int
    filename: str
    size_kb: float
    downloaded_at: str
    file_path: str


class CivitaiCache:
    """Local file cache with hash-based deduplication."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.metadata_file = cache_dir / "metadata.json"
        self._metadata: dict[str, CacheEntry] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    data = json.load(f)
                for sha256, entry_data in data.get("files", {}).items():
                    self._metadata[sha256] = CacheEntry(
                        sha256=sha256,
                        model_id=entry_data["model_id"],
                        version_id=entry_data["version_id"],
                        filename=entry_data["filename"],
                        size_kb=entry_data["size_kb"],
                        downloaded_at=entry_data["downloaded_at"],
                        file_path=entry_data["file_path"],
                    )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Invalid cache metadata, starting fresh: {e}")
                self._metadata = {}

    def _save_metadata(self) -> None:
        """Persist cache metadata to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "files": {
                sha256: {
                    "model_id": entry.model_id,
                    "version_id": entry.version_id,
                    "filename": entry.filename,
                    "size_kb": entry.size_kb,
                    "downloaded_at": entry.downloaded_at,
                    "file_path": entry.file_path,
                }
                for sha256, entry in self._metadata.items()
            }
        }
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def get(self, sha256: str) -> Path | None:
        """Get cached file path by SHA256 hash."""
        if sha256 in self._metadata:
            path = Path(self._metadata[sha256].file_path)
            if path.exists():
                return path
            # File missing, remove stale entry
            del self._metadata[sha256]
            self._save_metadata()
        return None

    def get_by_version(self, version_id: int) -> Path | None:
        """Get cached file path by version ID."""
        for entry in self._metadata.values():
            if entry.version_id == version_id:
                path = Path(entry.file_path)
                if path.exists():
                    return path
        return None

    def add(
        self,
        file_path: Path,
        sha256: str,
        model_id: int,
        version_id: int,
        filename: str,
        size_kb: float,
    ) -> None:
        """Add a file to the cache."""
        self._metadata[sha256] = CacheEntry(
            sha256=sha256,
            model_id=model_id,
            version_id=version_id,
            filename=filename,
            size_kb=size_kb,
            downloaded_at=datetime.now().isoformat(),
            file_path=str(file_path),
        )
        self._save_metadata()

    def remove(self, sha256: str) -> bool:
        """Remove a file from the cache."""
        if sha256 in self._metadata:
            path = Path(self._metadata[sha256].file_path)
            if path.exists():
                path.unlink()
            del self._metadata[sha256]
            self._save_metadata()
            return True
        return False

    def list_entries(self) -> list[CacheEntry]:
        """List all cache entries."""
        return list(self._metadata.values())

    def total_size_kb(self) -> float:
        """Get total size of cached files in KB."""
        return sum(entry.size_kb for entry in self._metadata.values())


class CivitaiClient:
    """Async client for the Civitai API with caching support.

    Args:
        api_key: Optional API key for authentication.
        cache_dir: Directory for local file cache.
        timeout: Request timeout in seconds (default: 30 for API, 3600 for downloads).
        verify_hashes: Whether to verify SHA256 hashes after download.
    """

    BASE_URL = "https://civitai.com/api/v1"
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "civitai"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        timeout: float = 30.0,
        download_timeout: float = 3600.0,
        verify_hashes: bool = True,
    ):
        self.api_key = api_key or os.environ.get("CIVITAI_API_KEY")
        # Support cache_dir from argument, env var, or default
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser()
        elif env_cache_dir := os.environ.get("CIVITAI_CACHE_DIR"):
            self.cache_dir = Path(env_cache_dir).expanduser()
        else:
            self.cache_dir = self.DEFAULT_CACHE_DIR
        self.timeout = timeout
        self.download_timeout = download_timeout
        self.verify_hashes = verify_hashes
        self._client: httpx.AsyncClient | None = None
        self._cache = CivitaiCache(self.cache_dir)

    @classmethod
    def from_config(cls, config: Any) -> "CivitaiClient":
        """Create client from Config object.

        Reads from [civitai] section:
            api_key: API key (or reads from CIVITAI_API_KEY env var)
            cache_dir: Cache directory path
            download_timeout: Download timeout in seconds
            verify_hashes: Whether to verify SHA256 after download
        """
        civitai_config = config.get("civitai", default={})

        api_key = civitai_config.get("api_key")
        # Support ${ENV_VAR} syntax
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var)

        # Environment variable takes precedence over config file for cache_dir
        cache_dir: Path | None = None
        if env_cache_dir := os.environ.get("CIVITAI_CACHE_DIR"):
            cache_dir = Path(env_cache_dir).expanduser()
        elif config_cache_dir := civitai_config.get("cache_dir"):
            cache_dir = Path(config_cache_dir).expanduser()

        return cls(
            api_key=api_key,
            cache_dir=cache_dir,
            download_timeout=civitai_config.get("download_timeout", 3600.0),
            verify_hashes=civitai_config.get("verify_hashes", True),
        )

    async def __aenter__(self) -> "CivitaiClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            headers = {"User-Agent": "oneiro/0.1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Make an API request with automatic retry on rate limit.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (without base URL)
            params: Query parameters
            max_retries: Maximum retry attempts for rate limiting

        Returns:
            JSON response as dict

        Raises:
            CivitaiAuthError: If authentication fails
            CivitaiRateLimitError: If rate limit exceeded after retries
            CivitaiNotFoundError: If resource not found
            CivitaiError: For other API errors
        """
        client = await self._ensure_client()
        url = f"{self.BASE_URL}{path}"

        for attempt in range(max_retries):
            try:
                response = await client.request(method, url, params=params)

                if response.status_code == 401:
                    raise CivitaiAuthError("Invalid or missing API key")

                if response.status_code == 404:
                    raise CivitaiNotFoundError(f"Resource not found: {path}")

                if response.status_code == 429:
                    # Rate limited - check for retry-after header
                    retry_after = int(response.headers.get("X-RateLimit-Reset", 60))
                    if attempt < max_retries - 1:
                        wait_time = min(retry_after, 2 ** (attempt + 1))
                        print(f"Rate limited, waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise CivitaiRateLimitError(
                        "Rate limit exceeded after retries",
                        retry_after=retry_after,
                    )

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                raise CivitaiError(f"HTTP error: {e.response.status_code}") from e
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise CivitaiError(f"Request failed: {e}") from e

        raise CivitaiError("Max retries exceeded")

    async def get_model(self, model_id: int) -> Model:
        """Fetch model metadata by ID.

        Args:
            model_id: The Civitai model ID

        Returns:
            Model object with versions and files
        """
        data = await self._request("GET", f"/models/{model_id}")
        return Model.from_dict(data)

    async def get_model_version(self, version_id: int) -> ModelVersion:
        """Fetch specific model version by ID.

        Args:
            version_id: The model version ID

        Returns:
            ModelVersion object with files
        """
        data = await self._request("GET", f"/model-versions/{version_id}")
        return ModelVersion.from_dict(data)

    async def get_model_version_by_hash(self, sha256: str) -> ModelVersion:
        """Lookup model version by file hash.

        Args:
            sha256: SHA256 hash of the model file

        Returns:
            ModelVersion object
        """
        data = await self._request("GET", f"/model-versions/by-hash/{sha256}")
        return ModelVersion.from_dict(data)

    async def search_models(
        self,
        query: str | None = None,
        types: list[ModelType | str] | None = None,
        base_models: list[BaseModel | str] | None = None,
        sort: str | None = None,
        limit: int = 20,
        page: int = 1,
        nsfw: bool | None = None,
    ) -> tuple[list[Model], dict[str, Any]]:
        """Search and filter models.

        Args:
            query: Search query string
            types: Filter by model types
            base_models: Filter by base model compatibility
            sort: Sort order (Highest Rated, Most Downloaded, Newest)
            limit: Results per page (max 100)
            page: Page number
            nsfw: Filter NSFW content (None = no filter)

        Returns:
            Tuple of (list of Models, metadata dict with pagination info)
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "page": page,
        }

        if query:
            params["query"] = query
        if types:
            params["types"] = ",".join(t.value if isinstance(t, ModelType) else t for t in types)
        if base_models:
            params["baseModels"] = ",".join(
                b.value if isinstance(b, BaseModel) else b for b in base_models
            )
        if sort:
            params["sort"] = sort
        if nsfw is not None:
            params["nsfw"] = str(nsfw).lower()

        data = await self._request("GET", "/models", params=params)
        models = [Model.from_dict(m) for m in data.get("items", [])]
        metadata = data.get("metadata", {})

        return models, metadata

    async def download_file(
        self,
        url: str,
        dest: Path | str,
        expected_hash: str | None = None,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
        resume: bool = True,
    ) -> Path:
        """Download a file with resume support and progress tracking.

        Args:
            url: Download URL
            dest: Destination file path
            expected_hash: Expected SHA256 hash for verification
            progress_callback: Async callback(downloaded_bytes, total_bytes)
            resume: Whether to attempt resume for partial downloads

        Returns:
            Path to downloaded file

        Raises:
            CivitaiError: If download or hash verification fails
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Check cache first if we have a hash
        if expected_hash:
            cached = self._cache.get(expected_hash)
            if cached:
                print(f"Found in cache: {cached}")
                if dest != cached:
                    shutil.copy(cached, dest)
                return dest

        # Use temp file for download, move on success
        temp_dir = dest.parent / ".downloads"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{dest.name}.partial"

        headers: dict[str, str] = {}
        start_byte = 0

        # Check for partial download
        if resume and temp_file.exists():
            start_byte = temp_file.stat().st_size
            headers["Range"] = f"bytes={start_byte}-"
            print(f"Resuming download from byte {start_byte}")

        client = await self._ensure_client()

        # Add API key as query param for download URLs if authenticated
        download_url = url
        if self.api_key and "civitai.com" in url:
            separator = "&" if "?" in url else "?"
            download_url = f"{url}{separator}token={self.api_key}"

        try:
            async with client.stream(
                "GET",
                download_url,
                headers=headers,
                timeout=httpx.Timeout(self.download_timeout),
            ) as response:
                # Handle resume response
                if response.status_code == 416:
                    # Range not satisfiable - file already complete or server doesn't support
                    if temp_file.exists():
                        shutil.move(temp_file, dest)
                        return dest
                    raise CivitaiError("Download failed: range not satisfiable")

                if response.status_code == 206:
                    # Partial content - resuming
                    total_size = start_byte + int(response.headers.get("content-length", 0))
                elif response.status_code == 200:
                    # Full download
                    total_size = int(response.headers.get("content-length", 0))
                    start_byte = 0  # Server doesn't support resume, start fresh
                    if temp_file.exists():
                        temp_file.unlink()
                else:
                    response.raise_for_status()
                    total_size = 0

                mode = "ab" if start_byte > 0 else "wb"
                downloaded = start_byte

                with open(temp_file, mode) as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            await progress_callback(downloaded, total_size)

        except httpx.HTTPStatusError as e:
            raise CivitaiError(f"Download failed: HTTP {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise CivitaiError(f"Download failed: {e}") from e

        # Verify hash if provided
        if self.verify_hashes and expected_hash:
            print("Verifying SHA256 hash...")
            actual_hash = await self._compute_hash(temp_file)
            if actual_hash.lower() != expected_hash.lower():
                temp_file.unlink()
                raise CivitaiError(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
            print("Hash verified successfully")

        # Move to final destination
        shutil.move(temp_file, dest)

        return dest

    async def download_model_version(
        self,
        version: ModelVersion,
        dest_dir: Path | str | None = None,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> Path:
        """Download the primary file for a model version.

        Args:
            version: ModelVersion to download
            dest_dir: Destination directory (default: cache_dir/models)
            progress_callback: Progress callback

        Returns:
            Path to downloaded file
        """
        primary = version.primary_file
        if not primary:
            raise CivitaiError(f"No files available for version {version.id}")

        # Check cache first
        if primary.sha256:
            cached = self._cache.get(primary.sha256)
            if cached:
                print(f"Model found in cache: {cached}")
                return cached

        dest_dir = Path(dest_dir) if dest_dir else self.cache_dir / "models"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / primary.name

        downloaded = await self.download_file(
            url=primary.download_url,
            dest=dest,
            expected_hash=primary.sha256,
            progress_callback=progress_callback,
        )

        # Add to cache if we have a hash
        if primary.sha256:
            self._cache.add(
                file_path=downloaded,
                sha256=primary.sha256,
                model_id=version.model_id,
                version_id=version.id,
                filename=primary.name,
                size_kb=primary.size_kb,
            )

        return downloaded

    async def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @property
    def cache(self) -> CivitaiCache:
        """Access the file cache."""
        return self._cache
