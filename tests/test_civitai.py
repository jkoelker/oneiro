"""Tests for CivitaiClient."""

from pathlib import Path

import httpx
import pytest
import respx

from oneiro.civitai import (
    BaseModel,
    CivitaiAuthError,
    CivitaiCache,
    CivitaiClient,
    CivitaiNotFoundError,
    CivitaiRateLimitError,
    Model,
    ModelFile,
    ModelType,
    ModelVersion,
)

# Sample API responses for testing
SAMPLE_MODEL_RESPONSE = {
    "id": 12345,
    "name": "Test Model",
    "description": "A test model for unit tests",
    "type": "LORA",
    "nsfw": False,
    "tags": ["test", "lora"],
    "creator": {"username": "testuser"},
    "modelVersions": [
        {
            "id": 67890,
            "modelId": 12345,
            "name": "v1.0",
            "description": "First version",
            "baseModel": "SDXL 1.0",
            "downloadUrl": "https://civitai.com/api/download/models/67890",
            "createdAt": "2025-01-01T00:00:00Z",
            "files": [
                {
                    "id": 11111,
                    "name": "test_model.safetensors",
                    "sizeKB": 1024000,
                    "type": "Model",
                    "metadata": {"format": "SafeTensor", "fp": "fp16"},
                    "downloadUrl": "https://civitai.com/api/download/models/67890",
                    "hashes": {
                        "SHA256": "abc123def456789abc123def456789abc123def456789abc123def456789abcd"
                    },
                    "primary": True,
                }
            ],
        }
    ],
}

SAMPLE_VERSION_RESPONSE = {
    "id": 67890,
    "modelId": 12345,
    "name": "v1.0",
    "description": "First version",
    "baseModel": "SDXL 1.0",
    "downloadUrl": "https://civitai.com/api/download/models/67890",
    "createdAt": "2025-01-01T00:00:00Z",
    "files": [
        {
            "id": 11111,
            "name": "test_model.safetensors",
            "sizeKB": 1024000,
            "type": "Model",
            "metadata": {"format": "SafeTensor", "fp": "fp16"},
            "downloadUrl": "https://civitai.com/api/download/models/67890",
            "hashes": {
                "SHA256": "abc123def456789abc123def456789abc123def456789abc123def456789abcd"
            },
            "primary": True,
        }
    ],
}

SAMPLE_SEARCH_RESPONSE = {
    "items": [SAMPLE_MODEL_RESPONSE],
    "metadata": {
        "totalItems": 1,
        "currentPage": 1,
        "pageSize": 20,
        "totalPages": 1,
    },
}


class TestModelFile:
    """Tests for ModelFile dataclass."""

    def test_from_dict(self):
        """ModelFile.from_dict creates instance from API response."""
        data = SAMPLE_VERSION_RESPONSE["files"][0]
        file = ModelFile.from_dict(data)

        assert file.id == 11111
        assert file.name == "test_model.safetensors"
        assert file.size_kb == 1024000
        assert file.type == "Model"
        assert file.format == "SafeTensor"
        assert file.fp == "fp16"
        assert file.sha256 == "abc123def456789abc123def456789abc123def456789abc123def456789abcd"
        assert file.primary is True

    def test_from_dict_missing_optional_fields(self):
        """ModelFile handles missing optional fields gracefully."""
        data = {
            "id": 1,
            "name": "model.bin",
            "downloadUrl": "https://example.com/download",
        }
        file = ModelFile.from_dict(data)

        assert file.id == 1
        assert file.name == "model.bin"
        assert file.sha256 is None
        assert file.format is None
        assert file.primary is False


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_from_dict(self):
        """ModelVersion.from_dict creates instance with files."""
        version = ModelVersion.from_dict(SAMPLE_VERSION_RESPONSE)

        assert version.id == 67890
        assert version.model_id == 12345
        assert version.name == "v1.0"
        assert version.base_model == "SDXL 1.0"
        assert len(version.files) == 1
        assert version.files[0].name == "test_model.safetensors"

    def test_primary_file(self):
        """primary_file returns the file marked as primary."""
        version = ModelVersion.from_dict(SAMPLE_VERSION_RESPONSE)
        primary = version.primary_file

        assert primary is not None
        assert primary.primary is True
        assert primary.name == "test_model.safetensors"

    def test_primary_file_fallback(self):
        """primary_file returns first file if none marked primary."""
        data = {
            "id": 1,
            "name": "v1",
            "files": [
                {"id": 1, "name": "first.bin", "downloadUrl": "http://x", "primary": False},
                {"id": 2, "name": "second.bin", "downloadUrl": "http://y", "primary": False},
            ],
        }
        version = ModelVersion.from_dict(data)

        assert version.primary_file is not None
        assert version.primary_file.name == "first.bin"


class TestModel:
    """Tests for Model dataclass."""

    def test_from_dict(self):
        """Model.from_dict creates instance with versions."""
        model = Model.from_dict(SAMPLE_MODEL_RESPONSE)

        assert model.id == 12345
        assert model.name == "Test Model"
        assert model.type == "LORA"
        assert model.nsfw is False
        assert "test" in model.tags
        assert model.creator == "testuser"
        assert len(model.versions) == 1

    def test_latest_version(self):
        """latest_version returns first version."""
        model = Model.from_dict(SAMPLE_MODEL_RESPONSE)

        assert model.latest_version is not None
        assert model.latest_version.name == "v1.0"


class TestCivitaiCache:
    """Tests for CivitaiCache."""

    def test_add_and_get(self, tmp_path):
        """Cache can add and retrieve entries."""
        cache = CivitaiCache(tmp_path)
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        cache.add(
            file_path=test_file,
            sha256="abc123",
            model_id=1,
            version_id=2,
            filename="test.bin",
            size_kb=100,
        )

        result = cache.get("abc123")
        assert result == test_file

    def test_get_missing_returns_none(self, tmp_path):
        """Cache returns None for missing entries."""
        cache = CivitaiCache(tmp_path)
        assert cache.get("nonexistent") is None

    def test_get_removes_stale_entry(self, tmp_path):
        """Cache removes entry if file no longer exists."""
        cache = CivitaiCache(tmp_path)
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        cache.add(
            file_path=test_file,
            sha256="abc123",
            model_id=1,
            version_id=2,
            filename="test.bin",
            size_kb=100,
        )

        # Delete the file
        test_file.unlink()

        # Should return None and clean up stale entry
        assert cache.get("abc123") is None
        assert "abc123" not in cache._metadata

    def test_get_by_version(self, tmp_path):
        """Cache can lookup by version ID."""
        cache = CivitaiCache(tmp_path)
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        cache.add(
            file_path=test_file,
            sha256="abc123",
            model_id=1,
            version_id=42,
            filename="test.bin",
            size_kb=100,
        )

        result = cache.get_by_version(42)
        assert result == test_file

    def test_remove(self, tmp_path):
        """Cache can remove entries and delete files."""
        cache = CivitaiCache(tmp_path)
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        cache.add(
            file_path=test_file,
            sha256="abc123",
            model_id=1,
            version_id=2,
            filename="test.bin",
            size_kb=100,
        )

        assert cache.remove("abc123") is True
        assert not test_file.exists()
        assert cache.get("abc123") is None

    def test_persistence(self, tmp_path):
        """Cache persists to disk and reloads."""
        cache1 = CivitaiCache(tmp_path)
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        cache1.add(
            file_path=test_file,
            sha256="abc123",
            model_id=1,
            version_id=2,
            filename="test.bin",
            size_kb=100,
        )

        # Create new cache instance (simulates restart)
        cache2 = CivitaiCache(tmp_path)

        assert cache2.get("abc123") == test_file

    def test_metadata_file_has_trailing_newline(self, tmp_path):
        """Metadata file should end with a newline."""
        cache = CivitaiCache(tmp_path)
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test content")

        cache.add(
            file_path=test_file,
            sha256="abc123",
            model_id=1,
            version_id=2,
            filename="test.bin",
            size_kb=100,
        )

        content = cache.metadata_file.read_text()
        assert content.endswith("\n"), "metadata.json should end with a newline"

    def test_total_size(self, tmp_path):
        """total_size_kb returns sum of all cached file sizes."""
        cache = CivitaiCache(tmp_path)

        for i, size in enumerate([100, 200, 300]):
            test_file = tmp_path / f"test{i}.bin"
            test_file.write_bytes(b"x" * 10)
            cache.add(
                file_path=test_file,
                sha256=f"hash{i}",
                model_id=1,
                version_id=i,
                filename=f"test{i}.bin",
                size_kb=size,
            )

        assert cache.total_size_kb() == 600


class TestCivitaiClientInit:
    """Tests for CivitaiClient initialization."""

    def test_default_init(self):
        """Client initializes with defaults."""
        client = CivitaiClient()

        assert client.api_key is None
        assert client.cache_dir == Path.home() / ".cache" / "civitai"
        assert client.timeout == 30.0
        assert client.verify_hashes is True

    def test_init_with_api_key(self):
        """Client accepts API key."""
        client = CivitaiClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_init_with_env_var(self, monkeypatch):
        """Client reads API key from environment."""
        monkeypatch.setenv("CIVITAI_API_KEY", "env-key")
        client = CivitaiClient()
        assert client.api_key == "env-key"

    def test_init_with_custom_cache_dir(self, tmp_path):
        """Client accepts custom cache directory."""
        client = CivitaiClient(cache_dir=tmp_path / "custom")
        assert client.cache_dir == tmp_path / "custom"

    def test_init_with_cache_dir_env_var(self, tmp_path, monkeypatch):
        """Client reads cache_dir from CIVITAI_CACHE_DIR environment variable."""
        monkeypatch.setenv("CIVITAI_CACHE_DIR", str(tmp_path / "env-cache"))
        client = CivitaiClient()
        assert client.cache_dir == tmp_path / "env-cache"

    def test_init_cache_dir_argument_takes_precedence(self, tmp_path, monkeypatch):
        """Explicit cache_dir argument takes precedence over env var."""
        monkeypatch.setenv("CIVITAI_CACHE_DIR", str(tmp_path / "env-cache"))
        client = CivitaiClient(cache_dir=tmp_path / "explicit-cache")
        assert client.cache_dir == tmp_path / "explicit-cache"

    def test_init_cache_dir_expands_user(self, monkeypatch):
        """Client expands ~ in cache_dir paths."""
        monkeypatch.setenv("CIVITAI_CACHE_DIR", "~/my-cache")
        client = CivitaiClient()
        assert client.cache_dir == Path.home() / "my-cache"

    def test_from_config(self, tmp_path, monkeypatch):
        """Client can be created from Config object."""
        monkeypatch.setenv("MY_API_KEY", "config-key")
        # Ensure CIVITAI_CACHE_DIR is not set for this test
        monkeypatch.delenv("CIVITAI_CACHE_DIR", raising=False)

        # Mock config object
        class MockConfig:
            def get(self, *keys, default=None):
                if keys == ("civitai",):
                    return {
                        "api_key": "${MY_API_KEY}",
                        "cache_dir": str(tmp_path / "cache"),
                        "download_timeout": 7200.0,
                        "verify_hashes": False,
                    }
                return default

        client = CivitaiClient.from_config(MockConfig())

        assert client.api_key == "config-key"
        assert client.cache_dir == tmp_path / "cache"
        assert client.download_timeout == 7200.0
        assert client.verify_hashes is False

    def test_from_config_env_var_precedence(self, tmp_path, monkeypatch):
        """CIVITAI_CACHE_DIR env var takes precedence over config file value."""
        monkeypatch.setenv("CIVITAI_CACHE_DIR", str(tmp_path / "env-cache"))

        # Mock config with a different cache_dir
        class MockConfig:
            def get(self, *keys, default=None):
                if keys == ("civitai",):
                    return {
                        "cache_dir": str(tmp_path / "config-cache"),
                    }
                return default

        client = CivitaiClient.from_config(MockConfig())

        # Env var should win over config
        assert client.cache_dir == tmp_path / "env-cache"

    def test_from_config_falls_back_to_config_cache_dir(self, tmp_path, monkeypatch):
        """from_config uses config cache_dir when env var is not set."""
        monkeypatch.delenv("CIVITAI_CACHE_DIR", raising=False)

        class MockConfig:
            def get(self, *keys, default=None):
                if keys == ("civitai",):
                    return {
                        "cache_dir": str(tmp_path / "config-cache"),
                    }
                return default

        client = CivitaiClient.from_config(MockConfig())

        assert client.cache_dir == tmp_path / "config-cache"

    def test_from_config_falls_back_to_default(self, tmp_path, monkeypatch):
        """from_config uses default cache_dir when neither env var nor config is set."""
        monkeypatch.delenv("CIVITAI_CACHE_DIR", raising=False)

        class MockConfig:
            def get(self, *keys, default=None):
                if keys == ("civitai",):
                    return {}  # No cache_dir in config
                return default

        client = CivitaiClient.from_config(MockConfig())

        # Should use the class default
        assert client.cache_dir == Path.home() / ".cache" / "civitai"


@pytest.mark.asyncio
class TestCivitaiClientAPI:
    """Tests for CivitaiClient API methods."""

    @respx.mock
    async def test_get_model(self):
        """get_model fetches model by ID."""
        respx.get("https://civitai.com/api/v1/models/12345").mock(
            return_value=httpx.Response(200, json=SAMPLE_MODEL_RESPONSE)
        )

        async with CivitaiClient() as client:
            model = await client.get_model(12345)

        assert model.id == 12345
        assert model.name == "Test Model"

    @respx.mock
    async def test_get_model_version(self):
        """get_model_version fetches version by ID."""
        respx.get("https://civitai.com/api/v1/model-versions/67890").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERSION_RESPONSE)
        )

        async with CivitaiClient() as client:
            version = await client.get_model_version(67890)

        assert version.id == 67890
        assert version.name == "v1.0"

    @respx.mock
    async def test_get_model_version_by_hash(self):
        """get_model_version_by_hash looks up by SHA256."""
        test_hash = "abc123def456"
        respx.get(f"https://civitai.com/api/v1/model-versions/by-hash/{test_hash}").mock(
            return_value=httpx.Response(200, json=SAMPLE_VERSION_RESPONSE)
        )

        async with CivitaiClient() as client:
            version = await client.get_model_version_by_hash(test_hash)

        assert version.id == 67890

    @respx.mock
    async def test_search_models(self):
        """search_models returns filtered results."""
        respx.get("https://civitai.com/api/v1/models").mock(
            return_value=httpx.Response(200, json=SAMPLE_SEARCH_RESPONSE)
        )

        async with CivitaiClient() as client:
            models, metadata = await client.search_models(
                query="test",
                types=[ModelType.LORA],
                base_models=[BaseModel.SDXL_1_0],
            )

        assert len(models) == 1
        assert models[0].name == "Test Model"
        assert metadata["totalItems"] == 1

    @respx.mock
    async def test_authentication_header(self):
        """Client sends Authorization header when API key provided."""
        route = respx.get("https://civitai.com/api/v1/models/1").mock(
            return_value=httpx.Response(200, json=SAMPLE_MODEL_RESPONSE)
        )

        async with CivitaiClient(api_key="test-key") as client:
            await client.get_model(1)

        assert route.called
        request = route.calls[0].request
        assert request.headers.get("Authorization") == "Bearer test-key"


@pytest.mark.asyncio
class TestCivitaiClientErrors:
    """Tests for CivitaiClient error handling."""

    @respx.mock
    async def test_auth_error(self):
        """Client raises CivitaiAuthError on 401."""
        respx.get("https://civitai.com/api/v1/models/1").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )

        async with CivitaiClient() as client:
            with pytest.raises(CivitaiAuthError, match="Invalid or missing API key"):
                await client.get_model(1)

    @respx.mock
    async def test_not_found_error(self):
        """Client raises CivitaiNotFoundError on 404."""
        respx.get("https://civitai.com/api/v1/models/999999").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )

        async with CivitaiClient() as client:
            with pytest.raises(CivitaiNotFoundError, match="Resource not found"):
                await client.get_model(999999)

    @respx.mock
    async def test_rate_limit_retry(self):
        """Client retries on rate limit then succeeds."""
        # First two calls return 429, third succeeds
        route = respx.get("https://civitai.com/api/v1/models/1")
        route.side_effect = [
            httpx.Response(429, headers={"X-RateLimit-Reset": "1"}),
            httpx.Response(429, headers={"X-RateLimit-Reset": "1"}),
            httpx.Response(200, json=SAMPLE_MODEL_RESPONSE),
        ]

        async with CivitaiClient() as client:
            model = await client.get_model(1)

        assert model.id == 12345
        assert route.call_count == 3

    @respx.mock
    async def test_rate_limit_exhausted(self):
        """Client raises CivitaiRateLimitError after max retries."""
        respx.get("https://civitai.com/api/v1/models/1").mock(
            return_value=httpx.Response(429, headers={"X-RateLimit-Reset": "60"})
        )

        async with CivitaiClient() as client:
            with pytest.raises(CivitaiRateLimitError, match="Rate limit exceeded"):
                await client.get_model(1)


@pytest.mark.asyncio
class TestCivitaiClientDownload:
    """Tests for CivitaiClient download functionality."""

    @respx.mock
    async def test_download_file(self, tmp_path):
        """download_file saves file to destination."""
        test_content = b"test file content"
        respx.get("https://civitai.com/api/download/models/1").mock(
            return_value=httpx.Response(
                200,
                content=test_content,
                headers={"Content-Length": str(len(test_content))},
            )
        )

        dest = tmp_path / "downloaded.bin"

        async with CivitaiClient(cache_dir=tmp_path, verify_hashes=False) as client:
            result = await client.download_file(
                url="https://civitai.com/api/download/models/1",
                dest=dest,
            )

        assert result == dest
        assert dest.read_bytes() == test_content

    @respx.mock
    async def test_download_with_progress(self, tmp_path):
        """download_file calls progress callback."""
        test_content = b"test file content"
        respx.get("https://civitai.com/api/download/models/1").mock(
            return_value=httpx.Response(
                200,
                content=test_content,
                headers={"Content-Length": str(len(test_content))},
            )
        )

        progress_calls = []

        async def progress_callback(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        dest = tmp_path / "downloaded.bin"

        async with CivitaiClient(cache_dir=tmp_path, verify_hashes=False) as client:
            await client.download_file(
                url="https://civitai.com/api/download/models/1",
                dest=dest,
                progress_callback=progress_callback,
            )

        assert len(progress_calls) > 0
        # Final call should have downloaded == total
        assert progress_calls[-1][0] == progress_calls[-1][1]

    @respx.mock
    async def test_download_uses_cache(self, tmp_path):
        """download_file returns cached file if hash matches."""
        # Create cached file
        cached_file = tmp_path / "cached.bin"
        cached_file.write_bytes(b"cached content")

        cache = CivitaiCache(tmp_path)
        cache.add(
            file_path=cached_file,
            sha256="abc123",
            model_id=1,
            version_id=1,
            filename="cached.bin",
            size_kb=100,
        )

        dest = tmp_path / "output.bin"

        async with CivitaiClient(cache_dir=tmp_path) as client:
            await client.download_file(
                url="https://civitai.com/api/download/models/1",
                dest=dest,
                expected_hash="abc123",
            )

        # Should copy from cache, not download
        assert dest.read_bytes() == b"cached content"

    @respx.mock
    async def test_download_model_version(self, tmp_path):
        """download_model_version downloads primary file."""
        test_content = b"model content"
        respx.get("https://civitai.com/api/download/models/67890").mock(
            return_value=httpx.Response(
                200,
                content=test_content,
                headers={"Content-Length": str(len(test_content))},
            )
        )

        version = ModelVersion.from_dict(SAMPLE_VERSION_RESPONSE)

        async with CivitaiClient(cache_dir=tmp_path, verify_hashes=False) as client:
            result = await client.download_model_version(version)

        assert result.name == "test_model.safetensors"
        assert result.read_bytes() == test_content
