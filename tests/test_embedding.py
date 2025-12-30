"""Tests for embedding configuration and loading functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from oneiro.pipelines.embedding import (
    PIPELINE_BASE_MODEL_MAP,
    EmbeddingConfig,
    EmbeddingIncompatibleError,
    EmbeddingSource,
    is_embedding_compatible,
    parse_civitai_url,
    parse_embedding_config,
    parse_embeddings_from_config,
)


class TestParseCivitaiUrl:
    """Tests for parse_civitai_url function."""

    def test_basic_model_url(self):
        """Parses basic model URL."""
        model_id, version_id = parse_civitai_url("https://civitai.com/models/12345")
        assert model_id == 12345
        assert version_id is None

    def test_model_url_with_name(self):
        """Parses model URL with name slug."""
        model_id, version_id = parse_civitai_url("https://civitai.com/models/12345/my-cool-model")
        assert model_id == 12345
        assert version_id is None

    def test_model_url_with_version(self):
        """Parses model URL with version ID in query string."""
        model_id, version_id = parse_civitai_url(
            "https://civitai.com/models/12345?modelVersionId=67890"
        )
        assert model_id == 12345
        assert version_id == 67890

    def test_model_url_with_name_and_version(self):
        """Parses model URL with name and version."""
        model_id, version_id = parse_civitai_url(
            "https://civitai.com/models/12345/model-name?modelVersionId=67890"
        )
        assert model_id == 12345
        assert version_id == 67890

    def test_invalid_url_raises(self):
        """Invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Civitai URL"):
            parse_civitai_url("https://example.com/something")


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_civitai_config_valid(self):
        """Valid Civitai config creates successfully."""
        config = EmbeddingConfig(
            name="easynegative",
            source=EmbeddingSource.CIVITAI,
            civitai_id=7808,
            token="easynegative",
        )
        assert config.source == EmbeddingSource.CIVITAI
        assert config.civitai_id == 7808
        assert config.token == "easynegative"

    def test_civitai_config_with_url(self):
        """Civitai config with URL is valid."""
        config = EmbeddingConfig(
            name="my-style",
            source=EmbeddingSource.CIVITAI,
            civitai_url="https://civitai.com/models/12345",
        )
        assert config.civitai_url == "https://civitai.com/models/12345"

    def test_civitai_config_requires_id_or_url(self):
        """Civitai config without ID or URL raises."""
        with pytest.raises(ValueError, match="civitai_id or civitai_url"):
            EmbeddingConfig(name="test", source=EmbeddingSource.CIVITAI)

    def test_huggingface_config_valid(self):
        """Valid HuggingFace config creates successfully."""
        config = EmbeddingConfig(
            name="cat-toy",
            source=EmbeddingSource.HUGGINGFACE,
            repo="sd-concepts-library/cat-toy",
        )
        assert config.source == EmbeddingSource.HUGGINGFACE
        assert config.repo == "sd-concepts-library/cat-toy"

    def test_huggingface_config_requires_repo(self):
        """HuggingFace config without repo raises."""
        with pytest.raises(ValueError, match="repo"):
            EmbeddingConfig(name="test", source=EmbeddingSource.HUGGINGFACE)

    def test_local_config_valid(self):
        """Valid local config creates successfully."""
        config = EmbeddingConfig(
            name="custom",
            source=EmbeddingSource.LOCAL,
            path="/path/to/embedding.safetensors",
        )
        assert config.source == EmbeddingSource.LOCAL
        assert config.path == "/path/to/embedding.safetensors"

    def test_local_config_requires_path(self):
        """Local config without path raises."""
        with pytest.raises(ValueError, match="path"):
            EmbeddingConfig(name="test", source=EmbeddingSource.LOCAL)

    def test_token_is_optional(self):
        """Token is optional and defaults to None."""
        config = EmbeddingConfig(
            name="test",
            source=EmbeddingSource.LOCAL,
            path="/path/to/file",
        )
        assert config.token is None


class TestParseEmbeddingConfig:
    """Tests for parse_embedding_config function."""

    def test_civitai_url_string(self):
        """Parses Civitai URL string."""
        config = parse_embedding_config("https://civitai.com/models/12345", name="test")
        assert config.source == EmbeddingSource.CIVITAI
        assert config.civitai_id == 12345
        assert config.name == "test"

    def test_civitai_url_with_version(self):
        """Parses Civitai URL with version."""
        config = parse_embedding_config(
            "https://civitai.com/models/12345?modelVersionId=67890",
            name="test",
        )
        assert config.civitai_id == 12345
        assert config.civitai_version == 67890

    def test_local_path_string(self):
        """Parses local path string."""
        config = parse_embedding_config("/path/to/embedding.safetensors", name="test")
        assert config.source == EmbeddingSource.LOCAL
        assert config.path == "/path/to/embedding.safetensors"

    def test_civitai_dict_with_id(self):
        """Parses Civitai dict with ID."""
        config = parse_embedding_config(
            {
                "source": "civitai",
                "id": 7808,
                "version": 12345,
                "token": "easynegative",
            },
            name="easynegative",
        )
        assert config.source == EmbeddingSource.CIVITAI
        assert config.civitai_id == 7808
        assert config.civitai_version == 12345
        assert config.token == "easynegative"

    def test_huggingface_dict(self):
        """Parses HuggingFace dict."""
        config = parse_embedding_config(
            {
                "source": "huggingface",
                "repo": "sd-concepts-library/cat-toy",
                "token": "<cat-toy>",
            },
            name="cat-toy",
        )
        assert config.source == EmbeddingSource.HUGGINGFACE
        assert config.repo == "sd-concepts-library/cat-toy"
        assert config.token == "<cat-toy>"

    def test_local_dict(self):
        """Parses local dict."""
        config = parse_embedding_config(
            {
                "source": "local",
                "path": "/path/to/embedding.safetensors",
                "token": "my-style",
            },
            name="custom",
        )
        assert config.source == EmbeddingSource.LOCAL
        assert config.path == "/path/to/embedding.safetensors"
        assert config.token == "my-style"

    def test_name_from_config_dict(self):
        """Name can come from config dict."""
        config = parse_embedding_config(
            {
                "name": "my-embedding",
                "source": "civitai",
                "id": 12345,
            }
        )
        assert config.name == "my-embedding"

    def test_invalid_source_raises(self):
        """Invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Invalid embedding source"):
            parse_embedding_config({"source": "invalid"}, name="test")

    def test_missing_name_raises(self):
        """Missing name raises ValueError."""
        with pytest.raises(ValueError, match="requires a name"):
            parse_embedding_config({"source": "civitai", "id": 123})


class TestParseEmbeddingsFromConfig:
    """Tests for parse_embeddings_from_config function."""

    def test_auto_load_embeddings(self):
        """Parses auto_load embeddings from global section."""
        full_config = {
            "embeddings": {
                "auto_load": ["easynegative"],
                "easynegative": {
                    "source": "civitai",
                    "id": 7808,
                    "token": "easynegative",
                },
            }
        }
        model_config = {"type": "flux2"}

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 1
        assert embeddings[0].name == "easynegative"
        assert embeddings[0].civitai_id == 7808

    def test_named_reference_embeddings(self):
        """Parses named reference embeddings from model config."""
        full_config = {
            "embeddings": {
                "bad-hands": {
                    "source": "civitai",
                    "id": 116230,
                    "token": "bad-hands-5",
                },
            }
        }
        model_config = {
            "type": "flux2",
            "embeddings": ["bad-hands"],
        }

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 1
        assert embeddings[0].name == "bad-hands"
        assert embeddings[0].civitai_id == 116230

    def test_inline_embeddings_in_array(self):
        """Parses inline embeddings defined in embeddings array."""
        full_config = {"embeddings": {}}
        model_config = {
            "type": "flux2",
            "embeddings": [
                {
                    "name": "inline-style",
                    "source": "civitai",
                    "id": 99999,
                    "token": "inline-token",
                }
            ],
        }

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 1
        assert embeddings[0].name == "inline-style"
        assert embeddings[0].civitai_id == 99999

    def test_inline_embeddings_section(self):
        """Parses inline_embeddings section in model config."""
        full_config = {"embeddings": {}}
        model_config = {
            "type": "flux2",
            "inline_embeddings": [
                {
                    "name": "model-specific",
                    "source": "local",
                    "path": "/path/to/embedding.safetensors",
                    "token": "custom",
                }
            ],
        }

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 1
        assert embeddings[0].name == "model-specific"
        assert embeddings[0].source == EmbeddingSource.LOCAL

    def test_combined_sources(self):
        """Parses embeddings from all sources combined."""
        full_config = {
            "embeddings": {
                "auto_load": ["easynegative"],
                "easynegative": {
                    "source": "civitai",
                    "id": 7808,
                    "token": "easynegative",
                },
                "bad-hands": {
                    "source": "civitai",
                    "id": 116230,
                },
            }
        }
        model_config = {
            "type": "flux2",
            "embeddings": ["bad-hands"],
            "inline_embeddings": [
                {
                    "name": "custom",
                    "source": "local",
                    "path": "/path/to/file.safetensors",
                }
            ],
        }

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 3
        names = [e.name for e in embeddings]
        assert "easynegative" in names
        assert "bad-hands" in names
        assert "custom" in names

    def test_no_duplicate_embeddings(self):
        """Same embedding is not loaded twice."""
        full_config = {
            "embeddings": {
                "auto_load": ["shared"],
                "shared": {
                    "source": "civitai",
                    "id": 12345,
                },
            }
        }
        model_config = {
            "type": "flux2",
            "embeddings": ["shared"],  # Also referenced here
        }

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 1
        assert embeddings[0].name == "shared"

    def test_missing_named_reference_warns(self, capsys):
        """Missing named reference prints warning."""
        full_config = {"embeddings": {}}
        model_config = {
            "type": "flux2",
            "embeddings": ["nonexistent"],
        }

        embeddings = parse_embeddings_from_config(full_config, model_config)

        assert len(embeddings) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_no_embeddings(self):
        """Returns empty list when no embeddings configured."""
        embeddings = parse_embeddings_from_config({}, {"type": "flux2"})
        assert embeddings == []


class TestIsEmbeddingCompatible:
    """Tests for is_embedding_compatible function."""

    def test_flux_compatible(self):
        """Flux.1 embeddings compatible with flux2 pipeline."""
        assert is_embedding_compatible("flux2", "Flux.1 Dev")
        assert is_embedding_compatible("flux2", "Flux.1 Schnell")
        assert is_embedding_compatible("flux2", "Flux.1 D")

    def test_sdxl_compatible(self):
        """SDXL embeddings compatible with sdxl pipeline."""
        assert is_embedding_compatible("sdxl", "SDXL 1.0")
        assert is_embedding_compatible("sdxl", "Pony")
        assert is_embedding_compatible("sdxl", "Illustrious")

    def test_incompatible_base_model(self):
        """Incompatible base model returns False."""
        assert not is_embedding_compatible("flux2", "SDXL 1.0")
        assert not is_embedding_compatible("sdxl", "Flux.1 Dev")
        assert not is_embedding_compatible("flux2", "SD 1.5")

    def test_none_base_model_is_compatible(self):
        """None base model assumed compatible."""
        assert is_embedding_compatible("flux2", None)

    def test_unknown_pipeline_is_compatible(self):
        """Unknown pipeline type assumed compatible."""
        assert is_embedding_compatible("unknown", "SDXL 1.0")

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert is_embedding_compatible("flux2", "flux.1 dev")
        assert is_embedding_compatible("flux2", "FLUX.1 DEV")


class TestEmbeddingIncompatibleError:
    """Tests for EmbeddingIncompatibleError exception."""

    def test_error_message(self):
        """Error message contains relevant info."""
        err = EmbeddingIncompatibleError("my_embedding", "flux2", "SDXL 1.0")
        assert "my_embedding" in str(err)
        assert "flux2" in str(err)
        assert "SDXL 1.0" in str(err)

    def test_error_attributes(self):
        """Error has correct attributes."""
        err = EmbeddingIncompatibleError("my_embedding", "flux2", "SDXL 1.0")
        assert err.embedding_name == "my_embedding"
        assert err.pipeline_type == "flux2"
        assert err.base_model == "SDXL 1.0"


class TestPipelineBaseModelMap:
    """Tests for PIPELINE_BASE_MODEL_MAP constant."""

    def test_all_pipeline_types_have_mappings(self):
        """All common pipeline types have base model mappings."""
        expected_types = ["flux2", "zimage", "qwen", "sdxl", "sd15"]
        for pipeline_type in expected_types:
            assert pipeline_type in PIPELINE_BASE_MODEL_MAP
            assert len(PIPELINE_BASE_MODEL_MAP[pipeline_type]) > 0


@pytest.mark.asyncio
class TestResolveEmbeddingPath:
    """Tests for resolve_embedding_path function."""

    async def test_local_path_exists(self, tmp_path):
        """Local path resolves when file exists."""
        from oneiro.pipelines.embedding import resolve_embedding_path

        emb_file = tmp_path / "test.safetensors"
        emb_file.write_bytes(b"test")

        config = EmbeddingConfig(
            name="test",
            source=EmbeddingSource.LOCAL,
            path=str(emb_file),
        )
        result = await resolve_embedding_path(config)

        assert result == emb_file
        assert config._resolved_path == emb_file

    async def test_local_path_not_exists(self, tmp_path):
        """Local path raises when file doesn't exist."""
        from oneiro.pipelines.embedding import resolve_embedding_path

        config = EmbeddingConfig(
            name="test",
            source=EmbeddingSource.LOCAL,
            path=str(tmp_path / "nonexistent.safetensors"),
        )

        with pytest.raises(FileNotFoundError):
            await resolve_embedding_path(config)

    async def test_huggingface_returns_repo_path(self):
        """HuggingFace source returns repo as path."""
        from oneiro.pipelines.embedding import resolve_embedding_path

        config = EmbeddingConfig(
            name="test",
            source=EmbeddingSource.HUGGINGFACE,
            repo="sd-concepts-library/cat-toy",
        )
        result = await resolve_embedding_path(config)

        assert result == Path("sd-concepts-library/cat-toy")
        assert config._resolved_path is None

    async def test_civitai_requires_client(self):
        """Civitai source requires CivitaiClient."""
        from oneiro.pipelines.embedding import resolve_embedding_path

        config = EmbeddingConfig(
            name="test",
            source=EmbeddingSource.CIVITAI,
            civitai_id=12345,
        )

        with pytest.raises(ValueError, match="CivitaiClient required"):
            await resolve_embedding_path(config)

    async def test_civitai_validates_compatibility(self):
        """Civitai source validates base model compatibility."""
        from oneiro.pipelines.embedding import resolve_embedding_path

        mock_client = AsyncMock()
        mock_model = Mock()
        mock_version = Mock()
        mock_version.base_model = "SDXL 1.0"
        mock_model.latest_version = mock_version
        mock_client.get_model = AsyncMock(return_value=mock_model)

        config = EmbeddingConfig(
            name="test",
            source=EmbeddingSource.CIVITAI,
            civitai_id=12345,
        )

        with pytest.raises(EmbeddingIncompatibleError):
            await resolve_embedding_path(
                config,
                civitai_client=mock_client,
                pipeline_type="flux2",
                validate_compatibility=True,
            )


class TestEmbeddingLoaderMixin:
    """Tests for EmbeddingLoaderMixin unload functionality."""

    def test_unload_single_embedding_success(self):
        """Unloading a loaded embedding removes it from tracking."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["token1", "token2"]
        pipeline._embedding_configs = [
            EmbeddingConfig(name="emb1", source=EmbeddingSource.LOCAL, path="/p", token="token1"),
            EmbeddingConfig(name="emb2", source=EmbeddingSource.LOCAL, path="/p", token="token2"),
        ]

        pipeline.unload_single_embedding("token1")

        assert "token1" not in pipeline._loaded_tokens
        assert "token2" in pipeline._loaded_tokens
        assert len(pipeline._embedding_configs) == 1
        pipeline.pipe.unload_textual_inversion.assert_called_once_with("token1")

    def test_unload_single_embedding_not_found_raises(self):
        """Unloading non-existent token raises ValueError."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["token1"]

        with pytest.raises(ValueError, match="not found"):
            pipeline.unload_single_embedding("nonexistent")

    def test_unload_single_embedding_no_pipeline_raises(self):
        """Unloading without pipeline raises RuntimeError."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = None
                self._init_embedding_state()

        pipeline = MockPipeline()

        with pytest.raises(RuntimeError, match="not loaded"):
            pipeline.unload_single_embedding("token1")

    def test_unload_all_embeddings_success(self):
        """Unloading all embeddings clears tracking lists."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["token1", "token2"]
        pipeline._embedding_configs = [
            EmbeddingConfig(name="emb1", source=EmbeddingSource.LOCAL, path="/p", token="token1"),
            EmbeddingConfig(name="emb2", source=EmbeddingSource.LOCAL, path="/p", token="token2"),
        ]

        pipeline.unload_all_embeddings()

        assert pipeline._loaded_tokens == []
        assert pipeline._embedding_configs == []
        pipeline.pipe.unload_textual_inversion.assert_called_once_with()

    def test_unload_all_embeddings_empty_is_noop(self):
        """Unloading when no embeddings loaded is a no-op."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self._init_embedding_state()

        pipeline = MockPipeline()

        pipeline.unload_all_embeddings()

        pipeline.pipe.unload_textual_inversion.assert_not_called()

    def test_unload_all_embeddings_no_pipeline_is_noop(self):
        """Unloading all without pipeline is a no-op."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = None
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["token1"]

        pipeline.unload_all_embeddings()

        assert pipeline._loaded_tokens == ["token1"]

    def test_unload_single_embedding_handles_exception(self, capsys):
        """Unloading handles exceptions gracefully."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self.pipe.unload_textual_inversion.side_effect = RuntimeError("API error")
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["token1"]
        pipeline._embedding_configs = [
            EmbeddingConfig(name="emb1", source=EmbeddingSource.LOCAL, path="/p", token="token1"),
        ]

        pipeline.unload_single_embedding("token1")

        assert "token1" not in pipeline._loaded_tokens
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "API error" in captured.out

    def test_unload_all_embeddings_handles_exception(self, capsys):
        """Unloading all handles exceptions gracefully."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self.pipe.unload_textual_inversion.side_effect = RuntimeError("API error")
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["token1"]
        pipeline._embedding_configs = [
            EmbeddingConfig(name="emb1", source=EmbeddingSource.LOCAL, path="/p", token="token1"),
        ]

        pipeline.unload_all_embeddings()

        assert pipeline._loaded_tokens == []
        assert pipeline._embedding_configs == []
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_unload_embedding_by_name_when_no_token(self):
        """Unloading works when embedding uses name as token."""
        from oneiro.pipelines.embedding import EmbeddingLoaderMixin

        class MockPipeline(EmbeddingLoaderMixin):
            def __init__(self):
                self.pipe = Mock()
                self._init_embedding_state()

        pipeline = MockPipeline()
        pipeline._loaded_tokens = ["my-embedding"]
        pipeline._embedding_configs = [
            EmbeddingConfig(
                name="my-embedding", source=EmbeddingSource.LOCAL, path="/p", token=None
            ),
        ]

        pipeline.unload_single_embedding("my-embedding")

        assert pipeline._loaded_tokens == []
        assert pipeline._embedding_configs == []
