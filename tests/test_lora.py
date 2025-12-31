"""Tests for LoRA configuration and loading functionality."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from oneiro.pipelines.lora import (
    PIPELINE_BASE_MODEL_MAP,
    LoraConfig,
    LoraIncompatibleError,
    LoraSource,
    is_lora_compatible,
    parse_civitai_url,
    parse_lora_config,
    parse_loras_from_config,
    parse_loras_from_model_config,
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


class TestLoraConfig:
    """Tests for LoraConfig dataclass."""

    def test_civitai_config_valid(self):
        """Valid Civitai config creates successfully."""
        config = LoraConfig(
            name="my-civitai-lora",
            source=LoraSource.CIVITAI,
            civitai_id=12345,
            weight=0.8,
        )
        assert config.name == "my-civitai-lora"
        assert config.source == LoraSource.CIVITAI
        assert config.civitai_id == 12345
        assert config.weight == 0.8

    def test_civitai_config_with_url(self):
        """Civitai config with URL is valid."""
        config = LoraConfig(
            name="url-lora",
            source=LoraSource.CIVITAI,
            civitai_url="https://civitai.com/models/12345",
        )
        assert config.civitai_url == "https://civitai.com/models/12345"

    def test_civitai_config_requires_id_or_url(self):
        """Civitai config without ID or URL raises."""
        with pytest.raises(ValueError, match="civitai_id or civitai_url"):
            LoraConfig(name="invalid", source=LoraSource.CIVITAI)

    def test_huggingface_config_valid(self):
        """Valid HuggingFace config creates successfully."""
        config = LoraConfig(
            name="my-hf-lora",
            source=LoraSource.HUGGINGFACE,
            repo="user/repo",
            weight_name="lora.safetensors",
        )
        assert config.source == LoraSource.HUGGINGFACE
        assert config.repo == "user/repo"

    def test_huggingface_config_requires_repo(self):
        """HuggingFace config without repo raises."""
        with pytest.raises(ValueError, match="repo"):
            LoraConfig(name="invalid", source=LoraSource.HUGGINGFACE)

    def test_local_config_valid(self):
        """Valid local config creates successfully."""
        config = LoraConfig(
            name="my-local-lora",
            source=LoraSource.LOCAL,
            path="/path/to/lora.safetensors",
        )
        assert config.source == LoraSource.LOCAL
        assert config.path == "/path/to/lora.safetensors"

    def test_local_config_requires_path(self):
        """Local config without path raises."""
        with pytest.raises(ValueError, match="path"):
            LoraConfig(name="invalid", source=LoraSource.LOCAL)

    def test_default_weight_is_one(self):
        """Default weight is 1.0."""
        config = LoraConfig(name="test", source=LoraSource.LOCAL, path="/path")
        assert config.weight == 1.0

    def test_adapter_name_defaults_to_name(self):
        """adapter_name defaults to name when not provided."""
        config = LoraConfig(name="my-lora", source=LoraSource.LOCAL, path="/path")
        assert config.adapter_name == "my-lora"

    def test_adapter_name_can_be_different_from_name(self):
        """adapter_name can be explicitly set different from name."""
        config = LoraConfig(
            name="my-lora",
            source=LoraSource.LOCAL,
            path="/path",
            adapter_name="custom_adapter",
        )
        assert config.name == "my-lora"
        assert config.adapter_name == "custom_adapter"

    def test_trigger_words_default_empty(self):
        """trigger_words defaults to empty list."""
        config = LoraConfig(name="test", source=LoraSource.LOCAL, path="/path")
        assert config.trigger_words == []

    def test_trigger_words_can_be_set(self):
        """trigger_words can be explicitly set."""
        config = LoraConfig(
            name="test",
            source=LoraSource.LOCAL,
            path="/path",
            trigger_words=["style", "subject"],
        )
        assert config.trigger_words == ["style", "subject"]

    def test_auto_trigger_default_true(self):
        """auto_trigger defaults to True."""
        config = LoraConfig(name="test", source=LoraSource.LOCAL, path="/path")
        assert config.auto_trigger is True

    def test_auto_trigger_can_be_disabled(self):
        """auto_trigger can be set to False."""
        config = LoraConfig(
            name="test",
            source=LoraSource.LOCAL,
            path="/path",
            auto_trigger=False,
        )
        assert config.auto_trigger is False


class TestParseLoraConfig:
    """Tests for parse_lora_config function."""

    def test_civitai_url_string(self):
        """Parses Civitai URL string."""
        config = parse_lora_config("https://civitai.com/models/12345")
        assert config.source == LoraSource.CIVITAI
        assert config.civitai_id == 12345
        assert config.name == "civitai_12345"
        assert config.adapter_name == "civitai_12345"

    def test_civitai_url_with_version(self):
        """Parses Civitai URL with version."""
        config = parse_lora_config("https://civitai.com/models/12345?modelVersionId=67890")
        assert config.civitai_id == 12345
        assert config.civitai_version == 67890

    def test_local_path_string(self):
        """Parses local path string."""
        config = parse_lora_config("/path/to/lora.safetensors")
        assert config.source == LoraSource.LOCAL
        assert config.path == "/path/to/lora.safetensors"
        assert config.name == "local_0"
        assert config.adapter_name == "local_0"

    def test_civitai_dict_with_id(self):
        """Parses Civitai dict with ID."""
        config = parse_lora_config(
            {
                "source": "civitai",
                "id": 12345,
                "version": 67890,
                "weight": 0.8,
            }
        )
        assert config.source == LoraSource.CIVITAI
        assert config.civitai_id == 12345
        assert config.civitai_version == 67890
        assert config.weight == 0.8

    def test_huggingface_dict(self):
        """Parses HuggingFace dict."""
        config = parse_lora_config(
            {
                "source": "huggingface",
                "repo": "user/repo",
                "weight_name": "lora.safetensors",
                "weight": 0.7,
            }
        )
        assert config.source == LoraSource.HUGGINGFACE
        assert config.repo == "user/repo"
        assert config.weight_name == "lora.safetensors"
        assert config.weight == 0.7

    def test_local_dict(self):
        """Parses local dict."""
        config = parse_lora_config(
            {
                "source": "local",
                "path": "/path/to/lora.safetensors",
            }
        )
        assert config.source == LoraSource.LOCAL
        assert config.path == "/path/to/lora.safetensors"

    def test_custom_adapter_name(self):
        """Custom adapter_name is preserved, name auto-generated."""
        config = parse_lora_config(
            {
                "source": "civitai",
                "id": 12345,
                "adapter_name": "my_style",
            }
        )
        assert config.name == "civitai_12345"
        assert config.adapter_name == "my_style"

    def test_custom_name_field(self):
        """Custom name field is used for both name and adapter_name."""
        config = parse_lora_config(
            {
                "source": "civitai",
                "id": 12345,
                "name": "my-lora",
            }
        )
        assert config.name == "my-lora"
        assert config.adapter_name == "my-lora"

    def test_custom_name_and_adapter_name(self):
        """Both name and adapter_name can be specified independently."""
        config = parse_lora_config(
            {
                "source": "civitai",
                "id": 12345,
                "name": "my-lora",
                "adapter_name": "custom_adapter",
            }
        )
        assert config.name == "my-lora"
        assert config.adapter_name == "custom_adapter"

    def test_invalid_source_raises(self):
        """Invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LoRA source"):
            parse_lora_config({"source": "invalid"})

    def test_trigger_words_from_config(self):
        """Parses trigger_words from dict config."""
        config = parse_lora_config(
            {
                "source": "civitai",
                "id": 12345,
                "trigger_words": ["style trigger", "subject"],
            }
        )
        assert config.trigger_words == ["style trigger", "subject"]

    def test_auto_trigger_from_config(self):
        """Parses auto_trigger from dict config."""
        config = parse_lora_config(
            {
                "source": "civitai",
                "id": 12345,
                "auto_trigger": False,
            }
        )
        assert config.auto_trigger is False

    def test_trigger_words_defaults_empty(self):
        """trigger_words defaults to empty list when not in config."""
        config = parse_lora_config({"source": "civitai", "id": 12345})
        assert config.trigger_words == []

    def test_auto_trigger_defaults_true(self):
        """auto_trigger defaults to True when not in config."""
        config = parse_lora_config({"source": "civitai", "id": 12345})
        assert config.auto_trigger is True


class TestParseLORAsFromModelConfig:
    """Tests for parse_loras_from_model_config function."""

    def test_loras_array(self):
        """Parses loras array format."""
        config = {
            "loras": [
                {"source": "civitai", "id": 12345, "weight": 0.8},
                {"source": "huggingface", "repo": "user/repo", "weight": 0.7},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 2
        assert loras[0].source == LoraSource.CIVITAI
        assert loras[1].source == LoraSource.HUGGINGFACE

    def test_civitai_lora_url(self):
        """Parses civitai_lora URL format."""
        config = {"civitai_lora": "https://civitai.com/models/12345"}
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1
        assert loras[0].source == LoraSource.CIVITAI
        assert loras[0].civitai_id == 12345

    def test_civitai_lora_id(self):
        """Parses civitai_lora_id format."""
        config = {
            "civitai_lora_id": 12345,
            "civitai_lora_version": 67890,
            "civitai_lora_weight": 0.9,
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1
        assert loras[0].civitai_id == 12345
        assert loras[0].civitai_version == 67890
        assert loras[0].weight == 0.9

    def test_legacy_lora_format(self):
        """Parses legacy lora/lora_weights format."""
        config = {
            "lora": "user/repo",
            "lora_weights": "lora.safetensors",
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1
        assert loras[0].source == LoraSource.HUGGINGFACE
        assert loras[0].repo == "user/repo"
        assert loras[0].weight_name == "lora.safetensors"

    def test_legacy_local_path(self):
        """Parses legacy lora as local path."""
        config = {"lora": "/path/to/lora.safetensors"}
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1
        assert loras[0].source == LoraSource.LOCAL

    def test_no_loras(self):
        """Returns empty list when no LoRAs configured."""
        loras = parse_loras_from_model_config({"type": "flux2"})
        assert loras == []

    def test_duplicate_civitai_lora_skipped(self):
        """Duplicate Civitai LoRAs by ID and version are skipped."""
        config = {
            "loras": [
                {"source": "civitai", "id": 12345, "version": 67890, "weight": 0.8},
                {"source": "civitai", "id": 12345, "version": 67890, "weight": 0.5},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1
        assert loras[0].civitai_id == 12345
        assert loras[0].weight == 0.8

    def test_same_civitai_id_different_version_not_duplicate(self):
        """Same Civitai ID with different versions are NOT duplicates."""
        config = {
            "loras": [
                {"source": "civitai", "id": 12345, "version": 67890},
                {"source": "civitai", "id": 12345, "version": 99999},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 2

    def test_duplicate_huggingface_lora_skipped(self):
        """Duplicate HuggingFace LoRAs by repo and weight_name are skipped."""
        config = {
            "loras": [
                {"source": "huggingface", "repo": "user/repo", "weight_name": "lora.safetensors"},
                {"source": "huggingface", "repo": "user/repo", "weight_name": "lora.safetensors"},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1

    def test_same_repo_different_weight_name_not_duplicate(self):
        """Same HF repo with different weight_name are NOT duplicates."""
        config = {
            "loras": [
                {"source": "huggingface", "repo": "user/repo", "weight_name": "lora1.safetensors"},
                {"source": "huggingface", "repo": "user/repo", "weight_name": "lora2.safetensors"},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 2

    def test_duplicate_local_lora_skipped(self):
        """Duplicate local LoRAs by path are skipped."""
        config = {
            "loras": [
                {"source": "local", "path": "/path/to/lora.safetensors"},
                {"source": "local", "path": "/path/to/lora.safetensors"},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 1

    def test_same_source_different_identifiers_not_duplicate(self):
        """Different identifiers within same source type are NOT duplicates."""
        config = {
            "loras": [
                {"source": "civitai", "id": 11111},
                {"source": "civitai", "id": 22222},
            ]
        }
        loras = parse_loras_from_model_config(config)
        assert len(loras) == 2


class TestParseLORAsFromConfig:
    """Tests for parse_loras_from_config function."""

    def test_auto_load_loras(self):
        """Parses auto_load LoRAs from global section."""
        full_config = {
            "loras": {
                "auto_load": ["realism-enhancer"],
                "realism-enhancer": {
                    "source": "civitai",
                    "id": 123456,
                    "weight": 0.5,
                },
            }
        }
        model_config = {"type": "flux2"}

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 1
        assert loras[0].adapter_name == "realism-enhancer"
        assert loras[0].civitai_id == 123456
        assert loras[0].weight == 0.5

    def test_named_reference_loras(self):
        """Parses named reference LoRAs from model config."""
        full_config = {
            "loras": {
                "detail-lora": {
                    "source": "huggingface",
                    "repo": "user/detail-lora",
                    "weight": 0.8,
                },
            }
        }
        model_config = {
            "type": "flux2",
            "loras": ["detail-lora"],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 1
        assert loras[0].adapter_name == "detail-lora"
        assert loras[0].repo == "user/detail-lora"
        assert loras[0].weight == 0.8

    def test_inline_loras_in_array(self):
        """Parses inline LoRAs defined in loras array."""
        full_config = {"loras": {}}
        model_config = {
            "type": "flux2",
            "loras": [
                {
                    "name": "inline-style",
                    "source": "civitai",
                    "id": 99999,
                    "weight": 0.7,
                }
            ],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 1
        assert loras[0].adapter_name == "inline-style"
        assert loras[0].civitai_id == 99999

    def test_inline_loras_section(self):
        """Parses inline_loras section in model config."""
        full_config = {"loras": {}}
        model_config = {
            "type": "flux2",
            "inline_loras": [
                {
                    "adapter_name": "model-specific",
                    "source": "local",
                    "path": "/path/to/lora.safetensors",
                    "weight": 0.9,
                }
            ],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 1
        assert loras[0].adapter_name == "model-specific"
        assert loras[0].source == LoraSource.LOCAL
        assert loras[0].weight == 0.9

    def test_combined_sources(self):
        """Parses LoRAs from all sources combined."""
        full_config = {
            "loras": {
                "auto_load": ["realism-enhancer"],
                "realism-enhancer": {
                    "source": "civitai",
                    "id": 123456,
                    "weight": 0.5,
                },
                "detail-lora": {
                    "source": "huggingface",
                    "repo": "user/detail",
                    "weight": 0.8,
                },
            }
        }
        model_config = {
            "type": "flux2",
            "loras": ["detail-lora"],
            "inline_loras": [
                {
                    "adapter_name": "custom",
                    "source": "local",
                    "path": "/path/to/file.safetensors",
                }
            ],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 3
        names = [lora.adapter_name for lora in loras]
        assert "realism-enhancer" in names
        assert "detail-lora" in names
        assert "custom" in names

    def test_no_duplicate_loras(self):
        """Same LoRA is not loaded twice."""
        full_config = {
            "loras": {
                "auto_load": ["shared"],
                "shared": {
                    "source": "civitai",
                    "id": 12345,
                },
            }
        }
        model_config = {
            "type": "flux2",
            "loras": ["shared"],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 1
        assert loras[0].adapter_name == "shared"

    def test_missing_named_reference_warns(self, capsys):
        """Missing named reference prints warning."""
        full_config = {"loras": {}}
        model_config = {
            "type": "flux2",
            "loras": ["nonexistent"],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_missing_auto_load_warns(self, capsys):
        """Missing auto_load LoRA prints warning."""
        full_config = {
            "loras": {
                "auto_load": ["nonexistent"],
            }
        }
        model_config = {"type": "flux2"}

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 0
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_no_loras(self):
        """Returns empty list when no LoRAs configured."""
        loras = parse_loras_from_config({}, {"type": "flux2"})
        assert loras == []

    def test_preserves_custom_adapter_name(self):
        """Custom adapter_name from config is preserved."""
        full_config = {
            "loras": {
                "my-lora": {
                    "source": "civitai",
                    "id": 12345,
                    "adapter_name": "custom_name",
                },
            }
        }
        model_config = {
            "type": "flux2",
            "loras": ["my-lora"],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 1
        assert loras[0].adapter_name == "custom_name"

    def test_mixed_string_and_dict_refs(self):
        """Handles mix of string references and inline dicts in loras array."""
        full_config = {
            "loras": {
                "named-lora": {
                    "source": "civitai",
                    "id": 11111,
                },
            }
        }
        model_config = {
            "type": "flux2",
            "loras": [
                "named-lora",
                {
                    "source": "civitai",
                    "id": 22222,
                    "name": "inline-lora",
                },
            ],
        }

        loras = parse_loras_from_config(full_config, model_config)

        assert len(loras) == 2
        assert loras[0].adapter_name == "named-lora"
        assert loras[0].civitai_id == 11111
        assert loras[1].adapter_name == "inline-lora"
        assert loras[1].civitai_id == 22222


class TestIsLoraCompatible:
    """Tests for is_lora_compatible function."""

    def test_flux_compatible(self):
        """Flux.1 LoRAs compatible with flux2 pipeline."""
        assert is_lora_compatible("flux2", "Flux.1 Dev")
        assert is_lora_compatible("flux2", "Flux.1 Schnell")
        assert is_lora_compatible("flux2", "Flux.1 D")

    def test_sdxl_compatible(self):
        """SDXL LoRAs compatible with sdxl pipeline."""
        assert is_lora_compatible("sdxl", "SDXL 1.0")
        assert is_lora_compatible("sdxl", "Pony")
        assert is_lora_compatible("sdxl", "Illustrious")

    def test_incompatible_base_model(self):
        """Incompatible base model returns False."""
        assert not is_lora_compatible("flux2", "SDXL 1.0")
        assert not is_lora_compatible("sdxl", "Flux.1 Dev")
        assert not is_lora_compatible("flux2", "SD 1.5")

    def test_none_base_model_is_compatible(self):
        """None base model assumed compatible."""
        assert is_lora_compatible("flux2", None)

    def test_unknown_pipeline_is_compatible(self):
        """Unknown pipeline type assumed compatible."""
        assert is_lora_compatible("unknown", "SDXL 1.0")

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert is_lora_compatible("flux2", "flux.1 dev")
        assert is_lora_compatible("flux2", "FLUX.1 DEV")


class TestLoraIncompatibleError:
    """Tests for LoraIncompatibleError exception."""

    def test_error_message(self):
        """Error message contains relevant info."""
        err = LoraIncompatibleError("my_lora", "flux2", "SDXL 1.0")
        assert "my_lora" in str(err)
        assert "flux2" in str(err)
        assert "SDXL 1.0" in str(err)

    def test_error_attributes(self):
        """Error has correct attributes."""
        err = LoraIncompatibleError("my_lora", "flux2", "SDXL 1.0")
        assert err.lora_name == "my_lora"
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
class TestResolveLoraPath:
    """Tests for resolve_lora_path function."""

    async def test_local_path_exists(self, tmp_path):
        """Local path resolves when file exists."""
        from oneiro.pipelines.lora import resolve_lora_path

        lora_file = tmp_path / "test.safetensors"
        lora_file.write_bytes(b"test")

        config = LoraConfig(name="test-lora", source=LoraSource.LOCAL, path=str(lora_file))
        result = await resolve_lora_path(config)

        assert result == lora_file
        assert config._resolved_path == lora_file

    async def test_local_path_not_exists(self, tmp_path):
        """Local path raises when file doesn't exist."""
        from oneiro.pipelines.lora import resolve_lora_path

        config = LoraConfig(
            name="nonexistent-lora",
            source=LoraSource.LOCAL,
            path=str(tmp_path / "nonexistent.safetensors"),
        )

        with pytest.raises(FileNotFoundError):
            await resolve_lora_path(config)

    async def test_huggingface_returns_repo_path(self):
        """HuggingFace source returns repo as path."""
        from oneiro.pipelines.lora import resolve_lora_path

        config = LoraConfig(name="hf-lora", source=LoraSource.HUGGINGFACE, repo="user/repo")
        result = await resolve_lora_path(config)

        assert result == Path("user/repo")
        assert config._resolved_path is None

    async def test_civitai_requires_client(self):
        """Civitai source requires CivitaiClient."""
        from oneiro.pipelines.lora import resolve_lora_path

        config = LoraConfig(name="civitai-lora", source=LoraSource.CIVITAI, civitai_id=12345)

        with pytest.raises(ValueError, match="CivitaiClient required"):
            await resolve_lora_path(config)

    async def test_civitai_validates_compatibility(self):
        """Civitai source validates base model compatibility."""
        from oneiro.pipelines.lora import resolve_lora_path

        mock_client = AsyncMock()
        mock_model = Mock()
        mock_version = Mock()
        mock_version.base_model = "SDXL 1.0"
        mock_model.latest_version = mock_version
        mock_client.get_model = AsyncMock(return_value=mock_model)

        config = LoraConfig(name="civitai-lora", source=LoraSource.CIVITAI, civitai_id=12345)

        with pytest.raises(LoraIncompatibleError):
            await resolve_lora_path(
                config,
                civitai_client=mock_client,
                pipeline_type="flux2",
                validate_compatibility=True,
            )
