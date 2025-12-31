"""Tests for LoRA auto-detection functionality."""

import pytest

from oneiro.lora_detector import (
    AutoLoraDetector,
    AutoLoraDetectorConfig,
    LoraMatch,
    TriggerIndex,
    create_detector_from_config,
)
from oneiro.pipelines.lora import LoraConfig, LoraSource


class TestAutoLoraDetectorConfig:
    """Tests for AutoLoraDetectorConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = AutoLoraDetectorConfig()
        assert config.enabled is True
        assert config.max_per_request == 4
        assert config.min_trigger_len == 3

    def test_custom_values(self):
        """Config accepts custom values."""
        config = AutoLoraDetectorConfig(
            enabled=False,
            max_per_request=2,
            min_trigger_len=5,
        )
        assert config.enabled is False
        assert config.max_per_request == 2
        assert config.min_trigger_len == 5


class TestLoraMatch:
    """Tests for LoraMatch dataclass."""

    def test_basic_match(self):
        """LoraMatch stores lora and trigger."""
        lora = LoraConfig(
            name="test-lora",
            source=LoraSource.LOCAL,
            path="/path/to/lora.safetensors",
            trigger_words=["shinkai"],
            base_model="SDXL 1.0",
        )
        match = LoraMatch(lora=lora, matched_trigger="shinkai")
        assert match.lora.name == "test-lora"
        assert match.matched_trigger == "shinkai"


class TestAutoLoraDetector:
    """Tests for AutoLoraDetector class."""

    @pytest.fixture
    def sdxl_lora(self):
        """Create an SDXL LoRA for testing."""
        return LoraConfig(
            name="shinkai-style",
            source=LoraSource.LOCAL,
            path="/loras/shinkai.safetensors",
            trigger_words=["shinkai style", "makoto shinkai"],
            base_model="SDXL 1.0",
            weight=0.8,
        )

    @pytest.fixture
    def pony_lora(self):
        """Create a Pony LoRA for testing."""
        return LoraConfig(
            name="detail-xl",
            source=LoraSource.LOCAL,
            path="/loras/detail.safetensors",
            trigger_words=["detailed", "high detail"],
            base_model="Pony",
            weight=0.5,
        )

    @pytest.fixture
    def flux_lora(self):
        """Create a Flux LoRA for testing."""
        return LoraConfig(
            name="realism-flux",
            source=LoraSource.LOCAL,
            path="/loras/realism.safetensors",
            trigger_words=["realistic", "photorealistic"],
            base_model="Flux.1",
            weight=1.0,
        )

    @pytest.fixture
    def detector(self, sdxl_lora, pony_lora, flux_lora):
        """Create detector with test LoRAs."""
        detector = AutoLoraDetector()
        detector.register_loras([sdxl_lora, pony_lora, flux_lora])
        return detector

    def test_match_single_trigger(self, detector):
        """Matches single trigger word in prompt."""
        matches = detector.match("a portrait in shinkai style", "sdxl")
        assert len(matches) == 1
        assert matches[0].lora.name == "shinkai-style"
        assert matches[0].matched_trigger == "shinkai style"

    def test_match_multiple_triggers_same_lora(self, detector):
        """Only matches once per LoRA even with multiple triggers."""
        matches = detector.match("shinkai style and makoto shinkai art", "sdxl")
        assert len(matches) == 1
        assert matches[0].lora.name == "shinkai-style"

    def test_match_multiple_loras(self, detector):
        """Matches multiple different LoRAs."""
        matches = detector.match("detailed portrait in shinkai style", "sdxl")
        assert len(matches) == 2
        names = {m.lora.name for m in matches}
        assert "shinkai-style" in names
        assert "detail-xl" in names

    def test_no_match_incompatible_base_model(self, detector):
        """Does not match LoRAs incompatible with pipeline type."""
        matches = detector.match("realistic portrait", "sdxl")
        assert len(matches) == 0

    def test_match_flux_pipeline(self, detector):
        """Matches Flux.1 LoRAs for flux1 pipeline."""
        matches = detector.match("photorealistic portrait", "flux1")
        assert len(matches) == 1
        assert matches[0].lora.name == "realism-flux"

    def test_case_insensitive_matching(self, detector):
        """Matching is case insensitive."""
        matches = detector.match("A portrait in SHINKAI STYLE", "sdxl")
        assert len(matches) == 1
        assert matches[0].lora.name == "shinkai-style"

    def test_word_boundary_matching(self, detector):
        """Does not match partial words."""
        matches = detector.match("a shinkaiX artwork", "sdxl")
        assert len(matches) == 0

    def test_hyphenated_trigger_words(self):
        """Matches trigger words containing hyphens."""
        detector = AutoLoraDetector()

        lora = LoraConfig(
            name="x-ray-lora",
            source=LoraSource.LOCAL,
            path="/loras/xray.safetensors",
            trigger_words=["x-ray", "x-ray style"],
            base_model="SDXL 1.0",
        )
        detector.register_loras([lora])

        matches = detector.match("an x-ray image of a hand", "sdxl")
        assert len(matches) == 1
        assert matches[0].matched_trigger == "x-ray"

    def test_hyphenated_trigger_no_partial_match(self):
        """Hyphenated triggers don't match partial words."""
        detector = AutoLoraDetector()

        lora = LoraConfig(
            name="x-ray-lora",
            source=LoraSource.LOCAL,
            path="/loras/xray.safetensors",
            trigger_words=["x-ray"],
            base_model="SDXL 1.0",
        )
        detector.register_loras([lora])

        matches = detector.match("a pixray image", "sdxl")
        assert len(matches) == 0

    def test_special_char_trigger_words(self):
        """Matches trigger words with special characters like underscores."""
        detector = AutoLoraDetector()

        lora = LoraConfig(
            name="special-lora",
            source=LoraSource.LOCAL,
            path="/loras/special.safetensors",
            trigger_words=["art_style", "foo.bar"],
            base_model="SDXL 1.0",
        )
        detector.register_loras([lora])

        matches = detector.match("generate an image in art_style", "sdxl")
        assert len(matches) == 1
        assert matches[0].matched_trigger == "art_style"

        matches = detector.match("generate foo.bar image", "sdxl")
        assert len(matches) == 1
        assert matches[0].matched_trigger == "foo.bar"

    def test_max_per_request_limit(self):
        """Respects max_per_request limit."""
        config = AutoLoraDetectorConfig(max_per_request=1)
        detector = AutoLoraDetector(config)

        loras = [
            LoraConfig(
                name=f"lora-{i}",
                source=LoraSource.LOCAL,
                path=f"/lora{i}.safetensors",
                trigger_words=[f"trigger{i}"],
                base_model="SDXL 1.0",
            )
            for i in range(5)
        ]
        detector.register_loras(loras)

        matches = detector.match("trigger0 trigger1 trigger2", "sdxl")
        assert len(matches) == 1

    def test_min_trigger_len_filter(self):
        """Filters out triggers shorter than min_trigger_len."""
        config = AutoLoraDetectorConfig(min_trigger_len=5)
        detector = AutoLoraDetector(config)

        lora = LoraConfig(
            name="test-lora",
            source=LoraSource.LOCAL,
            path="/lora.safetensors",
            trigger_words=["ab", "abc", "abcde", "abcdef"],
            base_model="SDXL 1.0",
        )
        detector.register_loras([lora])

        matches = detector.match("ab abc abcde abcdef", "sdxl")
        assert len(matches) == 1
        assert matches[0].matched_trigger in ("abcde", "abcdef")

    def test_disabled_detector_returns_empty(self):
        """Disabled detector returns no matches."""
        config = AutoLoraDetectorConfig(enabled=False)
        detector = AutoLoraDetector(config)

        lora = LoraConfig(
            name="test-lora",
            source=LoraSource.LOCAL,
            path="/lora.safetensors",
            trigger_words=["trigger"],
            base_model="SDXL 1.0",
        )
        detector.register_loras([lora])

        matches = detector.match("prompt with trigger", "sdxl")
        assert len(matches) == 0

    def test_auto_detect_false_excludes_lora(self):
        """LoRA with auto_detect=False is excluded."""
        detector = AutoLoraDetector()

        lora = LoraConfig(
            name="no-auto-lora",
            source=LoraSource.LOCAL,
            path="/lora.safetensors",
            trigger_words=["trigger"],
            base_model="SDXL 1.0",
            auto_detect=False,
        )
        detector.register_loras([lora])

        matches = detector.match("prompt with trigger", "sdxl")
        assert len(matches) == 0

    def test_auto_detect_true_includes_lora(self):
        """LoRA with auto_detect=True is included."""
        detector = AutoLoraDetector()

        lora = LoraConfig(
            name="yes-auto-lora",
            source=LoraSource.LOCAL,
            path="/lora.safetensors",
            trigger_words=["trigger"],
            base_model="SDXL 1.0",
            auto_detect=True,
        )
        detector.register_loras([lora])

        matches = detector.match("prompt with trigger", "sdxl")
        assert len(matches) == 1

    def test_auto_detect_none_uses_trigger_words(self):
        """LoRA with auto_detect=None uses trigger_words presence."""
        detector = AutoLoraDetector()

        lora_with_triggers = LoraConfig(
            name="has-triggers",
            source=LoraSource.LOCAL,
            path="/lora1.safetensors",
            trigger_words=["trigger"],
            base_model="SDXL 1.0",
        )
        lora_without_triggers = LoraConfig(
            name="no-triggers",
            source=LoraSource.LOCAL,
            path="/lora2.safetensors",
            trigger_words=[],
            base_model="SDXL 1.0",
        )
        detector.register_loras([lora_with_triggers, lora_without_triggers])

        matches = detector.match("prompt with trigger", "sdxl")
        assert len(matches) == 1
        assert matches[0].lora.name == "has-triggers"

    def test_no_base_model_excluded(self):
        """LoRA without base_model is excluded from auto-detect."""
        detector = AutoLoraDetector()

        lora = LoraConfig(
            name="no-base",
            source=LoraSource.LOCAL,
            path="/lora.safetensors",
            trigger_words=["trigger"],
            base_model=None,
        )
        detector.register_loras([lora])

        matches = detector.match("prompt with trigger", "sdxl")
        assert len(matches) == 0

    def test_invalidate_cache(self, detector):
        """Invalidating cache clears indexes."""
        _ = detector.match("shinkai style", "sdxl")
        assert "sdxl" in detector._indexes

        detector.invalidate_cache()
        assert len(detector._indexes) == 0

    def test_empty_prompt_returns_empty(self, detector):
        """Empty prompt returns no matches."""
        matches = detector.match("", "sdxl")
        assert len(matches) == 0

    def test_unknown_pipeline_type(self, detector):
        """Unknown pipeline type still matches (assumes compatible)."""
        detector._all_loras[0].base_model = "Unknown Model"
        matches = detector.match("shinkai style", "unknown_pipeline")
        # Unknown pipeline = empty compatible_bases = permissive matching
        assert len(matches) == 1
        assert matches[0].matched_trigger == "shinkai style"


class TestCreateDetectorFromConfig:
    """Tests for create_detector_from_config function."""

    def test_creates_detector_from_config(self):
        """Creates detector from config dict."""
        config = {
            "loras": {
                "auto_detect_enabled": True,
                "auto_detect_max_per_request": 3,
                "auto_detect_min_trigger_len": 4,
                "shinkai-style": {
                    "source": "local",
                    "path": "/loras/shinkai.safetensors",
                    "trigger_words": ["shinkai style"],
                    "base_model": "SDXL 1.0",
                    "weight": 0.8,
                },
            }
        }

        detector = create_detector_from_config(config)

        assert detector.config.enabled is True
        assert detector.config.max_per_request == 3
        assert detector.config.min_trigger_len == 4
        assert len(detector._all_loras) == 1
        assert detector._all_loras[0].name == "shinkai-style"

    def test_skips_auto_load_key(self):
        """Skips 'auto_load' key in loras section."""
        config = {
            "loras": {
                "auto_load": ["some-lora"],
                "actual-lora": {
                    "source": "local",
                    "path": "/loras/test.safetensors",
                    "trigger_words": ["test"],
                    "base_model": "SDXL 1.0",
                },
            }
        }

        detector = create_detector_from_config(config)
        assert len(detector._all_loras) == 1
        assert detector._all_loras[0].name == "actual-lora"

    def test_empty_config(self):
        """Handles empty config."""
        detector = create_detector_from_config({})
        assert len(detector._all_loras) == 0

    def test_default_config_values(self):
        """Uses default values when not specified."""
        config = {"loras": {}}
        detector = create_detector_from_config(config)

        assert detector.config.enabled is True
        assert detector.config.max_per_request == 4
        assert detector.config.min_trigger_len == 3


class TestTriggerIndex:
    """Tests for TriggerIndex dataclass."""

    def test_default_values(self):
        """TriggerIndex has correct defaults."""
        index = TriggerIndex()
        assert index.pattern is None
        assert index.trigger_to_loras == {}


class TestPipelineBaseModelMap:
    """Tests for base model compatibility mapping."""

    def test_sdxl_compatible_with_pony(self):
        """SDXL pipeline is compatible with Pony LoRAs."""
        from oneiro.lora_detector import PIPELINE_BASE_MODEL_MAP

        assert "Pony" in PIPELINE_BASE_MODEL_MAP["sdxl"]

    def test_flux_compatible_variants(self):
        """Flux1 pipeline is compatible with various Flux.1 variants."""
        from oneiro.lora_detector import PIPELINE_BASE_MODEL_MAP

        flux_bases = PIPELINE_BASE_MODEL_MAP["flux1"]
        assert "Flux.1" in flux_bases
        assert "Flux.1 Dev" in flux_bases
