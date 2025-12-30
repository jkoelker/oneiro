"""Tests for bot.py helper functions."""

import pytest

from oneiro.bot import (
    MAX_LORA_WEIGHT,
    MIN_LORA_WEIGHT,
    parse_lora_param,
    slugify,
    validate_lora_weight,
)


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_text(self):
        """Slugify basic text."""
        assert slugify("My Cool Model") == "my-cool-model"

    def test_special_characters_removed(self):
        """Special characters are removed."""
        assert slugify("Model (v1.2)") == "model-v12"

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces become single hyphen."""
        assert slugify("model   with   spaces") == "model-with-spaces"

    def test_underscores_removed(self):
        """Underscores are removed (filtered out by special char regex)."""
        # Underscores are removed by the [^a-z0-9\s-] regex before space handling
        assert slugify("model_name") == "modelname"

    def test_empty_string_returns_unnamed(self):
        """Empty string returns 'unnamed'."""
        assert slugify("") == "unnamed"

    def test_only_special_chars_returns_unnamed(self):
        """String with only special chars returns 'unnamed'."""
        assert slugify("!!!@@@") == "unnamed"


class TestValidateLoraWeight:
    """Tests for validate_lora_weight function."""

    def test_valid_weight_zero(self):
        """Weight 0.0 is valid."""
        validate_lora_weight(0.0, "test-lora")  # Should not raise

    def test_valid_weight_one(self):
        """Weight 1.0 is valid."""
        validate_lora_weight(1.0, "test-lora")  # Should not raise

    def test_valid_weight_max(self):
        """Weight at MAX_LORA_WEIGHT is valid."""
        validate_lora_weight(MAX_LORA_WEIGHT, "test-lora")  # Should not raise

    def test_valid_weight_min(self):
        """Weight at MIN_LORA_WEIGHT is valid."""
        validate_lora_weight(MIN_LORA_WEIGHT, "test-lora")  # Should not raise

    def test_valid_weight_negative(self):
        """Negative weight within range is valid."""
        validate_lora_weight(-1.5, "test-lora")  # Should not raise

    def test_invalid_weight_too_high(self):
        """Weight above MAX_LORA_WEIGHT raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_lora_weight(2.5, "test-lora")

    def test_invalid_weight_too_low(self):
        """Weight below MIN_LORA_WEIGHT raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_lora_weight(-3.0, "test-lora")

    def test_invalid_weight_extreme(self):
        """Extremely large weight raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            validate_lora_weight(100.0, "test-lora")

    def test_error_message_includes_lora_name(self):
        """Error message includes the LoRA name."""
        with pytest.raises(ValueError, match="my-custom-lora"):
            validate_lora_weight(5.0, "my-custom-lora")

    def test_error_message_includes_weight(self):
        """Error message includes the invalid weight value."""
        with pytest.raises(ValueError, match="5.0"):
            validate_lora_weight(5.0, "test-lora")


class TestParseLoraParam:
    """Tests for parse_lora_param function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        assert parse_lora_param("") == []

    def test_single_lora_name_only(self):
        """Single LoRA name without weight defaults to 1.0."""
        result = parse_lora_param("my-lora")
        assert result == [("my-lora", 1.0)]

    def test_single_lora_with_weight(self):
        """Single LoRA with weight parses correctly."""
        result = parse_lora_param("my-lora:0.8")
        assert result == [("my-lora", 0.8)]

    def test_civitai_id_only(self):
        """Civitai ID without weight defaults to 1.0."""
        result = parse_lora_param("civitai:12345")
        assert result == [("civitai:12345", 1.0)]

    def test_civitai_id_with_weight(self):
        """Civitai ID with weight parses correctly."""
        result = parse_lora_param("civitai:12345:0.7")
        assert result == [("civitai:12345", 0.7)]

    def test_multiple_loras(self):
        """Multiple comma-separated LoRAs parse correctly."""
        result = parse_lora_param("lora1:0.8,lora2:0.5")
        assert result == [("lora1", 0.8), ("lora2", 0.5)]

    def test_multiple_loras_mixed_formats(self):
        """Multiple LoRAs with mixed formats parse correctly."""
        result = parse_lora_param("my-lora,civitai:12345:0.7,another:0.5")
        assert result == [
            ("my-lora", 1.0),
            ("civitai:12345", 0.7),
            ("another", 0.5),
        ]

    def test_spaces_trimmed(self):
        """Spaces around entries are trimmed."""
        result = parse_lora_param("  lora1:0.8  ,  lora2:0.5  ")
        assert result == [("lora1", 0.8), ("lora2", 0.5)]

    def test_invalid_weight_falls_back_to_name(self):
        """Non-numeric weight treats whole thing as name."""
        result = parse_lora_param("lora-with-invalid:abc")
        assert result == [("lora-with-invalid:abc", 1.0)]

    def test_negative_weight_valid(self):
        """Negative weight within range is accepted."""
        result = parse_lora_param("my-lora:-1.5")
        assert result == [("my-lora", -1.5)]

    def test_weight_at_max_bound(self):
        """Weight at maximum bound is accepted."""
        result = parse_lora_param("my-lora:2.0")
        assert result == [("my-lora", 2.0)]

    def test_weight_at_min_bound(self):
        """Weight at minimum bound is accepted."""
        result = parse_lora_param("my-lora:-2.0")
        assert result == [("my-lora", -2.0)]

    def test_weight_exceeds_max_raises(self):
        """Weight exceeding maximum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("my-lora:2.5")

    def test_weight_below_min_raises(self):
        """Weight below minimum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("my-lora:-3.0")

    def test_civitai_weight_exceeds_max_raises(self):
        """Civitai LoRA weight exceeding maximum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("civitai:12345:5.0")

    def test_civitai_weight_below_min_raises(self):
        """Civitai LoRA weight below minimum raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("civitai:12345:-10.0")

    def test_error_message_includes_lora_name(self):
        """Error for invalid weight includes LoRA name."""
        with pytest.raises(ValueError, match="test-lora"):
            parse_lora_param("test-lora:10.0")

    def test_mixed_valid_and_invalid_raises_on_invalid(self):
        """Parsing stops with error when invalid weight encountered."""
        with pytest.raises(ValueError, match="out of range"):
            parse_lora_param("valid-lora:0.8,invalid-lora:100.0")

    def test_empty_parts_ignored(self):
        """Empty parts from multiple commas are ignored."""
        result = parse_lora_param("lora1,,lora2")
        assert result == [("lora1", 1.0), ("lora2", 1.0)]
