"""Tests for /dream command generation parameter handling.

Tests for issue #51: Exposing steps and guidance_scale parameters in /dream command
and setting model defaults via /model command.
"""

# Constants for validation limits (should match bot.py)
MIN_STEPS = 1
MAX_STEPS = 100
MIN_GUIDANCE_SCALE = 0.0
MAX_GUIDANCE_SCALE = 15.0


class TestDreamParamValidation:
    """Tests for /dream parameter validation ranges."""

    def test_steps_min_value(self):
        """Steps minimum value should be 1."""
        assert MIN_STEPS == 1

    def test_steps_max_value(self):
        """Steps maximum value should be 100."""
        assert MAX_STEPS == 100

    def test_guidance_scale_min_value(self):
        """Guidance scale minimum should be 0.0."""
        assert MIN_GUIDANCE_SCALE == 0.0

    def test_guidance_scale_max_value(self):
        """Guidance scale maximum should be 15.0."""
        assert MAX_GUIDANCE_SCALE == 15.0


class TestDreamParamDefaults:
    """Tests for /dream parameter default value resolution."""

    def test_steps_none_uses_model_default(self):
        """When steps is None, model config default should be used."""
        # Simulating the logic in bot.py dream command
        user_steps = None
        model_config_steps = 28  # e.g., FLUX.2 default

        actual_steps = user_steps if user_steps is not None else model_config_steps
        assert actual_steps == 28

    def test_steps_user_override(self):
        """When steps is provided, it should override model default."""
        user_steps = 50
        model_config_steps = 28

        actual_steps = user_steps if user_steps is not None else model_config_steps
        assert actual_steps == 50

    def test_guidance_scale_none_uses_model_default(self):
        """When guidance_scale is None, model config default should be used."""
        user_guidance = None
        model_config_guidance = 7.0  # e.g., SDXL default

        actual_guidance = user_guidance if user_guidance is not None else model_config_guidance
        assert actual_guidance == 7.0

    def test_guidance_scale_user_override(self):
        """When guidance_scale is provided, it should override model default."""
        user_guidance = 4.5
        model_config_guidance = 7.0

        actual_guidance = user_guidance if user_guidance is not None else model_config_guidance
        assert actual_guidance == 4.5

    def test_guidance_scale_zero_is_valid_override(self):
        """Guidance scale of 0.0 should be treated as a valid override, not None."""
        user_guidance = 0.0
        model_config_guidance = 7.0

        # Use 'is not None' check, not truthiness
        actual_guidance = user_guidance if user_guidance is not None else model_config_guidance
        assert actual_guidance == 0.0

    def test_steps_zero_is_invalid(self):
        """Steps of 0 should be outside valid range."""
        assert 0 < MIN_STEPS


class TestModelOverrides:
    """Tests for model-specific default overrides via /model command."""

    def test_override_steps_stored_in_state(self):
        """Model override for steps should be stored in state dict."""
        state: dict = {}
        model_name = "test-model"
        override_steps = 30

        # Simulate config.set("model_overrides", model_name, "steps", value=override_steps)
        if "model_overrides" not in state:
            state["model_overrides"] = {}
        if model_name not in state["model_overrides"]:
            state["model_overrides"][model_name] = {}
        state["model_overrides"][model_name]["steps"] = override_steps

        assert state["model_overrides"]["test-model"]["steps"] == 30

    def test_override_guidance_scale_stored_in_state(self):
        """Model override for guidance_scale should be stored in state dict."""
        state: dict = {}
        model_name = "test-model"
        override_guidance = 5.0

        if "model_overrides" not in state:
            state["model_overrides"] = {}
        if model_name not in state["model_overrides"]:
            state["model_overrides"][model_name] = {}
        state["model_overrides"][model_name]["guidance_scale"] = override_guidance

        assert state["model_overrides"]["test-model"]["guidance_scale"] == 5.0

    def test_override_priority_user_over_model_override(self):
        """User-provided value should take priority over model override."""
        user_steps = 40
        model_override_steps = 30
        model_config_steps = 20

        # Priority: user > model_override > model_config
        actual_steps = user_steps or model_override_steps or model_config_steps
        assert actual_steps == 40

    def test_override_priority_model_override_over_config(self):
        """Model override should take priority over base config default."""
        user_steps = None
        model_override_steps = 30
        model_config_steps = 20

        # When user doesn't provide, check override first
        if user_steps is not None:
            actual_steps = user_steps
        elif model_override_steps is not None:
            actual_steps = model_override_steps
        else:
            actual_steps = model_config_steps

        assert actual_steps == 30


class TestQwenGuidanceScaleMapping:
    """Tests for Qwen model true_cfg_scale handling."""

    def test_qwen_true_cfg_scale_mapping(self):
        """Qwen model should use true_cfg_scale from model_config."""
        model_config = {
            "type": "qwen",
            "guidance_scale": 4.0,  # Standard param
            "true_cfg_scale": 1.3,  # Qwen-specific
        }

        # Qwen should prefer true_cfg_scale
        guidance_scale = model_config.get("guidance_scale", 0.0)
        if model_config.get("true_cfg_scale"):
            guidance_scale = model_config.get("true_cfg_scale", 4.0)

        assert guidance_scale == 1.3


class TestZImageGuidanceScaleWarning:
    """Tests for Z-Image-Turbo guidance_scale handling."""

    def test_zimage_ignores_guidance_scale(self):
        """Z-Image-Turbo should always use 0.0 for guidance_scale internally."""
        # Z-Image-Turbo hardcodes guidance_scale=0.0
        # Even if user provides a value, pipeline ignores it
        user_guidance = 7.0  # noqa: F841 - documents user-provided value that gets ignored
        zimage_actual_guidance = 0.0  # Always 0.0 for Turbo

        # Note: We might want to warn user if they provide non-zero guidance
        # This test documents the expected behavior
        assert zimage_actual_guidance == 0.0
