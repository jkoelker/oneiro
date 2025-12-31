"""LoRA auto-detection based on trigger words in prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oneiro.pipelines.lora import LoraConfig

PIPELINE_BASE_MODEL_MAP: dict[str, list[str]] = {
    # FLUX.1 and FLUX.2 use different transformer architectures and are NOT LoRA-compatible:
    # - FLUX.1: FluxTransformer2DModel (19 double-stream + 38 single-stream blocks, T5+CLIP encoders)
    # - FLUX.2: Flux2Transformer2DModel (8 double-stream + 48 single-stream blocks, Mistral encoder)
    "flux1": ["Flux.1 D", "Flux.1 S", "Flux.1", "Flux.1 Dev", "Flux.1 Schnell"],
    "flux2": ["Flux.2"],
    "zimage": ["ZImageTurbo", "ZImageBase", "Z-Image"],
    "qwen": ["Qwen", "Qwen-Image"],
    "sdxl": ["SDXL 1.0", "SDXL Turbo", "SDXL Lightning", "Pony", "Illustrious"],
    "sd15": ["SD 1.5", "SD 1.4"],
    "sd3": ["SD 3", "SD 3.5"],
    "civitai": ["SDXL 1.0", "SDXL Turbo", "Pony", "SD 1.5", "Illustrious"],
}


@dataclass
class LoraMatch:
    """Represents a LoRA that matched trigger words in a prompt."""

    lora: LoraConfig
    matched_trigger: str


@dataclass
class AutoLoraDetectorConfig:
    """Configuration for auto-detection behavior."""

    enabled: bool = True
    max_per_request: int = 4
    min_trigger_len: int = 3


@dataclass
class TriggerIndex:
    """Compiled index for fast trigger word matching."""

    pattern: re.Pattern[str] | None = None
    trigger_to_loras: dict[str, list[LoraConfig]] = field(default_factory=dict)


class AutoLoraDetector:
    """Detects which LoRAs to load based on trigger words in prompts.

    Uses a compiled regex index for fast matching. Filters LoRAs by base model
    compatibility with the current pipeline type.
    """

    def __init__(self, config: AutoLoraDetectorConfig | None = None):
        self.config = config or AutoLoraDetectorConfig()
        self._indexes: dict[str, TriggerIndex] = {}
        self._all_loras: list[LoraConfig] = []

    def register_loras(self, loras: list[LoraConfig]) -> None:
        """Register LoRAs for auto-detection. Rebuilds indexes."""
        self._all_loras = list(loras)
        self._indexes.clear()

    def _build_index_for_pipeline(self, pipeline_type: str) -> TriggerIndex:
        """Build a trigger word index filtered by pipeline compatibility."""
        trigger_to_loras: dict[str, list[LoraConfig]] = {}

        for lora in self._all_loras:
            if not self._should_auto_detect(lora):
                continue

            if not self._is_compatible(lora, pipeline_type):
                continue

            for trigger in lora.trigger_words:
                normalized = trigger.lower().strip()
                if len(normalized) < self.config.min_trigger_len:
                    continue

                if normalized not in trigger_to_loras:
                    trigger_to_loras[normalized] = []
                trigger_to_loras[normalized].append(lora)

        if not trigger_to_loras:
            return TriggerIndex(pattern=None, trigger_to_loras={})

        sorted_triggers = sorted(trigger_to_loras.keys(), key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_triggers]
        # Use word boundary that handles hyphens and special characters in triggers.
        # Matches at: start of string, whitespace, or common punctuation (not mid-word).
        pattern_str = r"(?<![A-Za-z0-9])(" + "|".join(escaped) + r")(?![A-Za-z0-9])"
        pattern = re.compile(pattern_str, re.IGNORECASE)

        return TriggerIndex(pattern=pattern, trigger_to_loras=trigger_to_loras)

    def _should_auto_detect(self, lora: LoraConfig) -> bool:
        """Check if a LoRA should be included in auto-detection."""
        if lora.auto_detect is False:
            return False
        if lora.auto_detect is True:
            return True
        return len(lora.trigger_words) > 0

    def _is_compatible(self, lora: LoraConfig, pipeline_type: str) -> bool:
        """Check if a LoRA is compatible with a pipeline type."""
        if not lora.base_model:
            return False

        compatible_bases = PIPELINE_BASE_MODEL_MAP.get(pipeline_type, [])
        if not compatible_bases:
            return True

        lora_base_lower = lora.base_model.lower()
        for base in compatible_bases:
            if base.lower() in lora_base_lower or lora_base_lower in base.lower():
                return True

        return False

    def _get_index(self, pipeline_type: str) -> TriggerIndex:
        """Get or build the trigger index for a pipeline type."""
        if pipeline_type not in self._indexes:
            self._indexes[pipeline_type] = self._build_index_for_pipeline(pipeline_type)
        return self._indexes[pipeline_type]

    def match(self, prompt: str, pipeline_type: str) -> list[LoraMatch]:
        """Find all LoRAs whose trigger words appear in the prompt.

        Args:
            prompt: User's generation prompt
            pipeline_type: Current pipeline type (flux2, sdxl, etc.)

        Returns:
            List of LoraMatch objects, capped at max_per_request
        """
        if not self.config.enabled:
            return []

        index = self._get_index(pipeline_type)
        if index.pattern is None:
            return []

        matches: list[LoraMatch] = []
        seen_loras: set[str] = set()

        for m in index.pattern.finditer(prompt):
            trigger = m.group(1).lower()
            loras_for_trigger = index.trigger_to_loras.get(trigger, [])

            for lora in loras_for_trigger:
                if lora.name in seen_loras:
                    continue

                seen_loras.add(lora.name)
                matches.append(LoraMatch(lora=lora, matched_trigger=trigger))

                if len(matches) >= self.config.max_per_request:
                    return matches

        return matches

    def invalidate_cache(self) -> None:
        """Clear all cached indexes (call after config changes)."""
        self._indexes.clear()


def create_detector_from_config(full_config: dict[str, Any]) -> AutoLoraDetector:
    """Create an AutoLoraDetector from the [loras] config section."""
    from oneiro.pipelines.lora import parse_lora_config

    loras_section = full_config.get("loras", {})

    detector_config = AutoLoraDetectorConfig(
        enabled=loras_section.get("auto_detect_enabled", True),
        max_per_request=loras_section.get("auto_detect_max_per_request", 4),
        min_trigger_len=loras_section.get("auto_detect_min_trigger_len", 3),
    )

    detector = AutoLoraDetector(detector_config)

    lora_configs: list[LoraConfig] = []
    for name, lora_def in loras_section.items():
        if name in (
            "auto_load",
            "auto_detect_enabled",
            "auto_detect_max_per_request",
            "auto_detect_min_trigger_len",
        ):
            continue

        if not isinstance(lora_def, dict):
            continue

        parsed = parse_lora_config(lora_def, index=len(lora_configs))
        if not lora_def.get("name"):
            object.__setattr__(parsed, "name", name)
            if lora_def.get("adapter_name") is None:
                object.__setattr__(parsed, "adapter_name", name)

        lora_configs.append(parsed)

    detector.register_loras(lora_configs)
    return detector
