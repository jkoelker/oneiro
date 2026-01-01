"""Business logic for image generation - LoRA parsing and resolution."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oneiro.pipelines import LoraConfig, LoraSource
from oneiro.pipelines.lora import is_resource_compatible

if TYPE_CHECKING:
    from oneiro.civitai import CivitaiClient
    from oneiro.config import Config
    from oneiro.lora_detector import AutoLoraDetector


# LoRA weight validation limits
MIN_LORA_WEIGHT = -2.0
MAX_LORA_WEIGHT = 2.0

# Steps validation limits (for /dream and /model commands)
MIN_STEPS = 1
MAX_STEPS = 100

# Guidance scale validation limits (for /dream and /model commands)
MIN_GUIDANCE_SCALE = 0.0
MAX_GUIDANCE_SCALE = 15.0


def validate_lora_weight(weight: float, lora_name: str) -> None:
    """Validate that a LoRA weight is within acceptable bounds.

    Args:
        weight: The LoRA weight value to validate
        lora_name: Name/identifier of the LoRA (for error messages)

    Raises:
        ValueError: If weight is outside the valid range [-2.0, 2.0]
    """
    if weight < MIN_LORA_WEIGHT or weight > MAX_LORA_WEIGHT:
        raise ValueError(
            f"LoRA weight {weight} for '{lora_name}' is out of range. "
            f"Valid range is [{MIN_LORA_WEIGHT}, {MAX_LORA_WEIGHT}]."
        )


def parse_lora_param(lora_str: str) -> list[tuple[str, float]]:
    """Parse lora parameter string into list of (name/id, weight) tuples.

    Supports formats:
    - "lora-name" -> ("lora-name", 1.0)
    - "lora-name:0.8" -> ("lora-name", 0.8)
    - "civitai:12345" -> ("civitai:12345", 1.0)
    - "civitai:12345:0.7" -> ("civitai:12345", 0.7)
    - "lora1:0.8,lora2:0.5" -> [("lora1", 0.8), ("lora2", 0.5)]

    Args:
        lora_str: Comma-separated lora specifications

    Returns:
        List of (identifier, weight) tuples

    Raises:
        ValueError: If a weight is outside the valid range [-2.0, 2.0]
    """
    results: list[tuple[str, float]] = []
    if not lora_str:
        return results

    for part in lora_str.split(","):
        part = part.strip()
        if not part:
            continue

        # Check if it's a civitai: reference
        if part.startswith("civitai:"):
            # civitai:12345 or civitai:12345:0.8
            segments = part.split(":")
            if len(segments) == 2:
                # civitai:12345
                results.append((part, 1.0))
            elif len(segments) >= 3:
                # civitai:12345:0.8
                try:
                    weight = float(segments[2])
                except ValueError:
                    # Couldn't parse weight as float, treat whole thing as name
                    results.append((part, 1.0))
                else:
                    lora_name = f"civitai:{segments[1]}"
                    validate_lora_weight(weight, lora_name)
                    results.append((lora_name, weight))
        else:
            # Regular name or name:weight
            if ":" in part:
                name, weight_str = part.rsplit(":", 1)
                try:
                    weight = float(weight_str)
                except ValueError:
                    # Couldn't parse weight as float, treat whole thing as name
                    results.append((part, 1.0))
                else:
                    lora_name = name.strip()
                    validate_lora_weight(weight, lora_name)
                    results.append((lora_name, weight))
            else:
                results.append((part, 1.0))

    return results


@dataclass
class LoraResolutionResult:
    """Result of LoRA resolution.

    Attributes:
        configs: Resolved LoRA configurations ready for use
        warnings: Compatibility warnings (soft warnings, don't block generation)
        auto_detected: List of (lora_name, matched_trigger) for auto-detected LoRAs
    """

    configs: list[LoraConfig] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auto_detected: list[tuple[str, str]] = field(default_factory=list)


class LoraNotFoundError(Exception):
    """Raised when a named LoRA reference is not found in config."""

    def __init__(self, lora_name: str) -> None:
        self.lora_name = lora_name
        super().__init__(f"Unknown LoRA: `{lora_name}`")


async def resolve_loras(
    lora_param: str | None,
    prompt: str,
    config: "Config | None",
    civitai_client: "CivitaiClient | None",
    lora_detector: "AutoLoraDetector | None",
    pipeline_type: str | None,
) -> LoraResolutionResult:
    """Resolve LoRA specifications to configs with compatibility warnings.

    If lora_param is provided, parses and resolves explicit LoRA references.
    Otherwise, if lora_detector is available, auto-detects LoRAs from prompt.

    Args:
        lora_param: User-provided lora parameter (None for auto-detect)
        prompt: User prompt (for auto-detection matching)
        config: Config instance for looking up named LoRAs
        civitai_client: Client for fetching Civitai model info (for compatibility checks)
        lora_detector: Auto-detector for prompt-based LoRA matching
        pipeline_type: Current pipeline type for compatibility checks

    Returns:
        LoraResolutionResult with configs, warnings, and auto-detected info

    Raises:
        ValueError: If LoRA weight is out of range
        LoraNotFoundError: If named LoRA not found in config
    """
    # Import here to avoid circular import
    from oneiro.civitai import CivitaiError

    result = LoraResolutionResult()

    if lora_param and config:
        # Explicit lora parameter provided - skip auto-detection
        parsed_loras = parse_lora_param(lora_param)
        loras_section = config.get("loras", default={})

        for lora_ref, weight in parsed_loras:
            if lora_ref.startswith("civitai:"):
                # Direct Civitai reference - on-demand download
                civitai_id = int(lora_ref.split(":")[1])
                lora_config = LoraConfig(
                    name=f"civitai_{civitai_id}",
                    source=LoraSource.CIVITAI,
                    civitai_id=civitai_id,
                    weight=weight,
                )

                # Check compatibility (soft warning)
                if civitai_client and pipeline_type:
                    try:
                        model_info = await civitai_client.get_model(civitai_id)
                        version = model_info.latest_version
                        if version and not is_resource_compatible(
                            pipeline_type, version.base_model
                        ):
                            result.warnings.append(
                                f"⚠️ LoRA `{model_info.name}` (base: {version.base_model}) "
                                f"may not be compatible with current model ({pipeline_type})"
                            )
                    except CivitaiError as e:
                        result.warnings.append(f"⚠️ Could not verify LoRA {civitai_id}: {e}")

                result.configs.append(lora_config)
            else:
                # Named reference from config
                if lora_ref in loras_section and isinstance(loras_section[lora_ref], dict):
                    lora_def: dict[str, Any] = loras_section[lora_ref]
                    source_str = lora_def.get("source", "civitai")
                    source = LoraSource(source_str)

                    lora_config = LoraConfig(
                        name=lora_ref,
                        source=source,
                        weight=weight,
                        civitai_id=lora_def.get("id") or lora_def.get("civitai_id"),
                        civitai_version=lora_def.get("version") or lora_def.get("civitai_version"),
                        civitai_url=lora_def.get("url") or lora_def.get("civitai_url"),
                        repo=lora_def.get("repo"),
                        weight_name=lora_def.get("weight_name"),
                        path=lora_def.get("path"),
                    )

                    # Check compatibility for Civitai LoRAs
                    if source == LoraSource.CIVITAI and civitai_client and pipeline_type:
                        civitai_id = lora_config.civitai_id
                        if civitai_id:
                            try:
                                model_info = await civitai_client.get_model(civitai_id)
                                version = model_info.latest_version
                                if version and not is_resource_compatible(
                                    pipeline_type, version.base_model
                                ):
                                    result.warnings.append(
                                        f"⚠️ LoRA `{lora_ref}` (base: {version.base_model}) "
                                        f"may not be compatible with current model ({pipeline_type})"
                                    )
                            except CivitaiError as e:
                                result.warnings.append(f"⚠️ Could not verify LoRA `{lora_ref}`: {e}")

                    result.configs.append(lora_config)
                else:
                    raise LoraNotFoundError(lora_ref)

    elif lora_detector and pipeline_type:
        # No explicit lora - run auto-detection
        matches = lora_detector.match(prompt, pipeline_type)
        for match in matches:
            result.configs.append(match.lora)
            result.auto_detected.append((match.lora.name, match.matched_trigger))

    return result
