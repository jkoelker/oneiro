"""Hosted Krea Raw/Turbo recipes using the isolated image-workflow backport."""

from typing import Any

from oneiro.pipelines.backports.krea2 import (
    BackportedKrea2Transformer2DModel,
    Krea2AutoBlocks,
    Krea2TurboAutoBlocks,
)
from oneiro.pipelines.modular import ModularPipelineWrapper


def load_krea2_tokenizer(repo: str) -> Any:
    """Load Krea's tokenizer from its published fast-tokenizer asset."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", use_fast=True)


class Krea2PipelineWrapper(ModularPipelineWrapper):
    """Retain native Turbo initialization before selecting any workflow."""

    family = "krea2"
    default_steps = 8
    default_guidance_scale = 0.0

    def validate_config(
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Resolve the native variant and placement without touching tokenizer/weights."""
        repo = model_config.get("repo", "krea/Krea-2-Turbo")
        known = {"krea/Krea-2-Turbo": "turbo", "krea/Krea-2-Raw": "raw"}
        variant = model_config.get("variant", known.get(repo))
        if variant not in {"turbo", "raw"} or (repo in known and variant != known[repo]):
            raise ValueError("Krea requires variant='raw' or 'turbo' matching the model")
        if (
            model_config.get("embeddings")
            or model_config.get("inline_embeddings")
            or (full_config or {}).get("embeddings", {}).get("auto_load")
        ):
            raise ValueError("Krea does not support textual inversion embeddings")
        self.default_steps, self.default_guidance_scale = (
            (8, 0.0) if variant == "turbo" else (28, 4.5)
        )
        self._recipe_guidance_scale = self.default_guidance_scale
        self._component_repo = repo
        self.blocks = Krea2TurboAutoBlocks() if variant == "turbo" else Krea2AutoBlocks()
        super().validate_config(model_config, full_config)

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Inject the released-compatible transformer and published fast tokenizer."""
        self.validate_config(model_config, full_config)
        self._configure_cpu_threads(model_config.get("cpu_utilization", 0.75))
        repo = self._component_repo
        tokenizer = load_krea2_tokenizer(repo)
        transformer = BackportedKrea2Transformer2DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=self.policy.dtype
        )
        self.initialize_pipeline(
            repo, self.blocks, {"tokenizer": tokenizer, "transformer": transformer}
        )
