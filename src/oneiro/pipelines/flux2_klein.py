"""Hosted FLUX.2 Klein distilled/base native recipes."""

from typing import Any

from diffusers import Flux2KleinAutoBlocks, Flux2KleinBaseAutoBlocks

from oneiro.pipelines.modular import ModularPipelineWrapper


class Flux2KleinPipelineWrapper(ModularPipelineWrapper):
    """Select the native distilled or CFG graph, never guessing from repo substrings."""

    family = "flux2-klein"
    default_steps = 4
    default_guidance_scale = 1.0

    def validate_config(
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Resolve distilled/base graph and placement without loading components."""
        repo = model_config.get("repo", "black-forest-labs/FLUX.2-klein-9B")
        known = {
            "black-forest-labs/FLUX.2-klein-4B": "distilled",
            "black-forest-labs/FLUX.2-klein-9B": "distilled",
            "black-forest-labs/FLUX.2-klein-base-4B": "base",
            "black-forest-labs/FLUX.2-klein-base-9B": "base",
        }
        variant = model_config.get("variant", known.get(repo))
        if variant not in {"distilled", "base"} or (repo in known and variant != known[repo]):
            raise ValueError("Klein requires variant='distilled' or 'base' matching the model")
        if (
            model_config.get("embeddings")
            or model_config.get("inline_embeddings")
            or (full_config or {}).get("embeddings", {}).get("auto_load")
        ):
            raise ValueError("Klein does not support textual inversion embeddings")
        distilled = variant == "distilled"
        self.variant = variant
        self.default_steps, self.default_guidance_scale = (4, 1.0) if distilled else (50, 4.0)
        self._recipe_guidance_scale = self.default_guidance_scale
        self._component_repo = repo
        self.blocks = Flux2KleinAutoBlocks() if distilled else Flux2KleinBaseAutoBlocks()
        super().validate_config(model_config, full_config)

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load the preflighted Klein variant under shared resource ownership."""
        self.validate_config(model_config, full_config)
        self._configure_cpu_threads(model_config.get("cpu_utilization", 0.75))
        self.initialize_pipeline(self._component_repo, self.blocks)
        self.pipe.register_to_config(is_distilled=self.variant == "distilled")
