"""Hosted FLUX.1 dev/schnell recipes over the released native workflows."""

from typing import Any

from diffusers import FluxAutoBlocks

from oneiro.pipelines.modular import ModularPipelineWrapper


class Flux1PipelineWrapper(ModularPipelineWrapper):
    """Use native text/img2img blocks and the selected model's sampling defaults."""

    family = "flux1"
    default_steps = 28
    default_guidance_scale = 3.5
    variant = "dev"

    def workflow_inputs(self, workflow: str) -> set[str]:
        """Schnell lacks guidance embeddings despite the common graph's input slot."""
        inputs = super().workflow_inputs(workflow)
        if self.variant == "schnell":
            inputs.discard("guidance_scale")
        return inputs

    def validate_config(
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Resolve a known variant and actual placement without loading components."""
        repo = model_config.get("repo", "black-forest-labs/FLUX.1-dev")
        known = {
            "black-forest-labs/FLUX.1-dev": "dev",
            "black-forest-labs/FLUX.1-schnell": "schnell",
        }
        variant = model_config.get("variant", known.get(repo))
        if variant not in {"dev", "schnell"} or (repo in known and variant != known[repo]):
            raise ValueError("FLUX.1 requires variant='dev' or 'schnell' matching the model")
        if model_config.get("embeddings") or model_config.get("inline_embeddings"):
            raise ValueError("FLUX.1 does not support textual inversion embeddings")
        self.variant = variant
        self.default_steps, self.default_guidance_scale = (
            (28, 3.5) if variant == "dev" else (4, 0.0)
        )
        self._recipe_guidance_scale = self.default_guidance_scale
        self._component_repo, self.blocks = repo, FluxAutoBlocks()
        super().validate_config(model_config, full_config)

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load the preflighted native variant through shared placement."""
        self.validate_config(model_config, full_config)
        self._configure_cpu_threads(model_config.get("cpu_utilization", 0.75))
        self.initialize_pipeline(self._component_repo, self.blocks)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
