"""Hosted FLUX.1 dev/schnell recipes over the released native workflows."""

from typing import Any

from diffusers import FluxAutoBlocks

from oneiro.device import DevicePolicy
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

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load a known official variant, or require explicit custom-source metadata."""
        repo = model_config.get("repo", "black-forest-labs/FLUX.1-dev")
        known = {
            "black-forest-labs/FLUX.1-dev": "dev",
            "black-forest-labs/FLUX.1-schnell": "schnell",
        }
        variant = model_config.get("variant", known.get(repo))
        if variant not in {"dev", "schnell"} or (repo in known and variant != known[repo]):
            raise ValueError("FLUX.1 requires variant='dev' or 'schnell' matching the model")
        if (
            model_config.get("embeddings")
            or model_config.get("inline_embeddings")
            or (full_config or {}).get("embeddings", {}).get("auto_load")
        ):
            raise ValueError("FLUX.1 does not support textual inversion embeddings")
        self.variant = variant
        self.default_steps, self.default_guidance_scale = (
            (28, 3.5) if variant == "dev" else (4, 0.0)
        )
        self._configure_cpu_threads(model_config.get("cpu_utilization", 0.75))
        self.policy = DevicePolicy.auto_detect(
            cpu_offload=model_config.get("cpu_offload", True),
            offload_type=model_config.get("offload_type", "group"),
            group_offload_type=model_config.get("group_offload_type", "leaf_level"),
            group_offload_use_stream=model_config.get("group_offload_use_stream", True),
            group_offload_num_blocks_per_group=model_config.get(
                "group_offload_num_blocks_per_group"
            ),
        )
        self.initialize_pipeline(repo, FluxAutoBlocks())
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
