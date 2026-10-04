"""Hosted FLUX.2 Klein distilled/base native recipes."""

from typing import Any

from diffusers import Flux2KleinAutoBlocks, Flux2KleinBaseAutoBlocks

from oneiro.device import DevicePolicy
from oneiro.pipelines.modular import ModularPipelineWrapper


class Flux2KleinPipelineWrapper(ModularPipelineWrapper):
    """Select the native distilled or CFG graph, never guessing from repo substrings."""

    family = "flux2-klein"
    default_steps = 4
    default_guidance_scale = 1.0

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load the published Klein variant, with explicit metadata for custom sources."""
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
        self.default_steps, self.default_guidance_scale = (4, 1.0) if distilled else (50, 4.0)
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
        blocks = Flux2KleinAutoBlocks() if distilled else Flux2KleinBaseAutoBlocks()
        self.initialize_pipeline(repo, blocks)
        self.pipe.register_to_config(is_distilled=distilled)
