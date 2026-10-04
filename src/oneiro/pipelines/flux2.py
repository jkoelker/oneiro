"""Hosted FLUX.2 recipe preserving repository-native quantization."""

from typing import Any

from diffusers import Flux2AutoBlocks, Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration

from oneiro.device import DevicePolicy
from oneiro.pipelines.modular import ModularPipelineWrapper


class Flux2PipelineWrapper(ModularPipelineWrapper):
    """Use native text/image-conditioned blocks with one shared resource lifecycle."""

    family = "flux2"
    default_steps = 28
    default_guidance_scale = 4.0

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load components on CPU, letting their hosted configs select BNB quantization."""
        repo = model_config.get("repo", "diffusers/FLUX.2-dev-bnb-4bit")
        known = {"diffusers/FLUX.2-dev-bnb-4bit", "black-forest-labs/FLUX.2-dev"}
        variant = model_config.get("variant", "dev" if repo in known else None)
        if variant != "dev":
            raise ValueError("FLUX.2 requires variant='dev' for custom model sources")
        if (
            model_config.get("embeddings")
            or model_config.get("inline_embeddings")
            or (full_config or {}).get("embeddings", {}).get("auto_load")
        ):
            raise ValueError("FLUX.2 does not support textual inversion embeddings")
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
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=self.policy.dtype
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo, subfolder="text_encoder", torch_dtype=self.policy.dtype
        )
        self.initialize_pipeline(
            repo, Flux2AutoBlocks(), {"transformer": transformer, "text_encoder": text_encoder}
        )
