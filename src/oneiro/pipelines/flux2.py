"""Hosted FLUX.2 recipe preserving repository-native quantization."""

from typing import Any

from diffusers import Flux2AutoBlocks, Flux2Transformer2DModel
from transformers import Mistral3ForConditionalGeneration

from oneiro.pipelines.modular import ModularPipelineWrapper


class Flux2PipelineWrapper(ModularPipelineWrapper):
    """Use native text/image-conditioned blocks with one shared resource lifecycle."""

    family = "flux2"
    default_steps = 28
    default_guidance_scale = 4.0

    def validate_config(
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Validate variant/resources and placement without touching hosted assets."""
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
        self._component_repo, self.blocks = repo, Flux2AutoBlocks()
        super().validate_config(model_config, full_config)

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load components on CPU, letting their hosted configs select BNB quantization."""
        self.validate_config(model_config, full_config)
        self._configure_cpu_threads(model_config.get("cpu_utilization", 0.75))
        repo = self._component_repo
        transformer = Flux2Transformer2DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=self.policy.dtype
        )
        text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
            repo, subfolder="text_encoder", torch_dtype=self.policy.dtype
        )
        self.initialize_pipeline(
            repo, self.blocks, {"transformer": transformer, "text_encoder": text_encoder}
        )
