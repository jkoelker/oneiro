"""Hosted Qwen-Image native recipe with single-file/GGUF transformer injection."""

import math
import os
from typing import Any

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageAutoBlocks

from oneiro.pipelines.modular import ModularPipelineWrapper


class QwenPipelineWrapper(ModularPipelineWrapper):
    """Use native Qwen CFG, with optional single-file/GGUF transformer injection."""

    family = "qwen"
    default_steps = 8
    default_guidance_scale = 4.0

    def _parse_transformer_path(self, transformer: str) -> tuple[str, bool]:
        """Parse transformer path, returning (resolved_path, is_gguf).

        Supports:
        - Local paths: /path/to/model.gguf, ./model.gguf, ~/model.gguf
        - HF Hub: repo_id:filename (e.g., unsloth/Qwen-Image-GGUF:qwen-image-Q4_K_S.gguf)

        Returns:
            Tuple of (path_to_file, is_gguf_format)
        """
        # Expand user home directory
        expanded = os.path.expanduser(transformer)

        # Check if it's a local path
        if os.path.exists(expanded) or transformer.startswith(("/", "./", "~/")):
            is_gguf = expanded.lower().endswith(".gguf")
            return expanded, is_gguf

        # Check for repo:file format
        if ":" in transformer:
            from huggingface_hub import hf_hub_download

            repo_id, filename = transformer.split(":", 1)
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            is_gguf = filename.lower().endswith(".gguf")
            return path, is_gguf

        raise ValueError(
            f"transformer must be 'repo_id:filename' or a local path, got: {transformer}"
        )

    def _load_transformer(self, transformer_path: str, base_repo: str) -> Any:
        """Load transformer from path, with GGUF quantization if applicable.

        Args:
            transformer_path: Path specification (local or repo:file)
            base_repo: Base repository for config (e.g., Qwen/Qwen-Image)

        Returns:
            Loaded transformer model
        """
        from diffusers import QwenImageTransformer2DModel

        path, is_gguf = self._parse_transformer_path(transformer_path)

        if is_gguf:
            from diffusers import GGUFQuantizationConfig

            print(f"Loading GGUF transformer from {path}")
            return QwenImageTransformer2DModel.from_single_file(
                path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.policy.dtype),
                torch_dtype=self.policy.dtype,
                config=base_repo,
                subfolder="transformer",
            )

        print(f"Loading transformer from {path}")
        return QwenImageTransformer2DModel.from_single_file(
            path,
            torch_dtype=self.policy.dtype,
            config=base_repo,
            subfolder="transformer",
        )

    def validate_config(
        self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None
    ) -> None:
        """Resolve Qwen Image and placement without loading checkpoint or hosted assets."""
        repo = model_config.get("repo", "Qwen/Qwen-Image")
        known = {"Qwen/Qwen-Image", "Qwen/Qwen-Image-2512"}
        variant = model_config.get("variant", "image" if repo in known else None)
        if variant != "image":
            raise ValueError("Qwen requires variant='image' for custom model sources")
        if model_config.get("embeddings") or model_config.get("inline_embeddings"):
            raise ValueError("Qwen does not support textual inversion embeddings")
        self._component_repo, self.blocks = repo, QwenImageAutoBlocks()
        super().validate_config(model_config, full_config)

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load native Qwen blocks and preserve the single-file transformer API.

        Config options:
            repo: Base model repository (default: Qwen/Qwen-Image)
            transformer: Custom transformer checkpoint. Supports:
                - Local path: /path/to/model.gguf
                - HF Hub: repo_id:filename (e.g., unsloth/Qwen-Image-GGUF:qwen-image-Q4_K_S.gguf)
                GGUF quantization is auto-detected from .gguf extension.
            variant: Explicit 'image' metadata for custom model sources.
            cpu_offload: Enable CPU offload (default: True)
            offload_type: Offload implementation: group, model, or sequential
        """
        self.validate_config(model_config, full_config)
        repo = self._component_repo
        transformer_path = model_config.get("transformer")

        print(f"Loading Qwen-Image from {repo}")

        # Create scheduler with Qwen-specific config
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # Load custom transformer if specified (supports GGUF)
        transformer = None
        if transformer_path:
            transformer = self._load_transformer(transformer_path, repo)

        components: dict[str, Any] = {"scheduler": scheduler}
        if transformer is not None:
            components["transformer"] = transformer
        self.initialize_pipeline(repo, self.blocks, components)
