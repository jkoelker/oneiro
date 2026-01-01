"""Qwen-Image pipeline wrapper with LoRA, GGUF, and embedding support."""

import math
import os
from typing import Any

import torch
from PIL import Image

from oneiro.device import DevicePolicy
from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.embedding import EmbeddingLoaderMixin, parse_embeddings_from_config
from oneiro.pipelines.lora import LoraConfig, LoraLoaderMixin, parse_loras_from_model_config


class QwenPipelineWrapper(LoraLoaderMixin, EmbeddingLoaderMixin, BasePipeline):
    """Wrapper for Qwen-Image with multi-LoRA, GGUF, and embedding support."""

    def __init__(self) -> None:
        super().__init__()
        self._static_lora_configs: list[LoraConfig] = []

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

    def load(self, model_config: dict[str, Any], full_config: dict[str, Any] | None = None) -> None:
        """Load Qwen-Image model with optional LoRA, GGUF, and embedding support.

        Config options:
            repo: Base model repository (default: Qwen/Qwen-Image)
            transformer: Custom transformer checkpoint. Supports:
                - Local path: /path/to/model.gguf
                - HF Hub: repo_id:filename (e.g., unsloth/Qwen-Image-GGUF:qwen-image-Q4_K_S.gguf)
                GGUF quantization is auto-detected from .gguf extension.
            lora: LoRA repository
            lora_weights: LoRA weights filename
            cpu_offload: Enable CPU offload (default: True)
        """
        from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

        repo = model_config.get("repo", "Qwen/Qwen-Image")
        transformer_path = model_config.get("transformer")
        cpu_offload = model_config.get("cpu_offload", True)

        self.policy = DevicePolicy.auto_detect(cpu_offload=cpu_offload)

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

        pipeline_kwargs: dict[str, Any] = {
            "scheduler": scheduler,
            "torch_dtype": self.policy.dtype,
        }
        if transformer is not None:
            pipeline_kwargs["transformer"] = transformer

        self.pipe = DiffusionPipeline.from_pretrained(repo, **pipeline_kwargs)

        self.policy.apply_to_pipeline(self.pipe)

        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)
            # Track static LoRAs loaded from config for post_generate reset
            self._static_lora_configs = list(loras)
        else:
            self._static_lora_configs = []

        if full_config:
            embeddings = parse_embeddings_from_config(full_config, model_config)
            if embeddings:
                print(f"  Loading {len(embeddings)} embedding(s)...")
                self.load_embeddings_sync(embeddings)

        print(f"Qwen-Image loaded from {repo}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 8,
        guidance_scale: float = 4.0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate image with Qwen-Image.

        Note: Qwen uses true_cfg_scale instead of guidance_scale,
        and requires a negative_prompt (even empty string).
        """
        return super().generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            **kwargs,
        )

    def build_generation_kwargs(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        init_image: Image.Image | None,
        strength: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build Qwen generation kwargs. Uses true_cfg_scale, requires neg_prompt."""
        # Qwen requires negative_prompt, default to single space
        neg_prompt = negative_prompt if negative_prompt else " "
        # Use true_cfg_scale for Qwen (from kwargs or use guidance_scale)
        true_cfg_scale = kwargs.get("true_cfg_scale", guidance_scale)

        if init_image:
            print(f"Qwen img2img: '{prompt[:50]}...' strength={strength}")
            return {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "image": init_image,
                "strength": strength,
                "num_inference_steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "generator": generator,
                "num_images_per_prompt": 1,
            }
        else:
            print(f"Qwen-Image generating: '{prompt[:50]}...'")
            return {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "generator": generator,
                "num_images_per_prompt": 1,
            }

    def build_result(
        self,
        result: Any,
        seed: int,
        prompt: str,
        negative_prompt: str | None,
        steps: int,
        guidance_scale: float,
    ) -> GenerationResult:
        """Build result with true_cfg_scale as guidance_scale."""
        output_image = result.images[0]
        return GenerationResult(
            image=output_image,
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output_image.width,
            height=output_image.height,
            steps=steps,
            guidance_scale=guidance_scale,
        )

    def post_generate(self, **kwargs: Any) -> None:
        """Reset LoRA state to static adapters loaded from config.

        This prevents state leakage between generation requests by ensuring
        only the LoRAs defined in the model config remain active after each
        generation.
        """
        # Build expected static adapter names
        static_names = [lora.adapter_name or lora.name for lora in self._static_lora_configs]

        # Check if current adapters match static config (names and count)
        adapters_match = self._loaded_adapters == static_names

        if adapters_match:
            # Adapters match - just reset weights in case they were modified
            if self._static_lora_configs:
                adapter_weights = [lora.weight for lora in self._static_lora_configs]
                self.set_lora_adapters(static_names, adapter_weights)
            return

        # Adapters don't match - full reset required
        self.unload_loras()
        if self._static_lora_configs:
            self.load_loras_sync(self._static_lora_configs)
            print(f"Restored {len(self._static_lora_configs)} static LoRA(s)")
