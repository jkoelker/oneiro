"""Qwen-Image pipeline wrapper with LoRA and GGUF support."""

import math
import os
from typing import Any

import torch

from oneiro.pipelines.base import BasePipeline, GenerationResult
from oneiro.pipelines.lora import LoraLoaderMixin, parse_loras_from_model_config


class QwenPipelineWrapper(LoraLoaderMixin, BasePipeline):
    """Wrapper for Qwen-Image with multi-LoRA and GGUF support."""

    def __init__(self) -> None:
        super().__init__()
        self._init_lora_state()

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
                quantization_config=GGUFQuantizationConfig(compute_dtype=self._dtype),
                torch_dtype=self._dtype,
                config=base_repo,
                subfolder="transformer",
            )

        print(f"Loading transformer from {path}")
        return QwenImageTransformer2DModel.from_single_file(
            path,
            torch_dtype=self._dtype,
            config=base_repo,
            subfolder="transformer",
        )

    def load(self, model_config: dict[str, Any]) -> None:
        """Load Qwen-Image model with optional LoRA and GGUF support.

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

        # Build pipeline
        pipeline_kwargs: dict[str, Any] = {
            "scheduler": scheduler,
            "torch_dtype": self._dtype,
        }
        if transformer is not None:
            pipeline_kwargs["transformer"] = transformer

        self.pipe = DiffusionPipeline.from_pretrained(repo, **pipeline_kwargs)

        loras = parse_loras_from_model_config(model_config)
        if loras:
            print(f"  Loading {len(loras)} LoRA(s)...")
            self.load_loras_sync(loras)

        if cpu_offload and self._device == "cuda":
            self.pipe.enable_model_cpu_offload()
        elif self._device == "cuda":
            self.pipe.to("cuda")

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
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        actual_seed, generator = self._prepare_seed(seed)

        # Qwen requires negative_prompt, default to empty string
        neg_prompt = negative_prompt if negative_prompt else " "

        # Use true_cfg_scale for Qwen (passed via kwargs or use guidance_scale)
        true_cfg_scale = kwargs.get("true_cfg_scale", guidance_scale)

        # Handle img2img
        init_image = self._load_init_image(kwargs.get("init_image"))
        strength = kwargs.get("strength", 0.75)

        if init_image:
            print(f"Qwen img2img: '{prompt[:50]}...' seed={actual_seed} strength={strength}")
            result = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                num_images_per_prompt=1,
            )
        else:
            print(f"Qwen-Image generating: '{prompt[:50]}...' seed={actual_seed}")
            result = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                num_images_per_prompt=1,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        output_image = result.images[0]
        return GenerationResult(
            image=output_image,
            seed=actual_seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=output_image.width,
            height=output_image.height,
            steps=steps,
            guidance_scale=true_cfg_scale,
        )
