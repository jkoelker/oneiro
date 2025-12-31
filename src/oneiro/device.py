"""Device management for pipeline placement."""

from dataclasses import dataclass
from enum import Enum

import torch


class OffloadMode(str, Enum):
    """CPU offload behavior for CUDA pipelines."""

    AUTO = "auto"  # Offload if CUDA available (default)
    ALWAYS = "always"  # Require offload (error if not CUDA)
    NEVER = "never"  # Never offload, use .to(device)


@dataclass(frozen=True)
class DevicePolicy:
    """Immutable device configuration for pipeline placement.

    Attributes:
        device: Target device ("cuda", "mps", "cpu")
        dtype: Torch dtype for model weights
        offload: CPU offload behavior for large models
    """

    device: str
    dtype: torch.dtype
    offload: OffloadMode = OffloadMode.AUTO

    @classmethod
    def auto_detect(cls, cpu_offload: bool = True) -> "DevicePolicy":
        """Create policy with auto-detected device and dtype.

        Args:
            cpu_offload: Enable CPU offloading when available (default: True)

        Returns:
            DevicePolicy configured for the best available device
        """
        if torch.cuda.is_available():
            device = "cuda"
            # Use bfloat16 only if supported, else float16
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float32  # MPS works best with float32
        else:
            device = "cpu"
            dtype = torch.float32

        offload = OffloadMode.AUTO if cpu_offload else OffloadMode.NEVER
        return cls(device=device, dtype=dtype, offload=offload)

    def apply_to_pipeline(self, pipe) -> None:
        """Apply device policy to a diffusers pipeline.

        Args:
            pipe: A diffusers pipeline instance

        Raises:
            ValueError: If offload=ALWAYS but device is not CUDA
        """
        should_offload = self.offload == OffloadMode.ALWAYS or (
            self.offload == OffloadMode.AUTO and self.device == "cuda"
        )

        if should_offload:
            if self.device != "cuda":
                raise ValueError(
                    f"CPU offload requires CUDA device, got '{self.device}'. "
                    f"Set cpu_offload=false in config or use a CUDA-enabled system."
                )
            pipe.enable_model_cpu_offload()
        elif self.device != "cpu":
            pipe.to(self.device)
        # CPU: no action needed, pipeline stays on CPU

    @staticmethod
    def clear_cache() -> None:
        """Clear device memory cache if applicable."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()
