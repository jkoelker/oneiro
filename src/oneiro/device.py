"""Device management for pipeline placement."""

from dataclasses import dataclass
from enum import StrEnum

import torch


class OffloadMode(StrEnum):
    """CPU offload behavior for CUDA pipelines."""

    AUTO = "auto"  # Offload if CUDA available (default)
    ALWAYS = "always"  # Require offload (error if not CUDA)
    NEVER = "never"  # Never offload, use .to(device)


class OffloadType(StrEnum):
    """Diffusers offload implementation to use when CPU offload is enabled."""

    MODEL = "model"
    GROUP = "group"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class DevicePolicy:
    """Immutable device configuration for pipeline placement.

    Attributes:
        device: Target device ("cuda", "mps", "cpu")
        dtype: Torch dtype for model weights
        offload: CPU offload behavior for large models
        offload_type: Diffusers offload implementation to use when offloading
        group_offload_type: Diffusers group-offload granularity
        group_offload_use_stream: Overlap CPU/GPU transfers with CUDA streams
        group_offload_num_blocks_per_group: Blocks per offload group for block-level mode
    """

    device: str
    dtype: torch.dtype
    offload: OffloadMode = OffloadMode.AUTO
    offload_type: OffloadType = OffloadType.GROUP
    group_offload_type: str = "leaf_level"
    group_offload_use_stream: bool = True
    group_offload_num_blocks_per_group: int | None = None

    @classmethod
    def auto_detect(
        cls,
        cpu_offload: bool = True,
        offload_type: str | OffloadType = OffloadType.GROUP,
        group_offload_type: str = "leaf_level",
        group_offload_use_stream: bool = True,
        group_offload_num_blocks_per_group: int | None = None,
    ) -> "DevicePolicy":
        """Create policy with auto-detected device and dtype.

        Args:
            cpu_offload: Enable CPU offloading when available (default: True)
            offload_type: Offload implementation: "group", "model", or "sequential"
            group_offload_type: Diffusers group-offload granularity
            group_offload_use_stream: Overlap CPU/GPU transfers with CUDA streams
            group_offload_num_blocks_per_group: Blocks per offload group for block-level mode

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
        return cls(
            device=device,
            dtype=dtype,
            offload=offload,
            offload_type=OffloadType(offload_type),
            group_offload_type=group_offload_type,
            group_offload_use_stream=group_offload_use_stream,
            group_offload_num_blocks_per_group=group_offload_num_blocks_per_group,
        )

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
            if self.offload_type == OffloadType.MODEL:
                pipe.enable_model_cpu_offload()
                pipe._oneiro_offload_type = OffloadType.MODEL.value
            elif self.offload_type == OffloadType.SEQUENTIAL:
                pipe.enable_sequential_cpu_offload()
                pipe._oneiro_offload_type = OffloadType.SEQUENTIAL.value
            else:
                group_offload_num_blocks_per_group = self.group_offload_num_blocks_per_group
                if (
                    self.group_offload_type == "block_level"
                    and group_offload_num_blocks_per_group is None
                ):
                    group_offload_num_blocks_per_group = 1
                pipe.enable_group_offload(
                    onload_device=torch.device(self.device),
                    offload_device=torch.device("cpu"),
                    offload_type=self.group_offload_type,
                    num_blocks_per_group=group_offload_num_blocks_per_group,
                    use_stream=self.group_offload_use_stream,
                )
                pipe._oneiro_offload_type = OffloadType.GROUP.value
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
