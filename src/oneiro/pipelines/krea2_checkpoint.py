"""Stream and validate Comfy Krea checkpoints without a second generation backend."""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from oneiro.device import DevicePolicy
from oneiro.pipelines.backports.krea2 import BackportedKrea2Transformer2DModel

COMFY_DIFFUSION_MODEL_PREFIX = "model.diffusion_model."
KREA2_DTYPE_PRECISIONS = {
    "BF16": "bf16",
    "F8_E4M3": "fp8",
    "F16": "fp16",
    "F32": "fp32",
    "F64": "fp64",
}
KREA2_REQUIRED_TENSOR_KEYS = {"first.weight", "last.linear.weight", "blocks.0.attn.wq.weight"}


class _Krea2FP8Linear(torch.nn.Linear):
    """Run Comfy FP8 weights with dynamically quantized activations."""

    _oneiro_full_precision_mm = False
    _oneiro_weight_dtype: torch.dtype
    _oneiro_weight_scale: torch.Tensor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the FP8 weight using its checkpoint scale."""
        from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

        weight = QuantizedTensor(
            self.weight,
            "TensorCoreFP8Layout",
            TensorCoreFP8Layout.Params(
                scale=self._oneiro_weight_scale,
                orig_dtype=self._oneiro_weight_dtype,
                orig_shape=tuple(self.weight.shape),
            ),
        )
        if self._oneiro_full_precision_mm:
            return torch.nn.functional.linear(input, weight.dequantize(), self.bias)

        quantized, params = TensorCoreFP8Layout.quantize(input)
        quantized_input = QuantizedTensor(quantized, "TensorCoreFP8Layout", params)
        return torch.nn.functional.linear(quantized_input, weight, self.bias)


def _get_krea2_fp8_layers(metadata: dict[str, Any]) -> dict[str, bool]:
    """Return Comfy FP8 layer names and their full-precision matmul flags."""
    raw_metadata = metadata.get("_quantization_metadata")
    if raw_metadata is None:
        return {}
    try:
        quantization = json.loads(raw_metadata)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Krea 2 quantization metadata") from error
    layers = quantization.get("layers") if isinstance(quantization, dict) else None
    if not isinstance(layers, dict) or not layers:
        raise ValueError("Invalid Krea 2 quantization metadata")

    result: dict[str, bool] = {}
    for name, config in layers.items():
        if not isinstance(name, str) or not isinstance(config, dict):
            raise ValueError("Invalid Krea 2 quantization metadata")
        if config.get("format") != "float8_e4m3fn":
            raise ValueError(f"Unsupported Krea 2 quantization format: {config.get('format')}")
        full_precision = config.get("full_precision_matrix_mult", False)
        if not isinstance(full_precision, bool):
            raise ValueError("Krea 2 full_precision_matrix_mult must be a boolean")
        result[name.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX)] = full_precision
    return result


def _get_krea2_comfy_quant_layers(checkpoint: Any) -> dict[str, bool]:
    """Read per-layer Comfy quantization descriptors from a checkpoint."""
    result: dict[str, bool] = {}
    for key in checkpoint.keys():
        if not key.endswith(".comfy_quant"):
            continue
        descriptor = checkpoint.get_tensor(key)
        if descriptor.dtype != torch.uint8 or descriptor.ndim != 1 or descriptor.numel() > 4096:
            raise ValueError(f"Invalid Krea 2 quantization descriptor: {key}")
        try:
            config = json.loads(bytes(descriptor.tolist()).decode())
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"Invalid Krea 2 quantization descriptor: {key}") from error
        if not isinstance(config, dict):
            raise ValueError(f"Invalid Krea 2 quantization descriptor: {key}")
        if config.get("format") != "float8_e4m3fn":
            raise ValueError(f"Unsupported Krea 2 quantization format: {config.get('format')}")
        layer = key.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX).removesuffix(".comfy_quant")
        full_precision = config.get("full_precision_matrix_mult", False)
        if not isinstance(full_precision, bool):
            raise ValueError("Krea 2 full_precision_matrix_mult must be a boolean")
        result[layer] = full_precision
    return result


def get_krea2_checkpoint_precision_from_header(header: dict[str, Any]) -> str:
    """Validate a Krea SafeTensor header and return its dominant precision."""
    metadata = header.get("__metadata__", {}) or {}
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if not isinstance(metadata, dict) or any(
        not isinstance(value, dict) for value in tensors.values()
    ):
        raise ValueError("Invalid SafeTensor header")
    source_keys = list(tensors)
    dtypes = {str(value.get("dtype", "")) for value in tensors.values()}
    fp8_layers = _get_krea2_fp8_layers(metadata)
    has_fp8 = "F8_E4M3" in dtypes
    if not source_keys:
        raise ValueError("Krea 2 checkpoint has no tensors")
    unsupported_tensors = [
        f"{key} ({value.get('dtype')} {value.get('shape')})"
        for key, value in sorted(tensors.items())
        if value.get("dtype") not in KREA2_DTYPE_PRECISIONS
        and not (value.get("dtype") == "U8" and key.endswith(".comfy_quant"))
    ]
    if unsupported_tensors:
        raise ValueError(f"Unsupported Krea 2 checkpoint tensors: {', '.join(unsupported_tensors)}")
    if fp8_layers and not has_fp8:
        raise ValueError("Krea 2 FP8 metadata has no FP8 weights")
    if any(key.endswith(".weight_scale") for key in source_keys) and not has_fp8:
        raise ValueError("Krea 2 FP8 scales have no FP8 weights")
    normalized_tensors = {
        key.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX): value for key, value in tensors.items()
    }
    normalized_keys = set(normalized_tensors)
    fp8_keys = {key for key, value in normalized_tensors.items() if value.get("dtype") == "F8_E4M3"}
    fp8_weights = {key for key in fp8_keys if key.endswith(".weight")}
    scale_keys = {key for key in normalized_keys if key.endswith(".weight_scale")}
    comfy_quant_keys = {key for key in normalized_keys if key.endswith(".comfy_quant")}
    if fp8_keys != fp8_weights:
        raise ValueError("Unexpected non-weight FP8 Krea 2 tensor")
    if fp8_layers and fp8_weights != {f"{name}.weight" for name in fp8_layers}:
        raise ValueError("Krea 2 FP8 metadata does not match checkpoint weights")
    if (
        comfy_quant_keys
        and {f"{key.removesuffix('.comfy_quant')}.weight" for key in comfy_quant_keys}
        != fp8_weights
    ):
        raise ValueError("Krea 2 Comfy quantization descriptors do not match FP8 weights")
    if any(normalized_tensors[key].get("dtype") != "U8" for key in comfy_quant_keys):
        raise ValueError("Krea 2 Comfy quantization descriptors must be U8 tensors")
    if any(
        normalized_tensors[key].get("shape") in (None, [])
        or len(normalized_tensors[key]["shape"]) != 1
        or normalized_tensors[key]["shape"][0] > 4096
        for key in comfy_quant_keys
    ):
        raise ValueError("Invalid Krea 2 Comfy quantization descriptor shape")
    if scale_keys and {key.removesuffix("_scale") for key in scale_keys} != fp8_weights:
        raise ValueError("Krea 2 FP8 checkpoint has missing or orphaned scales")
    if fp8_layers and not scale_keys:
        raise ValueError("Krea 2 FP8 checkpoint is missing weight scales")
    if any(
        normalized_tensors[key].get("dtype") != "F32" or normalized_tensors[key].get("shape") != []
        for key in scale_keys
    ):
        raise ValueError("Krea 2 FP8 weight scales must be scalar FP32 tensors")
    if not KREA2_REQUIRED_TENSOR_KEYS.issubset(normalized_keys):
        raise ValueError("SafeTensor file does not contain Krea 2 transformer weights")
    element_counts = {
        dtype: sum(
            math.prod(value.get("shape", []))
            for value in tensors.values()
            if value.get("dtype") == dtype
        )
        for dtype in dtypes
    }
    dominant_dtype = max(
        KREA2_DTYPE_PRECISIONS,
        key=lambda dtype: element_counts.get(dtype, 0),
    )
    return KREA2_DTYPE_PRECISIONS[dominant_dtype]


def _get_krea2_checkpoint_precision(checkpoint: Any) -> str:
    """Validate an open Krea checkpoint and return its tensor precision."""
    header: dict[str, Any] = {"__metadata__": checkpoint.metadata() or {}}
    for key in checkpoint.keys():
        tensor_slice = checkpoint.get_slice(key)
        header[key] = {"dtype": tensor_slice.get_dtype(), "shape": tensor_slice.get_shape()}
    return get_krea2_checkpoint_precision_from_header(header)


def get_krea2_checkpoint_precision(checkpoint_path: Path) -> str:
    """Read and validate Krea checkpoint precision from its SafeTensor header."""
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        precision = _get_krea2_checkpoint_precision(checkpoint)
        metadata_layers = _get_krea2_fp8_layers(checkpoint.metadata() or {})
        descriptor_layers = _get_krea2_comfy_quant_layers(checkpoint)
        if any(
            layer in metadata_layers and metadata_layers[layer] != full_precision
            for layer, full_precision in descriptor_layers.items()
        ):
            raise ValueError("Conflicting Krea 2 quantization metadata")
        return precision


def convert_krea2_checkpoint_tensor(
    key: str,
    tensor: torch.Tensor,
) -> tuple[str, torch.Tensor]:
    """Convert one Comfy Krea 2 tensor to its Diffusers name and shape."""
    key = key.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX)
    prefix_replacements = (
        ("first.", "img_in."),
        ("tmlp.0.", "time_embed.linear_1."),
        ("tmlp.2.", "time_embed.linear_2."),
        ("tproj.1.", "time_mod_proj."),
        ("txtmlp.0.scale", "txt_in.norm.weight"),
        ("txtmlp.1.", "txt_in.linear_1."),
        ("txtmlp.3.", "txt_in.linear_2."),
        ("txtfusion.", "text_fusion."),
        ("blocks.", "transformer_blocks."),
        ("last.modulation.lin", "final_layer.scale_shift_table"),
        ("last.norm.scale", "final_layer.norm.weight"),
        ("last.linear.", "final_layer.linear."),
    )
    for source, target in prefix_replacements:
        if key.startswith(source):
            key = target + key.removeprefix(source)
            break

    key = key.replace(".attn.qknorm.qnorm.scale", ".attn.norm_q.weight")
    key = key.replace(".attn.qknorm.knorm.scale", ".attn.norm_k.weight")
    key = key.replace(".prenorm.scale", ".norm1.weight")
    key = key.replace(".postnorm.scale", ".norm2.weight")
    key = key.replace(".attn.wq.", ".attn.to_q.")
    key = key.replace(".attn.wk.", ".attn.to_k.")
    key = key.replace(".attn.wv.", ".attn.to_v.")
    key = key.replace(".attn.gate.", ".attn.to_gate.")
    key = key.replace(".attn.wo.", ".attn.to_out.0.")
    key = key.replace(".mlp.", ".ff.")

    if key == "final_layer.scale_shift_table":
        tensor = tensor.reshape(2, -1)
    elif key.endswith(".mod.lin"):
        key = key.removesuffix(".mod.lin") + ".scale_shift_table"
        tensor = tensor.reshape(6, -1)
    return key, tensor


def load_krea2_transformer(
    checkpoint_path: Path,
    component_repo: str,
    transformer_subfolder: str,
    policy: DevicePolicy,
) -> tuple[BackportedKrea2Transformer2DModel, DevicePolicy]:
    """Stream Comfy tensors into the compatible local transformer and return placement policy."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    try:
        transformer_config = BackportedKrea2Transformer2DModel.load_config(
            component_repo,
            subfolder=transformer_subfolder,
        )
    except OSError as error:
        raise RuntimeError(
            f"Unable to load Krea 2 components from '{component_repo}'; "
            "accept its license on Hugging Face and set HF_TOKEN"
        ) from error
    with init_empty_weights():
        transformer = BackportedKrea2Transformer2DModel.from_config(transformer_config)
    expected_shapes = {key: tuple(tensor.shape) for key, tensor in transformer.state_dict().items()}
    loaded_keys: set[str] = set()
    keep_in_fp32_modules = set(transformer._keep_in_fp32_modules)
    fp8_loaded = False

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        source_keys = list(checkpoint.keys())
        _get_krea2_checkpoint_precision(checkpoint)
        fp8_layers = _get_krea2_fp8_layers(checkpoint.metadata() or {})
        comfy_quant_layers = _get_krea2_comfy_quant_layers(checkpoint)
        for layer, full_precision in comfy_quant_layers.items():
            if layer in fp8_layers and fp8_layers[layer] != full_precision:
                raise ValueError(f"Conflicting Krea 2 quantization metadata: {layer}")
            fp8_layers[layer] = full_precision

        for source_key in source_keys:
            if source_key.endswith((".weight_scale", ".comfy_quant")):
                continue
            tensor = checkpoint.get_tensor(source_key)
            key, tensor = convert_krea2_checkpoint_tensor(source_key, tensor)
            expected_shape = expected_shapes.get(key)
            if expected_shape is None:
                raise ValueError(f"Unexpected Krea 2 checkpoint tensor: {source_key}")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Invalid shape for Krea 2 tensor {source_key}: "
                    f"expected {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tensor.dtype == torch.float8_e4m3fn:
                if not source_key.endswith(".weight"):
                    raise ValueError(f"Unexpected FP8 Krea 2 tensor: {source_key}")
                scale_key = f"{source_key.removesuffix('.weight')}.weight_scale"
                scale = (
                    checkpoint.get_tensor(scale_key)
                    if scale_key in source_keys
                    else torch.ones((), dtype=torch.float32)
                )
                if scale.numel() != 1 or not torch.isfinite(scale).item() or scale.item() <= 0:
                    raise ValueError(f"Invalid Krea 2 FP8 weight scale: {scale_key}")
                module = transformer.get_submodule(key.removesuffix(".weight"))
                if not isinstance(module, torch.nn.Linear):
                    raise ValueError(
                        f"FP8 Krea 2 tensor does not target a linear layer: {source_key}"
                    )
                module.__class__ = _Krea2FP8Linear
                module.register_buffer("_oneiro_weight_scale", scale.float(), persistent=False)
                module._oneiro_weight_dtype = policy.dtype
                layer_name = source_key.removeprefix(COMFY_DIFFUSION_MODEL_PREFIX).removesuffix(
                    ".weight"
                )
                module._oneiro_full_precision_mm = fp8_layers.get(layer_name, False)
                fp8_loaded = True

            set_module_tensor_to_device(
                transformer,
                key,
                "cpu",
                value=tensor,
                dtype=(
                    tensor.dtype
                    if tensor.dtype == torch.float8_e4m3fn
                    else (
                        torch.float32
                        if keep_in_fp32_modules.intersection(key.split("."))
                        else policy.dtype
                    )
                ),
            )
            loaded_keys.add(key)
    missing_keys = expected_shapes.keys() - loaded_keys
    if missing_keys:
        missing = ", ".join(sorted(missing_keys)[:3])
        raise ValueError(f"Krea 2 checkpoint is missing tensors: {missing}")
    if fp8_loaded and policy.group_offload_use_stream:
        policy = replace(policy, group_offload_use_stream=False)
    return transformer, policy
