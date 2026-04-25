"""Benchmark Diffusers offload modes for a configured Oneiro model.

This script is intended for CUDA hosts. It reloads the same configured model with
each requested offload mode, runs warmup + measured generations, and emits JSON
with wall-clock latency plus CUDA peak memory metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from oneiro.config import Config
from oneiro.device import DevicePolicy
from oneiro.pipelines import PipelineManager

DEFAULT_PROMPT = "a cinematic portrait of a robot blacksmith, dramatic light"
DEFAULT_OFFLOAD_TYPES = ("model", "group")


@dataclass
class OffloadBenchmarkResult:
    """Result for one offload mode benchmark run."""

    model: str
    pipeline_type: str
    offload_type: str
    status: str
    load_seconds: float | None
    warmup_runs: int
    measured_runs: int
    average_seconds: float | None
    median_seconds: float | None
    images_per_minute: float | None
    load_peak_vram_mib: float | None
    generation_peak_vram_mib: float | None
    total_peak_vram_mib: float | None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml", help="Path to base config.toml")
    parser.add_argument("--overlay", default=None, help="Optional overlay config path")
    parser.add_argument("--state", default=None, help="Optional runtime state JSON path")
    parser.add_argument("--model", default=None, help="Configured model name to benchmark")
    parser.add_argument(
        "--offload-types",
        nargs="+",
        default=list(DEFAULT_OFFLOAD_TYPES),
        choices=("model", "group", "sequential"),
        help="Offload modes to compare. Defaults to old model offload vs new group offload.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for generation")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--width", type=int, default=None, help="Generation width override")
    parser.add_argument("--height", type=int, default=None, help="Generation height override")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps override")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance override")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for measured runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup generations per mode")
    parser.add_argument("--runs", type=int, default=3, help="Measured generations per mode")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def clear_cuda() -> None:
    """Release Python and CUDA memory before the next benchmark mode."""
    gc.collect()
    DevicePolicy.clear_cache()


def peak_vram_mib() -> float:
    """Return the current CUDA peak allocation in MiB."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_peak_vram() -> None:
    """Reset CUDA peak-memory accounting."""
    torch.cuda.reset_peak_memory_stats()


def load_config(args: argparse.Namespace) -> Config:
    """Load Oneiro layered config from CLI arguments."""
    config = Config(
        base_path=Path(args.config),
        overlay_path=Path(args.overlay) if args.overlay else None,
        state_path=Path(args.state) if args.state else None,
    )
    config.load()
    return config


def resolve_model_config(config: Config, model_name: str | None) -> tuple[str, dict[str, Any]]:
    """Return the selected model name and a mutable copy of its config."""
    resolved_name = model_name or config.get("defaults", "model", default="zimage-turbo")
    model_config = config.get("models", resolved_name)
    if not isinstance(model_config, dict):
        raise ValueError(f"Unknown model: {resolved_name}")
    return resolved_name, deepcopy(model_config)


def generation_kwargs(args: argparse.Namespace, model_config: dict[str, Any]) -> dict[str, Any]:
    """Build generation kwargs from CLI overrides and model defaults."""
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width or model_config.get("width", 1024),
        "height": args.height or model_config.get("height", 1024),
        "steps": args.steps or model_config.get("steps", 9),
        "guidance_scale": args.guidance_scale
        if args.guidance_scale is not None
        else model_config.get("guidance_scale", model_config.get("true_cfg_scale", 0.0)),
    }


def run_generation_batch(
    pipeline: Any,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    measured: bool,
) -> list[float]:
    """Run warmup or measured generations and return per-image durations."""
    count = args.runs if measured else args.warmup_runs
    durations: list[float] = []
    kwargs = generation_kwargs(args, model_config)

    for index in range(count):
        started = time.perf_counter()
        pipeline.generate(seed=args.seed + index, **kwargs)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)

    return durations


def benchmark_offload_type(
    config: Config,
    model_name: str,
    base_model_config: dict[str, Any],
    offload_type: str,
    args: argparse.Namespace,
) -> OffloadBenchmarkResult:
    """Benchmark one offload mode for the selected model."""
    model_config = deepcopy(base_model_config)
    pipeline_type = model_config.get("type")
    if not isinstance(pipeline_type, str) or pipeline_type not in PipelineManager.PIPELINE_TYPES:
        raise ValueError(f"Unsupported pipeline type for {model_name}: {pipeline_type}")

    model_config["cpu_offload"] = True
    model_config["offload_type"] = offload_type

    clear_cuda()
    reset_peak_vram()
    wrapper = PipelineManager.PIPELINE_TYPES[pipeline_type]()

    load_started = time.perf_counter()
    try:
        wrapper.load(model_config, config.data)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        load_peak = peak_vram_mib()

        run_generation_batch(wrapper, args, model_config, measured=False)
        reset_peak_vram()
        durations = run_generation_batch(wrapper, args, model_config, measured=True)
        generation_peak = peak_vram_mib()
        average_seconds = statistics.mean(durations)
        median_seconds = statistics.median(durations)
        images_per_minute = 60 / average_seconds if average_seconds else None

        return OffloadBenchmarkResult(
            model=model_name,
            pipeline_type=pipeline_type,
            offload_type=offload_type,
            status="ok",
            load_seconds=load_seconds,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
            average_seconds=average_seconds,
            median_seconds=median_seconds,
            images_per_minute=images_per_minute,
            load_peak_vram_mib=load_peak,
            generation_peak_vram_mib=generation_peak,
            total_peak_vram_mib=max(load_peak, generation_peak),
        )
    except Exception as error:
        return OffloadBenchmarkResult(
            model=model_name,
            pipeline_type=pipeline_type,
            offload_type=offload_type,
            status="error",
            load_seconds=None,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
            average_seconds=None,
            median_seconds=None,
            images_per_minute=None,
            load_peak_vram_mib=None,
            generation_peak_vram_mib=None,
            total_peak_vram_mib=None,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        wrapper.unload()
        clear_cuda()


def write_results(results: list[OffloadBenchmarkResult], output: str | None) -> None:
    """Write benchmark results to stdout and optionally to a JSON file."""
    payload = [asdict(result) for result in results]
    text = json.dumps(payload, indent=2)
    print(text)
    if output:
        Path(output).write_text(text + "\n")


def main() -> int:
    """Run offload benchmarks for the selected model."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Diffusers CPU offload benchmarking")

    config = load_config(args)
    model_name, model_config = resolve_model_config(config, args.model)
    results = [
        benchmark_offload_type(config, model_name, model_config, offload_type, args)
        for offload_type in args.offload_types
    ]
    write_results(results, args.output)
    return 1 if any(result.status != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
