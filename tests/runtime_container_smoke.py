"""Installed-package acceptance; pipe into a network-disabled, no-GPU container."""

import importlib
import subprocess
import sys
import sysconfig
from importlib.metadata import distributions, version
from importlib.resources import files
from pathlib import Path

import torch
from diffusers import (
    Flux2AutoBlocks,
    Flux2KleinAutoBlocks,
    Flux2KleinBaseAutoBlocks,
    FluxAutoBlocks,
    QwenImageAutoBlocks,
    StableDiffusion3AutoBlocks,
    StableDiffusionXLAutoBlocks,
    ZImageAutoBlocks,
)

import oneiro
from oneiro.pipelines.backports import krea2


def main() -> None:
    """Verify imports, native workflow declarations, license data, and base Torch."""
    installed = Path(sysconfig.get_path("purelib"))
    assert Path(oneiro.__file__).is_relative_to(installed), oneiro.__file__
    assert oneiro.__version__ == version("oneiro"), "Installed package must include its initializer"
    assert Path(torch.__file__).is_relative_to(installed), torch.__file__
    torch_distributions = [
        distribution
        for distribution in distributions()
        if distribution.metadata["Name"].lower() == "torch"
    ]
    assert len(torch_distributions) == 1, torch_distributions
    constraints = Path("/app/runtime-constraints.txt").read_text().splitlines()
    expected_torch = next(line for line in constraints if line.startswith("torch=="))
    assert expected_torch == "torch==" + version("torch")
    assert torch.__version__ == version("torch")
    assert torch.version.cuda is not None, "Runtime image must retain CUDA-enabled Torch"
    assert not torch.cuda.is_available(), "Acceptance must not allocate a GPU"
    print("Python:", sys.version)
    print("Installed Oneiro:", version("oneiro"), oneiro.__file__)
    print("Torch:", version("torch"), torch.__file__, "distributions:", len(torch_distributions))
    print("Base constraint:", expected_torch, "CUDA:", torch.version.cuda, "GPU available: False")
    print("Click:", version("click"))
    for name in (
        "diffusers",
        "transformers",
        "accelerate",
        "comfy-kitchen",
        "bitsandbytes",
        "gguf",
    ):
        print(name + ":", version(name))
    for name in (
        "oneiro.device",
        "oneiro.discord.commands",
        "oneiro.pipelines.krea2_checkpoint",
        "diffusers.quantizers.gguf",
        "diffusers.quantizers.bitsandbytes",
        "comfy_kitchen.tensor",
        "bitsandbytes",
        "gguf",
        "kernels",
        "peft",
    ):
        importlib.import_module(name)
        print("Import:", name, "OK")
    for graph in (
        FluxAutoBlocks,
        Flux2AutoBlocks,
        Flux2KleinAutoBlocks,
        Flux2KleinBaseAutoBlocks,
        QwenImageAutoBlocks,
        StableDiffusionXLAutoBlocks,
        StableDiffusion3AutoBlocks,
        ZImageAutoBlocks,
        krea2.Krea2AutoBlocks,
        krea2.Krea2TurboAutoBlocks,
    ):
        workflows = graph().available_workflows
        assert "text2image" in workflows
        if graph in (krea2.Krea2AutoBlocks, krea2.Krea2TurboAutoBlocks):
            assert {"text2image", "image2image", "inpainting", "reference"} <= set(workflows)
        print(graph.__name__ + ":", sorted(workflows))
    license_text = files(krea2).joinpath("LICENSE").read_text()
    provenance = files(krea2).joinpath("PROVENANCE.md").read_text()
    assert "Apache License" in license_text and "Version 2.0" in license_text
    assert provenance.strip(), "Installed backport must include its source provenance"
    print("Backport data: Apache-2.0 LICENSE and PROVENANCE.md OK")
    subprocess.run(["uv", "pip", "check", "--system"], check=True)
    print("Installed-package acceptance: PASS (no model assets loaded)")


if __name__ == "__main__":
    main()
