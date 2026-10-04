"""Global embedding compatibility must not make unrelated native models unusable."""

import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from oneiro.config import Config
from oneiro.pipelines import PipelineManager
from oneiro.pipelines.civitai_checkpoint import CivitaiCheckpointPipeline
from oneiro.pipelines.embedding import EmbeddingIncompatibleError
from tests.test_civitai_checkpoint import checkpoint_wrapper
from tests.test_pipelines_modular import (
    capture_generation,
    load_hosted,
    local_embedding_wrapper,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only actual local token loading and native metadata are permitted."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access is forbidden in embedding regressions")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("family", ["zimage", "krea2", "flux1", "flux2", "flux2-klein", "qwen"])
async def test_global_embedding_does_not_block_unsupported_hosted_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], family: str
) -> None:
    """Optional globals must not reach hosted loaders that explicitly reject textual inversion."""
    prepared, _ = load_hosted(
        PipelineManager.PIPELINE_TYPES[family], monkeypatch, {"cpu_offload": False}
    )
    prepared.unload()
    base = tmp_path / "config.toml"
    base.write_text(
        f'[models.selected]\ntype = "{family}"\ncpu_offload = false\n'
        '[embeddings]\nauto_load = ["easynegative"]\n'
        '[embeddings.easynegative]\nsource = "huggingface"\nrepo = "unused/offline"\n'
    )
    config = Config(base)
    config.load()
    manager = PipelineManager(config)
    await manager.load_model("selected")
    assert manager.pipeline.active_embeddings == []
    assert config.get("embeddings", "auto_load") == ["easynegative"]
    warning = capsys.readouterr().out
    assert "Warning" in warning and "easynegative" in warning
    capture_generation(manager.pipeline, monkeypatch)
    assert (await manager.generate("still usable", width=32, height=32)).model_name == "selected"


@pytest.mark.parametrize("family", ["zimage", "qwen"])
async def test_model_specific_embedding_remains_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """A model-specific resource cannot become optional just because it is also auto-loaded."""
    previous, _ = load_hosted(
        PipelineManager.PIPELINE_TYPES[family], monkeypatch, {"cpu_offload": False}
    )
    base = tmp_path / "config.toml"
    base.write_text(
        f'[models.selected]\ntype = "{family}"\nembeddings = ["easynegative"]\n'
        '[embeddings]\nauto_load = ["easynegative"]\n'
        '[embeddings.easynegative]\nsource = "huggingface"\nrepo = "unused/offline"\n'
    )
    config = Config(base)
    config.load()
    manager = PipelineManager(config)
    manager.pipeline, manager.current_model = previous, "old"
    with pytest.raises(ValueError, match="does not support.*embeddings"):
        await manager.load_model("selected")
    assert manager.pipeline is previous and manager.current_model == "old"


async def test_unsupported_checkpoint_does_not_reload_skipped_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty preflight list must suppress the checkpoint loader's raw-config fallback."""
    _, profile, components = checkpoint_wrapper(
        tmp_path, monkeypatch, "ZImageTurbo", variant="turbo"
    )
    components["vae"].config = SimpleNamespace(block_out_channels=[8, 8])
    monkeypatch.setattr(
        CivitaiCheckpointPipeline, "_load_checkpoint_components", lambda *args: components
    )
    profile["type"] = "civitai"
    full_config = {
        "models": {"selected": profile},
        "embeddings": {
            "auto_load": ["optional"],
            "optional": {"source": "local", "path": "absent"},
        },
    }
    config = Mock(data=full_config)
    config.get.side_effect = lambda *keys, default=None: (
        profile if keys == ("models", "selected") else default
    )
    manager = PipelineManager(config)
    await manager.load_model("selected")
    assert manager.current_model == "selected" and manager.pipeline.active_embeddings == []


@pytest.mark.parametrize("explicit", [False, True])
async def test_sdxl_skips_incompatible_global_but_keeps_compatible_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Remote SD1.5 globals are skipped before download; explicit requests remain fatal."""
    _, profile, components = checkpoint_wrapper(tmp_path, monkeypatch)
    previous, embedding = local_embedding_wrapper(tmp_path)
    components.update(text_encoder=previous.pipe.text_encoder, tokenizer=previous.pipe.tokenizer)
    monkeypatch.setattr(
        CivitaiCheckpointPipeline, "_load_checkpoint_components", lambda *args: components
    )
    base = tmp_path / "config.toml"
    base.write_text(
        '[models.selected]\ntype = "civitai"\n'
        + "\n".join(f"{key} = {json.dumps(value)}" for key, value in profile.items())
        + ('\nembeddings = ["incompatible"]' if explicit else "")
        + '\n[embeddings]\nauto_load = ["incompatible", "style"]\n'
        + '[embeddings.incompatible]\nsource = "civitai"\nid = 1\n'
        + '[embeddings.style]\nsource = "local"\ntoken = "<style>"\n'
        + f"path = {json.dumps(embedding.path)}\n"
    )
    config = Config(base)
    config.load()
    manager = PipelineManager(config)
    manager.pipeline, manager.current_model = previous, "old"
    client = AsyncMock()
    client.get_model.return_value = SimpleNamespace(
        latest_version=SimpleNamespace(base_model="SD 1.5", name="incompatible")
    )
    manager.set_civitai_client(client)
    if explicit:
        with pytest.raises(EmbeddingIncompatibleError):
            await manager.load_model("selected")
        assert manager.pipeline is previous and manager.current_model == "old"
    else:
        await manager.load_model("selected")
        assert manager.pipeline.active_embeddings == ["<style>"]
        encoder, tokenizer = manager.pipeline.pipe.text_encoder, manager.pipeline.pipe.tokenizer
        token_id = tokenizer.convert_tokens_to_ids("<style>")
        assert torch.equal(encoder.get_input_embeddings().weight[token_id], torch.arange(8).float())
    client.download_model_version.assert_not_awaited()
