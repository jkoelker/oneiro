"""Queue admission and execution regressions using real native metadata and ownership."""

import asyncio
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

from oneiro.config import Config
from oneiro.lora_detector import AutoLoraDetector
from oneiro.pipelines import PipelineManager
from oneiro.pipelines.civitai_checkpoint import CivitaiCheckpointPipeline
from oneiro.pipelines.lora import LoraConfig, LoraSource
from oneiro.pipelines.zimage import ZImagePipelineWrapper
from tests.test_bot import _attachment, _dream_context, _register_test_commands
from tests.test_pipelines_modular import capture_generation, load_hosted


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forbid accidental downloads and Discord connections in this local regression gate."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network access is forbidden in the request regression gate")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def sampling_config(tmp_path: Path) -> Config:
    """Use actual persistent profiles; model assets and Discord alone are replaced."""
    base = tmp_path / "config.toml"
    base.write_text(
        '[defaults]\nmodel = "zimage"\n'
        '[models.zimage]\ntype = "zimage"\nsteps = 9\nguidance_scale = 0.0\n'
        '[models.zimage-turbo]\ntype = "zimage"\nsteps = 9\nguidance_scale = 0.0\n'
        '[models.klein]\ntype = "flux2-klein"\nsteps = 4\nguidance_scale = 1.0\n'
        '[models.raw]\ntype = "krea2"\nrepo = "krea/Krea-2-Raw"\n'
        "steps = 28\nguidance_scale = 4.5\n"
        '[models.turbo]\ntype = "krea2"\nsteps = 8\nguidance_scale = 0.0\n'
    )
    config = Config(base, state_path=tmp_path / "state.json")
    config.load()
    return config


async def loaded_context(config: Config, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep real loaders and graphs while substituting asset retrieval and inference."""
    ctx = await _dream_context()
    wrapper, _ = load_hosted(ZImagePipelineWrapper, monkeypatch, {"cpu_offload": False})
    manager = PipelineManager(config)
    manager.pipeline, manager.current_model = wrapper, "zimage"
    ctx.bot.pipeline_manager, ctx.bot.config = manager, config
    ctx.respond = AsyncMock()
    return ctx


@pytest.mark.parametrize("target,expected", [("klein", (4, 1.0)), ("raw", (28, 4.5))])
@pytest.mark.parametrize("controls", [{}, {"steps": 7}, {"guidance_scale": 1.0}])
async def test_dream_during_load_uses_execution_profile(
    sampling_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    expected: tuple[int, float],
    controls: dict[str, Any],
) -> None:
    """The unowned current_model=None interval must not freeze Z-Image's sampling values."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    entered, release = threading.Event(), threading.Event()
    load = ComponentSpec.load

    def blocked_asset(spec: ComponentSpec, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(5), "Timed out waiting to release the local asset double"
        return load(spec, **kwargs)

    monkeypatch.setattr(ComponentSpec, "load", blocked_asset)
    task = asyncio.create_task(manager.load_model(target))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert manager.current_model is None
        await _register_test_commands()["dream"](ctx, "prompt", **controls)
        request = ctx.bot.generation_queue._pending_requests[0].request
    finally:
        release.set()
        await task
    capture_generation(manager.pipeline, monkeypatch)
    result = await manager.generate(**request)
    assert (result.steps, result.guidance_scale) == (
        controls.get("steps", expected[0]),
        controls.get("guidance_scale", expected[1]),
    )
    assert result.model_name == target
    assert ("steps" in request) is ("steps" in controls)
    assert ("guidance_scale" in request) is ("guidance_scale" in controls)


async def test_queued_dream_switches_from_raw_to_turbo_defaults(
    sampling_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid Raw admission must still use Turbo's native defaults after a switch."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    await manager.load_model("raw")
    await _register_test_commands()["dream"](ctx, "prompt")
    request = ctx.bot.generation_queue._pending_requests[0].request
    await manager.load_model("turbo")
    capture_generation(manager.pipeline, monkeypatch)
    result = await manager.generate(**request)
    assert (result.steps, result.guidance_scale, result.model_name) == (8, 0.0, "turbo")


async def test_queued_controls_revalidate_before_resource_resolution(
    sampling_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative prompt valid for Raw must fail on Turbo before any LoRA file access."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    await manager.load_model("raw")
    await _register_test_commands()["dream"](ctx, "prompt", negative_prompt="blurry")
    request = ctx.bot.generation_queue._pending_requests[0].request
    await manager.load_model("turbo")
    request["loras"] = [LoraConfig(name="missing", source=LoraSource.LOCAL, path="/nonexistent")]
    with pytest.raises(ValueError, match="Negative prompts are not supported"):
        await manager.generate(**request)
    assert manager.current_model == "turbo"


async def test_dream_after_failed_switch_lazy_recovers_default(
    sampling_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real failed load must not make Discord permanently bypass manager lazy recovery."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    load = ComponentSpec.load

    def unavailable_asset(spec: ComponentSpec, **kwargs: Any) -> Any:
        raise OSError("simulated gated repository")

    monkeypatch.setattr(ComponentSpec, "load", unavailable_asset)
    with pytest.raises((OSError, RuntimeError)):
        await manager.load_model("raw")
    assert manager.pipeline is None and manager.current_model is None
    monkeypatch.setattr(ComponentSpec, "load", load)
    await _register_test_commands()["dream"](ctx, "prompt")
    assert len(ctx.bot.generation_queue._pending_requests) == 1
    request = ctx.bot.generation_queue._pending_requests[0].request
    inference = ZImagePipelineWrapper.run_inference

    def capture_after_load(
        owner: ZImagePipelineWrapper, values: dict[str, Any], image: bool
    ) -> Any:
        capture_generation(owner, monkeypatch)
        return owner.run_inference(values, image)

    monkeypatch.setattr(ZImagePipelineWrapper, "run_inference", capture_after_load)
    result = await manager.generate(**request)
    assert (result.steps, result.guidance_scale, result.model_name) == (9, 0.0, "zimage")
    monkeypatch.setattr(ZImagePipelineWrapper, "run_inference", inference)


@pytest.mark.parametrize(
    "controls", [{"negative_prompt": "blurry"}, {"scheduler": "euler"}, {"guidance_scale": 2.0}]
)
@pytest.mark.parametrize("recovering", [False, True])
async def test_dream_rejects_native_controls_before_attachment_or_queue(
    sampling_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    controls: dict[str, Any],
    recovering: bool,
) -> None:
    """Native rejection must happen before upload I/O, resource resolution, or queue entry."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    if recovering:
        ctx.bot.pipeline_manager.pipeline.unload()
        ctx.bot.pipeline_manager.pipeline = None
        ctx.bot.pipeline_manager.current_model = None
    image = _attachment()
    await _register_test_commands()["dream"](ctx, "prompt", image=image, **controls)
    assert not ctx.bot.generation_queue._pending_requests
    image.read.assert_not_awaited()
    assert ctx.followup.send.await_args.kwargs["ephemeral"] is True


@pytest.mark.parametrize("target", ["zimage", "turbo"])
@pytest.mark.parametrize("controls", [{"guidance_scale": 2.0}, {"steps": 0}])
async def test_model_rejects_invalid_overrides_before_state_or_switch(
    sampling_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    controls: dict[str, Any],
) -> None:
    """Both /model branches preserve the usable owner and persistent defaults on rejection."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    previous = manager.pipeline
    await _register_test_commands()["model"](ctx, target, **controls)
    assert sampling_config.get("model_overrides", target) is None
    assert sampling_config.get("defaults", "model") == "zimage"
    assert manager.pipeline is previous and manager.current_model == "zimage"


@pytest.mark.parametrize(
    "controls", [{"negative_prompt": "blurry"}, {"scheduler": "euler"}, {"guidance_scale": 2.0}]
)
async def test_lazy_execution_rejects_native_controls_before_assets(
    sampling_config: Config, monkeypatch: pytest.MonkeyPatch, controls: dict[str, Any]
) -> None:
    """Invalid queued controls cannot download components merely to discover incompatibility."""
    wrapper, records = load_hosted(ZImagePipelineWrapper, monkeypatch)
    wrapper.unload()
    records["spec_loads"].clear()
    manager = PipelineManager(sampling_config)
    with pytest.raises(ValueError):
        await manager.generate("prompt", **controls)
    assert not records["spec_loads"]
    assert manager.pipeline is None and manager.current_model is None


async def test_recovery_preserves_automatically_matched_lora(
    sampling_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery admission uses the default's actual family rather than disabling matching."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    manager.pipeline.unload()
    manager.pipeline = manager.current_model = None
    lora = LoraConfig(
        name="style",
        source=LoraSource.LOCAL,
        path="unused",
        base_model="ZImageTurbo",
        trigger_words=["style"],
    )
    detector = AutoLoraDetector()
    detector.register_loras([lora])
    ctx.bot.lora_detector = detector
    await _register_test_commands()["dream"](ctx, "style")
    assert ctx.bot.generation_queue._pending_requests[0].request["loras"] == [lora]


@pytest.mark.parametrize("target", ["zimage", "turbo"])
@pytest.mark.parametrize("correction", [None, 0.0])
async def test_model_preflights_saved_defaults_and_accepts_correction(
    sampling_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    correction: float | None,
) -> None:
    """Old invalid saved guidance cannot select an unusable owner; an explicit fix can."""
    ctx = await loaded_context(sampling_config, monkeypatch)
    manager = ctx.bot.pipeline_manager
    previous = manager.pipeline
    sampling_config.set("model_overrides", target, "guidance_scale", value=2.0)
    await _register_test_commands()["model"](ctx, target, guidance_scale=correction)
    if correction is None:
        assert manager.pipeline is previous and manager.current_model == "zimage"
        assert sampling_config.get("defaults", "model") == "zimage"
        assert sampling_config.get("model_overrides", target, "guidance_scale") == 2.0
        failure = ctx.followup.send.await_args
        assert failure.kwargs["ephemeral"] is True
        assert "Guidance is controlled" in failure.args[0]
    else:
        assert manager.current_model == target
        assert sampling_config.get("model_overrides", target, "guidance_scale") == 0.0
        capture_generation(manager.pipeline, monkeypatch)
        assert (await manager.generate("prompt")).guidance_scale == 0.0


async def test_latest_checkpoint_change_revalidates_all_controls_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpinned latest version can change between admission and the actual owned load."""
    profile = {"type": "civitai", "civitai_model_id": 1, "cpu_offload": False}
    config = Mock(data={"models": {"remote": profile}})
    config.get.side_effect = lambda *keys, default=None: {
        ("defaults", "model"): "remote",
        ("models", "remote"): profile,
    }.get(keys, default)
    manager = PipelineManager(config)
    client = AsyncMock()
    client.get_model.side_effect = [
        SimpleNamespace(latest_version=SimpleNamespace(base_model="Pony")),
        SimpleNamespace(latest_version=SimpleNamespace(base_model="ZImageTurbo")),
    ]
    client.download_model_version.return_value = Path("unused-checkpoint")
    manager.set_civitai_client(client)
    monkeypatch.setattr(CivitaiCheckpointPipeline, "load", lambda *args: None)
    with pytest.raises(ValueError, match="not compatible with zimage"):
        await manager.generate("prompt", scheduler="euler")
    assert client.get_model.await_count == 2
    client.download_model_version.assert_not_awaited()
    assert manager.pipeline is None and manager.current_model is None
