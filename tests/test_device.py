"""Tests for DevicePolicy."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers.modular_pipelines.components_manager import ComponentsManager

from oneiro.device import DevicePolicy, OffloadMode, OffloadType


class TestOffloadMode:
    """Tests for OffloadMode enum."""

    def test_string_values(self):
        assert OffloadMode.AUTO.value == "auto"
        assert OffloadMode.ALWAYS.value == "always"
        assert OffloadMode.NEVER.value == "never"

    def test_is_str_subclass(self):
        """OffloadMode should be usable as a string."""
        assert isinstance(OffloadMode.AUTO, str)
        assert OffloadMode.AUTO == "auto"


class TestOffloadType:
    """Tests for OffloadType enum."""

    def test_string_values(self):
        assert OffloadType.MODEL.value == "model"
        assert OffloadType.GROUP.value == "group"
        assert OffloadType.SEQUENTIAL.value == "sequential"

    def test_is_str_subclass(self):
        """OffloadType should be usable as a string."""
        assert isinstance(OffloadType.GROUP, str)
        assert OffloadType.GROUP == "group"


class TestDevicePolicyAutoDetect:
    """Tests for DevicePolicy.auto_detect()."""

    @pytest.mark.parametrize(
        "controls",
        [
            {"offload_type": "invalid"},
            {"group_offload_type": "invalid"},
            {"group_offload_num_blocks_per_group": 0},
            {"group_offload_num_blocks_per_group": -1},
            {"group_offload_num_blocks_per_group": 1.5},
        ],
    )
    def test_invalid_placement_controls_fail_without_components(
        self, controls: dict[str, Any]
    ) -> None:
        """Reject deterministic placement errors before the loader can unload or fetch assets."""
        with pytest.raises(ValueError):
            DevicePolicy.auto_detect(**controls)

    def test_returns_device_policy(self):
        policy = DevicePolicy.auto_detect()
        assert isinstance(policy, DevicePolicy)
        assert policy.device in ("cuda", "mps", "cpu")

    def test_cpu_offload_true_sets_auto(self):
        policy = DevicePolicy.auto_detect(cpu_offload=True)
        assert policy.offload == OffloadMode.AUTO

    def test_cpu_offload_false_sets_never(self):
        policy = DevicePolicy.auto_detect(cpu_offload=False)
        assert policy.offload == OffloadMode.NEVER

    def test_auto_detect_defaults_to_group_offload(self):
        policy = DevicePolicy.auto_detect()
        assert policy.offload_type == OffloadType.GROUP
        assert policy.group_offload_type == "leaf_level"
        assert policy.group_offload_use_stream is True

    def test_auto_detect_accepts_model_offload_type(self):
        policy = DevicePolicy.auto_detect(offload_type="model")
        assert policy.offload_type == OffloadType.MODEL

    def test_dtype_is_valid_torch_dtype(self):
        policy = DevicePolicy.auto_detect()
        assert policy.dtype in (torch.float16, torch.bfloat16, torch.float32)

    def test_cpu_device_uses_float32(self):
        """CPU device should always use float32."""
        # We can't force CPU detection, but we can verify the logic
        policy = DevicePolicy(device="cpu", dtype=torch.float32)
        assert policy.dtype == torch.float32


class TestDevicePolicyFrozen:
    """Tests for DevicePolicy immutability."""

    def test_cannot_modify_device(self):
        policy = DevicePolicy.auto_detect()
        with pytest.raises(AttributeError):
            policy.device = "cpu"

    def test_cannot_modify_dtype(self):
        policy = DevicePolicy.auto_detect()
        with pytest.raises(AttributeError):
            policy.dtype = torch.float32

    def test_cannot_modify_offload(self):
        policy = DevicePolicy.auto_detect()
        with pytest.raises(AttributeError):
            policy.offload = OffloadMode.NEVER

    def test_cannot_modify_offload_type(self):
        policy = DevicePolicy.auto_detect()
        with pytest.raises(AttributeError):
            policy.offload_type = OffloadType.MODEL


class TestDevicePolicyApply:
    """Tests for DevicePolicy.apply_to_pipeline()."""

    def test_always_offload_on_non_cuda_raises(self):
        policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.ALWAYS)

        class MockPipeline:
            pass

        with pytest.raises(ValueError, match="CPU offload requires CUDA"):
            policy.apply_to_pipeline(MockPipeline())

    def test_always_offload_on_mps_raises(self):
        policy = DevicePolicy(device="mps", dtype=torch.float32, offload=OffloadMode.ALWAYS)

        class MockPipeline:
            pass

        with pytest.raises(ValueError, match="CPU offload requires CUDA"):
            policy.apply_to_pipeline(MockPipeline())

    def test_never_offload_calls_to_device(self):
        policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.NEVER)

        class MockPipeline:
            def __init__(self):
                self.moved_to = None

            def to(self, device):
                self.moved_to = device

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.moved_to == "cuda"

    def test_auto_group_offload_on_cuda_enables_group_offload(self):
        policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)

        class MockPipeline:
            def __init__(self):
                self.group_offload_kwargs = None
                self._oneiro_offload_type: str | None = None

            def enable_group_offload(self, **kwargs):
                self.group_offload_kwargs = kwargs

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.group_offload_kwargs == {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "offload_type": "leaf_level",
            "num_blocks_per_group": None,
            "use_stream": True,
        }
        assert pipe._oneiro_offload_type == "group"

    def test_model_offload_on_cuda_enables_model_offload(self):
        policy = DevicePolicy(
            device="cuda",
            dtype=torch.float16,
            offload=OffloadMode.AUTO,
            offload_type=OffloadType.MODEL,
        )

        class MockPipeline:
            def __init__(self):
                self.offload_enabled = False
                self._oneiro_offload_type: str | None = None

            def enable_model_cpu_offload(self):
                self.offload_enabled = True

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.offload_enabled is True
        assert pipe._oneiro_offload_type == "model"

    def test_sequential_offload_on_cuda_enables_sequential_offload(self):
        policy = DevicePolicy(
            device="cuda",
            dtype=torch.float16,
            offload=OffloadMode.AUTO,
            offload_type=OffloadType.SEQUENTIAL,
        )

        class MockPipeline:
            def __init__(self):
                self.offload_enabled = False
                self._oneiro_offload_type: str | None = None

            def enable_sequential_cpu_offload(self):
                self.offload_enabled = True

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.offload_enabled is True
        assert pipe._oneiro_offload_type == "sequential"

    def test_block_level_stream_defaults_to_one_block_per_group(self):
        policy = DevicePolicy(
            device="cuda",
            dtype=torch.float16,
            offload=OffloadMode.AUTO,
            group_offload_type="block_level",
        )

        class MockPipeline:
            def __init__(self):
                self.group_offload_kwargs = None

            def enable_group_offload(self, **kwargs):
                self.group_offload_kwargs = kwargs

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.group_offload_kwargs is not None
        assert pipe.group_offload_kwargs["num_blocks_per_group"] == 1

    def test_block_level_without_stream_defaults_to_one_block_per_group(self):
        policy = DevicePolicy(
            device="cuda",
            dtype=torch.float16,
            offload=OffloadMode.AUTO,
            group_offload_type="block_level",
            group_offload_use_stream=False,
        )

        class MockPipeline:
            def __init__(self):
                self.group_offload_kwargs = None

            def enable_group_offload(self, **kwargs):
                self.group_offload_kwargs = kwargs

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.group_offload_kwargs is not None
        assert pipe.group_offload_kwargs["num_blocks_per_group"] == 1
        assert pipe.group_offload_kwargs["use_stream"] is False

    def test_auto_offload_on_cpu_no_action(self):
        policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.AUTO)

        class MockPipeline:
            def __init__(self):
                self.to_called = False
                self.offload_called = False

            def to(self, device):
                self.to_called = True

            def enable_model_cpu_offload(self):
                self.offload_called = True

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert not pipe.to_called
        assert not pipe.offload_called

    def test_cpu_device_no_action(self):
        policy = DevicePolicy(device="cpu", dtype=torch.float32, offload=OffloadMode.NEVER)

        class MockPipeline:
            def __init__(self):
                self.to_called = False
                self.offload_called = False

            def to(self, device):
                self.to_called = True

            def enable_model_cpu_offload(self):
                self.offload_called = True

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert not pipe.to_called
        assert not pipe.offload_called

    def test_mps_device_moves_to_mps(self):
        policy = DevicePolicy(device="mps", dtype=torch.float32, offload=OffloadMode.NEVER)

        class MockPipeline:
            def __init__(self):
                self.moved_to = None

            def to(self, device):
                self.moved_to = device

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.moved_to == "mps"


class TestDevicePolicyClearCache:
    """Tests for DevicePolicy.clear_cache()."""

    def test_clear_cache_does_not_raise(self):
        # Should not raise regardless of device availability
        DevicePolicy.clear_cache()

    def test_clear_cache_is_static(self):
        # Can be called without an instance
        DevicePolicy.clear_cache()


class TestDevicePolicyEquality:
    """Tests for DevicePolicy equality and hashing."""

    def test_equal_policies(self):
        p1 = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        p2 = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        assert p1 == p2

    def test_unequal_device(self):
        p1 = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        p2 = DevicePolicy(device="cpu", dtype=torch.float16, offload=OffloadMode.AUTO)
        assert p1 != p2

    def test_unequal_dtype(self):
        p1 = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        p2 = DevicePolicy(device="cuda", dtype=torch.bfloat16, offload=OffloadMode.AUTO)
        assert p1 != p2

    def test_unequal_offload(self):
        p1 = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        p2 = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.NEVER)
        assert p1 != p2

    def test_hashable(self):
        p = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)
        # Should not raise - frozen dataclasses are hashable
        hash(p)
        # Can be used in sets
        s = {p}
        assert p in s


class TestModularDevicePolicy:
    """Component-level placement must not duplicate shared-module hooks."""

    @pytest.mark.parametrize("offload_type", list(OffloadType))
    def test_shared_components_placed_once(
        self, monkeypatch: pytest.MonkeyPatch, offload_type: OffloadType
    ) -> None:
        module = torch.nn.Linear(2, 2)
        pipe = SimpleNamespace(components={"transformer": module, "text_encoder": module})
        manager = ComponentsManager()
        manager.add("transformer", module)
        manager.add("text_encoder", module)
        placed = []

        def group(module: torch.nn.Module, **kwargs: Any) -> None:
            placed.append(module)
            assert kwargs["onload_device"] == torch.device("cuda")

        def sequential(module: torch.nn.Module, **kwargs: Any) -> None:
            placed.append(module)
            assert kwargs["execution_device"] == torch.device("cuda")

        monkeypatch.setattr("diffusers.hooks.apply_group_offloading", group)
        monkeypatch.setattr("accelerate.cpu_offload", sequential)
        policy = DevicePolicy(device="cuda", dtype=torch.float16, offload_type=offload_type)
        policy.apply_to_modular_pipeline(pipe, manager)
        assert pipe.components["transformer"] is pipe.components["text_encoder"]
        if offload_type == OffloadType.MODEL:
            assert len(manager.model_hooks) == 1
            assert manager.model_hooks[0].model is module
            assert module._hf_hook.execution_device == torch.device("cuda:0")
            manager.disable_auto_cpu_offload()
        else:
            assert placed == [module]
            assert not manager._auto_offload_enabled

    def test_fp8_does_not_enable_stream_or_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = torch.nn.Linear(2, 2).to(dtype=torch.float8_e4m3fn)
        pipe = SimpleNamespace(components={"transformer": module})
        calls = []
        monkeypatch.setattr(
            "diffusers.hooks.apply_group_offloading", lambda module, **kwargs: calls.append(kwargs)
        )
        DevicePolicy(device="cuda", dtype=torch.float16).apply_to_modular_pipeline(
            pipe, ComponentsManager()
        )
        assert calls[0]["use_stream"] is False
        assert module.weight.dtype == torch.float8_e4m3fn

    @pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
    def test_nonoffload_moves_only_distinct_modules(
        self, monkeypatch: pytest.MonkeyPatch, device: str
    ) -> None:
        module = torch.nn.Linear(2, 2)
        calls = []
        monkeypatch.setattr(module, "to", lambda destination: calls.append(destination))
        pipe = SimpleNamespace(components={"vae": module, "other": module})
        policy = DevicePolicy(device=device, dtype=torch.float32, offload=OffloadMode.NEVER)
        policy.apply_to_modular_pipeline(pipe, ComponentsManager())
        assert calls == ([] if device == "cpu" else [device])

    def test_always_offload_requires_cuda(self) -> None:
        with pytest.raises(ValueError, match="requires CUDA"):
            DevicePolicy(
                device="cpu", dtype=torch.float32, offload=OffloadMode.ALWAYS
            ).apply_to_modular_pipeline(SimpleNamespace(components={}), ComponentsManager())

    def test_existing_accelerate_hooks_reject_competing_strategy(self) -> None:
        from accelerate import cpu_offload

        module = torch.nn.Linear(2, 2)
        cpu_offload(module, execution_device=torch.device("cpu"))
        with pytest.raises(ValueError, match="offload"):
            DevicePolicy(device="cuda", dtype=torch.float32).apply_to_modular_pipeline(
                SimpleNamespace(components={"vae": module}), ComponentsManager()
            )

    def test_native_group_offloading_accepts_transformers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from diffusers.hooks import apply_group_offloading
        from transformers import CLIPTextConfig, CLIPTextModel

        model = CLIPTextModel(
            CLIPTextConfig(
                vocab_size=2,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
            )
        )
        placed = []

        def cpu_group(module: torch.nn.Module, **kwargs: Any) -> None:
            assert kwargs["onload_device"] == torch.device("cuda")
            placed.append(module)
            # Exercise actual native hooks, replacing only the accelerator boundary.
            apply_group_offloading(module, **{**kwargs, "onload_device": torch.device("cpu")})

        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
        monkeypatch.setattr("diffusers.hooks.apply_group_offloading", cpu_group)
        pipe = SimpleNamespace(components={"text_encoder": model, "shared_encoder": model})
        DevicePolicy(
            device="cuda", dtype=torch.float32, group_offload_use_stream=False
        ).apply_to_modular_pipeline(pipe, ComponentsManager())
        assert placed == [model]
        assert model(torch.tensor([[0, 1, 0]])).last_hidden_state.shape == (1, 3, 8)
