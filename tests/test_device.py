"""Tests for DevicePolicy."""

import pytest
import torch

from oneiro.device import DevicePolicy, OffloadMode


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


class TestDevicePolicyAutoDetect:
    """Tests for DevicePolicy.auto_detect()."""

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

    def test_auto_offload_on_cuda_enables_offload(self):
        policy = DevicePolicy(device="cuda", dtype=torch.float16, offload=OffloadMode.AUTO)

        class MockPipeline:
            def __init__(self):
                self.offload_enabled = False

            def enable_model_cpu_offload(self):
                self.offload_enabled = True

        pipe = MockPipeline()
        policy.apply_to_pipeline(pipe)
        assert pipe.offload_enabled is True

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
