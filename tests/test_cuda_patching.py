"""
Tests for torch.cuda patching functionality.

These tests verify that torch.cuda.* APIs work transparently on MUSA.
"""

import sys

import pytest


class TestTorchCudaModule:
    """Test torch.cuda module patching."""

    def test_torch_cuda_is_patched_on_musa(self):
        """Test that torch.cuda is patched to torch.musa on MUSA platform."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # torch.cuda should now be torch_musa
            assert "torch_musa" in torch.cuda.__name__ or "musa" in str(torch.cuda)
        else:
            # On CUDA/CPU, torch.cuda should remain unchanged
            assert "cuda" in torch.cuda.__name__

    def test_torch_cuda_is_available(self):
        """Test torch.cuda.is_available() is NOT redirected on MUSA.

        This is intentionally NOT redirected to allow downstream projects
        to detect the MUSA platform using patterns like:
            if torch.cuda.is_available():  # CUDA
            elif torch.musa.is_available():  # MUSA
        """
        import torch

        import torchada

        result = torch.cuda.is_available()
        assert isinstance(result, bool)

        if torchada.is_musa_platform():
            # On MUSA platform, torch.cuda.is_available() should return False
            # because CUDA is not available - only MUSA is
            assert result is False
            # But torch.musa.is_available() should return True
            import torch_musa

            assert torch_musa.is_available() is True

    def test_torch_cuda_device_count(self):
        """Test torch.cuda.device_count() works."""
        import torch

        import torchada

        count = torch.cuda.device_count()
        assert isinstance(count, int)
        assert count >= 0

        if torchada.is_musa_platform() and torch.cuda.is_available():
            import torch_musa

            assert count == torch_musa.device_count()

    def test_torch_cuda_device_count_nvml(self):
        """Test torch.cuda._device_count_nvml() maps to torch.musa.device_count()."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # _device_count_nvml is NVIDIA-specific, should map to device_count on MUSA
            nvml_count = torch.cuda._device_count_nvml()
            musa_count = torch.musa.device_count()
            assert nvml_count == musa_count
            assert isinstance(nvml_count, int)
            assert nvml_count >= 0

    def test_torch_cuda_current_device(self):
        """Test torch.cuda.current_device() works when GPU available."""
        import torch

        import torchada

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            assert isinstance(device_id, int)
            assert device_id >= 0
            assert device_id < torch.cuda.device_count()

    def test_torch_cuda_get_device_name(self):
        """Test torch.cuda.get_device_name() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            assert isinstance(name, str)
            assert len(name) > 0

            if torchada.is_musa_platform():
                # MUSA device names typically contain "MTT" or "Moore"
                # but this is hardware-dependent, so just check it's not empty
                pass

    def test_torch_cuda_synchronize(self):
        """Test torch.cuda.synchronize() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            # Should not raise
            torch.cuda.synchronize()

    def test_torch_cuda_empty_cache(self):
        """Test torch.cuda.empty_cache() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            # Should not raise
            torch.cuda.empty_cache()

    def test_torch_cuda_memory_allocated(self):
        """Test torch.cuda.memory_allocated() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated()
            assert isinstance(mem, int)
            assert mem >= 0

    def test_torch_cuda_memory_reserved(self):
        """Test torch.cuda.memory_reserved() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved()
            assert isinstance(mem, int)
            assert mem >= 0

    def test_torch_cuda_set_device(self):
        """Test torch.cuda.set_device() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            torch.cuda.set_device(current)
            assert torch.cuda.current_device() == current


class TestTorchCudaLazyInit:
    """Test torch.cuda lazy initialization functions patching."""

    def test_import_lazy_call(self):
        """Test that _lazy_call can be imported from torch.cuda.

        This is needed because torch_musa only maps _lazy_init but not _lazy_call.
        torchada patches this to make from torch.cuda import _lazy_call work.
        """
        from torch.cuda import _lazy_call

        import torchada

        assert _lazy_call is not None
        assert callable(_lazy_call)

    def test_lazy_call_functionality(self):
        """Test that _lazy_call works correctly."""
        from torch.cuda import _lazy_call

        import torchada

        # _lazy_call should accept a callable and queue it for lazy execution
        called = []

        def test_callback():
            called.append(True)

        # Should not raise
        _lazy_call(test_callback)


class TestTorchCudaAmp:
    """Test torch.cuda.amp module patching."""

    def test_import_autocast(self):
        """Test that autocast can be imported from torch.cuda.amp."""
        from torch.cuda.amp import autocast

        import torchada

        assert autocast is not None

    def test_import_grad_scaler(self):
        """Test that GradScaler can be imported from torch.cuda.amp."""
        from torch.cuda.amp import GradScaler

        import torchada

        assert GradScaler is not None

    def test_autocast_context_manager(self):
        """Test autocast works as context manager."""
        import torch
        from torch.cuda.amp import autocast

        import torchada

        if torch.cuda.is_available():
            try:
                with autocast():
                    x = torch.randn(2, 2, device="cuda")
                    assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                # MUSA driver issue, not torchada
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_grad_scaler_creation(self):
        """Test GradScaler can be created."""
        import torch
        from torch.cuda.amp import GradScaler

        import torchada

        if torch.cuda.is_available():
            scaler = GradScaler()
            assert scaler is not None


class TestCUDAGraph:
    """Test CUDAGraph aliasing."""

    def test_cuda_graph_alias(self):
        """Test torch.cuda.CUDAGraph is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "CUDAGraph")
        assert hasattr(torch.cuda, "MUSAGraph")

    def test_cuda_graph_is_musa_graph(self):
        """Test torch.cuda.CUDAGraph is aliased to MUSAGraph on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            assert torch.cuda.CUDAGraph is torch.cuda.MUSAGraph

    def test_cuda_graph_creation(self):
        """Test CUDAGraph can be created."""
        import torch

        import torchada

        if torch.cuda.is_available():
            try:
                g = torch.cuda.CUDAGraph()
                assert g is not None
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_graphs_module(self):
        """Test torch.cuda.graphs module is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "graphs")

    def test_make_graphed_callables(self):
        """Test make_graphed_callables is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "make_graphed_callables")

    def test_graph_pool_handle(self):
        """Test graph_pool_handle is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "graph_pool_handle")


class TestDistributedBackend:
    """Test distributed backend patching."""

    def test_original_init_available(self):
        """Test original init_process_group is saved."""
        import torchada

        if torchada.is_musa_platform():
            original = torchada.get_original_init_process_group()
            assert original is not None

    def test_mccl_backend_available(self):
        """Test MCCL backend is registered."""
        import torch.distributed as dist

        import torchada

        if torchada.is_musa_platform():
            assert hasattr(dist.Backend, "MCCL")

    def test_nccl_backend_available(self):
        """Test NCCL backend constant is still available."""
        import torch.distributed as dist

        import torchada

        assert hasattr(dist.Backend, "NCCL")

    def test_new_group_patched(self):
        """Test new_group is patched to translate nccl->mccl."""
        import torch.distributed as dist

        import torchada

        if torchada.is_musa_platform():
            # Check new_group is wrapped
            assert hasattr(dist.new_group, "__wrapped__")


class TestNCCLModule:
    """Test NCCL to MCCL module patching."""

    def test_mccl_module_available(self):
        """Test torch.cuda.mccl is available."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            assert hasattr(torch.cuda, "mccl")


class TestRNGFunctions:
    """Test RNG functions are available through torch.cuda."""

    def test_get_rng_state(self):
        """Test torch.cuda.get_rng_state is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "get_rng_state")

    def test_set_rng_state(self):
        """Test torch.cuda.set_rng_state is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "set_rng_state")

    def test_manual_seed(self):
        """Test torch.cuda.manual_seed is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "manual_seed")

    def test_manual_seed_all(self):
        """Test torch.cuda.manual_seed_all is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "manual_seed_all")

    def test_seed(self):
        """Test torch.cuda.seed is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "seed")

    def test_initial_seed(self):
        """Test torch.cuda.initial_seed is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "initial_seed")

    def test_manual_seed_works(self):
        """Test torch.cuda.manual_seed actually works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)


class TestMemoryFunctions:
    """Test additional memory functions."""

    def test_max_memory_allocated(self):
        """Test torch.cuda.max_memory_allocated is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "max_memory_allocated")

    def test_max_memory_reserved(self):
        """Test torch.cuda.max_memory_reserved is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "max_memory_reserved")

    def test_memory_stats(self):
        """Test torch.cuda.memory_stats is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "memory_stats")

    def test_memory_summary(self):
        """Test torch.cuda.memory_summary is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "memory_summary")

    def test_memory_snapshot(self):
        """Test torch.cuda.memory_snapshot is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "memory_snapshot")

    def test_reset_peak_memory_stats(self):
        """Test torch.cuda.reset_peak_memory_stats is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "reset_peak_memory_stats")

    def test_mem_get_info(self):
        """Test torch.cuda.mem_get_info is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "mem_get_info")

        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                assert free >= 0
                assert total > 0
            except RuntimeError as e:
                if "MUSA" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestStreamAndEvent:
    """Test Stream and Event classes."""

    def test_stream_class(self):
        """Test torch.cuda.Stream is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "Stream")

    def test_event_class(self):
        """Test torch.cuda.Event is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "Event")

    def test_external_stream_class(self):
        """Test torch.cuda.ExternalStream is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "ExternalStream")

    def test_current_stream(self):
        """Test torch.cuda.current_stream is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "current_stream")

    def test_default_stream(self):
        """Test torch.cuda.default_stream is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "default_stream")

    def test_set_stream(self):
        """Test torch.cuda.set_stream is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "set_stream")

    def test_stream_context_manager(self):
        """Test torch.cuda.stream is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "stream")

    def test_stream_context_class(self):
        """Test torch.cuda.StreamContext is available on MUSA.

        StreamContext is not at top level of torch_musa, it's in
        torch_musa.core.stream.StreamContext, so we need special handling.
        """
        import torch

        import torchada

        if torchada.is_musa_platform():
            # StreamContext should be accessible via torch.cuda.StreamContext
            assert hasattr(torch.cuda, "StreamContext")
            # It should be the MUSA StreamContext class
            import torch_musa.core.stream

            assert torch.cuda.StreamContext is torch_musa.core.stream.StreamContext

    def test_stream_cuda_stream_property(self):
        """Test stream.cuda_stream returns same value as stream.musa_stream on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform() and torch.cuda.is_available():
            try:
                stream = torch.cuda.Stream()
                assert hasattr(stream, "cuda_stream")
                assert hasattr(stream, "musa_stream")
                assert stream.cuda_stream == stream.musa_stream
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestContextManagers:
    """Test device context managers."""

    def test_device_context_manager(self):
        """Test torch.cuda.device is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "device")

    def test_device_of_context_manager(self):
        """Test torch.cuda.device_of is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "device_of")


class TestDeviceFunctions:
    """Test additional device functions."""

    def test_get_device_properties(self):
        """Test torch.cuda.get_device_properties is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "get_device_properties")

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            assert props is not None
            assert hasattr(props, "name")
            assert hasattr(props, "total_memory")

    def test_get_device_capability(self):
        """Test torch.cuda.get_device_capability is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "get_device_capability")

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            assert isinstance(cap, tuple)
            assert len(cap) == 2

    def test_is_initialized(self):
        """Test torch.cuda.is_initialized is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "is_initialized")


class TestTorchGenerator:
    """Test torch.Generator works with cuda device on MUSA platform."""

    def test_generator_cuda_device(self):
        """Test torch.Generator(device='cuda') works on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # Should create a MUSA generator instead of failing
            gen = torch.Generator(device="cuda")
            assert gen is not None
            assert gen.device.type == "musa"

    def test_generator_cuda_device_index(self):
        """Test torch.Generator(device='cuda:0') works on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            gen = torch.Generator(device="cuda:0")
            assert gen is not None
            assert gen.device.type == "musa"
            assert gen.device.index == 0

    def test_generator_musa_device(self):
        """Test torch.Generator(device='musa') still works."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            gen = torch.Generator(device="musa")
            assert gen is not None
            assert gen.device.type == "musa"

    def test_generator_no_device(self):
        """Test torch.Generator() without device works."""
        import torch

        import torchada  # noqa: F401

        gen = torch.Generator()
        assert gen is not None

    def test_generator_manual_seed_chain(self):
        """Test torch.Generator(device='cuda').manual_seed(seed) works."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            gen = torch.Generator(device="cuda").manual_seed(42)
            assert gen is not None
            assert gen.device.type == "musa"


class TestTorchVersionCuda:
    """Test torch.version.cuda is NOT patched.

    This is intentionally NOT patched to allow downstream projects
    to detect the MUSA platform using patterns like:
        if torch.version.cuda is not None:  # CUDA
        elif hasattr(torch.version, 'musa'):  # MUSA
    """

    def test_torch_version_cuda_not_patched(self):
        """Test torch.version.cuda is NOT patched on MUSA platform."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # torch.version.cuda should remain None on MUSA
            # This allows downstream projects to detect the platform
            assert torch.version.cuda is None
            # But torch.version.musa should be set
            assert torch.version.musa is not None
            assert isinstance(torch.version.musa, str)


class TestTensorIsCuda:
    """Test tensor.is_cuda patching."""

    def test_cpu_tensor_is_cuda_false(self):
        """Test CPU tensor.is_cuda returns False."""
        import torch

        import torchada

        cpu_tensor = torch.empty(10, 10)
        assert cpu_tensor.is_cuda is False

    def test_musa_tensor_is_cuda_true(self):
        """Test MUSA tensor.is_cuda returns True on MUSA platform."""
        import torch

        import torchada

        if torchada.is_musa_platform() and torch.cuda.is_available():
            try:
                musa_tensor = torch.empty(10, 10, device="cuda:0")
                assert musa_tensor.is_cuda is True
                assert musa_tensor.is_musa is True
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestAutotuneProcess:
    """Test torch._inductor.autotune_process patching."""

    def test_cuda_visible_devices_patched(self):
        """Test CUDA_VISIBLE_DEVICES is patched to MUSA_VISIBLE_DEVICES on MUSA platform."""
        import torchada

        try:
            import torch._inductor.autotune_process as autotune_process
        except ImportError:
            pytest.skip("torch._inductor.autotune_process not available")

        if torchada.is_musa_platform():
            # On MUSA platform, CUDA_VISIBLE_DEVICES should be patched to MUSA_VISIBLE_DEVICES
            assert autotune_process.CUDA_VISIBLE_DEVICES == "MUSA_VISIBLE_DEVICES"
        else:
            # On CUDA/CPU, it should remain as CUDA_VISIBLE_DEVICES
            assert autotune_process.CUDA_VISIBLE_DEVICES == "CUDA_VISIBLE_DEVICES"

    def test_cuda_visible_devices_is_string(self):
        """Test CUDA_VISIBLE_DEVICES constant is a string."""
        import torchada

        try:
            import torch._inductor.autotune_process as autotune_process
        except ImportError:
            pytest.skip("torch._inductor.autotune_process not available")

        assert isinstance(autotune_process.CUDA_VISIBLE_DEVICES, str)

    def test_cuda_visible_devices_env_var_format(self):
        """Test CUDA_VISIBLE_DEVICES constant is in correct format for env vars."""
        import torchada

        try:
            import torch._inductor.autotune_process as autotune_process
        except ImportError:
            pytest.skip("torch._inductor.autotune_process not available")

        env_var = autotune_process.CUDA_VISIBLE_DEVICES
        # Env var should be uppercase and use underscores
        assert env_var.isupper()
        assert "_" in env_var
        # Should end with VISIBLE_DEVICES
        assert env_var.endswith("VISIBLE_DEVICES")


class TestNvtxStub:
    """Test torch.cuda.nvtx stub module."""

    def test_nvtx_module_available(self):
        """Test torch.cuda.nvtx is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "nvtx")

    def test_nvtx_mark(self):
        """Test torch.cuda.nvtx.mark is available and callable."""
        import torch.cuda.nvtx as nvtx

        import torchada

        assert hasattr(nvtx, "mark")
        # Should not raise
        nvtx.mark("test")

    def test_nvtx_range_push_pop(self):
        """Test torch.cuda.nvtx.range_push and range_pop are available."""
        import torch.cuda.nvtx as nvtx

        import torchada

        assert hasattr(nvtx, "range_push")
        assert hasattr(nvtx, "range_pop")
        # Should not raise
        nvtx.range_push("test")
        nvtx.range_pop()

    def test_nvtx_range_context_manager(self):
        """Test torch.cuda.nvtx.range context manager works."""
        import torch.cuda.nvtx as nvtx

        import torchada

        assert hasattr(nvtx, "range")
        # Should not raise
        with nvtx.range("test"):
            pass


class TestPatchDecorators:
    """Test the decorator-based patch registration system."""

    def test_patch_registry_is_populated(self):
        """Test that @patch_function decorator populates the registry."""
        from torchada._patch import _patch_registry

        # Registry should have at least 8 registered patches
        assert len(_patch_registry) >= 8

        # All entries should be callable
        for fn in _patch_registry:
            assert callable(fn)

    def test_patch_registry_contains_expected_functions(self):
        """Test that registry contains the expected patch functions."""
        from torchada._patch import _patch_registry

        # Get function names from registry
        fn_names = [fn.__name__ for fn in _patch_registry]

        # Check expected functions are registered
        expected_fns = [
            "_patch_torch_device",
            "_patch_torch_cuda_module",
            "_patch_distributed_backend",
            "_patch_tensor_is_cuda",
            "_patch_stream_cuda_stream",
            "_patch_autocast",
            "_patch_cpp_extension",
            "_patch_autotune_process",
        ]

        for expected in expected_fns:
            assert expected in fn_names, f"{expected} not found in registry"

    def test_requires_import_decorator_guards_import(self):
        """Test that @requires_import returns early when import fails."""
        from torchada._patch import requires_import

        @requires_import("nonexistent_module_that_does_not_exist")
        def test_func():
            raise AssertionError("Should not be called when import fails")

        # Should return None without raising
        result = test_func()
        assert result is None

    def test_requires_import_decorator_allows_execution(self):
        """Test that @requires_import allows execution when import succeeds."""
        from torchada._patch import requires_import

        @requires_import("sys")  # 'sys' always exists
        def test_func():
            return "executed"

        result = test_func()
        assert result == "executed"

    def test_requires_import_multiple_modules(self):
        """Test @requires_import with multiple module names."""
        from torchada._patch import requires_import

        @requires_import("sys", "os")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

        @requires_import("sys", "nonexistent_module_xyz")
        def test_func_fails():
            raise AssertionError("Should not run")

        result = test_func_fails()
        assert result is None


class TestIsCompiledAndBackends:
    """Test _is_compiled and backends patches."""

    def test_torch_musa_is_compiled(self):
        """Test torch.musa._is_compiled() exists and returns True."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            assert hasattr(torch.musa, "_is_compiled")
            assert torch.musa._is_compiled() is True

    def test_torch_cuda_is_compiled_redirects(self):
        """Test torch.cuda._is_compiled() works via redirect."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # torch.cuda is redirected to torch.musa
            result = torch.cuda._is_compiled()
            assert result is True

    def test_torch_backends_cuda_is_built(self):
        """Test torch.backends.cuda.is_built() returns True on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # Should return True since MUSA is available
            assert torch.backends.cuda.is_built() is True

    def test_torch_backends_cuda_matmul_allow_tf32(self):
        """Test torch.backends.cuda.matmul.allow_tf32 is accessible."""
        import torch

        import torchada

        # Should be accessible and settable
        original = torch.backends.cuda.matmul.allow_tf32
        assert isinstance(original, bool)

        # Test setting
        torch.backends.cuda.matmul.allow_tf32 = True
        assert torch.backends.cuda.matmul.allow_tf32 is True

        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.backends.cuda.matmul.allow_tf32 is False

        # Restore original
        torch.backends.cuda.matmul.allow_tf32 = original

    def test_torch_backends_cuda_matmul_fp32_precision(self):
        """Test torch.backends.cuda.matmul.fp32_precision is accessible."""
        import torch

        import torchada  # noqa: F401 - ensure patches are applied

        # Should be accessible
        original = torch.backends.cuda.matmul.fp32_precision
        assert original in ("highest", "high", "medium")

        # Test setting (use torch.get/set_float32_matmul_precision values)
        torch.backends.cuda.matmul.fp32_precision = "highest"
        assert torch.backends.cuda.matmul.fp32_precision == "highest"
        assert torch.get_float32_matmul_precision() == "highest"

        torch.backends.cuda.matmul.fp32_precision = "high"
        assert torch.backends.cuda.matmul.fp32_precision == "high"
        assert torch.get_float32_matmul_precision() == "high"

        # Restore original
        torch.backends.cuda.matmul.fp32_precision = original

    def test_torch_c_storage_use_count(self):
        """Test torch._C._storage_Use_Count is accessible on MUSA."""
        import torchada

        if torchada.is_musa_platform():
            # Should be able to import from torch._C after patching
            from torch._C import _storage_Use_Count as use_count

            assert use_count is not None
            assert callable(use_count)
