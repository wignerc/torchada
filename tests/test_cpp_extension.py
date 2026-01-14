"""
Tests for C++ extension building utilities.

These tests verify that after importing torchada, the standard torch imports
work correctly:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
"""

import os
import shutil
import tempfile

import pytest

# Import torchada first to apply patches
import torchada  # noqa: F401


class TestCppExtensionImports:
    """Test cpp_extension module imports using standard torch imports."""

    def test_import_cuda_home(self):
        """Test CUDA_HOME can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import CUDA_HOME

        # CUDA_HOME should be a string or None
        assert CUDA_HOME is None or isinstance(CUDA_HOME, str)

    def test_import_cuda_extension(self):
        """Test CUDAExtension can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import CUDAExtension

        assert CUDAExtension is not None

    def test_import_build_extension(self):
        """Test BuildExtension can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import BuildExtension

        assert BuildExtension is not None

    def test_cuda_home_on_musa(self):
        """Test CUDA_HOME points to MUSA on MUSA platform."""
        from torch.utils.cpp_extension import CUDA_HOME

        if torchada.is_musa_platform() and CUDA_HOME is not None:
            # On MUSA platform, CUDA_HOME should point to MUSA installation
            assert "musa" in CUDA_HOME.lower() or os.path.exists(
                os.path.join(CUDA_HOME, "bin", "mcc")
            )

    def test_torch_cpp_extension_is_patched(self):
        """Test that torch.utils.cpp_extension is patched correctly on MUSA."""
        import torch.utils.cpp_extension as torch_cpp_ext

        if torchada.is_musa_platform():
            # Verify CUDAExtension is our patched version
            assert torch_cpp_ext.CUDAExtension.__module__ == "torchada.utils.cpp_extension"

    def test_torch_cpp_extension_cuda_home_same_as_torchada(self):
        """Test that torch.utils.cpp_extension.CUDA_HOME matches torchada's."""
        import torch.utils.cpp_extension as torch_cpp_ext

        from torchada.utils.cpp_extension import CUDA_HOME as torchada_cuda_home

        if torchada.is_musa_platform():
            assert torch_cpp_ext.CUDA_HOME == torchada_cuda_home


class TestCUDAExtension:
    """Test CUDAExtension class using standard torch imports."""

    def test_create_extension_basic(self):
        """Test basic CUDAExtension creation."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
        )
        assert ext.name == "test_ext"
        assert "test.cu" in ext.sources

    def test_create_extension_with_include_dirs(self):
        """Test CUDAExtension with include_dirs."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
            include_dirs=["/usr/include"],
        )
        assert "/usr/include" in ext.include_dirs

    def test_create_extension_with_extra_compile_args(self):
        """Test CUDAExtension with extra_compile_args."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-arch=sm_70"]},
        )
        assert ext.extra_compile_args is not None


class TestMusaPatches:
    """Test patches applied to torch_musa for extension building."""

    def test_is_musa_file_recognizes_cu(self):
        """Test _is_musa_file recognizes .cu files."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.cu")
            assert musa_ext._is_musa_file("path/to/kernel.cu")

    def test_is_musa_file_recognizes_cuh(self):
        """Test _is_musa_file recognizes .cuh files."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.cuh")
            assert musa_ext._is_musa_file("path/to/header.cuh")

    def test_is_musa_file_recognizes_mu(self):
        """Test _is_musa_file still recognizes .mu files."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.mu")

    def test_ext_replaced_mapping(self):
        """Test EXT_REPLACED_MAPPING keeps .cu/.cuh."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            # Extensions are converted: .cu -> .mu, .cuh -> .muh for mcc compiler
            assert musa_sp.EXT_REPLACED_MAPPING["cu"] == "mu"
            assert musa_sp.EXT_REPLACED_MAPPING["cuh"] == "muh"

    def test_mapping_rule_exists(self):
        """Test _MAPPING_RULE is set."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            assert hasattr(musa_sp, "_MAPPING_RULE")
            assert len(musa_sp._MAPPING_RULE) > 0

    def test_mapping_rule_has_expected_entries(self):
        """Test _MAPPING_RULE has expected entries."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            rules = musa_sp._MAPPING_RULE

            # Check some key mappings
            assert rules.get("cudaMalloc") == "musaMalloc"
            assert rules.get("cudaFree") == "musaFree"
            assert rules.get("cudaStream_t") == "musaStream_t"
            assert rules.get("at::cuda") == "at::musa"
            assert rules.get("c10::cuda") == "c10::musa"
