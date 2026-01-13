"""
Tests for building CUDA extensions with torchada.

These tests verify that CUDAExtension and BuildExtension work correctly
on MUSA platforms, including source code porting.

The key point is that after importing torchada, the standard torch imports
should work transparently:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
"""

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

# Import torchada first to apply patches
import torchada  # noqa: F401

# Get the path to the test CUDA source file
CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
VECTOR_ADD_CU = os.path.join(CSRC_DIR, "vector_add.cu")


class TestExtensionBuildSetup:
    """Test extension build setup and configuration."""

    def test_vector_add_cu_exists(self):
        """Test that vector_add.cu test file exists."""
        assert os.path.exists(VECTOR_ADD_CU), f"Test file not found: {VECTOR_ADD_CU}"

    def test_can_create_setup_py(self):
        """Test that we can create a setup.py for the extension."""
        # Use standard torch imports - torchada patches make them work on MUSA
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        # Create a temporary directory for the test
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(VECTOR_ADD_CU, tmpdir)

            # Create setup.py content - uses standard torch imports
            setup_content = f"""
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_vector_add",
    ext_modules=[
        CUDAExtension(
            name="test_vector_add",
            sources=["vector_add.cu"],
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            assert os.path.exists(setup_path)

            # Verify the setup.py is valid Python
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", setup_path],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Setup.py syntax error: {result.stderr}"


def _is_gpu_available():
    """Check if CUDA or MUSA GPU is available."""
    import torch

    if torch.cuda.is_available():
        return True
    if hasattr(torch, "musa") and torch.musa.is_available():
        return True
    return False


@pytest.mark.skipif(
    not os.environ.get("TORCHADA_TEST_BUILD", "0") == "1",
    reason="Extension build tests are slow; set TORCHADA_TEST_BUILD=1 to run",
)
class TestExtensionBuild:
    """Test actual extension building (slow, opt-in)."""

    def test_build_vector_add_extension(self):
        """Test building the vector_add extension."""
        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(VECTOR_ADD_CU, tmpdir)

            # Create setup.py using standard torch imports
            setup_content = """
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_vector_add",
    ext_modules=[
        CUDAExtension(
            name="test_vector_add",
            sources=["vector_add.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Check that the extension was built
            ext_files = [
                f for f in os.listdir(tmpdir) if f.endswith(".so") or f.endswith(".pyd")
            ]
            assert len(ext_files) > 0, "No extension file was built"

    def test_run_vector_add_extension(self):
        """Test running the vector_add extension after building."""
        import torch

        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(VECTOR_ADD_CU, tmpdir)

            # Create setup.py using standard torch imports
            setup_content = """
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_vector_add",
    ext_modules=[
        CUDAExtension(
            name="test_vector_add",
            sources=["vector_add.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Add tmpdir to Python path and import the extension
            sys.path.insert(0, tmpdir)
            try:
                import test_vector_add

                try:
                    # Create test tensors
                    a = torch.randn(1000, device="cuda")
                    b = torch.randn(1000, device="cuda")

                    # Run vector add
                    c = test_vector_add.vector_add(a, b)

                    # Verify result
                    expected = a + b
                    assert torch.allclose(c, expected), "Vector add result incorrect"
                except RuntimeError as e:
                    # Skip if GPU is not working or kernel was compiled for different architecture
                    if "invalid device function" in str(e):
                        pytest.skip(
                            "GPU not available or kernel compiled for different architecture"
                        )
                    raise
            finally:
                sys.path.remove(tmpdir)
