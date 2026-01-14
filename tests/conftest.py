"""
Pytest configuration and fixtures for torchada tests.
"""

import os
import sys

import pytest

# Ensure torchada is imported first to apply patches
import torchada


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "musa: mark test as requiring MUSA platform")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA platform")
    config.addinivalue_line("markers", "gpu: mark test as requiring any GPU (CUDA or MUSA)")
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture(scope="session")
def platform():
    """Return the detected platform."""
    return torchada.detect_platform()


@pytest.fixture(scope="session")
def is_musa():
    """Return True if running on MUSA platform."""
    return torchada.is_musa_platform()


@pytest.fixture(scope="session")
def is_cuda():
    """Return True if running on CUDA platform."""
    return torchada.is_cuda_platform()


@pytest.fixture(scope="session")
def has_gpu():
    """Return True if any GPU is available."""
    import torch

    return torch.cuda.is_available()


@pytest.fixture(scope="function")
def gpu_tensor():
    """Create a GPU tensor fixture."""
    import torch

    if torch.cuda.is_available():
        return torch.randn(10, 10, device="cuda")
    else:
        pytest.skip("No GPU available")


@pytest.fixture(scope="function")
def cpu_tensor():
    """Create a CPU tensor fixture."""
    import torch

    return torch.randn(10, 10)


def pytest_collection_modifyitems(config, items):
    """Skip tests based on platform markers."""
    import torch

    skip_musa = pytest.mark.skip(reason="MUSA platform required")
    skip_cuda = pytest.mark.skip(reason="CUDA platform required")
    skip_gpu = pytest.mark.skip(reason="GPU required")

    for item in items:
        if "musa" in item.keywords and not torchada.is_musa_platform():
            item.add_marker(skip_musa)
        if "cuda" in item.keywords and not torchada.is_cuda_platform():
            item.add_marker(skip_cuda)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
