"""
Automatic patching module for torchada.

This module patches PyTorch to automatically translate 'cuda' device strings
to 'musa' when running on Moore Threads hardware.

Usage:
    import torchada  # This applies all patches automatically
    import torch

    # Then use torch.cuda as normal - it will work on MUSA
    torch.cuda.is_available()
    x = torch.randn(3, 3).cuda()
    from torch.cuda.amp import autocast, GradScaler

    # Distributed training with NCCL also works transparently
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")  # Uses MCCL on MUSA

    # CUDA Graphs work transparently
    g = torch.cuda.CUDAGraph()  # Uses MUSAGraph on MUSA
"""

import functools
import sys
from types import ModuleType
from typing import Any, Callable, List, Optional

import torch

from ._platform import is_musa_platform

_patched = False
_original_init_process_group = None

# Registry for patch functions
_patch_registry: List[Callable[[], None]] = []


def patch_function(func: Callable[[], None]) -> Callable[[], None]:
    """
    Decorator to register a function to be called during patching.

    This follows the registration pattern used in frameworks like Flask (@app.route),
    pytest (@pytest.fixture), and Django (@receiver). It allows patch functions
    to be defined anywhere in the module and automatically collected for application.

    Usage:
        @patch_function
        def _patch_something():
            # patching logic
            pass

    The decorated function will be called by apply_patches() in registration order.
    """
    _patch_registry.append(func)
    return func


def requires_import(*module_names: str) -> Callable[[Callable], Callable]:
    """
    Decorator to guard a patch function with import checks.

    If any of the specified modules cannot be imported, the decorated function
    returns early without executing. This replaces repetitive try/except patterns.

    Usage:
        @patch_function
        @requires_import('torch_musa')
        def _patch_something():
            # This only runs if torch_musa is importable
            import torch_musa
            # ... patching logic

        @patch_function
        @requires_import('torch._inductor.autotune_process')
        def _patch_autotune():
            import torch._inductor.autotune_process as ap
            # ... patching logic

    Args:
        *module_names: Variable number of module names to check for importability

    Returns:
        A decorator that wraps the function with import guards
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for module_name in module_names:
                try:
                    __import__(module_name)
                except ImportError:
                    return None
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _translate_device(device: Any) -> Any:
    """
    Translate 'cuda' device references to 'musa' on MUSA platform.

    Args:
        device: Device specification (string, torch.device, int, or None)

    Returns:
        Translated device specification
    """
    if not is_musa_platform():
        return device

    if device is None:
        return device

    if isinstance(device, str):
        # Handle 'cuda', 'cuda:0', 'cuda:1', etc.
        if device == "cuda" or device.startswith("cuda:"):
            return device.replace("cuda", "musa")
        return device

    if isinstance(device, torch.device):
        if device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    # For integer device IDs, keep as-is (context determines device type)
    return device


def _wrap_to_method(original_to: Callable) -> Callable:
    """Wrap tensor.to() to translate device strings."""

    @functools.wraps(original_to)
    def wrapped_to(self, *args, **kwargs):
        # Translate device in positional args
        if args and len(args) >= 1:
            first_arg = args[0]
            # Check if first arg looks like a device
            if isinstance(first_arg, (str, torch.device)):
                args = (_translate_device(first_arg),) + args[1:]
            elif isinstance(first_arg, torch.dtype):
                # .to(dtype) case, check for device in kwargs or second arg
                if len(args) >= 2:
                    args = (first_arg, _translate_device(args[1])) + args[2:]

        # Translate device in keyword args
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])

        return original_to(self, *args, **kwargs)

    return wrapped_to


def _wrap_tensor_cuda(original_cuda: Callable) -> Callable:
    """Wrap tensor.cuda() to use musa on MUSA platform."""

    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None, non_blocking=False):
        if is_musa_platform():
            # Use .musa() instead
            if hasattr(self, "musa"):
                return self.musa(device=device, non_blocking=non_blocking)
            else:
                # Fallback to .to()
                target_device = f"musa:{device}" if device is not None else "musa"
                return self.to(target_device, non_blocking=non_blocking)
        return original_cuda(self, device=device, non_blocking=non_blocking)

    return wrapped_cuda


def _wrap_module_cuda(original_cuda: Callable) -> Callable:
    """Wrap nn.Module.cuda() to use musa on MUSA platform."""

    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None):
        if is_musa_platform():
            if hasattr(self, "musa"):
                return self.musa(device=device)
            else:
                target_device = f"musa:{device}" if device is not None else "musa"
                return self.to(target_device)
        return original_cuda(self, device=device)

    return wrapped_cuda


_original_torch_device = None


class _DeviceFactoryMeta(type):
    """Metaclass to make isinstance(x, torch.device) work with our factory."""

    def __instancecheck__(cls, instance):
        if _original_torch_device is not None:
            return isinstance(instance, _original_torch_device)
        return False

    def __subclasscheck__(cls, subclass):
        if _original_torch_device is not None:
            return issubclass(subclass, _original_torch_device)
        return False


class DeviceFactoryWrapper(metaclass=_DeviceFactoryMeta):
    """
    A wrapper class that acts as torch.device but translates cuda to musa.

    Uses a metaclass to properly handle isinstance() checks.
    """

    _original = None

    def __new__(cls, device=None, index=None):
        original = cls._original
        if original is None:
            raise RuntimeError("DeviceFactoryWrapper not initialized")

        # Handle the case where device is already a torch.device
        if isinstance(device, original):
            if device.type == "cuda":
                device = "musa"
                index = device.index if index is None else index
            else:
                return device

        # Handle string device
        if isinstance(device, str):
            device = _translate_device(device)

        # Create the actual device
        if index is not None:
            return original(device, index)
        elif device is not None:
            return original(device)
        else:
            return original()


@patch_function
def _patch_torch_device():
    """
    Patch torch.device to translate 'cuda' to 'musa' on MUSA platform.

    This ensures that torch.device("cuda:0") creates a musa device when on MUSA.
    """
    global _original_torch_device

    if _original_torch_device is not None:
        return  # Already patched

    _original_torch_device = torch.device
    DeviceFactoryWrapper._original = _original_torch_device

    # Replace torch.device with our wrapper
    torch.device = DeviceFactoryWrapper


# Store original torch.Generator for patching
_original_torch_generator = None


@patch_function
def _patch_torch_generator():
    """
    Patch torch.Generator to translate 'cuda' device to 'musa' on MUSA platform.

    This ensures that torch.Generator(device="cuda") creates a MUSA generator
    instead of failing with "Cannot get CUDA generator without ATen_cuda library".
    """
    global _original_torch_generator

    if _original_torch_generator is not None:
        return  # Already patched

    _original_torch_generator = torch.Generator

    class GeneratorWrapper:
        """Wrapper for torch.Generator that translates cuda -> musa."""

        def __new__(cls, device=None):
            # Translate device if needed
            if device is not None:
                device = _translate_device(device)
            return _original_torch_generator(device=device)

        # Ensure isinstance checks work
        @classmethod
        def __instancecheck__(cls, instance):
            return isinstance(instance, _original_torch_generator)

    # Copy over class attributes
    GeneratorWrapper.__doc__ = _original_torch_generator.__doc__
    GeneratorWrapper.__module__ = _original_torch_generator.__module__

    torch.Generator = GeneratorWrapper


def _wrap_factory_function(original_fn: Callable) -> Callable:
    """Wrap tensor factory functions (empty, zeros, ones, etc.) to translate device."""

    @functools.wraps(original_fn)
    def wrapped_fn(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])
        return original_fn(*args, **kwargs)

    return wrapped_fn


# List of torch factory functions that accept a device argument
_FACTORY_FUNCTIONS = [
    "empty",
    "zeros",
    "ones",
    "full",
    "rand",
    "randn",
    "randint",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "tensor",
    "as_tensor",
    "from_numpy",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
    "randint_like",
    "empty_strided",
    "sparse_coo_tensor",
    "sparse_csr_tensor",
]


class _CudaModuleWrapper(ModuleType):
    """
    A wrapper module that redirects torch.cuda to torch.musa,
    but keeps certain attributes (like is_available) pointing to the original.

    This allows downstream projects to detect MUSA platform using:
        torch.cuda.is_available()  # Returns False on MUSA (original behavior)
    While still using torch.cuda.* APIs that redirect to torch.musa.
    """

    # Attributes that should NOT be redirected to torch.musa
    _NO_REDIRECT = {"is_available"}

    # Special attribute mappings for attributes not at top level of torch_musa
    # Maps attribute name -> dot-separated path within torch_musa
    _SPECIAL_ATTRS = {
        "StreamContext": "core.stream.StreamContext",
    }

    # Attribute name remappings (CUDA name -> MUSA name)
    # For CUDA-specific APIs that have different names in MUSA
    _REMAP_ATTRS = {
        "_device_count_nvml": "device_count",  # NVML is NVIDIA-specific
    }

    def __init__(self, original_cuda, musa_module):
        super().__init__("torch.cuda")
        self._original_cuda = original_cuda
        self._musa_module = musa_module

    def __getattr__(self, name):
        # Keep original is_available behavior
        if name in self._NO_REDIRECT:
            return getattr(self._original_cuda, name)

        # Handle special attributes that need nested lookup
        if name in self._SPECIAL_ATTRS:
            obj = self._musa_module
            for part in self._SPECIAL_ATTRS[name].split("."):
                obj = getattr(obj, part)
            return obj

        # Handle attribute name remapping (CUDA-specific names -> MUSA equivalents)
        if name in self._REMAP_ATTRS:
            return getattr(self._musa_module, self._REMAP_ATTRS[name])

        # Redirect everything else to torch.musa
        return getattr(self._musa_module, name)

    def __dir__(self):
        # Combine attributes from both modules
        attrs = set(dir(self._musa_module))
        attrs.update(self._NO_REDIRECT)
        return list(attrs)


# Store original torch.cuda module before patching
_original_torch_cuda = None


@patch_function
@requires_import("torch_musa")
def _patch_torch_cuda_module():
    """
    Patch torch.cuda to redirect to torch.musa on MUSA platform.

    This allows developers to use torch.cuda.* APIs transparently.

    Note: torch.cuda.is_available() is NOT redirected - it keeps the original
    behavior to allow downstream projects to detect the platform properly.
    """
    global _original_torch_cuda

    # torch_musa registers itself as torch.musa when imported
    # Now patch torch.cuda to point to torch.musa (which is torch_musa)
    if hasattr(torch, "musa"):
        # Save original torch.cuda before patching
        if _original_torch_cuda is None:
            _original_torch_cuda = torch.cuda

        # Create wrapper module that redirects most things to torch.musa
        # but keeps is_available pointing to the original
        cuda_wrapper = _CudaModuleWrapper(_original_torch_cuda, torch.musa)

        # Replace torch.cuda with our wrapper in sys.modules
        # This makes 'from torch.cuda import ...' work
        sys.modules["torch.cuda"] = cuda_wrapper

        # Also patch torch.cuda attribute directly
        torch.cuda = cuda_wrapper

        # Patch torch.cuda.amp
        if hasattr(torch.musa, "amp"):
            sys.modules["torch.cuda.amp"] = torch.musa.amp

        # Patch torch.cuda.graphs - MUSAGraph should be accessible as CUDAGraph
        if hasattr(torch.musa, "graphs"):
            sys.modules["torch.cuda.graphs"] = torch.musa.graphs

        # Add CUDAGraph alias pointing to MUSAGraph
        if hasattr(torch.musa, "MUSAGraph") and not hasattr(torch.musa, "CUDAGraph"):
            torch.musa.CUDAGraph = torch.musa.MUSAGraph

        # Patch torch.cuda.nccl -> torch.musa.mccl
        if hasattr(torch.musa, "mccl"):
            sys.modules["torch.cuda.nccl"] = torch.musa.mccl

        # Patch torch.cuda.profiler
        if hasattr(torch.musa, "profiler"):
            sys.modules["torch.cuda.profiler"] = torch.musa.profiler

        # Patch torch.cuda.nvtx - use our stub since MUSA doesn't have nvtx
        try:
            from .cuda import nvtx as nvtx_stub

            sys.modules["torch.cuda.nvtx"] = nvtx_stub
            torch.musa.nvtx = nvtx_stub
        except ImportError:
            pass

        # Patch missing _lazy_call from torch_musa.core._lazy_init
        # torch_musa only maps _lazy_init but not _lazy_call
        # This is needed for code that does: from torch.cuda import _lazy_call
        # We add it to torch.musa so _CudaModuleWrapper can redirect it
        try:
            from torch_musa.core._lazy_init import _lazy_call

            # Only add if not already present (forward compatible with torch_musa fix)
            if not hasattr(torch.musa, "_lazy_call"):
                torch.musa._lazy_call = _lazy_call
        except ImportError:
            pass

        # Add _is_compiled to torch_musa if not present
        # This is needed for code that checks torch.cuda._is_compiled()
        # (e.g., vLLM's CUDA kernel availability checks)
        if not hasattr(torch.musa, "_is_compiled"):
            torch.musa._is_compiled = lambda: True


@patch_function
@requires_import("torch.distributed")
def _patch_distributed_backend():
    """
    Patch torch.distributed to automatically use MCCL when NCCL is requested.

    This allows code using 'nccl' backend to work transparently on MUSA.
    """
    global _original_init_process_group

    import torch.distributed as dist

    if _original_init_process_group is not None:
        # Already patched
        return

    _original_init_process_group = dist.init_process_group

    @functools.wraps(_original_init_process_group)
    def patched_init_process_group(
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        timeout=None,
        world_size: int = -1,
        rank: int = -1,
        store=None,
        group_name: str = "",
        pg_options=None,
        device_id=None,
    ):
        # Translate 'nccl' to 'mccl' on MUSA platform
        if is_musa_platform() and backend is not None:
            if backend.lower() == "nccl":
                backend = "mccl"

        # Translate device_id if it's a cuda device
        if device_id is not None:
            device_id = _translate_device(device_id)

        # Build kwargs for the original function
        kwargs = {
            "backend": backend,
            "init_method": init_method,
            "world_size": world_size,
            "rank": rank,
            "store": store,
            "group_name": group_name,
            "pg_options": pg_options,
            "device_id": device_id,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout

        return _original_init_process_group(**kwargs)

    dist.init_process_group = patched_init_process_group

    # Also patch new_group to translate 'nccl' to 'mccl'
    original_new_group = dist.new_group

    @functools.wraps(original_new_group)
    def patched_new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
        device_id=None,
    ):
        # Translate 'nccl' to 'mccl' on MUSA platform
        if is_musa_platform() and backend is not None:
            if isinstance(backend, str) and backend.lower() == "nccl":
                backend = "mccl"

        # Translate device_id if it's a cuda device
        if device_id is not None:
            device_id = _translate_device(device_id)

        # Build kwargs for the original function
        kwargs = {
            "ranks": ranks,
            "backend": backend,
            "pg_options": pg_options,
            "use_local_synchronization": use_local_synchronization,
            "group_desc": group_desc,
            "device_id": device_id,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout

        return original_new_group(**kwargs)

    dist.new_group = patched_new_group


@patch_function
def _patch_tensor_is_cuda():
    """
    Patch torch.Tensor.is_cuda property to return True for MUSA tensors.

    This allows code that checks tensor.is_cuda to work on MUSA.
    We patch the is_cuda property to also return True for MUSA tensors.
    """
    # Store the original is_cuda property (it's a getset_descriptor)
    original_is_cuda = torch.Tensor.is_cuda

    @property
    def patched_is_cuda(self):
        """Return True if tensor is on CUDA or MUSA device."""
        # Check original is_cuda first
        try:
            if original_is_cuda.__get__(self):
                return True
        except Exception:
            pass
        # Also return True for MUSA tensors
        return getattr(self, "is_musa", False)

    # Replace is_cuda with our patched version
    torch.Tensor.is_cuda = patched_is_cuda


@patch_function
@requires_import("torch_musa.core.stream")
def _patch_stream_cuda_stream():
    """
    Patch MUSA Stream class to add cuda_stream property.

    This allows code that accesses stream.cuda_stream to work on MUSA.
    The cuda_stream property returns the same value as musa_stream.
    """
    from torch_musa.core.stream import Stream as MUSAStream

    # Add cuda_stream property that returns musa_stream
    if not hasattr(MUSAStream, "cuda_stream"):

        @property
        def cuda_stream(self):
            """Return the underlying stream pointer (same as musa_stream)."""
            return self.musa_stream

        MUSAStream.cuda_stream = cuda_stream


@patch_function
@requires_import("torch_musa")
def _patch_autocast():
    """
    Ensure torch.amp.autocast works with 'cuda' device_type on MUSA.
    """
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        return

    original_autocast = torch.amp.autocast

    class PatchedAutocast(original_autocast):
        def __init__(self, device_type, *args, **kwargs):
            # Translate 'cuda' to 'musa'
            if device_type == "cuda":
                device_type = "musa"
            super().__init__(device_type, *args, **kwargs)

    torch.amp.autocast = PatchedAutocast


@patch_function
@requires_import("torch_musa")
def _patch_torch_c_exports():
    """
    Patch torch._C to include MUSA-specific functions from torch_musa._MUSAC.

    Some functions like _storage_Use_Count exist in torch_musa._MUSAC but not
    in torch._C. Code that tries to do:
        from torch._C import _storage_Use_Count
    will fail without this patch.

    This patch adds missing functions from torch_musa._MUSAC to torch._C.
    """
    import torch_musa

    if not hasattr(torch_musa, "_MUSAC"):
        return

    musac = torch_musa._MUSAC

    # List of functions/classes to copy from _MUSAC to torch._C
    # These are commonly imported by downstream code
    _MUSAC_EXPORTS = [
        "_storage_Use_Count",
        # Add more as needed
    ]

    for name in _MUSAC_EXPORTS:
        if hasattr(musac, name) and not hasattr(torch._C, name):
            setattr(torch._C, name, getattr(musac, name))


@patch_function
@requires_import("torch_musa")
def _patch_backends_cuda():
    """
    Patch torch.backends.cuda to work on MUSA platform.

    This patches:
    - is_built() to return True when MUSA is available (since we're using
      torch.cuda APIs that are redirected to MUSA)
    - torch.backends.cuda.matmul.fp32_precision to use torch.get/set_float32_matmul_precision
      (this attribute is missing in some torch_musa versions)

    Note: Other torch.backends.cuda.matmul properties (allow_tf32, etc.) work
    as-is because they are settings that apply to the internal PyTorch
    operations regardless of backend.
    """
    if not hasattr(torch, "backends") or not hasattr(torch.backends, "cuda"):
        return

    # Patch is_built() to return True when MUSA is available
    # This allows code that checks torch.backends.cuda.is_built() to proceed
    original_is_built = torch.backends.cuda.is_built

    def patched_is_built():
        # If MUSA is available, report as "built" since we redirect cuda->musa
        if hasattr(torch, "musa") and torch.musa.is_available():
            return True
        return original_is_built()

    torch.backends.cuda.is_built = patched_is_built

    # Patch cuBLASModule to support fp32_precision attribute
    # This attribute is in newer PyTorch but may be missing in torch_musa's version
    matmul = torch.backends.cuda.matmul
    matmul_class = matmul.__class__

    # Check if fp32_precision is already supported
    try:
        _ = matmul.fp32_precision
        # Already supported, no need to patch
        return
    except AttributeError:
        pass

    # Store original methods
    original_getattr = matmul_class.__getattr__
    original_setattr = matmul_class.__setattr__

    def patched_getattr(self, name):
        if name == "fp32_precision":
            return torch.get_float32_matmul_precision()
        return original_getattr(self, name)

    def patched_setattr(self, name, value):
        if name == "fp32_precision":
            return torch.set_float32_matmul_precision(value)
        return original_setattr(self, name, value)

    matmul_class.__getattr__ = patched_getattr
    matmul_class.__setattr__ = patched_setattr


@patch_function
@requires_import("torchada.utils.cpp_extension", "torch.utils.cpp_extension")
def _patch_cpp_extension():
    """
    Patch torch.utils.cpp_extension to use torchada's MUSA-compatible versions.

    This allows developers to use standard imports like:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    And have them work transparently on MUSA platform.
    """
    import torch.utils.cpp_extension as torch_cpp_ext

    from .utils import cpp_extension as torchada_cpp_ext

    # Patch the key classes and functions
    torch_cpp_ext.CUDAExtension = torchada_cpp_ext.CUDAExtension
    torch_cpp_ext.BuildExtension = torchada_cpp_ext.BuildExtension
    torch_cpp_ext.CUDA_HOME = torchada_cpp_ext.CUDA_HOME

    # Also update sys.modules entry
    sys.modules["torch.utils.cpp_extension"] = torch_cpp_ext


@patch_function
@requires_import("torch._inductor.autotune_process")
def _patch_autotune_process():
    """
    Patch torch._inductor.autotune_process to use MUSA_VISIBLE_DEVICES on MUSA platform.

    The autotune subprocess uses CUDA_VISIBLE_DEVICES to control GPU visibility.
    On MUSA platform, we need to use MUSA_VISIBLE_DEVICES instead.

    Reference: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/autotune_process.py#L61
    """
    import torch._inductor.autotune_process as autotune_process

    # Patch the CUDA_VISIBLE_DEVICES constant to use MUSA_VISIBLE_DEVICES
    if hasattr(autotune_process, "CUDA_VISIBLE_DEVICES"):
        autotune_process.CUDA_VISIBLE_DEVICES = "MUSA_VISIBLE_DEVICES"


def apply_patches():
    """
    Apply all necessary patches for CUDA to MUSA translation.

    After calling this, developers can use torch.cuda.* APIs normally
    and they will be transparently redirected to torch.musa on MUSA platform.

    This includes:
    - torch.device("cuda") -> torch.device("musa")
    - torch.cuda.* API -> torch.musa.*
    - torch.cuda.nvtx -> no-op stub
    - torch.cuda.Stream.cuda_stream -> musa_stream
    - torch.Tensor.cuda() -> torch.Tensor.musa()
    - torch.Tensor.is_cuda -> True for MUSA tensors
    - torch.nn.Module.cuda() -> torch.nn.Module.musa()
    - Device string translation ("cuda" -> "musa")
    - torch.distributed with 'nccl' backend -> 'mccl'
    - torch.cuda.CUDAGraph -> torch.musa.MUSAGraph
    - torch.cuda.nccl -> torch.musa.mccl
    - torch.amp.autocast(device_type='cuda') -> 'musa'
    - torch.utils.cpp_extension (CUDAExtension, BuildExtension) -> MUSA versions
    - torch._inductor.autotune_process.CUDA_VISIBLE_DEVICES -> MUSA_VISIBLE_DEVICES

    This function should be called once at import time.

    Patch functions are registered via the @patch_function decorator and
    can be guarded with @requires_import for optional module dependencies.
    """
    global _patched

    if _patched:
        return

    if not is_musa_platform():
        _patched = True
        return

    # Import torch_musa to ensure it's initialized
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        _patched = True
        return

    # Apply all registered patch functions
    # These are registered via @patch_function decorator in definition order
    for patch_fn in _patch_registry:
        patch_fn()

    # Patch torch.Tensor.to()
    if hasattr(torch.Tensor, "to"):
        torch.Tensor.to = _wrap_to_method(torch.Tensor.to)

    # Patch torch.Tensor.cuda()
    if hasattr(torch.Tensor, "cuda"):
        torch.Tensor.cuda = _wrap_tensor_cuda(torch.Tensor.cuda)

    # Patch torch.nn.Module.cuda()
    if hasattr(torch.nn.Module, "cuda"):
        torch.nn.Module.cuda = _wrap_module_cuda(torch.nn.Module.cuda)

    # Patch tensor factory functions to translate device argument
    # We also need to update _device_constructors cache to include
    # the original (unwrapped) functions, because PyTorch's __torch_function__
    # dispatch receives the original C function, not our Python wrapper.
    original_fns = []
    for fn_name in _FACTORY_FUNCTIONS:
        if hasattr(torch, fn_name):
            original_fn = getattr(torch, fn_name)
            original_fns.append(original_fn)
            setattr(torch, fn_name, _wrap_factory_function(original_fn))

    # Update _device_constructors to include original functions
    # This ensures the device context manager (with torch.device(...):) works
    # because __torch_function__ receives the original C function
    try:
        from torch.utils._device import _device_constructors

        # Get the current set of constructors
        constructors = _device_constructors()

        # Add original (unwrapped) functions to the constructors set
        # PyTorch's __torch_function__ receives these, not our wrappers
        for orig_fn in original_fns:
            constructors.add(orig_fn)

    except (ImportError, AttributeError):
        pass  # Older PyTorch versions may not have this

    _patched = True


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _patched


# Additional exports for advanced usage
def get_original_init_process_group():
    """Get the original torch.distributed.init_process_group function."""
    return _original_init_process_group
