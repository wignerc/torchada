<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/MooreThreads/torchada/main/assets/logo.png" alt="logo" width="250" margin="10px"></img>
</div>

--------------------------------------------------------------------------------

# torchada

**Adapter package for torch_musa to act exactly like PyTorch CUDA**

torchada provides a unified interface that works transparently on both Moore Threads GPUs (MUSA) and NVIDIA GPUs (CUDA). Write your code once using standard PyTorch CUDA APIs, and it will run on MUSA hardware without any changes.

## Features

- **Zero Code Changes**: Just `import torchada` once, then use standard `torch.cuda.*` APIs
- **Automatic Platform Detection**: Detects whether you're running on CUDA or MUSA
- **Transparent Device Mapping**: `tensor.cuda()` and `tensor.to("cuda")` work on MUSA
- **Extension Building**: Standard `torch.utils.cpp_extension` works on MUSA after importing torchada
- **Source Code Porting**: Automatic CUDA → MUSA symbol mapping for C++/CUDA extensions
- **Distributed Training**: `torch.distributed` with `nccl` backend automatically uses `mccl` on MUSA
- **Mixed Precision**: `torch.cuda.amp` autocast and GradScaler work transparently
- **CUDA Graphs**: `torch.cuda.CUDAGraph` maps to `MUSAGraph` on MUSA
- **Inductor Support**: `torch._inductor` autotune uses `MUSA_VISIBLE_DEVICES` on MUSA

## Installation

```bash
pip install torchada

# Or install from source
git clone https://github.com/MooreThreads/torchada.git
cd torchada
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torchada  # Import once to apply patches - that's it!
import torch

# Platform detection (sglang-style)
def _is_cuda():
    return torch.version.cuda is not None

def _is_musa():
    return hasattr(torch.version, 'musa') and torch.version.musa is not None

# Check for GPU availability (works on both CUDA and MUSA)
if _is_cuda() or _is_musa():
    # Use standard torch.cuda APIs - they work transparently on MUSA:
    device = torch.device("cuda")  # Creates musa device on MUSA platform
    tensor = torch.randn(10, 10).cuda()  # Moves to MUSA on MUSA platform
    model = MyModel().cuda()

    # All torch.cuda.* APIs work transparently
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    torch.cuda.synchronize()
```

### Building C++ Extensions

```python
# setup.py - Use standard torch imports!
import torchada  # Import first to apply patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

print(f"Building with CUDA/MUSA home: {CUDA_HOME}")

ext_modules = [
    CUDAExtension(
        name="my_extension",
        sources=[
            "my_extension.cpp",
            "my_extension_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],  # Automatically mapped to mcc on MUSA
        },
    ),
]

setup(
    name="my_package",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
```

### JIT Compilation

```python
import torchada  # Import first to apply patches
from torch.utils.cpp_extension import load

# Load extension at runtime (works on both CUDA and MUSA)
my_extension = load(
    name="my_extension",
    sources=["my_extension.cpp", "my_extension_kernel.cu"],
    verbose=True,
)
```

### Mixed Precision Training

```python
import torchada  # Import first to apply patches
import torch

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler()

for data, target in dataloader:
    data, target = data.cuda(), target.cuda()

    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Distributed Training

```python
import torchada  # Import first to apply patches
import torch.distributed as dist

# Use 'nccl' backend as usual - torchada maps it to 'mccl' on MUSA
dist.init_process_group(backend='nccl')
```

### CUDA Graphs

```python
import torchada  # Import first to apply patches
import torch

# Use standard torch.cuda.CUDAGraph - works on MUSA too
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = model(x)
```

## Platform Detection

torchada automatically detects the platform:

```python
import torchada
from torchada import detect_platform, Platform

platform = detect_platform()
if platform == Platform.MUSA:
    print("Running on Moore Threads GPU")
elif platform == Platform.CUDA:
    print("Running on NVIDIA GPU")

# Or use torch.version-based detection (sglang-style)
def _is_musa():
    return hasattr(torch.version, 'musa') and torch.version.musa is not None

if _is_musa():
    print("MUSA platform detected")
```

## What Gets Patched

After `import torchada`, the following standard PyTorch APIs work on MUSA:

| Standard API | Description |
|--------------|-------------|
| `torch.cuda.*` | All APIs redirected to `torch.musa` |
| `torch.cuda.amp.*` | autocast, GradScaler |
| `torch.cuda.CUDAGraph` | Maps to MUSAGraph |
| `torch.cuda.nccl` | Maps to `torch.musa.mccl` |
| `torch.cuda.nvtx` | No-op stub (MUSA doesn't have nvtx) |
| `torch.cuda._lazy_call` | Patched for lazy initialization |
| `torch.distributed` (backend='nccl') | Automatically uses MCCL |
| `torch.device("cuda")` | Creates MUSA device on MUSA platform |
| `tensor.cuda()` | Moves to MUSA device |
| `tensor.is_cuda` | Returns True for MUSA tensors |
| `model.cuda()` | Moves model to MUSA device |
| `torch.amp.autocast(device_type='cuda')` | Uses 'musa' device type |
| `torch.utils.cpp_extension.*` | CUDAExtension, BuildExtension, CUDA_HOME |
| `torch._inductor.autotune_process` | Uses MUSA_VISIBLE_DEVICES |

## API Reference

### torchada

| Function | Description |
|----------|-------------|
| `detect_platform()` | Returns the detected platform (CUDA, MUSA, or CPU) |
| `is_musa_platform()` | Check if running on MUSA |
| `is_cuda_platform()` | Check if running on CUDA |
| `is_cpu_platform()` | Check if running on CPU only |
| `get_device_name()` | Get device name string ("cuda", "musa", or "cpu") |
| `get_platform()` | Alias for `detect_platform()` |
| `get_backend()` | Get the underlying torch device module |
| `is_patched()` | Check if patches have been applied |
| `get_version()` | Get torchada version string |
| `is_gpu_device(device)` | Check if device is CUDA or MUSA (see below) |
| `is_cuda_like_device(device)` | Alias for `is_gpu_device()` |
| `CUDA_HOME` | Path to CUDA/MUSA installation |

### torch.cuda (after importing torchada)

All standard `torch.cuda` APIs work, including:
- `device_count()`, `current_device()`, `set_device()`, `get_device_name()`
- `memory_allocated()`, `memory_reserved()`, `empty_cache()`, `reset_peak_memory_stats()`
- `synchronize()`, `Stream`, `Event`, `CUDAGraph`
- `amp.autocast()`, `amp.GradScaler()`
- `_lazy_call()`, `_lazy_init()`

**Note**: `torch.cuda.is_available()` is intentionally NOT redirected. It returns `False` on MUSA to allow proper platform detection. Use `hasattr(torch.version, 'musa') and torch.version.musa is not None` or `torch.musa.is_available()` instead.

### torch.utils.cpp_extension (after importing torchada)

| Symbol | Description |
|--------|-------------|
| `CUDAExtension` | Creates CUDA or MUSA extension based on platform |
| `CppExtension` | Creates C++ extension (no GPU code) |
| `BuildExtension` | Build command for extensions |
| `CUDA_HOME` | Path to CUDA/MUSA installation |
| `load()` | JIT compile and load extension |

## Symbol Mapping

torchada automatically maps CUDA symbols to MUSA equivalents when building extensions:

| CUDA | MUSA |
|------|------|
| `cudaMalloc` | `musaMalloc` |
| `cudaMemcpy` | `musaMemcpy` |
| `cudaStream_t` | `musaStream_t` |
| `cublasHandle_t` | `mublasHandle_t` |
| `curandState` | `murandState` |
| `at::cuda` | `at::musa` |
| `c10::cuda` | `c10::musa` |
| `cutlass::*` | `mutlass::*` |
| `#include <cuda/*>` | `#include <musa/*>` |
| ... | ... |

See `src/torchada/_mapping.py` for the complete mapping table (380+ mappings).

**Note**: Many CUDA constructs like atomic operations (`atomicAdd`, `atomicCAS`), shuffle intrinsics (`__shfl_sync`), and half-precision math (`__float2half`) are identical in MUSA and don't require mapping.

## Known Limitations

### Device Type String Comparisons

On MUSA platform, `torch.device("cuda")` is translated to a device with type `"musa"`, not `"cuda"`. This means direct string comparisons like `device.type == "cuda"` will fail.

**Problem:**
```python
device = torch.device("cuda:0")  # On MUSA, this becomes musa:0
if device.type == "cuda":  # Returns False on MUSA!
    ...
```

**Solution:** Use `torchada.is_gpu_device()` for portable code:
```python
import torchada

# Works on both CUDA and MUSA platforms
if torchada.is_gpu_device(device):
    ...

# Also works with tensors
if torchada.is_gpu_device(tensor):
    ...
```

This is a fundamental limitation because `device.type` is a C-level property that cannot be patched from Python. Downstream projects that check `device.type == "cuda"` need to be patched to use `torchada.is_gpu_device()` or check for both types: `device.type in ("cuda", "musa")`.

## Real-World Integrations

torchada has been successfully integrated into several popular PyTorch-based projects. Below are examples demonstrating the typical integration patterns.

### Integrated Projects

| Project | Category | PR | Status |
|---------|----------|--------|--------|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Image/Video Generation | [#11618](https://github.com/comfyanonymous/ComfyUI/pull/11618) | Open |
| [LightLLM](https://github.com/ModelTC/LightLLM) | LLM Inference | [#1162](https://github.com/ModelTC/LightLLM/pull/1162) | ✅ Merged |
| [Xinference](https://github.com/xorbitsai/inference) | Model Serving | [#4425](https://github.com/xorbitsai/inference/pull/4425) | ✅ Merged |
| [LightX2V](https://github.com/ModelTC/LightX2V) | Image/Video Generation | [#678](https://github.com/ModelTC/LightX2V/pull/678) | ✅ Merged |

### Integration Patterns

#### Pattern 1: Early Import with Platform Detection

The most common pattern is to import `torchada` early in the application lifecycle:

```python
# In __init__.py or main entry point
from your_app.device_utils import is_musa

if is_musa():
    import torchada  # noqa: F401

# Platform detection function
def is_musa():
    import torch
    return hasattr(torch.version, "musa") and torch.version.musa is not None
```

This pattern is used by **LightLLM** and **LightX2V**.

#### Pattern 2: Add to Dependencies

Add `torchada` to your project's dependencies:

```python
# pyproject.toml
dependencies = [
    "torchada>=0.1.11",
]

# Or requirements.txt
torchada>=0.1.11
```

#### Pattern 3: Device Availability Check

Create a device availability function that checks for MUSA:

```python
def is_musa_available() -> bool:
    try:
        import torch
        import torch_musa  # noqa: F401
        import torchada  # noqa: F401
        return torch.musa.is_available()
    except ImportError:
        return False

def get_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif is_musa_available():
        return "musa"
    return "cpu"
```

This pattern is used by **Xinference**.

#### Pattern 4: Platform-Specific Feature Flags

Enable features based on platform capabilities:

```python
import torchada

musa_available = hasattr(torch, "musa") and torch.musa.is_available()

def is_musa():
    return musa_available

# Enable NVIDIA-like optimizations on MUSA
if is_nvidia() or is_musa():
    ENABLE_PYTORCH_ATTENTION = True
    NUM_STREAMS = 2  # Async weight offloading
    MAX_PINNED_MEMORY = get_total_memory(torch.device("cpu")) * 0.9
```

This pattern is used by **ComfyUI**.

#### Pattern 5: Platform Device Classes

For projects with a device abstraction layer:

```python
from your_platform.base.nvidia import CudaDevice
from your_platform.registry import PLATFORM_DEVICE_REGISTER

@PLATFORM_DEVICE_REGISTER("musa")
class MusaDevice(CudaDevice):
    name = "cuda"  # Use CUDA APIs (redirected by torchada)

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
            import torchada  # noqa: F401
            return hasattr(torch, "musa") and torch.musa.is_available()
        except ImportError:
            return False
```

This pattern is used by **LightX2V**.

### Common Integration Steps

1. **Add dependency**: Add `torchada>=0.1.11` to your project dependencies

2. **Import early**: Import `torchada` before using any `torch.cuda` APIs
   ```python
   import torchada  # Apply patches
   import torch
   ```

3. **Add platform detection**: Create `is_musa()` function for platform-specific code
   ```python
   def is_musa():
       return hasattr(torch.version, "musa") and torch.version.musa is not None
   ```

4. **Update feature flags**: Include MUSA in capability checks
   ```python
   if is_nvidia() or is_musa():
       # Enable GPU-specific features
   ```

5. **Handle device type checks**: Use `torchada.is_gpu_device()` or check both types
   ```python
   # Instead of: device.type == "cuda"
   # Use: device.type in ("cuda", "musa")
   # Or: torchada.is_gpu_device(device)
   ```

## Architecture

torchada uses a decorator-based patch registration system:

```python
from torchada._patch import patch_function, requires_import

@patch_function  # Registers for automatic application
@requires_import('torch_musa')  # Guards against missing dependencies
def _patch_my_feature():
    # Patching logic here
    pass
```

All registered patches are applied automatically when you `import torchada`.

## License

MIT License