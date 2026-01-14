"""
Tests for CUDA to MUSA mapping rules.
"""

import pytest


class TestMappingImports:
    """Test mapping module imports."""

    def test_import_mapping_rule(self):
        """Test _MAPPING_RULE can be imported."""
        from torchada._mapping import _MAPPING_RULE

        assert isinstance(_MAPPING_RULE, dict)
        assert len(_MAPPING_RULE) > 0

    def test_import_ext_replaced_mapping(self):
        """Test EXT_REPLACED_MAPPING can be imported."""
        from torchada._mapping import EXT_REPLACED_MAPPING

        assert isinstance(EXT_REPLACED_MAPPING, dict)


class TestATenMappings:
    """Test ATen CUDA to MUSA mappings."""

    def test_aten_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["at::cuda"] == "at::musa"

    def test_aten_cuda_includes(self):
        from torchada._mapping import _MAPPING_RULE

        # Check for specific ATen include mappings
        assert "#include <ATen/cuda/CUDAContext.h>" in _MAPPING_RULE


class TestC10Mappings:
    """Test C10 CUDA to MUSA mappings."""

    def test_c10_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["c10::cuda"] == "c10::musa"
        assert _MAPPING_RULE["c10/cuda"] == "c10/musa"

    def test_c10_device_type(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["c10::DeviceType::CUDA"] == "c10::DeviceType::MUSA"


class TestTorchMappings:
    """Test torch namespace mappings."""

    def test_torch_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["torch::cuda"] == "torch::musa"
        assert _MAPPING_RULE["torch.cuda"] == "torch.musa"

    def test_torch_device_type(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["at::kCUDA"] == "at::kMUSA"
        assert _MAPPING_RULE["at::DeviceType::CUDA"] == "at::DeviceType::MUSA"
        # torch::kCUDA maps to PrivateUse1 for MUSA compatibility
        assert _MAPPING_RULE["torch::kCUDA"] == "c10::DeviceType::PrivateUse1"


class TestCuBLASMappings:
    """Test cuBLAS to muBLAS mappings."""

    def test_cublas_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublas"] == "mublas"
        assert _MAPPING_RULE["CUBLAS"] == "MUBLAS"

    def test_cublas_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasHandle_t"] == "mublasHandle_t"
        assert _MAPPING_RULE["cublasStatus_t"] == "mublasStatus_t"

    def test_cublas_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasCreate"] == "mublasCreate"
        assert _MAPPING_RULE["cublasDestroy"] == "mublasDestroy"
        assert _MAPPING_RULE["cublasSetStream"] == "mublasSetStream"
        assert _MAPPING_RULE["cublasGetStream"] == "mublasGetStream"

    def test_cublas_gemm(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasSgemm"] == "mublasSgemm"
        assert _MAPPING_RULE["cublasDgemm"] == "mublasDgemm"
        assert _MAPPING_RULE["cublasHgemm"] == "mublasHgemm"
        assert _MAPPING_RULE["cublasGemmEx"] == "mublasGemmEx"

    def test_cublas_batched(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasGemmBatchedEx"] == "mublasGemmBatchedEx"
        assert _MAPPING_RULE["cublasGemmStridedBatchedEx"] == "mublasGemmStridedBatchedEx"
        assert _MAPPING_RULE["cublasSgemmBatched"] == "mublasSgemmBatched"
        assert _MAPPING_RULE["cublasDgemmBatched"] == "mublasDgemmBatched"

    def test_cublaslt(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasLtCreate"] == "mublasLtCreate"
        assert _MAPPING_RULE["cublasLtDestroy"] == "mublasLtDestroy"
        assert _MAPPING_RULE["cublasLtHandle_t"] == "mublasLtHandle_t"
        assert _MAPPING_RULE["cublasLtMatmul"] == "mublasLtMatmul"


class TestCuRANDMappings:
    """Test cuRAND to muRAND mappings."""

    def test_curand_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["curand"] == "murand"
        assert _MAPPING_RULE["CURAND"] == "MURAND"

    def test_curand_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["curandState"] == "murandState"
        assert _MAPPING_RULE["curandStatePhilox4_32_10_t"] == "murandStatePhilox4_32_10_t"

    def test_curand_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["curand_init"] == "murand_init"
        assert _MAPPING_RULE["curand_uniform"] == "murand_uniform"
        assert _MAPPING_RULE["curand_uniform4"] == "murand_uniform4"
        assert _MAPPING_RULE["curand_normal"] == "murand_normal"
        assert _MAPPING_RULE["curand_normal4"] == "murand_normal4"


class TestCuDNNMappings:
    """Test cuDNN to muDNN mappings."""

    def test_cudnn_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudnn"] == "mudnn"
        assert _MAPPING_RULE["CUDNN"] == "MUDNN"

    def test_cudnn_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudnnHandle_t"] == "mudnnHandle_t"
        assert _MAPPING_RULE["cudnnStatus_t"] == "mudnnStatus_t"

    def test_cudnn_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudnnCreate"] == "mudnnCreate"
        assert _MAPPING_RULE["cudnnDestroy"] == "mudnnDestroy"
        assert _MAPPING_RULE["cudnnSetStream"] == "mudnnSetStream"


class TestCUDARuntimeMappings:
    """Test CUDA runtime to MUSA runtime mappings."""

    def test_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaMalloc"] == "musaMalloc"
        assert _MAPPING_RULE["cudaFree"] == "musaFree"
        assert _MAPPING_RULE["cudaMemcpy"] == "musaMemcpy"
        assert _MAPPING_RULE["cudaMemcpyAsync"] == "musaMemcpyAsync"
        assert _MAPPING_RULE["cudaMemset"] == "musaMemset"
        assert _MAPPING_RULE["cudaMemsetAsync"] == "musaMemsetAsync"

    def test_host_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaHostAlloc"] == "musaHostAlloc"
        assert _MAPPING_RULE["cudaHostFree"] == "musaHostFree"
        assert _MAPPING_RULE["cudaMallocHost"] == "musaMallocHost"
        assert _MAPPING_RULE["cudaFreeHost"] == "musaFreeHost"
        assert _MAPPING_RULE["cudaMallocManaged"] == "musaMallocManaged"

    def test_async_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaMallocAsync"] == "musaMallocAsync"
        assert _MAPPING_RULE["cudaFreeAsync"] == "musaFreeAsync"
        assert _MAPPING_RULE["cudaMemcpy2D"] == "musaMemcpy2D"
        assert _MAPPING_RULE["cudaMemcpy2DAsync"] == "musaMemcpy2DAsync"
        assert _MAPPING_RULE["cudaMemcpy3D"] == "musaMemcpy3D"
        assert _MAPPING_RULE["cudaMemcpy3DAsync"] == "musaMemcpy3DAsync"

    def test_device_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaDeviceSynchronize"] == "musaDeviceSynchronize"
        assert _MAPPING_RULE["cudaGetDevice"] == "musaGetDevice"
        assert _MAPPING_RULE["cudaSetDevice"] == "musaSetDevice"
        assert _MAPPING_RULE["cudaGetDeviceCount"] == "musaGetDeviceCount"
        assert _MAPPING_RULE["cudaGetDeviceProperties"] == "musaGetDeviceProperties"
        assert _MAPPING_RULE["cudaDeviceGetAttribute"] == "musaDeviceGetAttribute"

    def test_memcpy_kinds(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaMemcpyHostToDevice"] == "musaMemcpyHostToDevice"
        assert _MAPPING_RULE["cudaMemcpyDeviceToHost"] == "musaMemcpyDeviceToHost"
        assert _MAPPING_RULE["cudaMemcpyDeviceToDevice"] == "musaMemcpyDeviceToDevice"
        assert _MAPPING_RULE["cudaMemcpyHostToHost"] == "musaMemcpyHostToHost"


class TestStreamEventMappings:
    """Test CUDA stream/event mappings."""

    def test_stream_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaStream_t"] == "musaStream_t"
        assert _MAPPING_RULE["cudaEvent_t"] == "musaEvent_t"

    def test_stream_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaStreamCreate"] == "musaStreamCreate"
        assert _MAPPING_RULE["cudaStreamDestroy"] == "musaStreamDestroy"
        assert _MAPPING_RULE["cudaStreamSynchronize"] == "musaStreamSynchronize"
        assert _MAPPING_RULE["cudaStreamQuery"] == "musaStreamQuery"
        assert _MAPPING_RULE["cudaStreamWaitEvent"] == "musaStreamWaitEvent"

    def test_stream_flags(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaStreamDefault"] == "musaStreamDefault"
        assert _MAPPING_RULE["cudaStreamNonBlocking"] == "musaStreamNonBlocking"
        assert _MAPPING_RULE["cudaStreamCreateWithFlags"] == "musaStreamCreateWithFlags"
        assert _MAPPING_RULE["cudaStreamCreateWithPriority"] == "musaStreamCreateWithPriority"

    def test_event_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaEventCreate"] == "musaEventCreate"
        assert _MAPPING_RULE["cudaEventDestroy"] == "musaEventDestroy"
        assert _MAPPING_RULE["cudaEventRecord"] == "musaEventRecord"
        assert _MAPPING_RULE["cudaEventSynchronize"] == "musaEventSynchronize"
        assert _MAPPING_RULE["cudaEventElapsedTime"] == "musaEventElapsedTime"
        assert _MAPPING_RULE["cudaEventQuery"] == "musaEventQuery"

    def test_event_flags(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaEventDefault"] == "musaEventDefault"
        assert _MAPPING_RULE["cudaEventBlockingSync"] == "musaEventBlockingSync"
        assert _MAPPING_RULE["cudaEventDisableTiming"] == "musaEventDisableTiming"
        assert _MAPPING_RULE["cudaEventCreateWithFlags"] == "musaEventCreateWithFlags"


class TestErrorHandlingMappings:
    """Test CUDA error handling mappings."""

    def test_error_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaError_t"] == "musaError_t"
        assert _MAPPING_RULE["cudaSuccess"] == "musaSuccess"

    def test_error_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaGetLastError"] == "musaGetLastError"
        assert _MAPPING_RULE["cudaGetErrorString"] == "musaGetErrorString"
        assert _MAPPING_RULE["cudaPeekAtLastError"] == "musaPeekAtLastError"


class TestNCCLMappings:
    """Test NCCL to MCCL mappings."""

    def test_nccl_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["nccl"] == "mccl"
        assert _MAPPING_RULE["NCCL"] == "MCCL"

    def test_nccl_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["ncclComm_t"] == "mcclComm_t"
        assert _MAPPING_RULE["ncclDataType_t"] == "mcclDataType_t"
        assert _MAPPING_RULE["ncclRedOp_t"] == "mcclRedOp_t"
        assert _MAPPING_RULE["ncclResult_t"] == "mcclResult_t"
        assert _MAPPING_RULE["ncclSuccess"] == "mcclSuccess"
        assert _MAPPING_RULE["ncclUniqueId"] == "mcclUniqueId"

    def test_nccl_comm_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["ncclCommInitRank"] == "mcclCommInitRank"
        assert _MAPPING_RULE["ncclCommInitAll"] == "mcclCommInitAll"
        assert _MAPPING_RULE["ncclCommDestroy"] == "mcclCommDestroy"
        assert _MAPPING_RULE["ncclCommCount"] == "mcclCommCount"
        assert _MAPPING_RULE["ncclCommCuDevice"] == "mcclCommCuDevice"
        assert _MAPPING_RULE["ncclCommUserRank"] == "mcclCommUserRank"
        assert _MAPPING_RULE["ncclGetUniqueId"] == "mcclGetUniqueId"

    def test_nccl_collective_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["ncclAllReduce"] == "mcclAllReduce"
        assert _MAPPING_RULE["ncclBroadcast"] == "mcclBroadcast"
        assert _MAPPING_RULE["ncclReduce"] == "mcclReduce"
        assert _MAPPING_RULE["ncclAllGather"] == "mcclAllGather"
        assert _MAPPING_RULE["ncclReduceScatter"] == "mcclReduceScatter"
        assert _MAPPING_RULE["ncclSend"] == "mcclSend"
        assert _MAPPING_RULE["ncclRecv"] == "mcclRecv"
        assert _MAPPING_RULE["ncclGroupStart"] == "mcclGroupStart"
        assert _MAPPING_RULE["ncclGroupEnd"] == "mcclGroupEnd"


class TestLibraryMappings:
    """Test other library mappings."""

    def test_cusparse(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cusparse"] == "musparse"
        assert _MAPPING_RULE["CUSPARSE"] == "MUSPARSE"
        assert _MAPPING_RULE["cusparseHandle_t"] == "musparseHandle_t"
        assert _MAPPING_RULE["cusparseCreate"] == "musparseCreate"
        assert _MAPPING_RULE["cusparseDestroy"] == "musparseDestroy"

    def test_cusolver(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cusolver"] == "musolver"
        assert _MAPPING_RULE["CUSOLVER"] == "MUSOLVER"
        assert _MAPPING_RULE["cusolverDnHandle_t"] == "musolverDnHandle_t"
        assert _MAPPING_RULE["cusolverDnCreate"] == "musolverDnCreate"
        assert _MAPPING_RULE["cusolverDnDestroy"] == "musolverDnDestroy"

    def test_cufft(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cufft"] == "mufft"
        assert _MAPPING_RULE["CUFFT"] == "MUFFT"
        assert _MAPPING_RULE["cufftHandle"] == "mufftHandle"
        assert _MAPPING_RULE["cufftPlan1d"] == "mufftPlan1d"
        assert _MAPPING_RULE["cufftPlan2d"] == "mufftPlan2d"
        assert _MAPPING_RULE["cufftPlan3d"] == "mufftPlan3d"
        assert _MAPPING_RULE["cufftExecC2C"] == "mufftExecC2C"
        assert _MAPPING_RULE["cufftExecR2C"] == "mufftExecR2C"
        assert _MAPPING_RULE["cufftExecC2R"] == "mufftExecC2R"

    def test_cutlass(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cutlass"] == "mutlass"
        assert _MAPPING_RULE["CUTLASS"] == "MUTLASS"
        assert _MAPPING_RULE["cutlass/"] == "mutlass/"
        assert _MAPPING_RULE["cutlass::"] == "mutlass::"

    def test_cub(self):
        """CUB is provided directly by MUSA, no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        # CUB doesn't need mapping - MUSA provides it directly
        assert "cub::" not in _MAPPING_RULE
        assert "cub/" not in _MAPPING_RULE

    def test_thrust(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["thrust::cuda"] == "thrust::musa"


class TestIntrinsicMappings:
    """Test CUDA intrinsic and math function mappings.

    Note: Many CUDA intrinsics (shuffle, vote, sync, atomics, half precision)
    are the same in MUSA and don't require mapping. These tests verify that
    we correctly do NOT include identity mappings for these.
    """

    def test_shuffle_intrinsics_not_mapped(self):
        """Shuffle intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        # These should NOT be in the mapping (same syntax in MUSA)
        assert "__shfl_sync" not in _MAPPING_RULE
        assert "__shfl_xor_sync" not in _MAPPING_RULE

    def test_vote_intrinsics_not_mapped(self):
        """Vote intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "__ballot_sync" not in _MAPPING_RULE
        assert "__any_sync" not in _MAPPING_RULE

    def test_sync_intrinsics_not_mapped(self):
        """Sync intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "__syncthreads" not in _MAPPING_RULE
        assert "__threadfence" not in _MAPPING_RULE

    def test_atomic_operations_not_mapped(self):
        """Atomic operations are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "atomicAdd" not in _MAPPING_RULE
        assert "atomicCAS" not in _MAPPING_RULE

    def test_half_precision_not_mapped(self):
        """Half precision intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "__float2half" not in _MAPPING_RULE
        assert "__half2float" not in _MAPPING_RULE
        assert "__hadd" not in _MAPPING_RULE


class TestIncludeMappings:
    """Test header include mappings."""

    def test_cuda_headers(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cuda_runtime.h"] == "musa_runtime.h"
        assert _MAPPING_RULE["cuda_runtime_api.h"] == "musa_runtime_api.h"
        assert _MAPPING_RULE["cuda.h"] == "musa.h"
        assert _MAPPING_RULE["cuda_fp16.h"] == "musa_fp16.h"
        assert _MAPPING_RULE["cuda_bf16.h"] == "musa_bf16.h"


class TestPyTorchCppMappings:
    """Test PyTorch C++ API mappings."""

    def test_pytorch_stream_utils(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["getCurrentCUDAStream"] == "getCurrentMUSAStream"
        assert _MAPPING_RULE["getDefaultCUDAStream"] == "getDefaultMUSAStream"
        assert _MAPPING_RULE["CUDAStream"] == "MUSAStream"

    def test_pytorch_guards(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["CUDAGuard"] == "MUSAGuard"
        assert _MAPPING_RULE["OptionalCUDAGuard"] == "OptionalMUSAGuard"
        assert _MAPPING_RULE["CUDAStreamGuard"] == "MUSAStreamGuard"
        assert _MAPPING_RULE["CUDAEvent"] == "MUSAEvent"

    def test_pytorch_torch_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["torch::cuda::getCurrentCUDAStream"]
            == "torch::musa::getCurrentMUSAStream"
        )
        assert (
            _MAPPING_RULE["torch::cuda::getDefaultCUDAStream"]
            == "torch::musa::getDefaultMUSAStream"
        )
        assert _MAPPING_RULE["torch::cuda::getStreamFromPool"] == "torch::musa::getStreamFromPool"


class TestMappingCount:
    """Test total mapping count."""

    def test_mapping_count(self):
        from torchada._mapping import _MAPPING_RULE

        # We should have a substantial number of mappings
        assert len(_MAPPING_RULE) >= 250

    def test_ext_replaced_mapping(self):
        from torchada._mapping import EXT_REPLACED_MAPPING

        # Extensions are converted: .cu -> .mu, .cuh -> .muh for mcc compiler
        assert EXT_REPLACED_MAPPING["cu"] == "mu"
        assert EXT_REPLACED_MAPPING["cuh"] == "muh"


class TestMappingRobustness:
    """Tests to ensure mapping rules are robust and don't have issues."""

    def test_no_identity_mappings(self):
        """Ensure no mapping maps a key to itself (identity mapping)."""
        from torchada._mapping import _MAPPING_RULE

        identity_mappings = [(k, v) for k, v in _MAPPING_RULE.items() if k == v]
        assert len(identity_mappings) == 0, (
            f"Found {len(identity_mappings)} identity mappings that should be removed: "
            f"{identity_mappings[:5]}"
        )

    def test_no_empty_keys_or_values(self):
        """Ensure no mapping has empty key or value."""
        from torchada._mapping import _MAPPING_RULE

        for key, value in _MAPPING_RULE.items():
            assert key, "Found empty key in mapping"
            assert value is not None, f"Found None value for key: {key}"
            # Empty value is allowed for deletion, but we don't use that

    def test_all_keys_and_values_are_strings(self):
        """Ensure all keys and values are strings."""
        from torchada._mapping import _MAPPING_RULE

        for key, value in _MAPPING_RULE.items():
            assert isinstance(key, str), f"Key is not a string: {key}"
            assert isinstance(value, str), f"Value is not a string for key {key}: {value}"

    def test_cuda_to_musa_consistency(self):
        """Test that CUDA terms consistently map to MUSA equivalents."""
        from torchada._mapping import _MAPPING_RULE

        # Check that 'cuda' in key generally maps to 'musa' in value
        # (with some exceptions for special cases like torch::kCUDA -> PrivateUse1)
        exceptions = {
            "torch::kCUDA",  # Maps to PrivateUse1
            ".is_cuda()",  # Maps to .is_privateuseone()
        }

        for key, value in _MAPPING_RULE.items():
            if key in exceptions:
                continue
            # If key contains 'cuda' (case insensitive), value should contain 'musa'
            if "cuda" in key.lower() and "musa" not in value.lower():
                # Allow mappings where cuda -> privateuseone or similar
                if "privateuseone" not in value.lower() and "private" not in value.lower():
                    # Check for special patterns
                    if not any(
                        x in value.lower()
                        for x in [
                            "musa",
                            "privateuseone",
                            "ignore",
                            "torch_musa",
                        ]
                    ):
                        pytest.fail(
                            f"Inconsistent mapping: '{key}' -> '{value}' "
                            "(expected 'musa' or special case in value)"
                        )


class TestMappingApplication:
    """Test that mappings are correctly applied to source code."""

    def test_port_cuda_source_basic(self):
        """Test basic source code porting."""
        from torchada.utils.cpp_extension import _port_cuda_source

        # Test basic namespace replacement
        source = "at::cuda::getCurrentCUDAStream()"
        result = _port_cuda_source(source)
        assert "at::musa" in result
        assert "at::cuda" not in result

    def test_port_cuda_source_includes(self):
        """Test include statement porting."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = '#include <cuda_runtime.h>\n#include "my_file.cuh"'
        result = _port_cuda_source(source)
        assert "musa_runtime.h" in result
        assert 'my_file.muh"' in result

    def test_port_cuda_source_types(self):
        """Test type replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = "cudaStream_t stream; cudaError_t err;"
        result = _port_cuda_source(source)
        assert "musaStream_t" in result
        assert "musaError_t" in result
        assert "cudaStream_t" not in result

    def test_port_cuda_source_functions(self):
        """Test function name replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = "cudaMalloc(&ptr, size); cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);"
        result = _port_cuda_source(source)
        assert "musaMalloc" in result
        assert "musaMemcpy" in result
        assert "cudaMalloc" not in result

    def test_port_cuda_source_preserves_non_cuda(self):
        """Test that non-CUDA code is preserved."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = """
int main() {
    int x = 42;
    float y = 3.14f;
    return 0;
}
"""
        result = _port_cuda_source(source)
        assert "int main()" in result
        assert "int x = 42" in result
        assert "float y = 3.14f" in result

    def test_port_cuda_source_longer_patterns_first(self):
        """Test that longer patterns are applied before shorter ones."""
        from torchada.utils.cpp_extension import _port_cuda_source

        # This tests that 'cudaMemcpyHostToDevice' is replaced before 'cuda'
        source = "cudaMemcpyHostToDevice"
        result = _port_cuda_source(source)
        # Should be musaMemcpyHostToDevice, not something like musaMemcpyHostToDevice
        assert result == "musaMemcpyHostToDevice"

    def test_port_cuda_source_c10_macros(self):
        """Test C10 macro replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = "C10_CUDA_KERNEL_LAUNCH_CHECK();"
        result = _port_cuda_source(source)
        assert "C10_MUSA_KERNEL_LAUNCH_CHECK" in result
        assert "C10_CUDA" not in result

    def test_port_cuda_source_nccl(self):
        """Test NCCL to MCCL replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = (
            "ncclComm_t comm; ncclAllReduce(buffer, buffer, count, datatype, op, comm, stream);"
        )
        result = _port_cuda_source(source)
        assert "mcclComm_t" in result
        assert "mcclAllReduce" in result
        assert "ncclComm_t" not in result


class TestMappingSubstringOrdering:
    """Test that mappings with substring relationships are handled correctly."""

    def test_specific_before_generic(self):
        """Test that specific mappings work correctly with generic ones."""
        from torchada._mapping import _MAPPING_RULE

        # The _port_cuda_source function sorts by length (longest first)
        # So specific mappings like 'cudaMemcpyHostToDevice' should be in the mapping
        # and the replacement algorithm should handle them correctly
        # Verify that both specific and generic patterns exist
        assert "cudaMemcpy" in _MAPPING_RULE
        assert "cudaMemcpyHostToDevice" in _MAPPING_RULE

        # Verify the specific one maps correctly
        assert _MAPPING_RULE["cudaMemcpyHostToDevice"] == "musaMemcpyHostToDevice"

    def test_include_specific_paths(self):
        """Test that specific include paths are correctly mapped."""
        from torchada._mapping import _MAPPING_RULE

        # Check that specific ATen includes exist
        assert "#include <ATen/cuda/CUDAContext.h>" in _MAPPING_RULE

        # And that generic at::cuda also exists
        assert "at::cuda" in _MAPPING_RULE
