#pragma once

#include <cuda_runtime_api.h>
#include "cutlass/cutlass.h"
#include "cutlass/trace.h"
#include "cutlass/device_kernel.h" // cutlass::device_kernel

namespace cutlass {

struct KernelLaunchConfiguration {

  dim3 grid;
  dim3 block;
  size_t dynamic_smem;

  CUTLASS_HOST_DEVICE
  KernelLaunchConfiguration(
    dim3 _grid = dim3(1,1,1),
    dim3 _block = dim3(1,1,1),
    size_t _dynamic_smem = 0
  ):
    grid(_grid),
    block(_block),
    dynamic_smem(_dynamic_smem) { }
};


template <typename GemmKernel, typename Params>
Status kernel_launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    const Params &kernel_params,
    bool launch_with_pdl) {

  if (not launch_with_pdl) {
    device_kernel<GemmKernel><<<grid_dims, block_dims, smem_size, cuda_stream>>>(kernel_params);
  }
  else {}

  cudaError_t result = cudaGetLastError();
  if (cudaSuccess == result) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
    CUTLASS_TRACE_HOST("cutlass::kernel_launch: cudaGetLastError reports success");
#endif
    return Status::kSuccess;
  }
  else {
    CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
    return Status::kErrorInternal;
  }
}

} // namespace cutlass
