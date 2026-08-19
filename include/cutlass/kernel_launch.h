#pragma once

#include <cuda_runtime_api.h>
#include "cutlass/cutlass.h"
#include "cutlass/trace.h"
#include "cutlass/device_kernel.h" 

namespace cutlass {

template <typename GemmKernel, typename Params>
Status kernel_launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    cudaStream_t cuda_stream,
    const Params &kernel_params,
    bool launch_with_pdl) {
    device_kernel<GemmKernel><<<grid_dims, block_dims, smem_size, cuda_stream>>>(kernel_params);
  } 
}