#pragma once

// common
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/mma.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/kernel_launch.h"
#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

// 2.x
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

// 3.x
#include "cutlass/gemm/kernel/gemm_universal.hpp"


namespace cutlass::gemm::device {

template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;


template <class GemmKernel_>
class GemmUniversalAdapter<GemmKernel_, void>
{
  public:
    using Arguments = typename GemmKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename GemmKernel::Params;

  private:
    Params params_;

  public:
    /// Access the Params structure
    Params const& params() const {
      return params_;
    }

    /// Gets the workspace size
    static size_t
    get_workspace_size(Arguments const& args) {
      size_t workspace_bytes = 0;
      if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
        workspace_bytes += sizeof(int) * size_t(cute::size<0>(TileShape{})) * size_t(cute::size<1>(TileShape{}));
      }
      workspace_bytes += GemmKernel::get_workspace_size(args);
      return workspace_bytes;
    }

    /// Computes the grid shape
    static dim3
    get_grid_shape(Arguments const& args, void* workspace = nullptr) {
      auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
      return GemmKernel::get_grid_shape(tmp_params);
    }

    /// Computes the grid shape
    static dim3
    get_grid_shape(Params const& params) {
      return GemmKernel::get_grid_shape(params);
    }

    /// Initializes GEMM state from arguments.
    Status
    initialize(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {

      Status status = GemmKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
      params_ = GemmKernel::to_underlying_arguments(args, workspace);
      int smem_size = GemmKernel::SharedStorageSize;

      if (smem_size >= (48 << 10)) {
        cudaError_t result = cudaFuncSetAttribute(
            device_kernel<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
      }
      return Status::kSuccess;
    }
    
    static Status
    run(Params& params,
        cudaStream_t stream = nullptr,
        CudaHostAdapter *cuda_adapter = nullptr,
        bool launch_with_pdl = false) {
      dim3 const block = GemmKernel::get_block_shape();
      dim3 const grid = get_grid_shape(params);

      // configure smem size and carveout
      int smem_size = GemmKernel::SharedStorageSize;

      Status launch_result{ Status::kSuccess };
      launch_result = cutlass::kernel_launch<GemmKernel>(grid, block, smem_size, stream, params, launch_with_pdl);     
          
      return Status::kSuccess;
    }

    Status
    run(
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr,
      bool launch_with_pdl = false) {
      return run(params_, stream, cuda_adapter, launch_with_pdl);
    }

};

} // namespace cutlass::gemm::device
