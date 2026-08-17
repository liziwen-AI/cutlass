#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemvKernel_>
class GemvBlockScaled {
public:

  using GemvKernel = GemvKernel_;


  using ElementA = typename GemvKernel::ElementA;
  using LayoutA  = typename GemvKernel::LayoutA;
  using ElementB = typename GemvKernel::ElementB;
  using ElementC = typename GemvKernel::ElementC;

  using ElementSFA = typename GemvKernel::ElementSFA;
  using ElementSFB = typename GemvKernel::ElementSFB;

  using ElementAccumulator = typename GemvKernel::ElementAccumulator;
  using EpilogueOutputOp = typename GemvKernel::EpilogueOutputOp;

  static ComplexTransform const kTransformA = GemvKernel::kTransformA;
  static ComplexTransform const kTransformB = GemvKernel::kTransformB;

  static int const kThreadCount = GemvKernel::kThreadCount;
  static int const kThreadsPerRow = GemvKernel::kThreadsPerRow;

  using Arguments = typename GemvKernel::Arguments;
  using Params = typename GemvKernel::Params;

private:

  Params params_;

public:

  /// Constructs the GemvBlockScaled.
  GemvBlockScaled() = default;

  /// Determines whether the GemvBlockScaled can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return GemvKernel::can_implement(args);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    return 0;
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args, dim3 const &block) { 
    if(platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      return dim3((args.problem_size.row() + (block.x - 1)) / block.x, 1, args.batch_count % 65536);
    }
    else {
      return dim3((args.problem_size.row() + (block.y - 1)) / block.y, 1, args.batch_count % 65536);
    }
  }

  /// Computes the block shape
  static dim3 get_block_shape() { 
    if(platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      return dim3(kThreadCount, 1, 1);
    }
    else {
      return dim3(kThreadsPerRow, kThreadCount / kThreadsPerRow, 1);
    }
  }

  /// Initializes GemvBlockScaled state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    params_ = Params(args);
    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    return params_.update(args);    
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {
    const dim3 block = get_block_shape();
    const dim3 grid = get_grid_shape(params_, block);

    int smem_size = int(sizeof(typename GemvKernel::SharedStorage));

    cutlass::arch::synclog_setup();
    cutlass::Kernel<GemvKernel><<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();
    if (result == cudaSuccess) {
        return Status::kSuccess;
    } else {
        return Status::kErrorInternal;
    }
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr) {
    
    Status status = initialize(args, workspace, stream);
    
    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
