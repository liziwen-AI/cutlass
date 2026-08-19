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

namespace detail {

template <class DispatchPolicy, class Enable = void>
struct has_Stages : cute::false_type {};

template <class DispatchPolicy>
struct has_Stages<DispatchPolicy, cute::void_t<decltype(DispatchPolicy::Stages)>> : cute::true_type {};

template<class DispatchPolicy>
constexpr int stages_member(DispatchPolicy) {
  if constexpr (has_Stages<DispatchPolicy>::value) {
    return DispatchPolicy::Stages;
  }
  else {
    return 0;
  }
}

} // namespace detail

template <class GemmKernel_>
class GemmUniversalAdapter<GemmKernel_,cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<GemmKernel_>>::value>>
{
  public:
    using GemmKernel = GetUnderlyingKernel_t<GemmKernel_>;
    using TileShape = typename GemmKernel::TileShape;
    using ElementA = typename GemmKernel::ElementA;
    using ElementB = typename GemmKernel::ElementB;
    using ElementC = typename GemmKernel::ElementC;
    using ElementD = typename GemmKernel::ElementD;
    using ElementAccumulator = typename GemmKernel::ElementAccumulator;
    using DispatchPolicy = typename GemmKernel::DispatchPolicy;
    using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
    using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

    // Map back to 2.x type as best as possible
    using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
    using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
    using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
    using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

    using OperatorClass = cutlass::detail::get_operator_class_t<typename CollectiveMainloop::TiledMma>;


    // Legacy: provide a correct warp count, but no reliable warp shape
    static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

    static constexpr int WarpsInMma = cute::max(4, CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);
    static constexpr int WarpsInMmaM = 4;
    static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
    using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
    using WarpShape = cutlass::gemm::GemmShape<
        CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
        CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
        CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;


    using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

    // Split-K preserves splits that are 128b aligned
    static int constexpr kSplitKAlignment = cute::max(
        128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

    /// Argument structure: User API
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
