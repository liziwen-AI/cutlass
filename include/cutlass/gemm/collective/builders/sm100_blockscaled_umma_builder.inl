#pragma once

#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/gemm/collective/builders/sm100_pipeline_carveout.inl"



namespace cutlass::gemm::collective {



namespace detail {

template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int stages
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled(StageCount<stages> stage_count) {
  return stages;
}

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template <
  int CapacityBytes,
  class ElementA,
  class ElementB,
  class TileShapeMNK,
  class TileShapeSFA,
  class TileShapeSFB,
  int carveout_bytes
>
constexpr int
sm100_compute_stage_count_or_override_blockscaled(StageCountAutoCarveout<carveout_bytes> stage_count) {
  // For MXF8F6F4 MMA, ElementA/B will be passed in as uint8_t
  // Each stage include (CollectiveMma::SharedStorage)
  // 1. smem for A and smem for B (CollectiveMma::SharedStorage::TensorStorage)
  // 2. one MainloopPipeline = PipelineTmaUmmaAsync (CollectiveMma::SharedStorage::SharedStorage)
  // 3. smem for SFB and smem for SFB (CollectiveMma::SharedStorage::TensorStorage, independent of input size b.c. sizeof(sf) is fixed)
  constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage);
  constexpr auto a_bits = cute::sizeof_bits_v<ElementA>;
  constexpr auto b_bits = cute::sizeof_bits_v<ElementB>;
  constexpr auto stage_sfa_bytes = size(filter_zeros(TileShapeSFA{}));
  constexpr auto stage_sfb_bytes = size(filter_zeros(TileShapeSFB{}));

  constexpr int stage_bytes =
    cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
    static_cast<int>(mainloop_pipeline_bytes + stage_sfa_bytes + stage_sfb_bytes);

  return (CapacityBytes - carveout_bytes) / stage_bytes;
}

} // namespace detail


template <
  class ElementPairA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementPairB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,        
  class ClusterShape_MNK,     
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassBlockScaledTensorOp,
    ElementPairA,
    GmemLayoutATag,
    AlignmentA,
    ElementPairB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    BuilderScheduleTag,
    cute::enable_if_t<
      // Blockscaled Gemm
      (not cute::is_same_v<KernelMixedTmaCpAsyncWarpSpecialized1SmBlockScaledSm100, BuilderScheduleTag>) &&
      (cute::is_base_of_v<KernelScheduleBlockScaledGemmSm100, BuilderScheduleTag> ||
       cute::is_same_v<KernelScheduleAuto, BuilderScheduleTag>) 
       &&
      // Alignment check
      detail::sm1xx_blockscaled_gemm_is_aligned<typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type,
                                                AlignmentA,
                                                typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type,
                                                AlignmentB,
                                                BuilderScheduleTag>()>>
{
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;
  using ElementA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::data_type;
  using ElementB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::data_type;
  using ElementSF = ElementSFA;

  static constexpr cute::UMMA::Major UmmaMajorA = cutlass::gemm::collective::detail::tag_to_umma_major_A<GmemLayoutATag>();
  static constexpr cute::UMMA::Major UmmaMajorB = cutlass::gemm::collective::detail::tag_to_umma_major_B<GmemLayoutBTag>();

  static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
  static_assert(detail::blockscaled::check_input_datatypes<BuilderScheduleTag, ElementPairA, ElementPairB, UmmaMajorA, UmmaMajorB>(), "Incorrect input types");

  static constexpr bool is_2sm = detail::blockscaled::is_2sm<TileShape_MNK, ClusterShape_MNK, BuilderScheduleTag>();
  static constexpr auto Instr = detail::blockscaled::select_instr<ElementPairA, ElementPairB, ElementAccumulator, UmmaMajorA, UmmaMajorB, BuilderScheduleTag>();

  using TiledMma = typename cutlass::gemm::collective::detail::TrivialBlockscaledMma<ElementPairA, ElementPairB, ElementAccumulator,
                                                                  TileShape_MNK, ClusterShape_MNK,
                                                                  UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>::type;

  static constexpr bool UseMxf8f6f4 = Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8;

  using ElementAMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementA, UseMxf8f6f4>());
  using ElementBMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<ElementB, UseMxf8f6f4>());


  static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;

  using AtomThrID = typename TiledMma::AtomThrID;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  using ElementAMma_SmemAllocType = cute::conditional_t<UseMxf8f6f4, uint8_t, ElementAMma>;
  using ElementBMma_SmemAllocType = cute::conditional_t<UseMxf8f6f4, uint8_t, ElementBMma>;

  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopySFA = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(
      ClusterShape_MNK{}, AtomThrID{}));

  using GmemTiledCopySFB = decltype(cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_SFB(
      ClusterShape_MNK{}, AtomThrID{})); 

  using GmemTiledCopyPairA = decltype(cute::make_tuple(GmemTiledCopyA{}, GmemTiledCopySFA{}));
  using GmemTiledCopyPairB = decltype(cute::make_tuple(GmemTiledCopyB{}, GmemTiledCopySFB{}));

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma_SmemAllocType, BlockTileA_M, BlockTileA_K>());

  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsA = decltype(cute::make_tuple(SmemLayoutAtomA{}, SmemLayoutAtomSFA{}));

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma_SmemAllocType, BlockTileB_N, BlockTileB_K>());
  using SmemLayoutAtomSFB = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape_MNK{}));
  using SmemLayoutAtomsB = decltype(cute::make_tuple(SmemLayoutAtomB{}, SmemLayoutAtomSFB{}));

  using StrideA = cutlass::gemm::TagToStrideA_t<GmemLayoutATag>;
  using StrideB = cutlass::gemm::TagToStrideB_t<GmemLayoutBTag>;
  using InternalStrideA  = cute::remove_pointer_t<StrideA>;
  using InternalStrideB  = cute::remove_pointer_t<StrideB>;
  using InternalLayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
  using InternalLayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());
  using LayoutSFA = cute::conditional_t<cute::is_same_v<InternalStrideA, StrideA>, InternalLayoutSFA, InternalLayoutSFA *>;
  using LayoutSFB = cute::conditional_t<cute::is_same_v<InternalStrideB, StrideB>, InternalLayoutSFB, InternalLayoutSFB *>;
  using StridePairA = decltype(cute::make_tuple(StrideA{}, LayoutSFA{}));
  using StridePairB = decltype(cute::make_tuple(StrideB{}, LayoutSFB{}));

  static constexpr int MMA_N = cute::size<1>(TileShape_MNK{});
  static constexpr uint32_t AccumulatorPipelineStageCount = (MMA_N == 256) ? 1 : 2;
  static constexpr bool IsArrayOfPointersGemm = cute::is_base_of_v<KernelSchedulePtrArrayBlockScaledGemmSm100, BuilderScheduleTag>;
  static constexpr bool IsGroupGemm = !cute::is_same_v<StrideA, InternalStrideA>;
  static constexpr uint32_t SchedulerPipelineStageCount = cute::conditional_return<IsGroupGemm>(8, 2);

  static constexpr uint32_t KernelSmemCarveout = detail::Sm100DenseGemmTmaUmmaCarveout<
      ClusterShape_MNK,
      AccumulatorPipelineStageCount,
      SchedulerPipelineStageCount,
      detail::CLCResponseSize,
      IsArrayOfPointersGemm,
      4
    >::KernelSmemCarveout;
  static constexpr int Sm100ReducedSmemCapacityBytes = cutlass::gemm::collective::detail::sm100_smem_capacity_bytes - KernelSmemCarveout;

  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  static constexpr int PipelineStages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override_blockscaled<
      Sm100ReducedSmemCapacityBytes, ElementAMma_SmemAllocType, ElementBMma_SmemAllocType, SmemTileShape, SmemLayoutAtomSFA, SmemLayoutAtomSFB>(StageCountType{});

  // using DispatchPolicy = 
  //   cute::conditional_t<IsArrayOfPointersGemm,
  //     cutlass::gemm::MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaled<
  //         PipelineStages,
  //         SchedulerPipelineStageCount,
  //         AccumulatorPipelineStageCount,
  //         ClusterShape_MNK
  //     >,
  //     cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<
  //         PipelineStages,
  //         SchedulerPipelineStageCount,
  //         AccumulatorPipelineStageCount,
  //         ClusterShape_MNK
  //     >
  //   >;

  using CollectiveOp = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      cute::tuple<ElementA, ElementSF>,
      StridePairA,
      cute::tuple<ElementB, ElementSF>,
      StridePairB,
      TiledMma,
      GmemTiledCopyPairA,
      SmemLayoutAtomsA,
      void,
      cute::identity,
      GmemTiledCopyPairB,
      SmemLayoutAtomsB,
      void,
      cute::identity
    >;
};

} 
