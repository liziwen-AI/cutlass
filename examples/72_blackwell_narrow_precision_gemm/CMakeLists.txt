
if(CUTLASS_NVCC_ARCHS STREQUAL "100a" OR CUTLASS_NVCC_ARCHS STREQUAL "100f" OR CUTLASS_NVCC_ARCHS STREQUAL "101a" OR CUTLASS_NVCC_ARCHS STREQUAL "101f" OR CUTLASS_NVCC_ARCHS STREQUAL "103a" OR CUTLASS_NVCC_ARCHS STREQUAL "103f")
cutlass_example_add_executable(
  72a_blackwell_nvfp4_bf16_gemm
  72a_blackwell_nvfp4_bf16_gemm.cu
  )

cutlass_example_add_executable(
  72b_blackwell_nvfp4_nvfp4_gemm
  72b_blackwell_nvfp4_nvfp4_gemm.cu
  )

cutlass_example_add_executable(
  72c_blackwell_mixed_mxfp8_bf16_gemm
  72c_blackwell_mixed_mxfp8_bf16_gemm.cu
  )
endif()
