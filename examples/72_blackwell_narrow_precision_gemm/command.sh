

# sudo apt update && sudo apt install -y cmake make build-essential git

sudo apt-get install -y cuda-toolkit-13-0
export PATH=/usr/local/cuda-13.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-13.0
nvcc --version
nvidia-smi

cd ..
git clone https://github.com/liziwen-AI/cutlass.git
cd cutlass
git checkout -f liz
git branch

mkdir build && cd build
cmake .. \
  -DCUTLASS_NVCC_ARCHS=100a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_LIBRARY=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build .\
  --target 72b_blackwell_nvfp4_nvfp4_gemm \
  --parallel 16


sudo /usr/local/cuda-13.0/bin/ncu --target-processes all \
./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm
