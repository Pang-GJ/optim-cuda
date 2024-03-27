#include <cuda_runtime.h>
#include <nvToolsExtCuda.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "common/result_helper.h"

constexpr int K_THREAD_PER_BLOCK = 256;

template <uint blockSize>
__device__ __forceinline__ void warp_reduce(volatile float* sdata, uint tid) {
  if constexpr (blockSize >= 64) sdata[tid] += sdata[tid + 32];
  if constexpr (blockSize >= 32) sdata[tid] += sdata[tid + 16];
  if constexpr (blockSize >= 16) sdata[tid] += sdata[tid + 8];
  if constexpr (blockSize >= 8) sdata[tid] += sdata[tid + 4];
  if constexpr (blockSize >= 4) sdata[tid] += sdata[tid + 2];
  if constexpr (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <uint blockSize, uint num_per_thread>
__global__ void reduce6(float* d_in, float* d_out) {
  __shared__ float s_data[blockSize];

  // each thread loads num_per_thread element from global memory to shared
  // memory
  const auto tid = threadIdx.x;
  const auto i = blockIdx.x * (num_per_thread * blockDim.x) + tid;
  s_data[tid] = 0;
#pragma unroll
  for (int j = 0; j < num_per_thread; ++j) {
    s_data[tid] += d_in[j * blockDim.x + i];
  }

  __syncthreads();

  // complete unrolling
  if constexpr (blockSize >= 512) {
    if (tid < 256) {
      s_data[tid] += s_data[tid + 256];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128) {
      s_data[tid] += s_data[tid + 128];
    }
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (tid < 64) {
      s_data[tid] += s_data[tid + 64];
    }
    __syncthreads();
  }

  // unroll last warp
  if (tid < 32) {
    warp_reduce<blockSize>(s_data, tid);
  }
  // write result for this block to global memory
  if (tid == 0) {
    d_out[blockIdx.x] = s_data[0];
  }
}

int main() {
  srand(time(0));
  const int N = 32 * 1024 * 1024;
  auto* a = new float[N];
  float* d_a;
  cudaMalloc(&d_a, N * sizeof(float));

  const int block_num = 1024;
  const int num_per_block = N / block_num;
  const int num_per_thread = num_per_block / K_THREAD_PER_BLOCK;
  auto* out = new float[block_num];
  float* d_out;
  cudaMalloc(&d_out, block_num * sizeof(float));
  auto* res = new float[block_num];

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 10;
  }

  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < num_per_block; ++j) {
      cur += a[i * num_per_block + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  dim3 block(K_THREAD_PER_BLOCK, 1);
  dim3 grid(block_num, 1);

  nvtxRangePushA("reduce0");
  reduce6<K_THREAD_PER_BLOCK, num_per_thread><<<grid, block>>>(d_a, d_out);
  nvtxRangePop();

  cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
  if (CheckResultSame(out, res, block_num)) {
    std::cout << "The answer is correct!" << std::endl;
  } else {
    std::cout << "The answer is wrong!" << std::endl;
    for (int i = 0; i < block_num; ++i) {
      std::cout << "out[" << i << "] = " << out[i] << " res[" << i
                << "] = " << res[i] << std::endl;
    }
    std::cout << "\n";
  }

  cudaFree(d_a);
  cudaFree(d_out);
  cudaDeviceReset();

  delete[] out;
  delete[] res;
  delete[] a;
  return 0;
}