#include <cuda_runtime.h>
#include <nvToolsExtCuda.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "common/result_helper.h"

constexpr int K_THREAD_PER_BLOCK = 256;
const int K_WARP_SIZE = 32;

template <uint blockSize>
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  if constexpr (blockSize >= 32) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);  // 0-16, 1-17, 2-18, etc.
  }
  if constexpr (blockSize >= 16) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);  // 0-8, 1-9, 2-10, etc.
  }
  if constexpr (blockSize >= 8) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);  // 0-4, 1-5, 2-6, etc.
  }
  if constexpr (blockSize >= 4) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);  // 0-2, 1-3, 4-6, etc.
  }
  if constexpr (blockSize >= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);  // 0-1, 2-3, 4-5, etc.
  }
  return sum;
}

template <uint blockSize, uint num_per_thread>
__global__ void reduce7(float* d_in, float* d_out) {
  float sum = 0.0F;

  // each thread loads num_per_thread element from global memory to shared
  // memory
  const auto tid = threadIdx.x;
  const auto i = blockIdx.x * (num_per_thread * blockDim.x) + tid;
#pragma unroll
  for (int j = 0; j < num_per_thread; ++j) {
    sum += d_in[j * blockDim.x + i];
  }

  // shared memory for partial sums (one per warp in the block)
  static __shared__ float warp_level_sums[K_WARP_SIZE];
  const auto warp_id = tid / K_WARP_SIZE;
  const auto lane_id = tid % K_WARP_SIZE;

  sum = warp_reduce_sum<blockSize>(sum);

  if (lane_id == 0) {
    warp_level_sums[warp_id] = sum;
  }
  __syncthreads();

  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / K_WARP_SIZE) ? warp_level_sums[lane_id]
                                                 : 0.0F;
  // final reduce using first warp
  if (warp_id == 0) {
    sum = warp_reduce_sum<blockSize / K_WARP_SIZE>(sum);
  }
  // write result for this block to global memory
  if (tid == 0) {
    d_out[blockIdx.x] = sum;
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
      if (i * num_per_block + j < N) {
        cur += a[i * num_per_block + j];
      }
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  dim3 block(K_THREAD_PER_BLOCK, 1);
  dim3 grid(block_num, 1);

  nvtxRangePushA("reduce7");
  reduce7<K_THREAD_PER_BLOCK, num_per_thread><<<grid, block>>>(d_a, d_out);
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