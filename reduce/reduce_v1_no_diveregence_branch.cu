#include <cuda_runtime.h>
#include <nvToolsExtCuda.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "common/result_helper.h"

constexpr int K_THREAD_PER_BLOCK = 256;

// 去除线程束分支
__global__ void reduce1(float* d_in, float* d_out) {
  __shared__ float s_data[K_THREAD_PER_BLOCK];

  // each thread loads one element from global memory to shared memory
  const auto tid = threadIdx.x;
  const auto i = blockIdx.x * blockDim.x + tid;
  s_data[tid] = d_in[i];
  __syncthreads();

  // do reduction in shared memory
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    const auto idx = tid * 2 * stride;
    if (idx < blockDim.x) {
      s_data[idx] += s_data[idx + stride];
    }
    __syncthreads();
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

  const int block_num = N / K_THREAD_PER_BLOCK;
  auto* out = new float[block_num];
  float* d_out;
  cudaMalloc(&d_out, block_num * sizeof(float));
  auto* res = new float[block_num];

  for (int i = 0; i < N; i++) {
    a[i] = rand() % 10000;
  }

  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < K_THREAD_PER_BLOCK; ++j) {
      cur += a[i * K_THREAD_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  dim3 block(K_THREAD_PER_BLOCK, 1);
  dim3 grid(block_num, 1);

  nvtxRangePushA("reduce0");
  reduce1<<<grid, block>>>(d_a, d_out);
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