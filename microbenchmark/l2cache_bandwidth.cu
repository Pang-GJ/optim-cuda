#include <cuda_runtime.h>

#include <cstdio>

#include "common/result_helper.h"

// access data in bytes, should be smaller than l2 cache size
constexpr int DATA_SIZE_IN_BYTES = (1lu << 20) * 2;
// number of LDG instructions
// LDG指令用于从全局内存中读取数据，不经过L1缓存
constexpr int N_LDG = (1lu << 20) * 512;

constexpr int WARMUP_ITER = 200;
constexpr int BENCH_ITER = 200;

__device__ __forceinline__ int ldg_cg(const void* ptr) {
  int ret;
  asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

template <int BLOCK, int UNROLL, int N_DATA>
__global__ void kernel(const int* x, int* y) {
  int offset = (BLOCK * UNROLL * blockIdx.x + threadIdx.x) % N_DATA;
  const int* ldg_ptr = x + offset;
  int reg[UNROLL];

#pragma unroll
  for (int i = 0; i < UNROLL; i++) {
    reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
  }

  int sum = 0;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    sum += reg[i];
  }
  if (sum != 0) {
    *y = sum;
  }
}

int main() {
  const int N_DATA = DATA_SIZE_IN_BYTES / sizeof(int);

  const int UNROLL = 16;
  const int BLOCK = 128;

  static_assert(N_DATA >= UNROLL * BLOCK && N_DATA % (UNROLL * BLOCK) == 0,
                "UNROLL or BLOCK is invalid");

  int *x, *y;
  CHECK_CUDA_ERRORS(cudaMalloc(&x, N_DATA * sizeof(int)));
  CHECK_CUDA_ERRORS(cudaMalloc(&y, N_DATA * sizeof(int)));
  CHECK_CUDA_ERRORS(cudaMemset(x, 0, N_DATA * sizeof(int)));

  int grid = N_LDG / (UNROLL * BLOCK);

  cudaEvent_t start, stop;
  CHECK_CUDA_ERRORS(cudaEventCreate(&start));
  CHECK_CUDA_ERRORS(cudaEventCreate(&stop));

  // warm up to cache data into L2
  for (int i = 0; i < WARMUP_ITER; i++) {
    kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
  }

  CHECK_CUDA_ERRORS(cudaEventRecord(start));
  for (int i = 0; i < BENCH_ITER; i++) {
    kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
  }
  cudaDeviceSynchronize();
  CHECK_CUDA_ERRORS(cudaEventRecord(stop));

  float time_ms = 0.f;
  CHECK_CUDA_ERRORS(cudaEventSynchronize(stop));
  CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time_ms, start, stop));
  double gbps = ((double)(N_LDG * sizeof(int)) / 1e9) /
                ((double)time_ms / BENCH_ITER / 1e3);
  printf("L2 cache bandwidth: %.2f GB/s\n", gbps);

  CHECK_CUDA_ERRORS(cudaEventDestroy(start));
  CHECK_CUDA_ERRORS(cudaEventDestroy(stop));

  CHECK_CUDA_ERRORS(cudaFree(x));
  CHECK_CUDA_ERRORS(cudaFree(y));

  return 0;
}