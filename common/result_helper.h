#pragma once

bool CheckResultSame(float* out, float* ref, int size, float eps = 1e-6);

#define CHECK_CUDA_ERRORS(func)                                 \
  do {                                                          \
    cudaError_t err = (func);                                   \
    if (err != cudaSuccess) {                                   \
      printf("%s %d CUDA error: %s (%d)\n", __FILE__, __LINE__, \
             cudaGetErrorString(err), err);                     \
      exit(1);                                                  \
    }                                                           \
  } while (false)
