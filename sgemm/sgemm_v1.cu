#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "common/result_helper.h"

// 计算偏移，在行主序中，ld是矩阵的width
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// float4 transfer
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <
    const int BLOCK_SIZE_M,  // height of block of C that each thread block
                             // computes
    const int BLOCK_SIZE_K,  // width of block of A that each thread block loads
    const int BLOCK_SIZE_N,  // width of block of C that each thread block
                             // computes
    const int THREAD_SIZE_Y,  // height of block of C that each thread computes
    const int THREAD_SIZE_X,  // width of block of C that each thread computes
    const bool ENABLE_DOUBLE_BUFFER  // whether to use double buffering or not
    >
__global__ void Sgemm(float* __restrict__ A, float* __restrict__ B,
                      float* __restrict__ C, const int M, const int N,
                      const int K) {
  // block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  // thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // the thread number in Block of X,Y
  const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
  const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
  const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

  // thread id in the block
  const int tid = ty * THREAD_X_PER_BLOCK + tx;

  // shared memory for A and B
  // 为什么A矩阵分块是转置存储？
  __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
  __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
  // register for C
  float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0.0f};
  // register for A and B
  float frag_a[2][THREAD_SIZE_Y];
  float frag_b[2][THREAD_SIZE_X];
  // registers load global memory
  // 计算读取寄存器次数, 采用float4读取
  const int ldg_num_a =
      BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
  const int ldg_num_b =
      BLOCK_SIZE_N * BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK * 4);
  // 读取global memory使用的寄存器
  float ldg_a_reg[4 * ldg_num_a];
  float ldg_b_reg[4 * ldg_num_b];

  // 读取A和B的块一行所需要的线程数
  const int A_TILE_TRHEAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_TRHEAD_PER_ROW = BLOCK_SIZE_N / 4;
  // 计算读取的行号和列号
  const int A_TILE_ROW_START = tid / A_TILE_TRHEAD_PER_ROW;
  const int A_TILE_COL = tid % A_TILE_TRHEAD_PER_ROW * 4;
  const int B_TILE_ROW_START = tid / B_TILE_TRHEAD_PER_ROW;
  const int B_TILE_COL = tid % B_TILE_TRHEAD_PER_ROW * 4;

  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_TRHEAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_TRHEAD_PER_ROW;

  // 获取当前块的起始地址
  A = &A[(BLOCK_SIZE_M * by) * K];
  B = &B[BLOCK_SIZE_N * bx];

  // transfer first tile from global memory to shared memory
  // load A from global memory to shared memory
  // 为什么A矩阵分块是转置存储？
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
    int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
    FETCH_FLOAT4(ldg_a_reg[ldg_idx]) =
        FETCH_FLOAT4(A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL, K)]);
    As[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx];
    As[0][A_TILE_COL + 1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 1];
    As[0][A_TILE_COL + 2][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 2];
    As[0][A_TILE_COL + 3][A_TILE_ROW_START + i] = ldg_a_reg[ldg_idx + 3];
  }
  // load B from global memory to shared memory
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
    FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) =
        FETCH_FLOAT4(B[OFFSET(B_TILE_ROW_START + i, B_TILE_COL, N)]);
  }
  __syncthreads();

  // load A form shared memory to register
#pragma unroll
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
    FETCH_FLOAT4(frag_a[0][thread_y]) =
        FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
  }
  // load B form shared memory to register
#pragma unroll
  for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
    FETCH_FLOAT4(frag_b[0][thread_x]) =
        FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
  }

  int write_stage_idx = 1;
  int tile_idx = 0;
  do {
    tile_idx += BLOCK_SIZE_K;
    // load next tile from global mem
    if (tile_idx < K) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        const int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_idx]) = FETCH_FLOAT4(
            A[OFFSET(A_TILE_ROW_START + i, A_TILE_COL + tile_idx, K)]);
      }
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        const int ldg_idx = i / B_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_b_reg[ldg_idx]) = FETCH_FLOAT4(
            B[OFFSET(tile_idx + B_TILE_ROW_START + i, B_TILE_COL, N)]);
      }
    }

    int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
    for (int j = 0; j < BLOCK_SIZE_K - 1; ++j) {
      // load next tile from shared mem to registers
      // load A from shared memory to register
#pragma unroll
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[(j + 1) % 2][thread_y]) = FETCH_FLOAT4(
            As[load_stage_idx][j + 1][THREAD_SIZE_Y * ty + thread_y]);
      }
#pragma unroll
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[(j + 1) % 2][thread_x]) = FETCH_FLOAT4(
            Bs[load_stage_idx][j + 1][THREAD_SIZE_X * tx + thread_x]);
      }

      // compute C THREAD_SIZE_X * THREAD_SIZE_Y times
#pragma unroll
      for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
          // 也许这里就是A矩阵转置读取的原因？让thread_y访存连续
          accum[thread_y][thread_x] +=
              frag_a[j % 2][thread_y] * frag_b[j % 2][thread_x];
        }
      }
    }

    if (tile_idx < K) {
      // load next tile from tmp registers to shared mem
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        const int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
        As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_idx];
        As[write_stage_idx][A_TILE_COL + 1][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_idx + 1];
        As[write_stage_idx][A_TILE_COL + 2][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_idx + 2];
        As[write_stage_idx][A_TILE_COL + 3][A_TILE_ROW_START + i] =
            ldg_a_reg[ldg_idx + 3];
      }
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        const int ldg_idx = i / B_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) =
            FETCH_FLOAT4(ldg_b_reg[ldg_idx]);
      }
      // use double buffer, only need one sync
      __syncthreads();
      // switch write stage idx
      write_stage_idx ^= 1;
    }

    // load first tile from shared mem to register of next iter
    // load A from shared memory to register
#pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
      FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(
          As[load_stage_idx ^ 1][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
#pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(
          Bs[load_stage_idx ^ 1][0][THREAD_SIZE_X * tx + thread_x]);
    }

    // compute last tile mma THREAD_SIZE_X * THREAD_SIZE_Y times
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
      for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
        accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
      }
    }
  } while (tile_idx < K);

  // store back to C
  for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
      FETCH_FLOAT4(
          C[OFFSET(BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                   BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x, N)]) =
          FETCH_FLOAT4(accum[thread_y][thread_x]);
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s <M> <N> <K>\n", argv[0]);
    exit(0);
  }
  size_t M = atoi(argv[1]);
  size_t N = atoi(argv[2]);
  size_t K = atoi(argv[3]);

  assert(M % 8 == 0);
  assert(N % 8 == 0);
  assert(K % 8 == 0);

  size_t bytes_A = M * K * sizeof(float);
  size_t bytes_B = K * N * sizeof(float);
  size_t bytes_C = M * N * sizeof(float);
  float* h_A = (float*)malloc(bytes_A);
  float* h_B = (float*)malloc(bytes_B);
  float* h_C = (float*)malloc(bytes_C);
  float* h_C1 = (float*)malloc(bytes_C);

  float* d_A;
  float* d_B;
  float* d_C;

  CHECK_CUDA_ERRORS(cudaMalloc(&d_A, bytes_A));
  CHECK_CUDA_ERRORS(cudaMalloc(&d_B, bytes_B));
  CHECK_CUDA_ERRORS(cudaMalloc(&d_C, bytes_C));
  double msecPerMatrixMul[2] = {0, 0};
  double gigaFlop[2] = {0, 0};
  double flopsPerMatrixMul = 2.0 * M * N * K;

  const int BLOCK_SIZE_M = 128;
  const int BLOCK_SIZE_K = 8;
  const int BLOCK_SIZE_N = 128;
  const int THREAD_SIZE_Y = 8;
  const int THREAD_SIZE_X = 8;
  const bool ENABLE_DOUBLE_BUFFER = true;

  // generate A
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = i / 13.0;
  }
  // generate B
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = i % 13;
  }

  CHECK_CUDA_ERRORS(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERRORS(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CHECK_CUDA_ERRORS(cudaEventCreate(&start));
  CHECK_CUDA_ERRORS(cudaEventCreate(&stop));
  float msecTotal = 0;
  int numIters = 1000;

  CHECK_CUDA_ERRORS(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));

  CHECK_CUDA_ERRORS(cudaEventRecord(start));
  for (int run = 0; run < numIters; ++run) {
    dim3 block_dim(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 grid_dim(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y,
          THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
        <<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
  }
  CHECK_CUDA_ERRORS(cudaEventRecord(stop));
  CHECK_CUDA_ERRORS(cudaEventSynchronize(stop));
  CHECK_CUDA_ERRORS(cudaEventElapsedTime(&msecTotal, start, stop));

  CHECK_CUDA_ERRORS(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

  msecPerMatrixMul[0] = msecTotal / numIters;
  gigaFlop[0] = (flopsPerMatrixMul / 1e9) / (msecPerMatrixMul[0] / 1e3);
  printf("Sgemm Performance: %.2f GFlops, Time= %0.3f msec, Size= %.0f Ops\n",
         gigaFlop[0], msecPerMatrixMul[0], flopsPerMatrixMul);

  // cublas
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUDA_ERRORS(cudaMemcpy(d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERRORS(cudaEventRecord(start));
  for (int run = 0; run < numIters; ++run) {
    cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K,
                d_B, N, &beta, d_C, N);
  }
  CHECK_CUDA_ERRORS(cudaEventRecord(stop));
  CHECK_CUDA_ERRORS(cudaEventSynchronize(stop));
  CHECK_CUDA_ERRORS(cudaEventElapsedTime(&msecTotal, start, stop));
  CHECK_CUDA_ERRORS(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));
  msecPerMatrixMul[1] = msecTotal / numIters;
  gigaFlop[1] = (flopsPerMatrixMul / 1e9) / (msecPerMatrixMul[1] / 1e3);
  printf("cuBlas Performance: %.2f GFlops, Time= %0.3f msec, Size= %.0f Ops\n",
         gigaFlop[1], msecPerMatrixMul[1], flopsPerMatrixMul);
  cublasDestroy(blas_handle);

  double eps = 1e-6;
  bool correct = true;
  for (int i = 0; i < M * N; ++i) {
    int row = i / N;
    int col = i % N;
    double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
    double dot_length = M;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;
    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f, error term is > %E\n", i,
             h_C[i], h_C1[col * M + row], eps);
      correct = false;
      break;
    }
  }

  // free memory
  CHECK_CUDA_ERRORS(cudaFree(d_A));
  CHECK_CUDA_ERRORS(cudaFree(d_B));
  CHECK_CUDA_ERRORS(cudaFree(d_C));

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C1);
}