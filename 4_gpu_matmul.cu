/*
 * gpu_matmul.cu
 * NVIDIA GPU (CUDA) matrix multiplication: C = A * B
 *
 * Each CUDA thread computes exactly one element C[row][col].
 * Threads are organised into 16×16 blocks, and blocks tile the output matrix.
 *
 * Workflow:
 *   1. Allocate and initialise matrices on the CPU (host).
 *   2. Copy A and B to GPU memory (device).
 *   3. Launch the kernel — the GPU computes C in parallel.
 *   4. Copy C back to the CPU.
 *
 * Usage: ./gpu_matmul N
 *   N  - dimension of the square NxN matrices
 */

#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------ *
 * CUDA kernel — runs on the GPU.                                      *
 * Every thread figures out which (row, col) it owns and computes      *
 * the dot product of row `row` of A with column `col` of B.          *
 * ------------------------------------------------------------------ */
__global__ void matmul_kernel(const double *A, const double *B, double *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    /* Guard: threads at the grid edge may exceed the matrix boundary. */
    if (row >= N || col >= N)
        return;

    double sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

/* Helper: print a CUDA error and exit if the call failed. */
static void cuda_check(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error — %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Error: N must be a positive integer.\n");
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(double);

    /* ----- Host (CPU) memory ----- */
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: host memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i*N + j] = (double)(i + j + 1) / N;
            h_B[i*N + j] = (double)(i - j + N) / N;
        }
    }

    /* ----- Device (GPU) memory ----- */
    double *d_A, *d_B, *d_C;
    cuda_check(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    cuda_check(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    cuda_check(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    cuda_check(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "memcpy A H->D");
    cuda_check(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "memcpy B H->D");

    /*
     * Thread block: 16×16 = 256 threads (a common, hardware-friendly size).
     * Grid: enough blocks to cover the entire N×N output matrix.
     * The ceiling division ( (N + 15) / 16 ) handles N not divisible by 16.
     */
    int BLOCK = 16;
    dim3 block_dim(BLOCK, BLOCK);
    dim3 grid_dim((N + BLOCK - 1) / BLOCK,
                  (N + BLOCK - 1) / BLOCK);

    /* CUDA events give us GPU-side timing in milliseconds. */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    /* ---- Timed region ---- */
    cudaEventRecord(ev_start);

    matmul_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);   /* wait for the GPU to finish */
    /* ---- End timed region ---- */

    cuda_check(cudaGetLastError(), "kernel launch");

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    /* Copy result back to host and print a spot-check value. */
    cuda_check(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "memcpy C D->H");

    printf("[GPU]     N = %d  |  Time = %.4f s  |  C[0][0] = %.6f\n",
           N, gpu_ms / 1000.0f, h_C[0]);

    /* ----- Cleanup ----- */
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
