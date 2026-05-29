/*
 * hybrid_gpu_matmul.cu
 * Multi-node GPU matrix multiplication using MPI + CUDA
 * Strategy:
 *   - One MPI rank per GPU
 *   - Rank 0 initializes A and B on host
 *   - B is broadcast to all ranks (replicated), A is scattered in row stripes
 *   - Each rank copies its stripe + full B to its GPU, runs the kernel
 *   - Results gathered back to rank 0
 *
 * Usage: srun --mpi=pmix -n <num_gpus> ./hybrid_gpu_matmul N
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ *
 * Same naive kernel as gpu_matmul.cu — each thread computes one C[i][j]
 * ------------------------------------------------------------------ */
__global__ void matmul_kernel(const double *A, const double *B, double *C, int N, int local_rows)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= local_rows || col >= N)
        return;

    double sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

static void cuda_check(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error — %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) fprintf(stderr, "Usage: srun -n P %s N\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0 || N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N must be positive and divisible by P (N=%d, P=%d)\n", N, size);
        MPI_Finalize();
        return 1;
    }

    /* ── Bind each MPI rank to a GPU ─────────────────────────────────
     * On a multi-GPU node, ranks are assigned round-robin to GPUs.
     * On single-GPU nodes (typical HPC), this always picks device 0.
     */
    int num_devices;
    cuda_check(cudaGetDeviceCount(&num_devices), "getDeviceCount");
    int my_device = rank % num_devices;
    cuda_check(cudaSetDevice(my_device), "setDevice");

    int rows_per_rank = N / size;
    size_t bytes_B      = (size_t)N * N * sizeof(double);
    size_t bytes_stripe = (size_t)rows_per_rank * N * sizeof(double);

    /* ── Host allocations ────────────────────────────────────────────*/
    double *A       = NULL;   /* full A, rank 0 only */
    double *C       = NULL;   /* full C, rank 0 only */
    double *B       = (double *)malloc(bytes_B);
    double *local_A = (double *)malloc(bytes_stripe);
    double *local_C = (double *)malloc(bytes_stripe);

    if (!B || !local_A || !local_C) {
        fprintf(stderr, "Rank %d: host malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = (double *)malloc(bytes_B);
        C = (double *)malloc(bytes_B);
        if (!A || !C) { fprintf(stderr, "Rank 0: malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i*N + j] = (double)(i + j + 1) / N;
                B[i*N + j] = (double)(i - j + N) / N;
            }
    }

    /* ── Distribute data ─────────────────────────────────────────────*/
    MPI_Bcast  (B,       N * N,            MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A,       rows_per_rank * N, MPI_DOUBLE,
                local_A, rows_per_rank * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── GPU allocations ─────────────────────────────────────────────*/
    double *d_A, *d_B, *d_C;
    cuda_check(cudaMalloc(&d_A, bytes_stripe), "malloc d_A");
    cuda_check(cudaMalloc(&d_B, bytes_B),      "malloc d_B");
    cuda_check(cudaMalloc(&d_C, bytes_stripe), "malloc d_C");

    cuda_check(cudaMemcpy(d_A, local_A, bytes_stripe, cudaMemcpyHostToDevice), "H->D A");
    cuda_check(cudaMemcpy(d_B, B,       bytes_B,      cudaMemcpyHostToDevice), "H->D B");

    /* ── Time just the compute + gather ─────────────────────────────*/
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Launch kernel */
    int BLOCK = 16;
    dim3 block_dim(BLOCK, BLOCK);
    dim3 grid_dim((N            + BLOCK - 1) / BLOCK,
                  (rows_per_rank + BLOCK - 1) / BLOCK);

    matmul_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N, rows_per_rank);
    cuda_check(cudaGetLastError(), "kernel launch");
    cudaDeviceSynchronize();   /* wait before MPI_Gather */

    /* Copy result stripe back to host */
    cuda_check(cudaMemcpy(local_C, d_C, bytes_stripe, cudaMemcpyDeviceToHost), "D->H C");

    /* Gather all stripes to rank 0 */
    MPI_Gather(local_C, rows_per_rank * N, MPI_DOUBLE,
               C,       rows_per_rank * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("[MPI+GPU]  N = %d  |  MPI ranks = %d  |  GPUs = %d"
               "  |  Time = %.4f s  |  C[0][0] = %.6f\n",
               N, size, size, t_end - t_start, C[0]);
        free(A);
        free(C);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────*/
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(B); free(local_A); free(local_C);

    MPI_Finalize();
    return 0;
}
