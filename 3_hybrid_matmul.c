/*
 * hybrid_matmul.c
 * Hybrid MPI + OpenMP matrix multiplication: C = A * B
 *
 * Strategy:
 *   - MPI distributes rows of A across processes (coarse-grain parallelism).
 *   - OpenMP threads parallelize the inner loops within each MPI process
 *     (fine-grain parallelism).
 *   - Matrix B is replicated on every process via MPI_Bcast.
 *   - Results are gathered back to rank 0 with MPI_Gather.
 *
 * Constraint: N must be evenly divisible by the number of MPI processes.
 *
 * Usage: mpirun -np <P> ./hybrid_matmul N
 *   N  - dimension of the square NxN matrices
 *   P  - number of MPI processes (N must be divisible by P)
 *
 * Set OpenMP threads with:  export OMP_NUM_THREADS=<T>
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    /*
     * MPI_THREAD_FUNNELED: the program is multi-threaded, but only the
     * main thread makes MPI calls.  This is safe and widely supported.
     */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np P %s N\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);

    if (N <= 0 || N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: N must be positive and divisible by P (N=%d, P=%d).\n", N, size);
        MPI_Finalize();
        return 1;
    }

    int rows_per_rank = N / size;   /* rows of A each process is responsible for */

    /* Only rank 0 holds the full matrices A and C. */
    double *A = NULL;
    double *C = NULL;

    /* Every rank needs a full copy of B to compute its portion of C. */
    double *B = malloc(N * N * sizeof(double));

    /* Local stripe of A and the corresponding stripe of C. */
    double *local_A = malloc(rows_per_rank * N * sizeof(double));
    double *local_C = malloc(rows_per_rank * N * sizeof(double));

    if (!B || !local_A || !local_C) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));
        if (!A || !C) {
            fprintf(stderr, "Rank 0: memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i*N + j] = (double)(i + j + 1) / N;
                B[i*N + j] = (double)(i - j + N) / N;
            }
        }
    }

    /* ---- Timed region (wall time measured on rank 0) ---- */
    double t_start = MPI_Wtime();

    /* Share B with every process so they can all compute their C rows. */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Give each process its stripe of A (rows_per_rank rows). */
    MPI_Scatter(A,       rows_per_rank * N, MPI_DOUBLE,
                local_A, rows_per_rank * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Each process multiplies its rows of A by the full B using OpenMP. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows_per_rank; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i*N + k] * B[k*N + j];
            }
            local_C[i*N + j] = sum;
        }
    }

    /* Collect the partial results into the full C matrix on rank 0. */
    MPI_Gather(local_C, rows_per_rank * N, MPI_DOUBLE,
               C,       rows_per_rank * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    /* ---- End timed region ---- */

    if (rank == 0) {
        printf("[MPI+OMP]  N = %d  |  MPI ranks = %d  |  OMP threads/rank = %d  "
               "|  Time = %.4f s  |  C[0][0] = %.6f\n",
               N, size, omp_get_max_threads(),
               t_end - t_start, C[0]);
        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
