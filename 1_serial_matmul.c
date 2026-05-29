/*
 * serial_matmul.c
 * Serial (single-core) matrix multiplication: C = A * B
 *
 * Usage: ./serial_matmul N
 *   N  - dimension of the square NxN matrices
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Return elapsed wall-clock time in seconds between two timespec snapshots. */
double elapsed(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec)
         + (end.tv_nsec - start.tv_nsec) * 1e-9;
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

    /* Allocate matrices as flat 1-D arrays; element (i,j) is at [i*N + j]. */
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: memory allocation failed for N=%d.\n", N);
        return 1;
    }

    /* Fill A and B with simple values so results are easy to sanity-check. */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (double)(i + j + 1) / N;
            B[i*N + j] = (double)(i - j + N) / N;
            C[i*N + j] = 0.0;
        }
    }

    /* ---- Timed region ---- */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    /* ---- End timed region ---- */

    printf("[Serial]  N = %d  |  Time = %.4f s  |  C[0][0] = %.6f\n",
           N, elapsed(t_start, t_end), C[0]);

    free(A);
    free(B);
    free(C);
    return 0;
}
