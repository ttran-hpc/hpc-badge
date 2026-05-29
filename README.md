# Matrix Multiplication

A five-step progression from serial to multi-GPU, all computing **C = A × B** for
dense square *N × N* matrices of `double`s.

| Program | Parallelism | Expected speedup vs serial |
|---|---|---|
| `1_serial_matmul.c`       | None | 1× (baseline) |
| `2_omp_matmul.c`          | OpenMP (shared-memory threads) | ~10–100× (scales with core count) |
| `3_hybrid_matmul.c`       | MPI + OpenMP (distributed + threaded) | ~100–1000× (scales with nodes × cores) |
| `4_gpu_matmul.cu`         | NVIDIA CUDA (thousands of GPU threads) | ~1000–100000× (depends on GPU model) |
| `5_hybrid_gpu_matmul.cu`  | MPI + CUDA (multi-node GPU) | scales with GPU count |

All programs print results in the same format so you can paste timings
side-by-side and discuss the scaling:

```
[Serial]   N = 8192  |  Time = 34183.3211 s  |  C[0][0] = 6827.166626
[OpenMP]   N = 8192  |  Threads = 64  |  Time = 664.5720 s  |  C[0][0] = 6827.166626
[MPI+OMP]  N = 8192  |  MPI ranks = 8  |  OMP threads/rank = 64  |  Time = 89.6699 s  |  C[0][0] = 6827.166626
[GPU]      N = 8192  |  Time = 0.3226 s  |  C[0][0] = 6827.166626
[MPI+GPU]  N = 8192  |  MPI ranks = 4  |  GPUs = 4  |  Time = 0.1539 s  |  C[0][0] = 6827.166626
[MPI+GPU]  N = 8192  |  MPI ranks = 16  |  GPUs = 16  |  Time = 0.0847 s  |  C[0][0] = 6827.166626
```

The `C[0][0]` value is a quick sanity check: all programs should print
the same number for the same *N*.

---

## Prerequisites

| Tool | Purpose |
|---|---|
| GCC 13 (`gnu13` module) | C compiler with OpenMP support |
| OpenMPI | MPI implementation |
| CUDA Toolkit 13+ | NVIDIA GPU compiler (`nvcc`) |

---

## Compiling

### Load modules (HPC cluster)

```bash
module load gnu14
module load openmpi5
module load cuda
```

### Serial

```bash
gcc -O2 -std=c11 -o serial_matmul 1_serial_matmul.c
```

### OpenMP

```bash
gcc -O2 -std=c11 -fopenmp -o omp_matmul 2_omp_matmul.c
```

### MPI + OpenMP (hybrid)

```bash
mpicc -O2 -std=c11 -fopenmp -o hybrid_matmul 3_hybrid_matmul.c
```

> `mpicc` is a thin wrapper around `gcc`; `-fopenmp` still needs to be
> passed explicitly.

### GPU (CUDA)

```bash
nvcc -O2 -o gpu_matmul 4_gpu_matmul.cu
```

### Multi-GPU (MPI + CUDA)

```bash
MPI_ROOT=$(dirname $(dirname $(which mpicc)))
nvcc -O2 -ccbin=g++ -o hybrid_gpu_matmul 5_hybrid_gpu_matmul.cu \
    -I${MPI_ROOT}/include \
    -L${MPI_ROOT}/lib \
    -lmpi
```

> **Note:** CUDA 12.4 supports GCC up to version 13. If you see host compiler
> errors, either upgrade to CUDA 13+ or point `-ccbin` at an older GCC:
> `nvcc -O2 -ccbin=/path/to/gcc-13 ...`

---

## Running

### Serial

```bash
./serial_matmul 8192
```

### OpenMP

```bash
export OMP_NUM_THREADS=64
./omp_matmul 8192
```

Try different thread counts (1, 2, 4, 8, 16, 32) and plot the speedup.

### Hybrid MPI + OpenMP

```bash
export OMP_NUM_THREADS=64
mpirun -np 8 ./hybrid_matmul 8192
```

**Constraint:** *N* must be divisible by the number of MPI processes.
For example, with `-np 4`, valid sizes include 1024, 2048, 4096.

When running on multiple nodes, use `srun` instead of `mpirun` (see SLURM
script below).

### GPU

```bash
./gpu_matmul 8192
```

### Multi-GPU

```bash
srun --mpi=pmix -n 16 ./hybrid_gpu_matmul 8192
```

**Constraint:** *N* must be divisible by the number of MPI ranks (= number of GPUs).

---

## Recommended matrix sizes

| Goal | Suggested N |
|---|---|
| Quick functional test | 256 |
| Visible timing differences | 1024 |
| Strong-scaling study | 2048, 4096 |
| GPU stress test | 4096, 8192 |
| Multi-GPU scaling study | 8192, 16384 |

> **Tip:** Naive matrix multiplication is *O(N³)*, so doubling *N* multiplies
> run time by roughly 8×.  Use this to predict and verify your timing results.

---

## Sample SLURM batch script

Save as `run_matmul.sh` and submit with `sbatch run_matmul.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=hpc_workshop
#SBATCH --nodes=8                  # total nodes for the MPI job
#SBATCH --ntasks-per-node=1        # MPI ranks per node (total ranks = nodes × ntasks-per-node)
#SBATCH --cpus-per-task=64         # OpenMP threads per rank (= OMP_NUM_THREADS)
#SBATCH --gres=gpu:1               # one GPU per node
#SBATCH -p h200                  # or h100, a30 for GPU jobs
#SBATCH --time=12:00:00
#SBATCH --output=matmul_%j.out
#SBATCH --error=matmul_%j.err

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load gnu13
module load openmpi5
module load cuda/12.4

# OMP_NUM_THREADS must match --cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Bind MPI processes to sockets for better memory locality
export OMP_PROC_BIND=close
export OMP_PLACES=cores

N=8192   # must be divisible by total MPI ranks (nodes × ntasks-per-node)

# ── Compile ───────────────────────────────────────────────────────────────────
echo "=== Compiling ==="
gcc  -O2 -std=c11            -o serial_matmul  1_serial_matmul.c
gcc  -O2 -std=c11 -fopenmp   -o omp_matmul     2_omp_matmul.c
mpicc -O2 -std=c11 -fopenmp  -o hybrid_matmul  3_hybrid_matmul.c
nvcc -O2                     -o gpu_matmul     4_gpu_matmul.cu
MPI_ROOT=$(dirname $(dirname $(which mpicc)))
nvcc -O2 -ccbin=g++ -o hybrid_gpu_matmul 5_hybrid_gpu_matmul.cu \
    -I${MPI_ROOT}/include -L${MPI_ROOT}/lib -lmpi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "=== Serial ==="
./serial_matmul $N

echo ""
echo "=== OpenMP ($OMP_NUM_THREADS threads) ==="
./omp_matmul $N

echo ""
echo "=== Hybrid MPI+OpenMP ($(( SLURM_NNODES * SLURM_NTASKS_PER_NODE )) ranks × $OMP_NUM_THREADS threads) ==="
srun --mpi=pmix ./hybrid_matmul $N

echo ""
echo "=== GPU ==="
./gpu_matmul $N

echo ""
echo "=== Multi-node GPU (${SLURM_NNODES} nodes × 1 GPU) ==="
srun --mpi=pmix ./hybrid_gpu_matmul $N
```

> **Adjusting for your cluster:**
> - Change `--nodes`, `--ntasks-per-node`, and `--cpus-per-task` to match
>   available resources. Remember to keep *N* divisible by total MPI ranks.

---

## Results (N = 8192)

| Implementation | Resources | Time | Speedup vs Serial |
|---|---|---|---|
| Serial | 1 core | ~9.5 hours | 1× |
| OpenMP | 64 threads | 11 min | 51× |
| Hybrid MPI+OpenMP | 8 nodes × 64 threads | 90 s | 381× |
| GPU | 1 GPU | 0.32 s | 106,000× |
| Multi-GPU | 4 GPUs | 0.15 s | 222,000× |
| Multi-GPU | 16 GPUs | 0.085 s | 404,000× |

---

## Discussion

### Problem Setup

We benchmark dense matrix multiplication C = A × B for N = 8192 — matrices of ~512 MB each. This is an O(N³) compute problem (8192³ ≈ 550 billion floating point operations), making it an ideal stress test for parallel hardware. All results are verified by the identical `C[0][0] = 6827.166626` across every implementation.

---

### Serial Baseline

The serial implementation takes nearly 10 hours — a vivid reminder of why parallelism exists. Every operation is sequential, and the CPU is barely keeping its memory bus busy since the naive triple-loop has poor cache reuse for large N.

---

### OpenMP: 51× on 64 Threads

Distributing the outer loop across 64 threads gives a 51× speedup. Theoretical maximum would be 64× (linear scaling), so we achieve about **80% parallel efficiency**. The gap comes from memory bandwidth contention — all 64 threads share the same L3 cache and memory bus, and at N=8192 the matrices far exceed cache capacity, making this a memory-bound problem rather than compute-bound.

---

### Hybrid MPI+OpenMP: 381× on 8 Nodes

Scaling to 8 nodes (8 MPI ranks × 64 OpenMP threads = 512 total threads) yields a 381× speedup — **7.4× faster than single-node OpenMP**. Theoretical scaling from 1 to 8 nodes would be 8×, so we achieve about **93% parallel efficiency** across nodes.

The improvement over single-node OpenMP efficiency reflects the key advantage of distributed memory: each node has its own independent memory bus and L3 cache, eliminating the bandwidth contention that limited the single-node case. The network overhead (broadcasting the 512 MB B matrix to all 8 nodes) is amortized well at this problem size, which is why efficiency stays high.

---

### Single GPU: 106,000× — The Biggest Jump

The single GPU result is the most dramatic in the benchmark — a **106,000× speedup** over serial, and roughly **200× faster than 64-thread OpenMP**. This jump illustrates the fundamental architectural difference: a modern GPU has thousands of CUDA cores designed specifically for this kind of embarrassingly parallel arithmetic, with memory bandwidth that dwarfs a CPU's.

Importantly, this is a *naive* kernel — one thread per output element, no shared memory tiling. Production code using `cublasDgemm` would be even faster. The takeaway: even an unoptimized GPU implementation dominates highly-tuned CPU parallelism for compute-dense problems.

---

### Multi-GPU Scaling: 4 and 16 GPUs

| GPUs | Time | Speedup vs 1 GPU | Parallel Efficiency |
|---|---|---|---|
| 1 | 0.323 s | 1× | — |
| 4 | 0.154 s | 2.1× | 52% |
| 16 | 0.085 s | 3.8× | 24% |

Scaling across GPUs via MPI shows diminishing returns. Going from 1→4 GPUs gives only 2.1× (vs ideal 4×), and 1→16 GPUs gives 3.8× (vs ideal 16×). This is expected — at N=8192, the per-GPU compute time is already only ~0.3 seconds, so the **MPI communication overhead** (scattering A stripes, broadcasting B, gathering C) becomes a significant fraction of total runtime. The computation is no longer the bottleneck; the network is.

This is a textbook illustration of **Amdahl's Law**: as the parallel portion shrinks toward zero, adding more parallelism yields diminishing returns. To see better multi-GPU efficiency, try N=32768 where compute time dominates communication time again.

---

### Key Takeaways

1. **Match your tool to your problem.** CPU threading helps but is memory-bandwidth-limited for large dense linear algebra. GPUs exist precisely for this workload.

2. **Multi-node CPU parallelism shines when compute dominates.** MPI+OpenMP scales efficiently (93%) because compute time is large relative to network overhead at this problem size.

3. **Multi-GPU efficiency depends on the compute-to-communication ratio.** The GPU is so fast at N=8192 that network overhead dominates — scaling improves significantly at larger N.

4. **Correctness is non-negotiable.** Every implementation produces `C[0][0] = 6827.166626` — a simple but essential sanity check when introducing each new layer of parallelism.

5. **Never write your own BLAS.** This benchmark uses a naive O(N³) kernel. `cublasDgemm` with tensor cores would push the GPU numbers by another order of magnitude.

---

## Discussion Questions

1. **Amdahl's Law**: the initialization and I/O parts of the code are serial.
   At what thread count does speedup plateau? Can you observe this in your results?

2. **Memory bandwidth**: matrix multiplication is compute-bound for large *N*
   but memory-bound for small *N*. At what size does OpenMP stop helping?

3. **Communication overhead**: for small *N*, the MPI hybrid version may be
   *slower* than the OpenMP version. Why? At what size does it become faster?

4. **GPU transfer cost**: the `[GPU]` timing covers only the kernel, not the
   host-to-device copies. Try timing those separately with `cudaEventRecord`
   around the `cudaMemcpy` calls. How does transfer time compare to compute time?

5. **Roofline model**: given the FLOP count (2N³ operations) and your measured
   time, what is the achieved GFLOP/s? How does it compare to the hardware peak?

6. **Multi-GPU crossover**: at what value of N does scaling from 1→4 GPUs become
   efficient (>75% parallel efficiency)? How does this change for 1→16 GPUs?
