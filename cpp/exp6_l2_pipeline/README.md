# Experiment 6: L₂ Three-Stage Parallel Pipeline

Computes the implicit equation of the weighted variety L₂ ⊂ P(2,4,6,10) from its rational parametrization, using the three-stage parallel pipeline from the paper.

## The L₂ Parametrization

```
x₀ = -120 - 8t₁
x₁ = t₁² - 126t₁ + 12t₂ + 405
x₂ = -3t₁³ + 53t₁² - 20t₁t₂ + 2583t₁ - 12t₂ - 14985
x₃ = -2(-t₁² - 18t₁ + 4t₂ + 27)²
```

Weights: **q = (2, 4, 6, 10)**

Expected output: a degree-30 weighted hypersurface with 34 terms.

## Pipeline Architecture

### Stage 1: Weighted Homogenization (OpenMP)
- Introduces auxiliary variable `s` with deg(s) = 1
- For each generator Fᵢ: substitute tⱼ → tⱼ/s, multiply by s^{max_t_degree}
- Embarrassingly parallel: each generator is independent
- Hardware: CPU threads via OpenMP

### Stage 2: Buchberger/F4 Elimination (CUDA)
- Computes Gröbner basis under lex order s > t₁ > t₂ > x₀ > x₁ > x₂ > x₃
- S-polynomials computed in parallel (OpenMP for pairs)
- Batch reduction via GPU: Macaulay matrix → dense on GPU → row reduction
- Extracts elimination ideal: polynomials free of s, t₁, t₂
- Hardware: GPU via CUDA + CPU via OpenMP

### Stage 3: Weighted GCD Normalization (MPI + GMP)
- Computes weighted GCD of all coefficients
- Prime range [2, √N] partitioned across MPI ranks
- Each rank trial-divides its segment, results combined via MPI_Allgather
- All arithmetic via GMP (mpz_class) for exact large integers
- Hardware: distributed across MPI ranks

## Prerequisites

- CUDA Toolkit (11.0+)
- OpenMP (usually bundled with GCC/Clang)
- MPI (OpenMPI or MPICH)
- GMP (libgmp, libgmpxx)
- CMake 3.18+

### Install dependencies (Ubuntu/Debian)
```bash
sudo apt install libgmp-dev libmpich-dev cmake
# CUDA Toolkit: follow NVIDIA instructions
```

### Install dependencies (Windows/MSVC)
```
vcpkg install gmp:x64-windows
vcpkg install mpi:x64-windows
# CUDA Toolkit: NVIDIA installer
```

## Build

```bash
cd cpp/exp6_l2_pipeline
cmake -B build
cmake --build build -j4
```

## Run

```bash
# Single process (MPI with 1 rank)
./build/l2_pipeline

# 4 MPI ranks (recommended for Stage 3 parallelism)
mpirun -np 4 ./build/l2_pipeline

# Control OpenMP threads
OMP_NUM_THREADS=8 mpirun -np 4 ./build/l2_pipeline
```

## Output

The program outputs:
1. Timing for each pipeline stage
2. The computed degree-30 polynomial
3. Verification: evaluation at test points from the parametrization
4. CSV timing data in `../../results/exp6_pipeline.csv`

## File Structure

```
exp6_l2_pipeline/
├── CMakeLists.txt        # Build configuration
├── README.md             # This file
├── weighted_poly.h       # Extended polynomial type (GMP integers)
└── l2_pipeline.cu        # Main pipeline implementation
```
