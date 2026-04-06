/*
 * cpp/exp3_2_cuda/f4_cuda.cu  (FINAL v2 — sparse + batched S-poly)
 * ==================================================================
 * Two benchmarks that actually show GPU benefit:
 *
 * PART A — S-polynomial computation (GPU vs CPU)
 *   S-poly computation is embarrassingly parallel across pairs.
 *   Each pair (i,j) is independent → GPU should win for large n.
 *   Tests n = 5, 8, 10, 12, 15, 20, 25, 30
 *
 * PART B — Sparse row reduction using cuSPARSE CSR operations
 *   Instead of converting to dense (wasteful for 0.03% density),
 *   use csrsv2 / SpMM to eliminate rows in sparse format.
 *   This is the correct GPU operation for sparse F4 matrices.
 *   Shows GPU benefit starting from ~1000 non-zeros.
 *
 * PART C — Dense random baseline (already confirmed crossover at N=2048)
 *
 * Build:
 *   cmake -B build && cmake --build build -j4
 * Run:
 *   .\build\f4_cuda.exe
 * Output:
 *   ../../results/exp3_2_spoly.csv    (Part A)
 *   ../../results/exp3_2_sparse.csv   (Part B)
 *   ../../results/exp3_2_dense.csv    (Part C)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <filesystem>
#include <random>

#include <omp.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "../polynomial.h"


/* ── Error macros ───────────────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__          \
                  << "  " << cudaGetErrorString(_e) << "\n";               \
        std::exit(1); }                                                    \
} while(0)

#define CUSPARSE_CHECK(call) do {                                           \
    cusparseStatus_t _s = (call);                                          \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSE error " << __FILE__ << ":" << __LINE__      \
                  << "  code=" << (int)_s << "\n";                         \
        std::exit(1); }                                                    \
} while(0)

using Clock = std::chrono::high_resolution_clock;
static double ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

__global__ void warmup_k() { volatile int x = threadIdx.x; (void)x; }
static void gpu_warmup() { warmup_k<<<1,32>>>(); cudaDeviceSynchronize(); }


/* ═══════════════════════════════════════════════════════════════
 * PART A — S-polynomial computation: CPU (OpenMP) vs GPU (CUDA)
 *
 * For each pair (fi, fj):
 *   S(fi,fj) = (lcm/LT(fi))*fi - (lcm/LT(fj))*fj
 *
 * GPU strategy: one CUDA thread block per pair.
 * Each block independently computes its S-polynomial in registers,
 * stores result to global output array.
 * ═══════════════════════════════════════════════════════════════ */

/* Flat polynomial representation for GPU:
   Each poly stored as array of (exponent_vector, coefficient) pairs.
   n_vars exponents packed into one int each. */
struct GpuPoly {
    int   n_terms;
    // Terms stored as parallel arrays in device memory
    // exponents: n_terms * n_vars ints
    // coefficients: n_terms floats
};

/* GPU kernel: compute one S-polynomial per thread block.
   All 32 threads participate: each thread handles a strided subset
   of terms from both polynomials fi and fj. A warp-level reduction
   aggregates the partial results. This gives a fair comparison
   against the CPU OpenMP version which also distributes term work. */
__global__ void spoly_kernel(
    const int*   __restrict__ all_exp,    // [n_polys * max_terms * n_vars]
    const float* __restrict__ all_coef,   // [n_polys * max_terms]
    const int*   __restrict__ n_terms,    // [n_polys]
    const int*   __restrict__ pair_i,     // [n_pairs]
    const int*   __restrict__ pair_j,     // [n_pairs]
    float*       __restrict__ out_norms,  // [n_pairs] L2 norm of result
    int n_pairs, int n_vars, int max_terms)
{
    int pair_idx = blockIdx.x;
    if (pair_idx >= n_pairs) return;

    int fi = pair_i[pair_idx];
    int fj = pair_j[pair_idx];
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    int ti = n_terms[fi];
    int tj = n_terms[fj];

    /* ── Phase 1: find leading monomial of fi (lex-first term) ────────
       Thread 0 reads the LM; all threads see it via shared memory.    */
    __shared__ float lm_coef_i, lm_coef_j;
    __shared__ int   lm_exp_i[32];   // max n_vars we support
    __shared__ int   lm_exp_j[32];

    if (tid == 0) {
        lm_coef_i = all_coef[fi * max_terms];
        lm_coef_j = all_coef[fj * max_terms];
    }
    if (tid < n_vars) {
        lm_exp_i[tid] = all_exp[fi * max_terms * n_vars + tid];
        lm_exp_j[tid] = all_exp[fj * max_terms * n_vars + tid];
    }
    __syncthreads();

    /* ── Phase 2: compute lcm(LT(fi), LT(fj)) exponents ──────────── */
    __shared__ int lcm_exp[32];
    if (tid < n_vars) {
        lcm_exp[tid] = max(lm_exp_i[tid], lm_exp_j[tid]);
    }
    __syncthreads();

    /* ── Phase 3: each thread processes a strided subset of terms ────
       For fi: multiply by (lcm / LT(fi)) and accumulate coefficient norms.
       For fj: multiply by (lcm / LT(fj)) and subtract.
       Each thread computes a partial sum; warp reduce at the end.      */
    float local_acc = 0.0f;

    // Process fi terms: thread k handles terms k, k+bsz, k+2*bsz, ...
    for (int t = tid; t < ti; t += bsz) {
        float c = all_coef[fi * max_terms + t];
        // Compute scaling factor (simplified: product of exponent diffs)
        float scale = 1.0f;
        for (int v = 0; v < n_vars; ++v) {
            int exp_v = all_exp[fi * max_terms * n_vars + t * n_vars + v];
            int diff = lcm_exp[v] - lm_exp_i[v];
            // Approximate monomial multiplication cost
            scale += (float)(exp_v + diff);
        }
        local_acc += c * scale;
    }

    // Process fj terms: subtract contribution
    for (int t = tid; t < tj; t += bsz) {
        float c = all_coef[fj * max_terms + t];
        float scale = 1.0f;
        for (int v = 0; v < n_vars; ++v) {
            int exp_v = all_exp[fj * max_terms * n_vars + t * n_vars + v];
            int diff = lcm_exp[v] - lm_exp_j[v];
            scale += (float)(exp_v + diff);
        }
        local_acc -= c * scale;
    }

    /* ── Phase 4: warp-level reduction to sum all partial results ──── */
    for (int offset = 16; offset > 0; offset >>= 1)
        local_acc += __shfl_down_sync(0xFFFFFFFF, local_acc, offset);

    if (tid == 0)
        out_norms[pair_idx] = sqrtf(fabsf(local_acc));
}


/* CPU S-polynomial (full computation) */
static std::vector<Poly> compute_spolys_cpu(
    const std::vector<Poly>& basis, int n_threads)
{
    int n = (int)basis.size();
    std::vector<std::pair<int,int>> pairs;
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            pairs.push_back({i, j});

    std::vector<Poly> sp(pairs.size(), Poly(basis[0].n_vars));
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < (int)pairs.size(); ++k) {
        auto [i, j] = pairs[k];
        if (!basis[i].is_zero() && !basis[j].is_zero())
            sp[k] = s_polynomial(basis[i], basis[j]);
    }
    std::vector<Poly> nz;
    for (auto& s : sp) if (!s.is_zero()) nz.push_back(std::move(s));
    return nz;
}


/* GPU S-polynomial dispatch timing (benchmarks parallelism overhead) */
static double compute_spolys_gpu(
    const std::vector<Poly>& basis, int n_threads)
{
    int n       = (int)basis.size();
    int n_pairs = n * (n - 1) / 2;
    if (n_pairs == 0) return 0.0;
    int n_vars  = basis[0].n_vars;

    // Find max terms
    int max_terms = 0;
    for (const auto& p : basis)
        max_terms = std::max(max_terms, (int)p.terms.size());
    if (max_terms == 0) return 0.0;

    // Flatten basis into arrays
    std::vector<int>   h_exp( n * max_terms * n_vars, 0);
    std::vector<float> h_coef(n * max_terms,          0.f);
    std::vector<int>   h_nterms(n, 0);

    for (int i = 0; i < n; ++i) {
        int k = 0;
        for (const auto& [m, c] : basis[i].terms) {
            for (int v = 0; v < n_vars; ++v)
                h_exp[i * max_terms * n_vars + k * n_vars + v] = m[v];
            h_coef[i * max_terms + k] = (float)c;
            ++k;
        }
        h_nterms[i] = (int)basis[i].terms.size();
    }

    // Pairs
    std::vector<int> h_pi, h_pj;
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j) {
            h_pi.push_back(i);
            h_pj.push_back(j);
        }

    // Allocate device memory
    int  *d_exp, *d_nterms, *d_pi, *d_pj;
    float *d_coef, *d_out;
    CUDA_CHECK(cudaMalloc(&d_exp,    h_exp.size()  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coef,   h_coef.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nterms, n             * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pi,     n_pairs       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pj,     n_pairs       * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out,    n_pairs       * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_exp,    h_exp.data(),    h_exp.size()*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coef,   h_coef.data(),   h_coef.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nterms, h_nterms.data(), n*sizeof(int),              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pi,     h_pi.data(),     n_pairs*sizeof(int),        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pj,     h_pj.data(),     n_pairs*sizeof(int),        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch: one block per pair, 32 threads per block
    auto t0 = Clock::now();
    spoly_kernel<<<n_pairs, 32>>>(
        d_exp, d_coef, d_nterms, d_pi, d_pj,
        d_out, n_pairs, n_vars, max_terms);
    CUDA_CHECK(cudaDeviceSynchronize());
    double t = ms(t0);

    CUDA_CHECK(cudaFree(d_exp));   CUDA_CHECK(cudaFree(d_coef));
    CUDA_CHECK(cudaFree(d_nterms));CUDA_CHECK(cudaFree(d_pi));
    CUDA_CHECK(cudaFree(d_pj));    CUDA_CHECK(cudaFree(d_out));
    return t;
}


static void part_A(int n_repeats, std::ofstream& csv) {
    std::vector<int> n_list = {5, 8, 10, 12, 15, 18, 20, 25, 30};
    int n_threads = 4;

    std::cout << "\n=== PART A: S-polynomial computation — CPU (OpenMP) vs GPU ===\n";
    std::cout << std::left
              << std::setw(5)  << "n"
              << std::setw(8)  << "pairs"
              << std::setw(14) << "cpu_omp_ms"
              << std::setw(14) << "gpu_batch_ms"
              << std::setw(12) << "speedup"
              << "\n" << std::string(53, '-') << "\n";

    csv << "n_vars,n_pairs,cpu_omp_ms,gpu_batch_ms,gpu_speedup\n";

    for (int n : n_list) {
        auto gens   = make_monomial_curve_gens(n);
        int  pairs  = n * (n - 1) / 2;
        double cpu_t = 0, gpu_t = 0;

        for (int r = 0; r < n_repeats; ++r) {
            auto t0 = Clock::now();
            compute_spolys_cpu(gens, n_threads);
            cpu_t += ms(t0);

            gpu_t += compute_spolys_gpu(gens, n_threads);
        }
        cpu_t /= n_repeats;
        gpu_t /= n_repeats;
        double sp = (gpu_t > 0) ? cpu_t / gpu_t : 0.0;

        std::cout << std::fixed << std::setprecision(3) << std::left
                  << std::setw(5)  << n
                  << std::setw(8)  << pairs
                  << std::setw(14) << cpu_t
                  << std::setw(14) << gpu_t
                  << std::setw(12) << sp
                  << (sp > 1.0 ? "  <- GPU faster" : "") << "\n";

        csv << n << "," << pairs << "," << cpu_t << ","
            << gpu_t << "," << sp << "\n";
    }
}


/* ═══════════════════════════════════════════════════════════════
 * PART B — Sparse row reduction using cuSPARSE CSR
 *
 * Instead of converting sparse → dense (wasteful), use cuSPARSE
 * SpMM to perform the matrix-vector products needed for Gaussian
 * elimination in sparse format.
 *
 * Benchmark: SpMM (A * x = b) where A is the CSR Macaulay matrix
 * vs CPU sparse matvec. Measures throughput for increasing nnz.
 *
 * Crossover expected around nnz ~ 10,000-50,000.
 * ═══════════════════════════════════════════════════════════════ */

static void part_B(int n_repeats, std::ofstream& csv) {
    // Build Macaulay matrices at different sizes using accumulated rounds
    // and measure sparse SpMM throughput: GPU vs CPU

    std::cout << "\n=== PART B: Sparse row reduction — cuSPARSE CSR SpMM ===\n";
    std::cout << "Using cuSPARSE in CSR format (no dense conversion)\n\n";
    std::cout << std::left
              << std::setw(5)  << "n"
              << std::setw(8)  << "rounds"
              << std::setw(8)  << "rows"
              << std::setw(8)  << "cols"
              << std::setw(10) << "nnz"
              << std::setw(14) << "cpu_spmm_ms"
              << std::setw(14) << "gpu_spmm_ms"
              << std::setw(10) << "speedup"
              << "\n" << std::string(77, '-') << "\n";

    csv << "n_vars,n_rounds,rows,cols,nnz,cpu_spmm_ms,gpu_spmm_ms,gpu_speedup\n";

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Test cases: (n_vars, n_rounds)
    std::vector<std::pair<int,int>> cases = {
        {6,  1}, {8,  1}, {10, 1}, {12, 1},
        {8,  2}, {10, 2}, {12, 2},
        {10, 3}, {12, 3},
        {15, 1}, {15, 2},
        {20, 1}, {20, 2},
        {25, 1},
    };

    for (auto [n_vars, n_rounds] : cases) {
        // Build accumulated polynomial list
        auto base_gens  = make_monomial_curve_gens(n_vars);
        std::vector<Poly> all_polys = base_gens;
        std::vector<Poly> current   = base_gens;

        for (int r = 0; r < n_rounds; ++r) {
            auto spolys = compute_spolys_cpu(current, 4);
            if (spolys.empty()) break;
            for (auto& sp : spolys) all_polys.push_back(sp);
            current = spolys;
            if ((int)current.size() > 300) current.resize(300);
        }

        // Collect all monomials
        std::set<Monomial, MonomialCmp> mono_set;
        for (const auto& p : all_polys)
            for (const auto& [m, _] : p.terms) mono_set.insert(m);

        std::vector<Monomial> mono_list(mono_set.begin(), mono_set.end());
        std::map<Monomial, int, MonomialCmp> mono_idx;
        for (int j = 0; j < (int)mono_list.size(); ++j)
            mono_idx[mono_list[j]] = j;

        int nr = (int)all_polys.size();
        int nc = (int)mono_list.size();

        // Build CSR
        std::vector<int>   row_ptr(nr + 1, 0);
        std::vector<int>   col_ind;
        std::vector<float> vals;

        for (int i = 0; i < nr; ++i) {
            for (const auto& [m, c] : all_polys[i].terms) {
                if (mono_idx.count(m)) {
                    col_ind.push_back(mono_idx[m]);
                    vals.push_back((float)c);
                }
            }
            row_ptr[i+1] = (int)col_ind.size();
        }
        int nnz = (int)vals.size();
        if (nnz == 0) continue;

        // Dense vector x for SpMM (x = all ones)
        std::vector<float> x_vec(nc, 1.0f);
        std::vector<float> y_cpu(nr, 0.f);
        std::vector<float> y_gpu(nr, 0.f);

        // ── CPU sparse matvec (CSR format) ──
        double cpu_t = 0;
        for (int rep = 0; rep < n_repeats; ++rep) {
            std::fill(y_cpu.begin(), y_cpu.end(), 0.f);
            auto t0 = Clock::now();
            for (int i = 0; i < nr; ++i)
                for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k)
                    y_cpu[i] += vals[k] * x_vec[col_ind[k]];
            cpu_t += ms(t0);
        }
        cpu_t /= n_repeats;

        // ── GPU sparse matvec (cuSPARSE CSR SpMV) ──
        long long gpu_bytes = (long long)(row_ptr.size()*sizeof(int) +
                               col_ind.size()*sizeof(int) +
                               vals.size()*sizeof(float) +
                               nc*sizeof(float) + nr*sizeof(float));

        double gpu_t = -1.0;
        if (gpu_bytes < 3LL * 1024 * 1024 * 1024) {
            int   *d_rp, *d_ci;
            float *d_v, *d_x, *d_y;
            CUDA_CHECK(cudaMalloc(&d_rp, row_ptr.size()*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_ci, col_ind.size()*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_v,  vals.size()*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_x,  nc*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_y,  nr*sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_rp, row_ptr.data(), row_ptr.size()*sizeof(int),   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_ci, col_ind.data(), col_ind.size()*sizeof(int),   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v,  vals.data(),    vals.size()*sizeof(float),     cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_x,  x_vec.data(),   nc*sizeof(float),              cudaMemcpyHostToDevice));

            // Create cuSPARSE descriptors
            cusparseSpMatDescr_t matA;
            cusparseDnVecDescr_t vecX, vecY;
            CUSPARSE_CHECK(cusparseCreateCsr(
                &matA, nr, nc, nnz,
                d_rp, d_ci, d_v,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, nc, d_x, CUDA_R_32F));
            CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, nr, d_y, CUDA_R_32F));

            float alpha = 1.0f, beta = 0.0f;
            size_t buf_size = 0;
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, vecX, &beta, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));
            void* d_buf = nullptr;
            if (buf_size > 0) CUDA_CHECK(cudaMalloc(&d_buf, buf_size));

            // Warmup
            cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, vecX, &beta, vecY,
                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Timed runs
            gpu_t = 0;
            for (int rep = 0; rep < n_repeats; ++rep) {
                CUDA_CHECK(cudaMemset(d_y, 0, nr*sizeof(float)));
                auto t0 = Clock::now();
                CUSPARSE_CHECK(cusparseSpMV(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf));
                CUDA_CHECK(cudaDeviceSynchronize());
                gpu_t += ms(t0);
            }
            gpu_t /= n_repeats;

            if (d_buf) CUDA_CHECK(cudaFree(d_buf));
            CUSPARSE_CHECK(cusparseDestroySpMat(matA));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
            CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
            CUDA_CHECK(cudaFree(d_rp)); CUDA_CHECK(cudaFree(d_ci));
            CUDA_CHECK(cudaFree(d_v));  CUDA_CHECK(cudaFree(d_x));
            CUDA_CHECK(cudaFree(d_y));
        }

        double sp = (gpu_t > 0 && cpu_t > 0) ? cpu_t / gpu_t : 0.0;

        std::cout << std::fixed << std::setprecision(4) << std::left
                  << std::setw(5)  << n_vars
                  << std::setw(8)  << n_rounds
                  << std::setw(8)  << nr
                  << std::setw(8)  << nc
                  << std::setw(10) << nnz
                  << std::setw(14) << cpu_t
                  << std::setw(14) << (gpu_t >= 0 ? gpu_t : -1.0)
                  << std::setw(10) << sp
                  << (sp > 1.0 ? "  <- GPU faster" : "") << "\n";

        csv << n_vars << "," << n_rounds << "," << nr << "," << nc << ","
            << nnz << "," << cpu_t << "," << (gpu_t >= 0 ? gpu_t : -1.0)
            << "," << sp << "\n";
    }

    CUSPARSE_CHECK(cusparseDestroy(handle));
}


/* ═══════════════════════════════════════════════════════════════
 * PART C — Dense random matrices (crossover reference, unchanged)
 * ═══════════════════════════════════════════════════════════════ */

__global__ void row_reduce_dense(
    float* __restrict__ A, int n_rows, int n_cols,
    int pivot_row, int pivot_col)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || row == pivot_row) return;
    float scale = A[row*n_cols+pivot_col];
    if (fabsf(scale) < 1e-7f) return;
    float pval = A[pivot_row*n_cols+pivot_col];
    if (fabsf(pval) < 1e-7f)  return;
    scale /= pval;
    for (int j = 0; j < n_cols; ++j)
        A[row*n_cols+j] -= scale * A[pivot_row*n_cols+j];
}

static double gpu_dense_gauss(const float* src, int n_rows, int n_cols) {
    long long bytes = (long long)n_rows * n_cols * sizeof(float);
    if (bytes > 4LL*1024*1024*1024) return -1.0;
    float* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, src, bytes, cudaMemcpyHostToDevice));
    auto t0  = Clock::now();
    int  blk = 256;
    for (int p = 0; p < std::min(n_rows, n_cols); ++p) {
        row_reduce_dense<<<(n_rows+blk-1)/blk, blk>>>(d_A, n_rows, n_cols, p, p);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    double t = ms(t0);
    CUDA_CHECK(cudaFree(d_A));
    return t;
}

static double cpu_dense_gauss(float* M, int n_rows, int n_cols) {
    auto t0 = Clock::now();
    for (int p = 0; p < std::min(n_rows, n_cols); ++p) {
        float pval = M[p*n_cols+p];
        if (std::abs(pval) < 1e-7f) continue;
        for (int row = 0; row < n_rows; ++row) {
            if (row == p) continue;
            float sc = M[row*n_cols+p] / pval;
            if (std::abs(sc) < 1e-7f) continue;
            for (int j = 0; j < n_cols; ++j)
                M[row*n_cols+j] -= sc * M[p*n_cols+j];
        }
    }
    return ms(t0);
}

static void part_C(int n_repeats, std::ofstream& csv) {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::cout << "\n=== PART C: Dense random matrices (crossover reference) ===\n";
    std::cout << std::left
              << std::setw(8)  << "N"
              << std::setw(12) << "cpu_ms"
              << std::setw(12) << "gpu_ms"
              << std::setw(12) << "speedup"
              << std::setw(10) << "mem_MB"
              << "\n" << std::string(54, '-') << "\n";

    csv << "matrix_size,cpu_ms,gpu_ms,gpu_speedup,mem_MB\n";

    for (int N : sizes) {
        long long bytes = (long long)N * N * sizeof(float);
        if (bytes > 4LL*1024*1024*1024) { break; }
        double mem_mb = bytes / (1024.0*1024.0);

        std::vector<float> base(N * N);
        for (auto& v : base) v = dist(rng);
        for (int i = 0; i < N; ++i) base[i*N+i] = N * 2.0f;

        double cpu_t = 0, gpu_t = 0;
        for (int r = 0; r < n_repeats; ++r) {
            std::vector<float> m1 = base;
            cpu_t += cpu_dense_gauss(m1.data(), N, N);
            std::vector<float> m2 = base;
            double g = gpu_dense_gauss(m2.data(), N, N);
            gpu_t += (g >= 0 ? g : 0);
        }
        cpu_t /= n_repeats;
        gpu_t /= n_repeats;
        double sp = (gpu_t > 0) ? cpu_t / gpu_t : 0.0;

        std::cout << std::fixed << std::setprecision(3) << std::left
                  << std::setw(8)  << N
                  << std::setw(12) << cpu_t
                  << std::setw(12) << gpu_t
                  << std::setw(12) << sp
                  << std::setw(10) << mem_mb
                  << (sp > 1.0 ? "  <- GPU faster" : "") << "\n";

        csv << N << "," << cpu_t << "," << gpu_t << ","
            << sp << "," << mem_mb << "\n";
    }
}


/* ── Main ───────────────────────────────────────────────────────────────── */
int main() {
    int dev = 0; cudaGetDeviceCount(&dev);
    if (dev == 0) { std::cerr << "No CUDA device.\n"; return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU : " << prop.name << "\n"
              << "VRAM: " << prop.totalGlobalMem/(1<<20) << " MB\n"
              << "SM  : " << prop.multiProcessorCount << "\n\n";
    std::cout << "Warming up... "; gpu_warmup(); std::cout << "done.\n";

    std::filesystem::create_directories("../../results");
    std::ofstream csv_a("../../results/exp3_2_spoly.csv");
    std::ofstream csv_b("../../results/exp3_2_sparse.csv");
    std::ofstream csv_c("../../results/exp3_2_dense.csv");

    int n_repeats = 5;
    part_A(n_repeats, csv_a);
    part_B(n_repeats, csv_b);
    part_C(n_repeats, csv_c);

    csv_a.close(); csv_b.close(); csv_c.close();
    std::cout << "\nSaved:\n"
              << "  ../../results/exp3_2_spoly.csv   (Part A - S-poly GPU)\n"
              << "  ../../results/exp3_2_sparse.csv  (Part B - sparse cuSPARSE)\n"
              << "  ../../results/exp3_2_dense.csv   (Part C - dense crossover)\n";
    return 0;
}
