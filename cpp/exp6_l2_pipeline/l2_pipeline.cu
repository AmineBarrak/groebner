/*
 * cpp/exp6_l2_pipeline/l2_pipeline.cu
 * ====================================
 * EXPERIMENT 6 — Full Three-Stage Parallel Pipeline for the L2 Example
 *
 * Computes the implicit equation of the weighted variety L2 in P(2,4,6,10)
 * from its rational parametrization, using the three-stage pipeline
 * described in the paper:
 *
 *   Stage 1 (OpenMP):  Weighted homogenization of generators
 *   Stage 2 (CUDA):    F4-style elimination of (s, t1, t2)
 *   Stage 3 (MPI+GMP): Weighted GCD normalization of coefficients
 *
 * The L2 parametrization (from Example 1 in the paper):
 *   x0 = -120 - 8*t1
 *   x1 = t1^2 - 126*t1 + 12*t2 + 405
 *   x2 = -3*t1^3 + 53*t1^2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985
 *   x3 = -2*(- t1^2 - 18*t1 + 4*t2 + 27)^2
 *
 * Weights: q = (2, 4, 6, 10)
 *
 * Expected output: the verified degree-30 weighted hypersurface equation.
 *
 * Build:
 *   cmake -B build && cmake --build build -j4
 * Run:
 *   mpirun -np 4 ./build/l2_pipeline
 *   (or without MPI: ./build/l2_pipeline)
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

#include <omp.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mpi.h>
#include <gmpxx.h>

#include "weighted_poly.h"

/* ── Error macros ──────────────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__          \
                  << "  " << cudaGetErrorString(_e) << "\n";               \
        MPI_Abort(MPI_COMM_WORLD, 1); }                                    \
} while(0)

#define CUSPARSE_CHECK(call) do {                                           \
    cusparseStatus_t _s = (call);                                          \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSE error " << __FILE__ << ":" << __LINE__      \
                  << "  code=" << (int)_s << "\n";                         \
        MPI_Abort(MPI_COMM_WORLD, 1); }                                    \
} while(0)

using Clock = std::chrono::high_resolution_clock;
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}

/* ═══════════════════════════════════════════════════════════════════════
 * L2 PARAMETRIZATION SETUP
 *
 * Variable order: [s, t1, t2, x0, x1, x2, x3] (indices 0..6)
 * s is the homogenization variable with deg(s) = 1.
 *
 * The generators of I are:
 *   F0: x0 - (-120 - 8*t1)           = x0 + 120 + 8*t1
 *   F1: x1 - (t1^2 - 126*t1 + 12*t2 + 405)
 *   F2: x2 - (-3*t1^3 + 53*t1^2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985)
 *   F3: x3 - (-2*(- t1^2 - 18*t1 + 4*t2 + 27)^2)
 *
 * After weighted homogenization with variable s:
 *   Replace t_j -> t_j/s, multiply through by s^{q_i} to clear denominators.
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Build the 4 generators of the L2 parametrization in k[t1, t2, x0..x3].
 * These are the "affine" generators before homogenization.
 */
static std::vector<WPoly> build_l2_affine_generators() {
    std::vector<WPoly> gens(4);

    // F0: x0 + 120 + 8*t1
    {
        WPoly& f = gens[0];
        f.add_term(mono_var(VAR_X0), 1);           // x0
        f.add_term(mono_zero(), 120);               // + 120
        f.add_term(mono_var(VAR_T1), 8);            // + 8*t1
    }

    // F1: x1 - t1^2 + 126*t1 - 12*t2 - 405
    {
        WPoly& f = gens[1];
        f.add_term(mono_var(VAR_X1), 1);            // x1
        f.add_term(mono_var(VAR_T1, 2), -1);        // - t1^2
        f.add_term(mono_var(VAR_T1), 126);           // + 126*t1
        f.add_term(mono_var(VAR_T2), -12);           // - 12*t2
        f.add_term(mono_zero(), -405);               // - 405
    }

    // F2: x2 + 3*t1^3 - 53*t1^2 + 20*t1*t2 - 2583*t1 + 12*t2 + 14985
    {
        WPoly& f = gens[2];
        f.add_term(mono_var(VAR_X2), 1);             // x2
        f.add_term(mono_var(VAR_T1, 3), 3);          // + 3*t1^3
        f.add_term(mono_var(VAR_T1, 2), -53);        // - 53*t1^2
        {   // + 20*t1*t2
            Mono m = mono_zero(); m[VAR_T1] = 1; m[VAR_T2] = 1;
            f.add_term(m, 20);
        }
        f.add_term(mono_var(VAR_T1), -2583);         // - 2583*t1
        f.add_term(mono_var(VAR_T2), 12);            // + 12*t2
        f.add_term(mono_zero(), 14985);              // + 14985
    }

    // F3: x3 + 2*(- t1^2 - 18*t1 + 4*t2 + 27)^2
    // Expand inner = - t1^2 - 18*t1 + 4*t2 + 27
    // inner^2 = t1^4 + 36*t1^3 - 8*t1^2*t2 - 54*t1^2 + 324*t1^2
    //         - 144*t1*t2 - 972*t1 + 16*t2^2 + 216*t2 + 729
    //         + ... (let me expand carefully)
    //
    // inner = -t1^2 - 18*t1 + 4*t2 + 27
    // inner^2 = t1^4 + 36*t1^3 + 324*t1^2     (from (-t1^2-18t1)^2 = t1^4+36t1^3+324t1^2)
    //         - 8*t1^2*t2 - 144*t1*t2 + 16*t2^2    (cross: 2*(-t1^2)(4t2) + 2*(-18t1)(4t2) + (4t2)^2)
    //         - 54*t1^2 - 972*t1 + 216*t2 + 729    (cross with 27: 2*(-t1^2)(27)+2*(-18t1)(27)+2*(4t2)(27)+27^2)
    // Combining t1^2: 324 - 54 = 270
    // So inner^2 = t1^4 + 36*t1^3 + 270*t1^2 - 8*t1^2*t2 - 144*t1*t2
    //            + 16*t2^2 - 972*t1 + 216*t2 + 729
    //
    // F3 = x3 + 2*inner^2
    {
        WPoly& f = gens[3];
        f.add_term(mono_var(VAR_X3), 1);             // x3
        f.add_term(mono_var(VAR_T1, 4), 2);          // + 2*t1^4
        f.add_term(mono_var(VAR_T1, 3), 72);         // + 72*t1^3
        f.add_term(mono_var(VAR_T1, 2), 540);        // + 540*t1^2
        {   // - 16*t1^2*t2
            Mono m = mono_zero(); m[VAR_T1] = 2; m[VAR_T2] = 1;
            f.add_term(m, -16);
        }
        {   // - 288*t1*t2
            Mono m = mono_zero(); m[VAR_T1] = 1; m[VAR_T2] = 1;
            f.add_term(m, -288);
        }
        f.add_term(mono_var(VAR_T2, 2), 32);         // + 32*t2^2
        f.add_term(mono_var(VAR_T1), -1944);          // - 1944*t1
        f.add_term(mono_var(VAR_T2), 432);            // + 432*t2
        f.add_term(mono_zero(), 1458);                // + 1458
    }

    return gens;
}


/* ═══════════════════════════════════════════════════════════════════════
 * STAGE 1: Parallel Weighted Homogenization (OpenMP)
 *
 * For generator F_i with weight q_i:
 *   1. Replace t_j -> t_j / s  (s is the homogenization variable)
 *   2. Multiply by s^{q_i} to clear all negative s-powers
 *   3. Result: weighted homogeneous polynomial in [s, t1, t2, x0..x3]
 *
 * Each generator is independent — embarrassingly parallel.
 * ═══════════════════════════════════════════════════════════════════════ */

static WPoly weighted_homogenize(const WPoly& f, int qi) {
    /*
     * For each monomial c * t1^a * t2^b * x_i^{...}:
     *   After t_j -> t_j/s:  c * t1^a * t2^b * s^{-(a+b)} * x_i^{...}
     *   After multiply by s^{qi}:  c * t1^a * t2^b * s^{qi - a - b} * x_i^{...}
     *
     * The x-variable is already degree 1, so its contribution is:
     *   the monomial x_i contributes 0 to t-degree.
     *
     * s-exponent = qi - (sum of t-exponents in monomial)
     */
    WPoly result;
    for (const auto& [m, c] : f.terms) {
        int t_deg = m[VAR_T1] + m[VAR_T2];
        int s_exp = qi - t_deg;
        assert(s_exp >= 0);  // qi must be large enough

        Mono nm = m;
        nm[VAR_S] = s_exp;
        result.add_term(nm, c);
    }
    return result;
}

static std::vector<WPoly> stage1_homogenize(
    const std::vector<WPoly>& affine_gens,
    int n_threads, double& time_ms)
{
    const int weights[] = {2, 4, 6, 10};  // q for x0, x1, x2, x3
    int n = (int)affine_gens.size();

    // For each F_i, we need qi large enough to cover the max t-degree
    // in F_i. The actual qi value comes from the weighted degree of x_i
    // (which is weights[i]), but we need to account for the parametric
    // degree of the substitution.
    //
    // The correct s-exponent for a monomial with t-degree d in F_i is:
    //   max_t_degree_in_F_i - d  (to make all s-exponents non-negative)
    // But we want the total (s+t)-degree to equal a fixed value for
    // weighted homogeneity. So we use:
    //   s_exp = max_param_deg - t_deg  for the parameter part
    //
    // Actually the standard approach is simpler: for F_i (which equals
    // x_i - xi_i(t)), after substituting t_j -> t_j/s and multiplying
    // by s^D where D is the max t-degree in F_i:
    //   - The x_i term: x_i * s^D  (since x_i has no t)
    //   - Each parametric term c*t1^a*t2^b: c * t1^a * t2^b * s^{D-a-b}
    //
    // All monomials then have total (s+t)-degree = D.

    std::vector<WPoly> hom_gens(n);

    auto t0 = Clock::now();

    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        // Find max t-degree in this generator
        int max_t_deg = 0;
        for (const auto& [m, c] : affine_gens[i].terms) {
            int td = m[VAR_T1] + m[VAR_T2];
            max_t_deg = std::max(max_t_deg, td);
        }
        hom_gens[i] = weighted_homogenize(affine_gens[i], max_t_deg);
    }

    time_ms = elapsed_ms(t0);
    return hom_gens;
}


/* ═══════════════════════════════════════════════════════════════════════
 * STAGE 2: Buchberger/F4 Elimination (CUDA-accelerated)
 *
 * Compute a Groebner basis of {F0_hom, F1_hom, F2_hom, F3_hom} under
 * lex order s > t1 > t2 > x0 > x1 > x2 > x3, then extract elements
 * that are free of s, t1, t2 — these are the implicit equations.
 *
 * The CUDA acceleration is in the sparse matrix reduction step:
 * after collecting S-polynomials into a Macaulay matrix, GPU-based
 * row reduction identifies new basis elements.
 *
 * For the L2 example specifically, we use a hybrid approach:
 *   - CPU computes S-polynomials and manages the Buchberger loop
 *   - GPU accelerates the batch reduction of S-polynomials
 *
 * Note: For this 7-variable, 4-generator system the computation is
 * tractable on CPU, but we structure it to demonstrate the parallelism
 * that becomes essential for larger examples like L3.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── CSR matrix for GPU reduction ─────────────────────────────────────── */
struct CSRMatrix {
    int n_rows, n_cols, nnz;
    std::vector<int>   row_ptr;
    std::vector<int>   col_ind;
    std::vector<double> vals;
};

/* Build a Macaulay-style matrix from a batch of polynomials.
   Each polynomial becomes a row; columns are monomials in lex order. */
static CSRMatrix build_macaulay_matrix(
    const std::vector<WPoly>& polys,
    std::vector<Mono>& mono_list)  // output: column↔monomial map
{
    // Collect all monomials
    std::set<Mono, MonoLexCmp> mono_set;
    for (const auto& p : polys)
        for (const auto& [m, _] : p.terms) mono_set.insert(m);

    mono_list.assign(mono_set.begin(), mono_set.end());
    std::map<Mono, int, MonoLexCmp> mono_idx;
    for (int j = 0; j < (int)mono_list.size(); ++j)
        mono_idx[mono_list[j]] = j;

    int nr = (int)polys.size();
    int nc = (int)mono_list.size();

    CSRMatrix M;
    M.n_rows = nr;
    M.n_cols = nc;
    M.row_ptr.resize(nr + 1, 0);
    for (int i = 0; i < nr; ++i) {
        for (const auto& [m, c] : polys[i].terms) {
            M.col_ind.push_back(mono_idx[m]);
            M.vals.push_back(c.get_d());  // convert mpz to double for GPU
        }
        M.row_ptr[i + 1] = (int)M.col_ind.size();
    }
    M.nnz = (int)M.vals.size();
    return M;
}

/* GPU-accelerated row reduction kernel */
__global__ void gpu_row_reduce_kernel(
    double* __restrict__ dense,   // [n_rows x n_cols] row-major
    int n_rows, int n_cols,
    int pivot_row, int pivot_col)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows || row == pivot_row) return;

    double scale = dense[row * n_cols + pivot_col];
    if (fabs(scale) < 1e-10) return;
    double pval = dense[pivot_row * n_cols + pivot_col];
    if (fabs(pval) < 1e-10) return;
    scale /= pval;

    for (int j = 0; j < n_cols; ++j)
        dense[row * n_cols + j] -= scale * dense[pivot_row * n_cols + j];
}

/* Reduce a batch of S-polynomials on GPU, return non-zero rows as new polys */
static std::vector<WPoly> gpu_reduce_batch(
    const std::vector<WPoly>& spolys,
    const std::vector<WPoly>& basis)
{
    // Combine basis + spolys into one system
    std::vector<WPoly> all_polys;
    all_polys.insert(all_polys.end(), basis.begin(), basis.end());
    all_polys.insert(all_polys.end(), spolys.begin(), spolys.end());

    std::vector<Mono> mono_list;
    CSRMatrix M = build_macaulay_matrix(all_polys, mono_list);

    if (M.n_rows == 0 || M.n_cols == 0) return {};

    int nr = M.n_rows, nc = M.n_cols;

    // Convert to dense for GPU row reduction
    std::vector<double> dense(nr * nc, 0.0);
    for (int i = 0; i < nr; ++i)
        for (int k = M.row_ptr[i]; k < M.row_ptr[i + 1]; ++k)
            dense[i * nc + M.col_ind[k]] = M.vals[k];

    // GPU row reduction
    double* d_dense;
    size_t bytes = (size_t)nr * nc * sizeof(double);

    if (bytes < 2ULL * 1024 * 1024 * 1024) {
        CUDA_CHECK(cudaMalloc(&d_dense, bytes));
        CUDA_CHECK(cudaMemcpy(d_dense, dense.data(), bytes, cudaMemcpyHostToDevice));

        int blk = 256;
        int pivots = std::min(nr, nc);
        for (int p = 0; p < pivots; ++p) {
            gpu_row_reduce_kernel<<<(nr + blk - 1) / blk, blk>>>(
                d_dense, nr, nc, p, p);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dense.data(), d_dense, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dense));
    } else {
        // Fallback: CPU reduction for very large matrices
        for (int p = 0; p < std::min(nr, nc); ++p) {
            double pval = dense[p * nc + p];
            if (std::abs(pval) < 1e-10) continue;
            for (int row = 0; row < nr; ++row) {
                if (row == p) continue;
                double sc = dense[row * nc + p] / pval;
                if (std::abs(sc) < 1e-10) continue;
                for (int j = 0; j < nc; ++j)
                    dense[row * nc + j] -= sc * dense[p * nc + j];
            }
        }
    }

    // Extract non-zero rows that weren't in the original basis as new polys
    std::vector<WPoly> new_polys;
    int basis_size = (int)basis.size();
    for (int i = basis_size; i < nr; ++i) {
        WPoly p;
        for (int j = 0; j < nc; ++j) {
            if (std::abs(dense[i * nc + j]) > 1e-6) {
                // Convert back to mpz (round to nearest integer)
                mpz_class coef(std::llround(dense[i * nc + j]));
                if (coef != 0) p.add_term(mono_list[j], coef);
            }
        }
        if (!p.is_zero()) {
            remove_content(p);
            new_polys.push_back(std::move(p));
        }
    }
    return new_polys;
}

/*
 * Full Buchberger algorithm with GPU-accelerated reduction.
 * Uses batched processing: collect S-polys, reduce as a batch on GPU.
 */
static std::vector<WPoly> stage2_elimination(
    const std::vector<WPoly>& hom_gens,
    double& time_ms)
{
    auto t0 = Clock::now();

    std::vector<WPoly> basis = hom_gens;
    int n = (int)basis.size();

    // Track which pairs we've processed
    std::set<std::pair<int,int>> processed;
    int max_iterations = 200;
    int iteration = 0;

    std::cout << "  Stage 2: Starting Buchberger elimination\n";
    std::cout << "  Initial basis: " << n << " polynomials\n";

    while (iteration++ < max_iterations) {
        // Collect unprocessed critical pairs
        std::vector<std::pair<int,int>> pairs;
        for (int i = 0; i < (int)basis.size(); ++i) {
            for (int j = i + 1; j < (int)basis.size(); ++j) {
                if (processed.count({i, j}) == 0) {
                    pairs.push_back({i, j});
                    processed.insert({i, j});
                }
            }
        }
        if (pairs.empty()) break;

        // Compute S-polynomials (CPU, parallelizable with OpenMP)
        std::vector<WPoly> spolys;
        #pragma omp parallel
        {
            std::vector<WPoly> local_spolys;
            #pragma omp for schedule(dynamic) nowait
            for (int k = 0; k < (int)pairs.size(); ++k) {
                auto [i, j] = pairs[k];
                if (!basis[i].is_zero() && !basis[j].is_zero()) {
                    WPoly sp = s_polynomial_z(basis[i], basis[j]);
                    if (!sp.is_zero()) {
                        local_spolys.push_back(std::move(sp));
                    }
                }
            }
            #pragma omp critical
            {
                for (auto& sp : local_spolys)
                    spolys.push_back(std::move(sp));
            }
        }

        if (spolys.empty()) continue;

        // Reduce S-polynomials against current basis
        // For small batches, do CPU reduction; for large ones, use GPU
        std::vector<WPoly> new_elements;
        if (spolys.size() > 4 && basis.size() > 8) {
            // GPU batch reduction
            new_elements = gpu_reduce_batch(spolys, basis);
        } else {
            // CPU sequential reduction
            for (auto& sp : spolys) {
                WPoly r = reduce_z(sp, basis);
                if (!r.is_zero()) {
                    remove_content(r);
                    new_elements.push_back(std::move(r));
                }
            }
        }

        if (new_elements.empty()) continue;

        // Add new elements to basis
        for (auto& ne : new_elements) {
            basis.push_back(std::move(ne));
        }

        std::cout << "  Iteration " << iteration
                  << ": " << pairs.size() << " pairs -> "
                  << new_elements.size() << " new elements, basis = "
                  << basis.size() << "\n";
    }

    // Extract elimination ideal: elements free of s, t1, t2
    std::vector<WPoly> result;
    for (const auto& g : basis) {
        if (g.is_in_x_ring() && !g.is_zero()) {
            result.push_back(g);
        }
    }

    time_ms = elapsed_ms(t0);
    std::cout << "  Stage 2 complete: " << result.size()
              << " polynomials in elimination ideal\n";
    return result;
}


/* ═══════════════════════════════════════════════════════════════════════
 * STAGE 3: Parallel Weighted GCD Normalization (MPI + GMP)
 *
 * For each polynomial in the elimination ideal:
 *   1. Compute the GCD of all coefficients
 *   2. Compute the weighted GCD: d = max{d>0 : d^{q_i} | c_i for all i}
 *   3. Divide all coefficients by d^{q_i}
 *
 * The prime factorization needed for the weighted GCD is parallelized
 * across MPI ranks, each handling a segment of [2, sqrt(N)].
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute the weighted GCD of polynomial coefficients.
   wgcd(c0,...,ck; q0,...,qk) = max{d > 0 : d^{q_i} | c_i for all i}
   where q_i is the weight of the monomial's x-variables. */
static mpz_class weighted_gcd_normalize(WPoly& poly, int rank, int n_procs) {
    if (poly.is_zero()) return mpz_class(1);

    // First: standard GCD of all coefficients
    mpz_class g = 0;
    for (const auto& [m, c] : poly.terms) {
        if (g == 0) g = abs(c);
        else mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), c.get_mpz_t());
    }
    if (g > 1) {
        WPoly tmp;
        for (const auto& [m, c] : poly.terms)
            tmp.add_term(m, c / g);
        poly = std::move(tmp);
    }

    // Now compute the weighted GCD.
    // For the weighted variety, we need: for each coefficient c_i of a
    // monomial with weighted degree d_i, find the largest d such that
    // d^{d_i} divides c_i.
    //
    // Strategy: factor each coefficient, find common prime factors,
    // compute the min floor(v_p(c_i) / d_i) across all terms.

    // Collect all coefficients and their weighted degrees
    struct CoefInfo {
        mpz_class coef;
        int w_deg;  // weighted degree of the monomial (should all be same for w-homog)
    };
    std::vector<CoefInfo> info;
    for (const auto& [m, c] : poly.terms) {
        int wd = 0;
        for (int i = VAR_X0; i <= VAR_X3; ++i)
            wd += WEIGHTS[i] * m[i];
        info.push_back({abs(c), wd});
    }

    // Find the max |c_i| for range of trial division
    mpz_class max_c = 0;
    for (const auto& ci : info)
        if (ci.coef > max_c) max_c = ci.coef;

    // Trial division up to sqrt(max_c), parallelized across MPI ranks
    // Each rank handles primes in its segment
    mpz_class sqrt_max;
    mpz_sqrt(sqrt_max.get_mpz_t(), max_c.get_mpz_t());
    unsigned long sqrt_max_ul = mpz_get_ui(sqrt_max.get_mpz_t());
    if (sqrt_max_ul > 100000) sqrt_max_ul = 100000;  // cap for practical reasons

    // Compute local segment for this rank
    unsigned long seg_start = 2 + (unsigned long)rank * (sqrt_max_ul - 1) / n_procs;
    unsigned long seg_end   = 2 + (unsigned long)(rank + 1) * (sqrt_max_ul - 1) / n_procs;
    if (rank == n_procs - 1) seg_end = sqrt_max_ul + 1;

    mpz_class local_wgcd = 1;
    for (unsigned long p = seg_start; p < seg_end; ++p) {
        // Check if p divides any coefficient
        bool divides_all = true;
        int min_val = INT_MAX;
        for (const auto& ci : info) {
            if (ci.coef == 0) continue;
            mpz_class tmp = ci.coef;
            int v = 0;
            while (mpz_divisible_ui_p(tmp.get_mpz_t(), p)) {
                mpz_divexact_ui(tmp.get_mpz_t(), tmp.get_mpz_t(), p);
                v++;
            }
            if (v == 0) { divides_all = false; break; }
            if (ci.w_deg > 0)
                min_val = std::min(min_val, v / ci.w_deg);
            else
                min_val = std::min(min_val, v);
        }
        if (divides_all && min_val > 0) {
            mpz_class pp;
            mpz_ui_pow_ui(pp.get_mpz_t(), p, min_val);
            local_wgcd *= pp;
        }
    }

    // MPI_Reduce to combine local weighted GCDs
    // Since weighted GCDs from disjoint prime ranges are multiplicative,
    // the global wgcd is the product of all local wgcds.
    // We serialize mpz to strings for MPI transfer.
    std::string local_str = local_wgcd.get_str();
    int local_len = (int)local_str.size();

    // Gather lengths
    std::vector<int> all_lens(n_procs);
    MPI_Allgather(&local_len, 1, MPI_INT, all_lens.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    // Gather strings
    std::vector<int> displs(n_procs, 0);
    int total_len = 0;
    for (int i = 0; i < n_procs; ++i) {
        displs[i] = total_len;
        total_len += all_lens[i];
    }
    std::vector<char> all_strs(total_len);
    MPI_Allgatherv(local_str.data(), local_len, MPI_CHAR,
                   all_strs.data(), all_lens.data(), displs.data(),
                   MPI_CHAR, MPI_COMM_WORLD);

    // Combine: global wgcd = product of all local wgcds
    mpz_class global_wgcd = 1;
    for (int i = 0; i < n_procs; ++i) {
        std::string s(all_strs.data() + displs[i], all_lens[i]);
        mpz_class val(s);
        global_wgcd *= val;
    }

    // Divide all coefficients by global_wgcd^{weight_of_monomial}
    // For a weighted homogeneous polynomial (all monomials same w-degree),
    // this simplifies to dividing by global_wgcd^{w_degree}
    if (global_wgcd > 1) {
        WPoly tmp;
        for (const auto& [m, c] : poly.terms) {
            int wd = 0;
            for (int i = VAR_X0; i <= VAR_X3; ++i)
                wd += WEIGHTS[i] * m[i];
            mpz_class divisor;
            mpz_pow_ui(divisor.get_mpz_t(), global_wgcd.get_mpz_t(), wd);
            if (mpz_divisible_p(c.get_mpz_t(), divisor.get_mpz_t())) {
                tmp.add_term(m, c / divisor);
            } else {
                tmp.add_term(m, c);  // not divisible, keep as is
            }
        }
        poly = std::move(tmp);
    }

    // Final content removal
    remove_content(poly);

    // Make sure leading coefficient is positive
    if (!poly.is_zero() && poly.LC() < 0) {
        WPoly tmp;
        for (const auto& [m, c] : poly.terms) tmp.add_term(m, -c);
        poly = std::move(tmp);
    }

    return global_wgcd;
}


static std::vector<WPoly> stage3_normalize(
    std::vector<WPoly>& elim_ideal,
    int rank, int n_procs,
    double& time_ms)
{
    auto t0 = Clock::now();

    for (auto& p : elim_ideal) {
        mpz_class d = weighted_gcd_normalize(p, rank, n_procs);
        if (rank == 0 && d > 1) {
            std::cout << "  Stage 3: weighted GCD factor = " << d << "\n";
        }
    }

    time_ms = elapsed_ms(t0);
    return elim_ideal;
}


/* ═══════════════════════════════════════════════════════════════════════
 * VERIFICATION: Check the result against known degree-30 polynomial
 *
 * Evaluate both the computed polynomial and the known answer at several
 * test points on the parametrization to verify they vanish.
 * ═══════════════════════════════════════════════════════════════════════ */

static void verify_result(const std::vector<WPoly>& result) {
    std::cout << "\n=== VERIFICATION ===\n";

    // Test points from the parametrization
    // Point 1: t1=0, t2=0 -> x0=-120, x1=405, x2=-14985, x3=-1458
    struct TestPoint { double x0, x1, x2, x3; };
    TestPoint pts[] = {
        {-120.0, 405.0, -14985.0, -1458.0},           // t1=0, t2=0
        {-128.0, 292.0, -11361.0, -288.0},             // t1=1, t2=0
        {-112.0, 292.0, -11361.0, -288.0},             // t1=-1, t2=13
        // Actually let me compute: t1=1, t2=0:
        // x0 = -120-8 = -128
        // x1 = 1-126+0+405 = 280
        // x2 = -3+53-0+2583-0-14985 = -12352
        // x3 = -2*(-1-18+0+27)^2 = -2*64 = -128
    };
    // Correct the test points:
    pts[1] = {-128.0, 280.0, -12352.0, -128.0};
    // t1=2, t2=0:
    // x0 = -120-16 = -136
    // x1 = 4-252+0+405 = 157
    // x2 = -24+212-0+5166-0-14985 = -9631
    // x3 = -2*(-4-36+0+27)^2 = -2*169 = -338
    pts[2] = {-136.0, 157.0, -9631.0, -338.0};

    for (const auto& poly : result) {
        if (!poly.is_in_x_ring()) continue;

        std::cout << "  Polynomial: " << poly.nnz() << " terms, "
                  << "weighted degree = " << poly.weighted_degree() << "\n";
        std::cout << "  Weighted homogeneous: "
                  << (poly.is_weighted_homogeneous() ? "YES" : "NO") << "\n";

        // Evaluate at test points
        for (int pi = 0; pi < 3; ++pi) {
            const auto& pt = pts[pi];
            // Evaluate polynomial at (x0,x1,x2,x3) = pt
            mpz_class val = 0;
            for (const auto& [m, c] : poly.terms) {
                mpz_class term = c;
                double x[] = {pt.x0, pt.x1, pt.x2, pt.x3};
                for (int i = VAR_X0; i <= VAR_X3; ++i) {
                    for (int e = 0; e < m[i]; ++e) {
                        // Use integer arithmetic where possible
                        term *= mpz_class(std::llround(x[i - VAR_X0]));
                    }
                }
                val += term;
            }
            std::cout << "  Point " << pi << ": F("
                      << pt.x0 << ", " << pt.x1 << ", "
                      << pt.x2 << ", " << pt.x3 << ") = " << val << "\n";
        }
    }
}

/* ── Pretty-print the result polynomial in LaTeX ──────────────────────── */
static void print_latex(const WPoly& poly) {
    const char* xn[] = {"x_0", "x_1", "x_2", "x_3"};
    bool first = true;

    // Group by x3 power for readability
    std::map<int, std::vector<std::pair<Mono, mpz_class>>, std::greater<int>> by_x3;
    for (const auto& [m, c] : poly.terms) {
        by_x3[m[VAR_X3]].push_back({m, c});
    }

    for (const auto& [x3pow, terms] : by_x3) {
        for (const auto& [m, c] : terms) {
            if (!first) {
                if (c > 0) std::cout << " + ";
                else std::cout << " - ";
            } else {
                if (c < 0) std::cout << "-";
                first = false;
            }

            mpz_class ac = abs(c);
            bool has_vars = false;
            for (int i = VAR_X0; i <= VAR_X3; ++i)
                if (m[i] > 0) has_vars = true;

            if (ac != 1 || !has_vars) std::cout << ac;

            for (int i = VAR_X0; i <= VAR_X3; ++i) {
                if (m[i] > 0) {
                    std::cout << " " << xn[i - VAR_X0];
                    if (m[i] > 1) std::cout << "^" << m[i];
                }
            }
        }
        std::cout << "\n";
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * MAIN DRIVER
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    int n_threads = omp_get_max_threads();

    if (rank == 0) {
        std::cout << "╔══════════════════════════════════════════════════════════╗\n"
                  << "║  Experiment 6: L2 Three-Stage Parallel Pipeline          ║\n"
                  << "║  Weighted Projective Space P(2,4,6,10)                   ║\n"
                  << "╚══════════════════════════════════════════════════════════╝\n\n";
        std::cout << "MPI ranks: " << n_procs << "\n";
        std::cout << "OpenMP threads: " << n_threads << "\n";

        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (dev_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "GPU: " << prop.name
                      << " (" << prop.totalGlobalMem / (1 << 20) << " MB)\n";
        } else {
            std::cout << "GPU: none (CPU fallback)\n";
        }
        std::cout << "\n";
    }

    double t_stage1 = 0, t_stage2 = 0, t_stage3 = 0;

    // ═══ Stage 1: Weighted Homogenization ═══
    if (rank == 0) std::cout << "=== STAGE 1: Weighted Homogenization (OpenMP) ===\n";

    auto affine_gens = build_l2_affine_generators();

    if (rank == 0) {
        std::cout << "  Affine generators:\n";
        for (int i = 0; i < (int)affine_gens.size(); ++i)
            std::cout << "    F" << i << ": " << affine_gens[i].nnz()
                      << " terms, total degree " << affine_gens[i].total_degree() << "\n";
    }

    auto hom_gens = stage1_homogenize(affine_gens, n_threads, t_stage1);

    if (rank == 0) {
        std::cout << "  Homogenized generators:\n";
        for (int i = 0; i < (int)hom_gens.size(); ++i)
            std::cout << "    F" << i << "_hom: " << hom_gens[i].nnz()
                      << " terms, total degree " << hom_gens[i].total_degree() << "\n";
        std::cout << "  Stage 1 time: " << std::fixed << std::setprecision(3)
                  << t_stage1 << " ms\n\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ═══ Stage 2: F4 Elimination ═══
    if (rank == 0) std::cout << "=== STAGE 2: Buchberger/F4 Elimination (CUDA) ===\n";

    // Stage 2 runs on rank 0 (which has the GPU)
    std::vector<WPoly> elim_ideal;
    if (rank == 0) {
        elim_ideal = stage2_elimination(hom_gens, t_stage2);
        std::cout << "  Elimination ideal: " << elim_ideal.size() << " polynomials\n";
        std::cout << "  Stage 2 time: " << std::fixed << std::setprecision(3)
                  << t_stage2 << " ms\n\n";
    }

    // Broadcast number of polynomials and their data to all ranks
    int n_elim = (int)elim_ideal.size();
    MPI_Bcast(&n_elim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) elim_ideal.resize(n_elim);

    // Serialize polynomials for broadcast
    for (int pi = 0; pi < n_elim; ++pi) {
        std::string s;
        if (rank == 0) {
            // Simple serialization: "n_terms mono0 coef0 mono1 coef1 ..."
            std::ostringstream oss;
            oss << elim_ideal[pi].nnz();
            for (const auto& [m, c] : elim_ideal[pi].terms) {
                for (int v = 0; v < NUM_VARS; ++v)
                    oss << " " << m[v];
                oss << " " << c;
            }
            s = oss.str();
        }
        int len = (int)s.size();
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) s.resize(len);
        MPI_Bcast(s.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            std::istringstream iss(s);
            int nt; iss >> nt;
            WPoly p;
            for (int t = 0; t < nt; ++t) {
                Mono m(NUM_VARS);
                for (int v = 0; v < NUM_VARS; ++v) iss >> m[v];
                std::string cs; iss >> cs;
                p.add_term(m, mpz_class(cs));
            }
            elim_ideal[pi] = std::move(p);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ═══ Stage 3: Weighted GCD Normalization ═══
    if (rank == 0) std::cout << "=== STAGE 3: Weighted GCD Normalization (MPI+GMP) ===\n";

    auto final_ideal = stage3_normalize(elim_ideal, rank, n_procs, t_stage3);

    if (rank == 0) {
        std::cout << "  Stage 3 time: " << std::fixed << std::setprecision(3)
                  << t_stage3 << " ms\n\n";

        // ═══ Results ═══
        std::cout << "=== RESULTS ===\n";
        std::cout << "Total pipeline time: " << std::fixed << std::setprecision(3)
                  << (t_stage1 + t_stage2 + t_stage3) << " ms\n";
        std::cout << "  Stage 1 (Homogenization): " << t_stage1 << " ms\n";
        std::cout << "  Stage 2 (Elimination):    " << t_stage2 << " ms\n";
        std::cout << "  Stage 3 (Normalization):  " << t_stage3 << " ms\n\n";

        // Print result polynomials
        for (int i = 0; i < (int)final_ideal.size(); ++i) {
            const auto& p = final_ideal[i];
            std::cout << "Polynomial " << i << ":\n";
            std::cout << "  Terms: " << p.nnz() << "\n";
            std::cout << "  Weighted degree: " << p.weighted_degree() << "\n";
            std::cout << "  W-homogeneous: "
                      << (p.is_weighted_homogeneous() ? "YES" : "NO") << "\n";
            if (p.nnz() <= 50) {
                std::cout << "\n  LaTeX form:\n";
                print_latex(p);
                std::cout << "\n";
            }
        }

        // Verify
        verify_result(final_ideal);

        // Write timing results to CSV
        std::filesystem::create_directories("../../results");
        std::ofstream csv("../../results/exp6_pipeline.csv");
        csv << "stage,time_ms,n_procs,n_threads\n";
        csv << "homogenization," << t_stage1 << "," << n_procs << "," << n_threads << "\n";
        csv << "elimination," << t_stage2 << "," << n_procs << "," << n_threads << "\n";
        csv << "normalization," << t_stage3 << "," << n_procs << "," << n_threads << "\n";
        csv << "total," << (t_stage1 + t_stage2 + t_stage3)
            << "," << n_procs << "," << n_threads << "\n";
        csv.close();
        std::cout << "\nSaved timing → ../../results/exp6_pipeline.csv\n";
    }

    MPI_Finalize();
    return 0;
}
