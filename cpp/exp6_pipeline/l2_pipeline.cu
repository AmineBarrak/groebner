/*
 * cpp/exp6_l2_pipeline/l2_pipeline.cu
 * ====================================
 * EXPERIMENT 6 -- Full Three-Stage Parallel Pipeline for the L2 Example
 *
 * Computes the implicit equation of the weighted variety L2 in P(2,4,6,10)
 * from its rational parametrization:
 *
 *   Stage 1 (OpenMP):  Weighted homogenization of generators
 *   Stage 2 (CUDA):    Buchberger elimination of (s, t1, t2)
 *   Stage 3 (MPI+GMP): Weighted GCD normalization of coefficients
 *
 * The L2 parametrization (Example 1 in the paper):
 *   x0 = -120 - 8*t1
 *   x1 = t1^2 - 126*t1 + 12*t2 + 405
 *   x2 = -3*t1^3 + 53*t1^2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985
 *   x3 = -2*(-t1^2 - 18*t1 + 4*t2 + 27)^2
 *
 * Weights: q = (2, 4, 6, 10)
 * Expected output: degree-30 weighted homogeneous polynomial, 34 terms.
 *
 * Target: NVIDIA Tesla V100 (sm_70) on Matilda HPC
 *
 * Build:
 *   cmake -B build && cmake --build build -j8
 * Run:
 *   mpirun ./build/l2_pipeline
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

/* -- Error macros ------------------------------------------------------ */
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


/* =====================================================================
 * L2 PARAMETRIZATION SETUP
 * ===================================================================== */

static std::vector<WPoly> build_l2_affine_generators() {
    std::vector<WPoly> gens(4);

    // F0: x0 + 120 + 8*t1
    {
        WPoly& f = gens[0];
        f.add_term(mono_var(VAR_X0), 1);
        f.add_term(mono_zero(), 120);
        f.add_term(mono_var(VAR_T1), 8);
    }

    // F1: x1 - t1^2 + 126*t1 - 12*t2 - 405
    {
        WPoly& f = gens[1];
        f.add_term(mono_var(VAR_X1), 1);
        f.add_term(mono_var(VAR_T1, 2), -1);
        f.add_term(mono_var(VAR_T1), 126);
        f.add_term(mono_var(VAR_T2), -12);
        f.add_term(mono_zero(), -405);
    }

    // F2: x2 + 3*t1^3 - 53*t1^2 + 20*t1*t2 - 2583*t1 + 12*t2 + 14985
    {
        WPoly& f = gens[2];
        f.add_term(mono_var(VAR_X2), 1);
        f.add_term(mono_var(VAR_T1, 3), 3);
        f.add_term(mono_var(VAR_T1, 2), -53);
        { Mono m = mono_zero(); m[VAR_T1] = 1; m[VAR_T2] = 1; f.add_term(m, 20); }
        f.add_term(mono_var(VAR_T1), -2583);
        f.add_term(mono_var(VAR_T2), 12);
        f.add_term(mono_zero(), 14985);
    }

    // F3: x3 + 2*(-t1^2 - 18*t1 + 4*t2 + 27)^2
    // Expanded: x3 + 2*t1^4 + 72*t1^3 - 16*t1^2*t2 + 540*t1^2
    //           - 288*t1*t2 - 1944*t1 + 32*t2^2 + 432*t2 + 1458
    {
        WPoly& f = gens[3];
        f.add_term(mono_var(VAR_X3), 1);
        f.add_term(mono_var(VAR_T1, 4), 2);
        f.add_term(mono_var(VAR_T1, 3), 72);
        f.add_term(mono_var(VAR_T1, 2), 540);
        { Mono m = mono_zero(); m[VAR_T1] = 2; m[VAR_T2] = 1; f.add_term(m, -16); }
        { Mono m = mono_zero(); m[VAR_T1] = 1; m[VAR_T2] = 1; f.add_term(m, -288); }
        f.add_term(mono_var(VAR_T2, 2), 32);
        f.add_term(mono_var(VAR_T1), -1944);
        f.add_term(mono_var(VAR_T2), 432);
        f.add_term(mono_zero(), 1458);
    }

    return gens;
}


/* =====================================================================
 * STAGE 1: Parallel Weighted Homogenization (OpenMP)
 *
 * For generator F_i: substitute t_j -> t_j/s, multiply by s^{max_t_deg}
 * Each generator is independent -- embarrassingly parallel.
 * ===================================================================== */

static WPoly weighted_homogenize(const WPoly& f, int qi) {
    WPoly result;
    for (const auto& [m, c] : f.terms) {
        int t_deg = m[VAR_T1] + m[VAR_T2];
        int s_exp = qi - t_deg;
        assert(s_exp >= 0);
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
    int n = (int)affine_gens.size();
    std::vector<WPoly> hom_gens(n);

    auto t0 = Clock::now();

    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
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


/* =====================================================================
 * STAGE 2: Buchberger/F4 Elimination (CUDA-accelerated)
 *
 * Compute Groebner basis under lex order s > t1 > t2 > x0..x3,
 * then extract elements free of s, t1, t2.
 *
 * GPU acceleration: batch row reduction of Macaulay matrices.
 * ===================================================================== */

struct CSRMatrix {
    int n_rows, n_cols, nnz;
    std::vector<int>   row_ptr;
    std::vector<int>   col_ind;
    std::vector<double> vals;
};

static CSRMatrix build_macaulay_matrix(
    const std::vector<WPoly>& polys,
    std::vector<Mono>& mono_list)
{
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
    M.n_rows = nr; M.n_cols = nc;
    M.row_ptr.resize(nr + 1, 0);
    for (int i = 0; i < nr; ++i) {
        for (const auto& [m, c] : polys[i].terms) {
            M.col_ind.push_back(mono_idx[m]);
            M.vals.push_back(c.get_d());
        }
        M.row_ptr[i + 1] = (int)M.col_ind.size();
    }
    M.nnz = (int)M.vals.size();
    return M;
}

/* GPU row reduction kernel */
__global__ void gpu_row_reduce_kernel(
    double* __restrict__ dense, int n_rows, int n_cols,
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

static std::vector<WPoly> gpu_reduce_batch(
    const std::vector<WPoly>& spolys,
    const std::vector<WPoly>& basis)
{
    std::vector<WPoly> all_polys;
    all_polys.insert(all_polys.end(), basis.begin(), basis.end());
    all_polys.insert(all_polys.end(), spolys.begin(), spolys.end());

    std::vector<Mono> mono_list;
    CSRMatrix M = build_macaulay_matrix(all_polys, mono_list);
    if (M.n_rows == 0 || M.n_cols == 0) return {};

    int nr = M.n_rows, nc = M.n_cols;
    std::vector<double> dense(nr * nc, 0.0);
    for (int i = 0; i < nr; ++i)
        for (int k = M.row_ptr[i]; k < M.row_ptr[i + 1]; ++k)
            dense[i * nc + M.col_ind[k]] = M.vals[k];

    size_t bytes = (size_t)nr * nc * sizeof(double);
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);

    if (dev_count > 0 && bytes < 2ULL * 1024 * 1024 * 1024) {
        /* --- GPU path (V100) --- */
        double* d_dense;
        CUDA_CHECK(cudaMalloc(&d_dense, bytes));
        CUDA_CHECK(cudaMemcpy(d_dense, dense.data(), bytes, cudaMemcpyHostToDevice));
        int blk = 256;
        for (int p = 0; p < std::min(nr, nc); ++p)
            gpu_row_reduce_kernel<<<(nr + blk - 1) / blk, blk>>>(d_dense, nr, nc, p, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(dense.data(), d_dense, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dense));
    } else {
        /* --- CPU fallback --- */
        for (int p = 0; p < std::min(nr, nc); ++p) {
            double pval = dense[p * nc + p];
            if (std::abs(pval) < 1e-10) continue;
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < nr; ++row) {
                if (row == p) continue;
                double sc = dense[row * nc + p] / pval;
                if (std::abs(sc) < 1e-10) continue;
                for (int j = 0; j < nc; ++j)
                    dense[row * nc + j] -= sc * dense[p * nc + j];
            }
        }
    }

    // Extract non-zero reduced S-polys
    std::vector<WPoly> new_polys;
    int basis_size = (int)basis.size();
    for (int i = basis_size; i < nr; ++i) {
        WPoly p;
        for (int j = 0; j < nc; ++j) {
            if (std::abs(dense[i * nc + j]) > 1e-6) {
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

static std::vector<WPoly> stage2_elimination(
    const std::vector<WPoly>& hom_gens, double& time_ms)
{
    auto t0 = Clock::now();
    std::vector<WPoly> basis = hom_gens;
    std::set<std::pair<int,int>> processed;
    int max_iterations = 200;
    int iteration = 0;

    std::cout << "  Stage 2: Starting Buchberger elimination\n";
    std::cout << "  Initial basis: " << basis.size() << " polynomials\n";

    while (iteration++ < max_iterations) {
        std::vector<std::pair<int,int>> pairs;
        for (int i = 0; i < (int)basis.size(); ++i)
            for (int j = i + 1; j < (int)basis.size(); ++j)
                if (!processed.count({i, j})) {
                    pairs.push_back({i, j});
                    processed.insert({i, j});
                }
        if (pairs.empty()) break;

        // Compute S-polynomials (OpenMP parallel)
        std::vector<WPoly> spolys;
        #pragma omp parallel
        {
            std::vector<WPoly> local;
            #pragma omp for schedule(dynamic) nowait
            for (int k = 0; k < (int)pairs.size(); ++k) {
                auto [i, j] = pairs[k];
                if (!basis[i].is_zero() && !basis[j].is_zero()) {
                    WPoly sp = s_polynomial_z(basis[i], basis[j]);
                    if (!sp.is_zero()) local.push_back(std::move(sp));
                }
            }
            #pragma omp critical
            for (auto& sp : local) spolys.push_back(std::move(sp));
        }
        if (spolys.empty()) continue;

        // Reduce: GPU batch for large, CPU for small
        std::vector<WPoly> new_elements;
        if (spolys.size() > 4 && basis.size() > 8) {
            new_elements = gpu_reduce_batch(spolys, basis);
        } else {
            for (auto& sp : spolys) {
                WPoly r = reduce_z(sp, basis);
                if (!r.is_zero()) {
                    remove_content(r);
                    new_elements.push_back(std::move(r));
                }
            }
        }
        if (new_elements.empty()) continue;

        for (auto& ne : new_elements) basis.push_back(std::move(ne));
        std::cout << "  Iter " << iteration << ": " << pairs.size()
                  << " pairs -> " << new_elements.size()
                  << " new, basis = " << basis.size() << "\n";
    }

    // Extract elimination ideal: elements free of s, t1, t2
    std::vector<WPoly> result;
    for (const auto& g : basis)
        if (g.is_in_x_ring() && !g.is_zero())
            result.push_back(g);

    time_ms = elapsed_ms(t0);
    std::cout << "  Stage 2 complete: " << result.size()
              << " polynomials in elimination ideal\n";
    return result;
}


/* =====================================================================
 * STAGE 3: Parallel Weighted GCD Normalization (MPI + GMP)
 *
 * Prime range [2, sqrt(N)] partitioned across MPI ranks.
 * Each rank trial-divides its segment, results combined via Allgather.
 * ===================================================================== */

static mpz_class weighted_gcd_normalize(WPoly& poly, int rank, int n_procs) {
    if (poly.is_zero()) return mpz_class(1);

    // Standard GCD first
    mpz_class g = 0;
    for (const auto& [m, c] : poly.terms) {
        if (g == 0) g = abs(c);
        else mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), c.get_mpz_t());
    }
    if (g > 1) {
        WPoly tmp;
        for (const auto& [m, c] : poly.terms) tmp.add_term(m, c / g);
        poly = std::move(tmp);
    }

    // Weighted GCD: find largest d such that d^{w_deg} divides all coefficients
    struct CoefInfo { mpz_class coef; int w_deg; };
    std::vector<CoefInfo> info;
    for (const auto& [m, c] : poly.terms) {
        int wd = 0;
        for (int i = VAR_X0; i <= VAR_X3; ++i) wd += WEIGHTS[i] * m[i];
        info.push_back({abs(c), wd});
    }

    mpz_class max_c = 0;
    for (const auto& ci : info) if (ci.coef > max_c) max_c = ci.coef;

    mpz_class sqrt_max;
    mpz_sqrt(sqrt_max.get_mpz_t(), max_c.get_mpz_t());
    unsigned long sqrt_max_ul = mpz_get_ui(sqrt_max.get_mpz_t());
    if (sqrt_max_ul > 100000) sqrt_max_ul = 100000;

    // Partition prime range across MPI ranks
    unsigned long seg_start = 2 + (unsigned long)rank * (sqrt_max_ul - 1) / n_procs;
    unsigned long seg_end   = 2 + (unsigned long)(rank + 1) * (sqrt_max_ul - 1) / n_procs;
    if (rank == n_procs - 1) seg_end = sqrt_max_ul + 1;

    mpz_class local_wgcd = 1;
    for (unsigned long p = seg_start; p < seg_end; ++p) {
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
            min_val = std::min(min_val, ci.w_deg > 0 ? v / ci.w_deg : v);
        }
        if (divides_all && min_val > 0) {
            mpz_class pp;
            mpz_ui_pow_ui(pp.get_mpz_t(), p, min_val);
            local_wgcd *= pp;
        }
    }

    // MPI combine: product of local wgcds (disjoint prime ranges)
    std::string local_str = local_wgcd.get_str();
    int local_len = (int)local_str.size();

    std::vector<int> all_lens(n_procs);
    MPI_Allgather(&local_len, 1, MPI_INT, all_lens.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(n_procs, 0);
    int total_len = 0;
    for (int i = 0; i < n_procs; ++i) { displs[i] = total_len; total_len += all_lens[i]; }

    std::vector<char> all_strs(total_len);
    MPI_Allgatherv(local_str.data(), local_len, MPI_CHAR,
                   all_strs.data(), all_lens.data(), displs.data(),
                   MPI_CHAR, MPI_COMM_WORLD);

    mpz_class global_wgcd = 1;
    for (int i = 0; i < n_procs; ++i) {
        std::string s(all_strs.data() + displs[i], all_lens[i]);
        global_wgcd *= mpz_class(s);
    }

    if (global_wgcd > 1) {
        WPoly tmp;
        for (const auto& [m, c] : poly.terms) {
            int wd = 0;
            for (int i = VAR_X0; i <= VAR_X3; ++i) wd += WEIGHTS[i] * m[i];
            mpz_class divisor;
            mpz_pow_ui(divisor.get_mpz_t(), global_wgcd.get_mpz_t(), wd);
            if (mpz_divisible_p(c.get_mpz_t(), divisor.get_mpz_t()))
                tmp.add_term(m, c / divisor);
            else
                tmp.add_term(m, c);
        }
        poly = std::move(tmp);
    }

    remove_content(poly);
    if (!poly.is_zero() && poly.LC() < 0) {
        WPoly tmp;
        for (const auto& [m, c] : poly.terms) tmp.add_term(m, -c);
        poly = std::move(tmp);
    }
    return global_wgcd;
}

static std::vector<WPoly> stage3_normalize(
    std::vector<WPoly>& elim_ideal, int rank, int n_procs, double& time_ms)
{
    auto t0 = Clock::now();
    for (auto& p : elim_ideal) {
        mpz_class d = weighted_gcd_normalize(p, rank, n_procs);
        if (rank == 0 && d > 1)
            std::cout << "  Stage 3: weighted GCD factor = " << d << "\n";
    }
    time_ms = elapsed_ms(t0);
    return elim_ideal;
}


/* =====================================================================
 * VERIFICATION
 * ===================================================================== */

static void verify_result(const std::vector<WPoly>& result) {
    std::cout << "\n=== VERIFICATION ===\n";

    // Test points from parametrization
    // t1=0, t2=0: x0=-120, x1=405, x2=-14985, x3=-1458
    // t1=1, t2=0: x0=-128, x1=280, x2=-12352, x3=-128
    // t1=2, t2=0: x0=-136, x1=157, x2=-9631, x3=-338
    struct Pt { long x0, x1, x2, x3; };
    Pt pts[] = {
        {-120, 405, -14985, -1458},
        {-128, 280, -12352, -128},
        {-136, 157, -9631,  -338},
    };

    for (const auto& poly : result) {
        if (!poly.is_in_x_ring()) continue;
        std::cout << "  Polynomial: " << poly.nnz() << " terms, wdeg="
                  << poly.weighted_degree()
                  << ", w-homog=" << (poly.is_weighted_homogeneous() ? "YES" : "NO") << "\n";

        for (int pi = 0; pi < 3; ++pi) {
            const auto& pt = pts[pi];
            mpz_class val = 0;
            for (const auto& [m, c] : poly.terms) {
                mpz_class term = c;
                long x[] = {pt.x0, pt.x1, pt.x2, pt.x3};
                for (int i = VAR_X0; i <= VAR_X3; ++i)
                    for (int e = 0; e < m[i]; ++e)
                        term *= mpz_class(x[i - VAR_X0]);
                val += term;
            }
            std::cout << "  Point " << pi << ": F(" << pt.x0 << "," << pt.x1
                      << "," << pt.x2 << "," << pt.x3 << ") = " << val
                      << (val == 0 ? "  [OK]" : "  [NONZERO]") << "\n";
        }
    }
}

static void print_latex(const WPoly& poly) {
    const char* xn[] = {"x_0", "x_1", "x_2", "x_3"};
    bool first = true;
    for (const auto& [m, c] : poly.terms) {
        if (!first) std::cout << (c > 0 ? " + " : " - ");
        else if (c < 0) std::cout << "-";
        first = false;
        mpz_class ac = abs(c);
        bool has_vars = false;
        for (int i = VAR_X0; i <= VAR_X3; ++i) if (m[i] > 0) has_vars = true;
        if (ac != 1 || !has_vars) std::cout << ac;
        for (int i = VAR_X0; i <= VAR_X3; ++i)
            if (m[i] > 0) {
                std::cout << " " << xn[i - VAR_X0];
                if (m[i] > 1) std::cout << "^" << m[i];
            }
    }
    std::cout << " = 0\n";
}


/* =====================================================================
 * MAIN DRIVER
 * ===================================================================== */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, n_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    int n_threads = omp_get_max_threads();

    if (rank == 0) {
        std::cout << "============================================================\n"
                  << "  Experiment 6: L2 Three-Stage Parallel Pipeline\n"
                  << "  Weighted Projective Space P(2,4,6,10)\n"
                  << "============================================================\n\n"
                  << "MPI ranks: " << n_procs << "\n"
                  << "OpenMP threads: " << n_threads << "\n";

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

    double t1_ms = 0, t2_ms = 0, t3_ms = 0;

    // === Stage 1 ===
    if (rank == 0) std::cout << "=== STAGE 1: Weighted Homogenization (OpenMP) ===\n";
    auto affine_gens = build_l2_affine_generators();
    if (rank == 0)
        for (int i = 0; i < 4; ++i)
            std::cout << "  F" << i << ": " << affine_gens[i].nnz()
                      << " terms, deg " << affine_gens[i].total_degree() << "\n";

    auto hom_gens = stage1_homogenize(affine_gens, n_threads, t1_ms);
    if (rank == 0) {
        for (int i = 0; i < 4; ++i)
            std::cout << "  F" << i << "_hom: " << hom_gens[i].nnz()
                      << " terms, deg " << hom_gens[i].total_degree() << "\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << t1_ms << " ms\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // === Stage 2 (rank 0 only, has GPU) ===
    if (rank == 0) std::cout << "=== STAGE 2: Buchberger/F4 Elimination (CUDA) ===\n";
    std::vector<WPoly> elim_ideal;
    if (rank == 0) {
        elim_ideal = stage2_elimination(hom_gens, t2_ms);
        std::cout << "  Elimination ideal: " << elim_ideal.size() << " polys\n"
                  << "  Time: " << t2_ms << " ms\n\n";
    }

    // Broadcast results to all ranks
    int n_elim = (int)elim_ideal.size();
    MPI_Bcast(&n_elim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) elim_ideal.resize(n_elim);

    for (int pi = 0; pi < n_elim; ++pi) {
        std::string s;
        if (rank == 0) {
            std::ostringstream oss;
            oss << elim_ideal[pi].nnz();
            for (const auto& [m, c] : elim_ideal[pi].terms) {
                for (int v = 0; v < NUM_VARS; ++v) oss << " " << m[v];
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

    // === Stage 3 ===
    if (rank == 0) std::cout << "=== STAGE 3: Weighted GCD Normalization (MPI+GMP) ===\n";
    auto final_ideal = stage3_normalize(elim_ideal, rank, n_procs, t3_ms);

    if (rank == 0) {
        std::cout << "  Time: " << t3_ms << " ms\n\n";

        std::cout << "=== RESULTS ===\n"
                  << "Total: " << (t1_ms + t2_ms + t3_ms) << " ms\n"
                  << "  Stage 1 (Homogenization): " << t1_ms << " ms\n"
                  << "  Stage 2 (Elimination):    " << t2_ms << " ms\n"
                  << "  Stage 3 (Normalization):  " << t3_ms << " ms\n\n";

        for (int i = 0; i < (int)final_ideal.size(); ++i) {
            const auto& p = final_ideal[i];
            std::cout << "Polynomial " << i << ": " << p.nnz() << " terms, wdeg="
                      << p.weighted_degree() << ", w-homog="
                      << (p.is_weighted_homogeneous() ? "YES" : "NO") << "\n";
            if (p.nnz() <= 50) { std::cout << "  "; print_latex(p); }
        }

        verify_result(final_ideal);

        // Write CSV
        std::filesystem::create_directories("../../results");
        std::ofstream csv("../../results/exp6_pipeline.csv");
        csv << "stage,time_ms,n_procs,n_threads\n"
            << "homogenization," << t1_ms << "," << n_procs << "," << n_threads << "\n"
            << "elimination," << t2_ms << "," << n_procs << "," << n_threads << "\n"
            << "normalization," << t3_ms << "," << n_procs << "," << n_threads << "\n"
            << "total," << (t1_ms + t2_ms + t3_ms) << "," << n_procs << "," << n_threads << "\n";
        csv.close();
        std::cout << "\nSaved -> ../../results/exp6_pipeline.csv\n";
    }

    MPI_Finalize();
    return 0;
}
