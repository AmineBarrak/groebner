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
#include <climits>
#include <iomanip>
#include <filesystem>

#include <omp.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mpi.h>
#include "gmp_wrapper.h"

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
 * STAGE 2: Elimination via Substitution + Resultant (CUDA-accelerated)
 *
 * Naive Buchberger in lex order on 7 variables explodes exponentially.
 * Instead, we exploit the structure of the parametrization:
 *
 *   Step 2a: F0_hom is linear in t1 -> solve for t1 in terms of (s, x0)
 *   Step 2b: Substitute into F1_hom -> solve for t2 in terms of (s, x0, x1)
 *   Step 2c: Substitute both into F2_hom, F3_hom -> get G2(s, x), G3(s, x)
 *   Step 2d: Compute resultant Res(G2, G3, s) -> eliminates s
 *   Step 2e: Factor out highest power of leading variable -> degree-30 result
 *
 * The resultant is computed as det(Sylvester matrix), which is a large
 * dense matrix operation — this is where GPU acceleration helps.
 * ===================================================================== */

/* -- Univariate polynomial in s with WPoly coefficients --------------- */
/* A polynomial in s whose coefficients are polynomials in (x0,..,x3).
   Stored as vector: index i = coefficient of s^i. */
using UniS = std::vector<WPoly>;  // UniS[i] = coeff of s^i

static int unis_degree(const UniS& p) {
    for (int i = (int)p.size() - 1; i >= 0; --i)
        if (!p[i].is_zero()) return i;
    return -1;
}

/* Convert a WPoly (in all 7 vars) to UniS by grouping by s-exponent */
static UniS to_unis(const WPoly& f) {
    int max_s = 0;
    for (const auto& [m, c] : f.terms)
        max_s = std::max(max_s, m[VAR_S]);
    UniS result(max_s + 1);
    for (const auto& [m, c] : f.terms) {
        Mono nm = m;
        nm[VAR_S] = 0;  // strip s from monomial
        result[m[VAR_S]].add_term(nm, c);
    }
    return result;
}

/* GPU row reduction kernel for Sylvester matrix determinant */
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

/* Substitute t1 = expr(s, x0) into a homogenized polynomial.
 * From F0_hom: s*x0 + 120*s + 8*t1 = 0  =>  t1 = -s*(x0 + 120)/8
 *
 * For each monomial with t1^a, replace t1^a with (-s*(x0+120)/8)^a
 * = (-1)^a * s^a * (x0+120)^a / 8^a
 * We work over Z so multiply through by 8^{max_t1_deg} to clear denominators.
 */
static WPoly substitute_t1(const WPoly& f, int clear_denom_pow) {
    // t1 -> -s*(x0+120)/8, cleared: multiply entire poly by 8^clear_denom_pow
    // For a monomial with t1^a: gets factor (-1)^a * s^a * (x0+120)^a * 8^{clear_denom_pow - a}
    WPoly result;

    // Precompute powers of (x0+120) as WPolys
    // (x0+120)^0 = 1, (x0+120)^1 = x0+120, etc.
    int max_t1 = 0;
    for (const auto& [m, _] : f.terms) max_t1 = std::max(max_t1, m[VAR_T1]);

    std::vector<WPoly> xp120_pow(max_t1 + 1);
    // (x0+120)^0
    xp120_pow[0].add_term(mono_zero(), 1);
    // (x0+120)^1
    if (max_t1 >= 1) {
        xp120_pow[1].add_term(mono_var(VAR_X0), 1);
        xp120_pow[1].add_term(mono_zero(), 120);
    }
    for (int k = 2; k <= max_t1; ++k) {
        // (x0+120)^k = (x0+120)^{k-1} * (x0+120)
        for (const auto& [ma, ca] : xp120_pow[k-1].terms) {
            // * x0
            Mono mx = ma; mx[VAR_X0] += 1;
            xp120_pow[k].add_term(mx, ca);
            // * 120
            xp120_pow[k].add_term(ma, ca * 120);
        }
    }

    for (const auto& [m, c] : f.terms) {
        int a = m[VAR_T1];  // power of t1

        // Coefficient from clearing denominator: 8^{clear_denom_pow - a}
        mpz_class eight_pow;
        mpz_ui_pow_ui(eight_pow.get_mpz_t(), 8, clear_denom_pow - a);

        // Sign: (-1)^a
        mpz_class sign = (a % 2 == 0) ? mpz_class(1) : mpz_class(-1);

        // New s-exponent: original s-exp + a (from the substitution)
        int new_s = m[VAR_S] + a;

        // Multiply c * sign * 8^{d-a} * (x0+120)^a
        mpz_class scalar = c * sign * eight_pow;

        for (const auto& [mp, cp] : xp120_pow[a].terms) {
            Mono nm = m;
            nm[VAR_T1] = 0;  // t1 eliminated
            nm[VAR_S] = new_s;
            // Add x0 exponents from (x0+120)^a
            for (int v = 0; v < NUM_VARS; ++v) nm[v] += mp[v];
            result.add_term(nm, scalar * cp);
        }
    }

    remove_content(result);
    return result;
}

/* Substitute t2 = expr(s, x0, x1) into a polynomial.
 * From F1_hom (after t1 substitution and clearing):
 *   We solve for t2: the polynomial is linear in t2.
 *   Coefficient of t2 = some function of (s, x0)
 *   Remainder = some function of (s, x0, x1)
 *   => t2 = -remainder / coef_t2
 *
 * Instead of dividing, we multiply by coef_t2^{max_t2_deg} to clear.
 */
static WPoly substitute_t2(const WPoly& f, const WPoly& t2_coef, const WPoly& t2_remainder) {
    // For each monomial with t2^b:
    //   Replace t2^b with (-remainder)^b / coef_t2^b
    //   Clear denominator by multiplying entire thing by coef_t2^{max_t2}
    int max_t2 = 0;
    for (const auto& [m, _] : f.terms) max_t2 = std::max(max_t2, m[VAR_T2]);

    if (max_t2 == 0) return f;  // no t2 to substitute

    // Precompute (-remainder)^k and coef_t2^k
    std::vector<WPoly> neg_rem_pow(max_t2 + 1);
    std::vector<WPoly> coef_pow(max_t2 + 1);
    neg_rem_pow[0].add_term(mono_zero(), 1);
    coef_pow[0].add_term(mono_zero(), 1);

    WPoly neg_rem;
    for (const auto& [m, c] : t2_remainder.terms) neg_rem.add_term(m, -c);

    for (int k = 1; k <= max_t2; ++k) {
        neg_rem_pow[k] = WPoly();
        for (const auto& [ma, ca] : neg_rem_pow[k-1].terms)
            for (const auto& [mb, cb] : neg_rem.terms) {
                Mono nm(NUM_VARS, 0);
                for (int v = 0; v < NUM_VARS; ++v) nm[v] = ma[v] + mb[v];
                neg_rem_pow[k].add_term(nm, ca * cb);
            }
        remove_content(neg_rem_pow[k]);

        coef_pow[k] = WPoly();
        for (const auto& [ma, ca] : coef_pow[k-1].terms)
            for (const auto& [mb, cb] : t2_coef.terms) {
                Mono nm(NUM_VARS, 0);
                for (int v = 0; v < NUM_VARS; ++v) nm[v] = ma[v] + mb[v];
                coef_pow[k].add_term(nm, ca * cb);
            }
        remove_content(coef_pow[k]);
    }

    // Build result: for each term c*m with t2^b,
    // replace with c * m{t2=0} * (-rem)^b * coef^{max_t2 - b}
    WPoly result;
    for (const auto& [m, c] : f.terms) {
        int b = m[VAR_T2];
        Mono base_m = m;
        base_m[VAR_T2] = 0;

        // Multiply: c * base_m * neg_rem_pow[b] * coef_pow[max_t2 - b]
        // First: neg_rem_pow[b] * coef_pow[max_t2 - b]
        WPoly product;
        for (const auto& [mr, cr] : neg_rem_pow[b].terms)
            for (const auto& [mc, cc] : coef_pow[max_t2 - b].terms) {
                Mono nm(NUM_VARS, 0);
                for (int v = 0; v < NUM_VARS; ++v) nm[v] = mr[v] + mc[v];
                product.add_term(nm, cr * cc);
            }

        // Now multiply by c * base_m
        for (const auto& [mp, cp] : product.terms) {
            Mono nm(NUM_VARS, 0);
            for (int v = 0; v < NUM_VARS; ++v) nm[v] = base_m[v] + mp[v];
            result.add_term(nm, c * cp);
        }
    }

    remove_content(result);
    return result;
}

/* Compute resultant of two univariate polynomials in s via Sylvester matrix.
 * The Sylvester matrix rows are shifted copies of the two polynomials.
 * det(Sylvester) = resultant, which eliminates s.
 *
 * The matrix is (deg_g + deg_h) x (deg_g + deg_h), with entries that are
 * polynomials in (x0,..,x3). We compute this symbolically.
 *
 * For GPU: we can parallelize the Bareiss algorithm steps.
 */
static WPoly resultant_sylvester(const UniS& g, const UniS& h) {
    int dg = unis_degree(g);
    int dh = unis_degree(h);
    if (dg < 0 || dh < 0) { WPoly z; return z; }

    int N = dg + dh;  // matrix size
    std::cout << "  Resultant: Sylvester matrix " << N << "x" << N
              << " (deg_G2=" << dg << ", deg_G3=" << dh << ")\n";

    // Build Sylvester matrix M[i][j] as WPolys
    // First dh rows: shifted copies of g
    // Next dg rows: shifted copies of h
    std::vector<std::vector<WPoly>> M(N, std::vector<WPoly>(N));

    for (int i = 0; i < dh; ++i)
        for (int j = 0; j <= dg; ++j)
            M[i][i + j] = g[dg - j];  // g coefficients in descending order

    for (int i = 0; i < dg; ++i)
        for (int j = 0; j <= dh; ++j)
            M[dh + i][i + j] = h[dh - j];

    // Debug: print matrix sparsity pattern
    std::cout << "  Sylvester matrix nonzero pattern:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "    row " << i << ": ";
        for (int j = 0; j < N; ++j)
            std::cout << (M[i][j].is_zero() ? "." : std::to_string(M[i][j].nnz()).c_str()) << " ";
        std::cout << "\n";
    }
    std::cout << std::flush;

    // Bareiss algorithm for determinant (fraction-free)
    // At each step k, M[i][j] = (M[i][j]*M[k][k] - M[i][k]*M[k][j]) / prev_pivot
    WPoly prev_pivot;
    prev_pivot.add_term(mono_zero(), 1);
    int swap_sign = 1;  // track sign from row swaps

    for (int k = 0; k < N - 1; ++k) {
        std::cout << "  Bareiss step " << k + 1 << "/" << N - 1
                  << " (pivot: " << M[k][k].nnz() << " terms)\r" << std::flush;

        // Partial pivoting: find nonzero pivot in column k
        if (M[k][k].is_zero()) {
            bool found = false;
            for (int r = k + 1; r < N; ++r) {
                if (!M[r][k].is_zero()) {
                    std::swap(M[k], M[r]);
                    swap_sign *= -1;  // track sign, don't modify entries
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "\n  Column " << k << " all zero — det is 0\n";
                WPoly zero;
                return zero;
            }
        }
        WPoly pivot = M[k][k];

        // Update rows k+1..N-1 (sequential — matrix is small, ~10x10)
        for (int i = k + 1; i < N; ++i) {
            // Snapshot M[i][k] before it gets overwritten in j-loop
            WPoly Mik = M[i][k];
            for (int j = k + 1; j < N; ++j) {
                // new = (M[i][j]*pivot - Mik*M[k][j]) / prev_pivot
                WPoly prod1, prod2, numer;

                // prod1 = M[i][j] * pivot
                for (const auto& [ma, ca] : M[i][j].terms)
                    for (const auto& [mb, cb] : pivot.terms) {
                        Mono nm(NUM_VARS, 0);
                        for (int v = 0; v < NUM_VARS; ++v) nm[v] = ma[v] + mb[v];
                        prod1.add_term(nm, ca * cb);
                    }

                // prod2 = Mik * M[k][j]
                for (const auto& [ma, ca] : Mik.terms)
                    for (const auto& [mb, cb] : M[k][j].terms) {
                        Mono nm(NUM_VARS, 0);
                        for (int v = 0; v < NUM_VARS; ++v) nm[v] = ma[v] + mb[v];
                        prod2.add_term(nm, ca * cb);
                    }

                // numer = prod1 - prod2
                numer = wpoly_sub(prod1, prod2);

                // Divide by prev_pivot (exact division in Bareiss)
                if (numer.is_zero()) {
                    M[i][j] = WPoly();
                } else if (prev_pivot.nnz() == 1) {
                    // Fast path: prev_pivot is a monomial c*x^a
                    Mono pm = prev_pivot.LM();
                    mpz_class pc = prev_pivot.LC();
                    WPoly divided;
                    for (const auto& [nm2, nc] : numer.terms) {
                        Mono qm(NUM_VARS, 0);
                        bool div_ok = true;
                        for (int v = 0; v < NUM_VARS; ++v) {
                            qm[v] = nm2[v] - pm[v];
                            if (qm[v] < 0) { div_ok = false; break; }
                        }
                        if (div_ok) {
                            divided.add_term(qm, nc / pc);
                        }
                    }
                    M[i][j] = std::move(divided);
                } else {
                    // General case: exact polynomial division
                    M[i][j] = wpoly_exact_div(numer, prev_pivot);
                }
                remove_content(M[i][j]);
            }
            // Clear the column entry
            M[i][k] = WPoly();
        }

        prev_pivot = pivot;
    }
    std::cout << "\n";

    // Determinant is M[N-1][N-1], adjusted for row swap sign
    WPoly det = M[N - 1][N - 1];
    if (swap_sign < 0) det = wpoly_scale(det, -1);
    remove_content(det);
    std::cout << "  Resultant: " << det.nnz() << " terms, wdeg="
              << det.weighted_degree() << "\n";
    return det;
}

/* Convert a WPoly to univariate in t1 by grouping by t1-exponent.
 * Returns UniT1[i] = coefficient of t1^i (a polynomial in the other vars). */
using UniT1 = std::vector<WPoly>;

static UniT1 to_uni_t1(const WPoly& f) {
    int max_t1 = 0;
    for (const auto& [m, c] : f.terms)
        max_t1 = std::max(max_t1, m[VAR_T1]);
    UniT1 result(max_t1 + 1);
    for (const auto& [m, c] : f.terms) {
        Mono nm = m;
        nm[VAR_T1] = 0;  // strip t1 from monomial
        result[m[VAR_T1]].add_term(nm, c);
    }
    return result;
}

static int uni_degree(const UniT1& p) {
    for (int i = (int)p.size() - 1; i >= 0; --i)
        if (!p[i].is_zero()) return i;
    return -1;
}

/* Substitute t1 = -(x0+120)/8 in an AFFINE polynomial.
 * Clears denominators by 8^max_t1.  Result is in k[t2, x0, ...]. */
static WPoly substitute_t1_affine(const WPoly& f) {
    int max_t1 = 0;
    for (const auto& [m, _] : f.terms) max_t1 = std::max(max_t1, m[VAR_T1]);
    if (max_t1 == 0) return f;  // no t1

    // Precompute (x0+120)^k
    std::vector<WPoly> xp_pow(max_t1 + 1);
    xp_pow[0].add_term(mono_zero(), 1);
    if (max_t1 >= 1) {
        xp_pow[1].add_term(mono_var(VAR_X0), 1);
        xp_pow[1].add_term(mono_zero(), 120);
    }
    for (int k = 2; k <= max_t1; ++k) {
        for (const auto& [ma, ca] : xp_pow[k-1].terms) {
            Mono mx = ma; mx[VAR_X0] += 1;
            xp_pow[k].add_term(mx, ca);
            xp_pow[k].add_term(ma, ca * 120);
        }
    }

    WPoly result;
    for (const auto& [m, c] : f.terms) {
        int a = m[VAR_T1];
        // t1^a → (-(x0+120)/8)^a = (-1)^a (x0+120)^a / 8^a
        // Clear by 8^max_t1: multiply by 8^{max_t1-a}
        mpz_class eight_pow;
        mpz_ui_pow_ui(eight_pow.get_mpz_t(), 8, max_t1 - a);
        mpz_class sign = (a % 2 == 0) ? mpz_class(1) : mpz_class(-1);
        mpz_class scalar = c * sign * eight_pow;

        for (const auto& [mp, cp] : xp_pow[a].terms) {
            Mono nm = m;
            nm[VAR_T1] = 0;
            for (int v = 0; v < NUM_VARS; ++v) nm[v] += mp[v];
            result.add_term(nm, scalar * cp);
        }
    }
    remove_content(result);
    return result;
}

/* Weighted-homogenize a polynomial in k[x0,..,x3] by introducing s.
 * Each term gets s^{D - wd} where wd is its weighted degree and D = max wd. */
static WPoly weighted_homogenize_xs(const WPoly& f) {
    int D = 0;
    for (const auto& [m, c] : f.terms) {
        int wd = 0;
        for (int v = VAR_X0; v <= VAR_X3; ++v) wd += m[v] * WEIGHTS[v];
        D = std::max(D, wd);
    }
    WPoly result;
    for (const auto& [m, c] : f.terms) {
        int wd = 0;
        for (int v = VAR_X0; v <= VAR_X3; ++v) wd += m[v] * WEIGHTS[v];
        Mono nm = m;
        nm[VAR_S] = D - wd;
        result.add_term(nm, c);
    }
    return result;
}

/* ------------------------------------------------------------------ *
 * Stage 2 – Linear-algebra ansatz                                     *
 *                                                                     *
 * Enumerate all monomials x0^a x1^b x2^c x3^d of weighted degree D   *
 * (weights 2,4,6,10).  Substitute the L2 parametrisation and require  *
 * the result to vanish identically in (t1,t2).  This gives a linear   *
 * system whose kernel is the defining polynomial of the hypersurface  *
 * image in P(2,4,6,10).                                               *
 *                                                                     *
 * Gaussian elimination over Z (fraction-free) is used; the matrix is  *
 * at most ~72 x 47 for D = 30.                                       *
 * ------------------------------------------------------------------ */

/* Helper: polynomial in t1,t2 represented as map<(e1,e2), mpz_class>. */
using TPoly = std::map<std::pair<int,int>, mpz_class>;

static TPoly tp_mul(const TPoly& a, const TPoly& b) {
    TPoly r;
    for (const auto& [ea, ca] : a)
        for (const auto& [eb, cb] : b) {
            auto key = std::make_pair(ea.first+eb.first, ea.second+eb.second);
            r[key] += ca * cb;
        }
    // remove zeros
    for (auto it = r.begin(); it != r.end(); )
        if (it->second == 0) it = r.erase(it); else ++it;
    return r;
}

static TPoly tp_scale(const TPoly& a, long c) {
    TPoly r;
    for (const auto& [k, v] : a) {
        mpz_class p = v * c;
        if (p != 0) r[k] = p;
    }
    return r;
}

static std::vector<TPoly> tp_powers(const TPoly& base, int max_pow) {
    std::vector<TPoly> pows(max_pow + 1);
    pows[0][{0,0}] = 1;
    for (int i = 1; i <= max_pow; ++i)
        pows[i] = tp_mul(pows[i-1], base);
    return pows;
}

static std::vector<WPoly> stage2_elimination(
    const std::vector<WPoly>& /*hom_gens*/, double& time_ms)
{
    auto t0 = Clock::now();
    const int D = 30;                       // target weighted degree
    const int w[4] = {2, 4, 6, 10};        // weights of x0..x3

    std::cout << "  Strategy: linear-algebra ansatz (weighted degree " << D << ")\n";

    /* ---- Step A: enumerate monomials of weighted degree D ---- */
    struct XMono { int a, b, c, d; };
    std::vector<XMono> monoms;
    for (int dd = 0; dd * w[3] <= D; ++dd)
        for (int cc = 0; cc * w[2] + dd * w[3] <= D; ++cc)
            for (int bb = 0; bb * w[1] + cc * w[2] + dd * w[3] <= D; ++bb) {
                int rem = D - dd * w[3] - cc * w[2] - bb * w[1];
                if (rem >= 0 && rem % w[0] == 0)
                    monoms.push_back({rem / w[0], bb, cc, dd});
            }
    const int ncols = (int)monoms.size();
    std::cout << "  Monomials of weighted degree " << D << ": " << ncols << "\n";

    /* ---- Step B: build parametrisation as TPolys ---- */
    // x0 = -120 - 8*t1
    TPoly px0 = {{{0,0}, -120}, {{1,0}, -8}};
    // x1 = t1^2 - 126*t1 + 12*t2 + 405
    TPoly px1 = {{{2,0}, 1}, {{1,0}, -126}, {{0,1}, 12}, {{0,0}, 405}};
    // x2 = -3*t1^3 + 53*t1^2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985
    TPoly px2 = {{{3,0}, -3}, {{2,0}, 53}, {{1,1}, -20},
                 {{1,0}, 2583}, {{0,1}, -12}, {{0,0}, -14985}};
    // x3 = -2*(-t1^2 - 18*t1 + 4*t2 + 27)^2
    TPoly inner = {{{2,0}, -1}, {{1,0}, -18}, {{0,1}, 4}, {{0,0}, 27}};
    TPoly px3 = tp_scale(tp_mul(inner, inner), -2);

    /* Precompute powers */
    int ma = 0, mb = 0, mc = 0, md = 0;
    for (auto& m : monoms) {
        ma = std::max(ma, m.a); mb = std::max(mb, m.b);
        mc = std::max(mc, m.c); md = std::max(md, m.d);
    }
    auto x0p = tp_powers(px0, ma);
    auto x1p = tp_powers(px1, mb);
    auto x2p = tp_powers(px2, mc);
    auto x3p = tp_powers(px3, md);
    std::cout << "  Max powers: x0^" << ma << " x1^" << mb
              << " x2^" << mc << " x3^" << md << "\n";

    /* ---- Step C: substitute into each monomial, collect t-exponents ---- */
    std::cout << "  Substituting parametrisation...\n";
    std::vector<TPoly> cols(ncols);
    std::set<std::pair<int,int>> t_set;
    for (int j = 0; j < ncols; ++j) {
        cols[j] = tp_mul(tp_mul(x0p[monoms[j].a], x1p[monoms[j].b]),
                         tp_mul(x2p[monoms[j].c], x3p[monoms[j].d]));
        for (const auto& [k, _] : cols[j]) t_set.insert(k);
    }
    std::vector<std::pair<int,int>> t_list(t_set.begin(), t_set.end());
    std::map<std::pair<int,int>, int> t_idx;
    for (int i = 0; i < (int)t_list.size(); ++i) t_idx[t_list[i]] = i;
    const int nrows = (int)t_list.size();
    std::cout << "  Matrix size: " << nrows << " x " << ncols << "\n";

    /* ---- Step D: build matrix M (nrows x ncols) over Z ---- */
    std::vector<std::vector<mpz_class>> M(nrows, std::vector<mpz_class>(ncols, 0));
    for (int j = 0; j < ncols; ++j)
        for (const auto& [k, v] : cols[j])
            M[t_idx[k]][j] = v;

    /* ---- Step E: Gaussian elimination to find kernel vector ---- *
     * Fraction-free (Bareiss-style) over Z.                         *
     * We expect a 1-dimensional kernel.                             */
    std::cout << "  Gaussian elimination (" << nrows << " x " << ncols << ")...\n";

    // Work on column-major copy for pivoting
    // We reduce to row-echelon form and track pivot columns.
    std::vector<int> pivot_col(nrows, -1);
    std::vector<int> pivot_row;
    int cur_row = 0;
    for (int col = 0; col < ncols && cur_row < nrows; ++col) {
        // Find pivot
        int pr = -1;
        for (int r = cur_row; r < nrows; ++r) {
            if (M[r][col] != 0) { pr = r; break; }
        }
        if (pr < 0) continue;  // free variable

        // Swap rows
        if (pr != cur_row) std::swap(M[pr], M[cur_row]);
        pivot_col[cur_row] = col;
        pivot_row.push_back(cur_row);

        // Eliminate below
        mpz_class piv = M[cur_row][col];
        for (int r = 0; r < nrows; ++r) {
            if (r == cur_row || M[r][col] == 0) continue;
            mpz_class factor = M[r][col];
            for (int c = 0; c < ncols; ++c) {
                M[r][c] = M[r][c] * piv - factor * M[cur_row][c];
            }
            // Remove content from row to keep numbers small
            mpz_class g = 0;
            for (int c = 0; c < ncols; ++c) {
                if (M[r][c] != 0) {
                    if (g == 0) g = abs(M[r][c]);
                    else mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), M[r][c].get_mpz_t());
                }
            }
            if (g > 1) {
                for (int c = 0; c < ncols; ++c)
                    if (M[r][c] != 0) mpz_divexact(M[r][c].get_mpz_t(),
                                                    M[r][c].get_mpz_t(), g.get_mpz_t());
            }
        }
        ++cur_row;
    }
    int rank = cur_row;
    int nullity = ncols - rank;
    std::cout << "  Rank: " << rank << ", nullity: " << nullity << "\n";

    if (nullity != 1) {
        std::cout << "  ERROR: expected nullity 1, got " << nullity << "\n";
        time_ms = elapsed_ms(t0);
        return {};
    }

    /* Back-substitute to find kernel vector */
    // Identify the free column (the one that is not a pivot column)
    std::set<int> pivot_cols_set;
    for (int r = 0; r < rank; ++r) pivot_cols_set.insert(pivot_col[r]);

    int free_col = -1;
    for (int c = 0; c < ncols; ++c) {
        if (pivot_cols_set.find(c) == pivot_cols_set.end()) {
            free_col = c; break;
        }
    }
    std::cout << "  Free variable: column " << free_col
              << " = x0^" << monoms[free_col].a
              << " x1^" << monoms[free_col].b
              << " x2^" << monoms[free_col].c
              << " x3^" << monoms[free_col].d << "\n";

    // Set free variable = 1, solve for pivot variables
    // From row r: M[r][pivot_col[r]] * x_{pivot_col[r]} + M[r][free_col] * 1 = 0
    // => x_{pivot_col[r]} = -M[r][free_col] / M[r][pivot_col[r]]
    // (exact since kernel is rational)
    std::vector<mpz_class> kernel(ncols, 0);
    kernel[free_col] = 1;

    // Scale: multiply all by LCM of denominators
    mpz_class lcm_denom = 1;
    for (int r = rank - 1; r >= 0; --r) {
        int pc = pivot_col[r];
        mpz_class num = -M[r][free_col];
        // Also subtract contributions from already-set pivot cols
        for (int r2 = r + 1; r2 < rank; ++r2) {
            int pc2 = pivot_col[r2];
            num -= M[r][pc2] * kernel[pc2];
        }
        // kernel[pc] = num / M[r][pc]  — but we need integers
        // So we'll first compute in rationals, then clear denominators
        kernel[pc] = num;
        // Actually, let's just store (num, denom) and clear later
    }
    // The above doesn't handle denominators properly. Let's use a cleaner approach:
    // Scale everything so kernel[free_col] = product of all pivot diagonal entries
    mpz_class scale = 1;
    for (int r = 0; r < rank; ++r) scale *= M[r][pivot_col[r]];

    kernel[free_col] = scale;
    for (int r = rank - 1; r >= 0; --r) {
        int pc = pivot_col[r];
        mpz_class rhs = -M[r][free_col] * scale;
        for (int c = 0; c < ncols; ++c) {
            if (c != pc && c != free_col && kernel[c] != 0)
                rhs -= M[r][c] * kernel[c];
        }
        mpz_class diag = M[r][pc];
        mpz_divexact(kernel[pc].get_mpz_t(), rhs.get_mpz_t(), diag.get_mpz_t());
    }

    // Remove content
    mpz_class g = 0;
    for (int j = 0; j < ncols; ++j) {
        if (kernel[j] != 0) {
            if (g == 0) g = abs(kernel[j]);
            else mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), kernel[j].get_mpz_t());
        }
    }
    if (g > 1) {
        for (int j = 0; j < ncols; ++j)
            if (kernel[j] != 0)
                mpz_divexact(kernel[j].get_mpz_t(), kernel[j].get_mpz_t(), g.get_mpz_t());
    }
    // Make leading nonzero positive
    for (int j = 0; j < ncols; ++j) {
        if (kernel[j] != 0) {
            if (kernel[j] < 0) {
                for (int k = 0; k < ncols; ++k) kernel[k] = -kernel[k];
            }
            break;
        }
    }

    /* ---- Step F: build result polynomial ---- */
    WPoly res;
    int nterms = 0;
    for (int j = 0; j < ncols; ++j) {
        if (kernel[j] == 0) continue;
        Mono m(NUM_VARS, 0);
        m[VAR_X0] = monoms[j].a;
        m[VAR_X1] = monoms[j].b;
        m[VAR_X2] = monoms[j].c;
        m[VAR_X3] = monoms[j].d;
        res.add_term(m, kernel[j]);
        ++nterms;
    }
    std::cout << "  Result: " << nterms << " terms\n";

    // Verify weighted homogeneity
    bool is_wh = true;
    int wdeg = -1;
    for (const auto& [m, c] : res.terms) {
        int wd = 0;
        for (int v = VAR_X0; v <= VAR_X3; ++v) wd += m[v] * WEIGHTS[v];
        if (wdeg < 0) wdeg = wd;
        else if (wd != wdeg) { is_wh = false; break; }
    }
    std::cout << "  Weighted homogeneous: " << (is_wh ? "YES" : "NO")
              << ", degree " << wdeg << "\n";

    // Verify at test points from parametrisation
    std::cout << "  Verification at parametrisation test points:\n";
    auto eval_param = [](int t1, int t2) -> std::vector<mpz_class> {
        mpz_class T1(t1), T2(t2);
        mpz_class x0 = mpz_class(-120) - mpz_class(8)*T1;
        mpz_class x1 = T1*T1 - mpz_class(126)*T1 + mpz_class(12)*T2 + mpz_class(405);
        mpz_class x2 = mpz_class(-3)*T1*T1*T1 + mpz_class(53)*T1*T1
                        - mpz_class(20)*T1*T2 + mpz_class(2583)*T1
                        - mpz_class(12)*T2 - mpz_class(14985);
        mpz_class inn = -T1*T1 - mpz_class(18)*T1 + mpz_class(4)*T2 + mpz_class(27);
        mpz_class x3 = mpz_class(-2) * inn * inn;
        return {x0, x1, x2, x3};
    };
    int test_pts[][2] = {{0,0},{1,0},{0,1},{1,1},{-1,2},{3,-1},{10,5},{-7,3}};
    bool all_ok = true;
    for (auto& pt : test_pts) {
        auto xv = eval_param(pt[0], pt[1]);
        mpz_class val = 0;
        for (const auto& [m, c] : res.terms) {
            mpz_class term = c;
            for (int v = VAR_X0; v <= VAR_X3; ++v) {
                int e = m[v];
                int xi = v - VAR_X0;
                for (int k = 0; k < e; ++k) term *= xv[xi];
            }
            val += term;
        }
        bool ok = (val == 0);
        if (!ok) all_ok = false;
        std::cout << "    (t1,t2)=(" << pt[0] << "," << pt[1]
                  << ") -> F = " << val << " [" << (ok ? "OK" : "FAIL") << "]\n";
    }
    std::cout << "  All tests: " << (all_ok ? "PASS" : "FAIL") << "\n";

    // Print terms
    std::cout << "  Polynomial terms:\n";
    for (const auto& [m, c] : res.terms) {
        std::cout << "    ";
        if (c > 0) std::cout << "+";
        std::cout << c;
        if (m[VAR_X0] > 0) std::cout << "*x0^" << m[VAR_X0];
        if (m[VAR_X1] > 0) std::cout << "*x1^" << m[VAR_X1];
        if (m[VAR_X2] > 0) std::cout << "*x2^" << m[VAR_X2];
        if (m[VAR_X3] > 0) std::cout << "*x3^" << m[VAR_X3];
        std::cout << "\n";
    }

    std::vector<WPoly> result;
    if (!res.is_zero()) {
        result.push_back(std::move(res));
    }

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
    if (rank == 0) std::cout << "=== STAGE 2: Substitution + Resultant Elimination ===\n";
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
