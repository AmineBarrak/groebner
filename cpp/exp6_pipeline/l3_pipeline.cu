/*
 * cpp/exp6_pipeline/l3_pipeline.cu
 * =================================
 * EXPERIMENT 6 -- Three-Stage Parallel Pipeline for the L3 Example
 *
 * Computes the implicit equation of the weighted variety L3 in P(2,4,6,10)
 * from its rational parametrization (2 parameters u,v):
 *
 *   x0 = 2v(4u^2 - 12uv + 3v^2 + 252u - 54v - 405)
 *   x1 = 4v(u^4*v - 24u^4 - 66u^3*v + 9u^2*v^2 + 1188u^3 + 297u^2*v
 *           + 138uv^2 - 36v^3 - 8424uv + 945v^2 + 14580v)
 *   x2 = 4v(2u^6*v^2 - 8u^5*v^3 + ... - 2821230v^2)  [27 terms inside]
 *   x3 = -16v^2(v-27)(4u^3 - u^2*v - 18uv + 4v^2 + 27v)^3
 *
 * Weights: q = (2, 4, 6, 10)
 * Expected output: degree-80 weighted homogeneous polynomial, 318 terms.
 *
 * Strategy: linear-algebra ansatz with fraction-free Gaussian elimination
 * over Z using GMP multiprecision integers.
 *
 * Target: NVIDIA Tesla V100 (sm_70) on Matilda HPC
 *
 * Build:
 *   cmake -B build -DL3=ON && cmake --build build -j8
 * Run:
 *   mpirun ./build/l3_pipeline
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
#include <sstream>
#include <random>

#include <omp.h>
#include <cuda_runtime.h>
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

using Clock = std::chrono::high_resolution_clock;
static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}


/* =====================================================================
 * POLYNOMIAL IN (u,v) OVER Z  --  the "TPoly" type
 * ===================================================================== */

using TPoly = std::map<std::pair<int,int>, mpz_class>;

static TPoly tp_add(const TPoly& a, const TPoly& b) {
    TPoly r = a;
    for (const auto& [k, v] : b) {
        r[k] += v;
        if (r[k] == 0) r.erase(k);
    }
    return r;
}

static TPoly tp_mul(const TPoly& a, const TPoly& b) {
    TPoly r;
    for (const auto& [ea, ca] : a)
        for (const auto& [eb, cb] : b) {
            auto key = std::make_pair(ea.first+eb.first, ea.second+eb.second);
            r[key] += ca * cb;
        }
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

static TPoly tp_from_mono(int eu, int ev, long c = 1) {
    TPoly r;
    if (c != 0) r[{eu, ev}] = c;
    return r;
}

static TPoly tp_pow(const TPoly& base, int n) {
    TPoly r; r[{0,0}] = 1;
    for (int i = 0; i < n; ++i)
        r = tp_mul(r, base);
    return r;
}

static std::vector<TPoly> tp_powers(const TPoly& base, int max_pow) {
    std::vector<TPoly> pows(max_pow + 1);
    pows[0][{0,0}] = 1;
    for (int i = 1; i <= max_pow; ++i)
        pows[i] = tp_mul(pows[i-1], base);
    return pows;
}


/* =====================================================================
 * L3 PARAMETRIZATION
 * ===================================================================== */

static void build_l3_parametrisation(TPoly& px0, TPoly& px1, TPoly& px2, TPoly& px3)
{
    TPoly U = tp_from_mono(1, 0);
    TPoly V = tp_from_mono(0, 1);

    // x0 = 2v(4u^2 - 12uv + 3v^2 + 252u - 54v - 405)
    TPoly inner_x0 = tp_add(
        tp_add(tp_add(tp_add(tp_add(
            tp_scale(tp_pow(U, 2), 4),
            tp_scale(tp_mul(U, V), -12)),
            tp_scale(tp_pow(V, 2), 3)),
            tp_scale(U, 252)),
            tp_scale(V, -54)),
        tp_from_mono(0, 0, -405));
    px0 = tp_scale(tp_mul(V, inner_x0), 2);

    // x1 = 4v * (inner terms)
    struct Term { long coeff; int eu, ev; };
    Term x1_terms[] = {
        {1, 4, 1}, {-24, 4, 0}, {-66, 3, 1}, {9, 2, 2}, {1188, 3, 0},
        {297, 2, 1}, {138, 1, 2}, {-36, 0, 3}, {-8424, 1, 1}, {945, 0, 2},
        {14580, 0, 1}
    };
    TPoly inner_x1;
    for (const auto& t : x1_terms)
        inner_x1 = tp_add(inner_x1, tp_scale(tp_mul(tp_pow(U, t.eu), tp_pow(V, t.ev)), t.coeff));
    px1 = tp_scale(tp_mul(V, inner_x1), 4);

    // x2 = 4v * (inner terms)
    Term x2_terms[] = {
        {2, 6, 2}, {-8, 5, 3}, {2, 4, 4}, {-40, 6, 1}, {106, 5, 2},
        {495, 4, 3}, {-204, 3, 4}, {18, 2, 5}, {-144, 6, 0}, {1476, 5, 1},
        {-18756, 4, 2}, {4280, 3, 3}, {-1038, 2, 4}, {564, 1, 5}, {-72, 0, 6},
        {160704, 4, 1}, {4464, 3, 2}, {75024, 2, 3}, {-33480, 1, 4}, {3186, 0, 5},
        {-104004, 3, 1}, {-1353996, 2, 2}, {315252, 1, 3}, {-4032, 0, 4},
        {3669786, 1, 2}, {-622323, 0, 3}, {-2821230, 0, 2}
    };
    TPoly inner_x2;
    for (const auto& t : x2_terms)
        inner_x2 = tp_add(inner_x2, tp_scale(tp_mul(tp_pow(U, t.eu), tp_pow(V, t.ev)), t.coeff));
    px2 = tp_scale(tp_mul(V, inner_x2), 4);

    // x3 = -16 * v^2 * (v - 27) * (4u^3 - u^2*v - 18uv + 4v^2 + 27v)^3
    TPoly v_minus_27 = tp_add(V, tp_from_mono(0, 0, -27));
    TPoly cubic = tp_add(tp_add(tp_add(tp_add(
        tp_scale(tp_pow(U, 3), 4),
        tp_scale(tp_mul(tp_pow(U, 2), V), -1)),
        tp_scale(tp_mul(U, V), -18)),
        tp_scale(tp_pow(V, 2), 4)),
        tp_scale(V, 27));
    px3 = tp_scale(tp_mul(tp_mul(tp_pow(V, 2), v_minus_27), tp_pow(cubic, 3)), -16);

    std::cout << "  L3 parametrisation built:\n"
              << "    x0: " << px0.size() << " terms\n"
              << "    x1: " << px1.size() << " terms\n"
              << "    x2: " << px2.size() << " terms\n"
              << "    x3: " << px3.size() << " terms\n";
}


/* =====================================================================
 * STAGE 1: Setup (trivial for ansatz approach)
 * ===================================================================== */

static void stage1_setup(double& time_ms)
{
    auto t0 = Clock::now();
    // For the ansatz approach, Stage 1 just prints info
    time_ms = elapsed_ms(t0);
}


/* =====================================================================
 * STAGE 2: Linear-Algebra Ansatz
 *
 * Enumerate all monomials x0^a x1^b x2^c x3^d of weighted degree 80
 * (weights 2,4,6,10). Substitute the L3 parametrisation and require
 * the result to vanish identically in (u,v). This gives a linear
 * system whose kernel is the defining polynomial.
 *
 * Matrix size: ~(num_uv_monomials) x 521, over Z.
 * Fraction-free Gaussian elimination with GMP.
 * ===================================================================== */

static std::vector<WPoly> stage2_elimination(double& time_ms)
{
    auto t0 = Clock::now();
    const int D = 80;
    const int w[4] = {2, 4, 6, 10};

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

    /* ---- Step B: build L3 parametrisation as TPolys ---- */
    TPoly px0, px1, px2, px3;
    build_l3_parametrisation(px0, px1, px2, px3);

    /* Precompute powers */
    int ma = 0, mb = 0, mc = 0, md = 0;
    for (auto& m : monoms) {
        ma = std::max(ma, m.a); mb = std::max(mb, m.b);
        mc = std::max(mc, m.c); md = std::max(md, m.d);
    }
    std::cout << "  Max powers: x0^" << ma << " x1^" << mb
              << " x2^" << mc << " x3^" << md << "\n";
    std::cout << "  Precomputing powers..." << std::flush;

    auto tp0 = Clock::now();
    auto x0p = tp_powers(px0, ma);
    auto x1p = tp_powers(px1, mb);
    auto x2p = tp_powers(px2, mc);
    auto x3p = tp_powers(px3, md);
    std::cout << " done (" << elapsed_ms(tp0) << " ms)\n";

    /* ---- Step C: substitute into each monomial, collect (u,v) exponents ---- */
    std::cout << "  Substituting parametrisation into " << ncols << " monomials...\n";
    auto ts0 = Clock::now();
    std::vector<TPoly> cols(ncols);
    std::set<std::pair<int,int>> t_set;

    #pragma omp parallel
    {
        std::set<std::pair<int,int>> local_set;
        #pragma omp for schedule(dynamic, 4)
        for (int j = 0; j < ncols; ++j) {
            cols[j] = tp_mul(tp_mul(x0p[monoms[j].a], x1p[monoms[j].b]),
                             tp_mul(x2p[monoms[j].c], x3p[monoms[j].d]));
            for (const auto& [k, _] : cols[j]) local_set.insert(k);
            if ((j+1) % 50 == 0) {
                #pragma omp critical
                std::cout << "    " << (j+1) << "/" << ncols
                          << " (" << cols[j].size() << " terms)\n";
            }
        }
        #pragma omp critical
        t_set.insert(local_set.begin(), local_set.end());
    }

    std::vector<std::pair<int,int>> t_list(t_set.begin(), t_set.end());
    std::map<std::pair<int,int>, int> t_idx;
    for (int i = 0; i < (int)t_list.size(); ++i) t_idx[t_list[i]] = i;
    const int nrows = (int)t_list.size();
    std::cout << "  Substitution done (" << elapsed_ms(ts0) << " ms)\n";
    std::cout << "  Matrix size: " << nrows << " x " << ncols << "\n";

    /* ---- Step D: build matrix M (nrows x ncols) over Z ---- */
    std::cout << "  Building matrix...\n";
    auto tm0 = Clock::now();
    std::vector<std::vector<mpz_class>> M(nrows, std::vector<mpz_class>(ncols, 0));
    for (int j = 0; j < ncols; ++j)
        for (const auto& [k, v] : cols[j])
            M[t_idx[k]][j] = v;
    // Free substitution data
    cols.clear(); cols.shrink_to_fit();
    std::cout << "  Matrix built (" << elapsed_ms(tm0) << " ms)\n";

    /* ---- Step E: Gaussian elimination (fraction-free) ---- */
    std::cout << "  Gaussian elimination (" << nrows << " x " << ncols << ")...\n";
    auto tg0 = Clock::now();

    std::vector<int> pivot_col(nrows, -1);
    int cur_row = 0;
    for (int col = 0; col < ncols && cur_row < nrows; ++col) {
        int pr = -1;
        for (int r = cur_row; r < nrows; ++r) {
            if (M[r][col] != 0) { pr = r; break; }
        }
        if (pr < 0) continue;
        if (pr != cur_row) std::swap(M[pr], M[cur_row]);
        pivot_col[cur_row] = col;

        mpz_class piv = M[cur_row][col];
        #pragma omp parallel for schedule(dynamic)
        for (int r = 0; r < nrows; ++r) {
            if (r == cur_row || M[r][col] == 0) continue;
            mpz_class factor = M[r][col];
            for (int c = 0; c < ncols; ++c)
                M[r][c] = M[r][c] * piv - factor * M[cur_row][c];
            // Remove content
            mpz_class g = 0;
            for (int c = 0; c < ncols; ++c) {
                if (M[r][c] != 0) {
                    if (g == 0) g = abs(M[r][c]);
                    else mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), M[r][c].get_mpz_t());
                }
            }
            if (g > 1)
                for (int c = 0; c < ncols; ++c)
                    if (M[r][c] != 0) mpz_divexact(M[r][c].get_mpz_t(),
                                                    M[r][c].get_mpz_t(), g.get_mpz_t());
        }
        ++cur_row;
        if (cur_row % 50 == 0)
            std::cout << "    pivot " << cur_row << "/" << ncols
                      << " (" << elapsed_ms(tg0) << " ms)\n";
    }
    int rank = cur_row;
    int nullity = ncols - rank;
    std::cout << "  Gaussian elimination done (" << elapsed_ms(tg0) << " ms)\n";
    std::cout << "  Rank: " << rank << ", nullity: " << nullity << "\n";

    if (nullity != 1) {
        std::cout << "  ERROR: expected nullity 1, got " << nullity << "\n";
        time_ms = elapsed_ms(t0);
        return {};
    }

    /* Back-substitute to find kernel vector */
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

    // Scale: set free var = product of diagonal pivots
    mpz_class scale = 1;
    for (int r = 0; r < rank; ++r) scale *= M[r][pivot_col[r]];

    std::vector<mpz_class> kernel(ncols, 0);
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
    if (g > 1)
        for (int j = 0; j < ncols; ++j)
            if (kernel[j] != 0)
                mpz_divexact(kernel[j].get_mpz_t(), kernel[j].get_mpz_t(), g.get_mpz_t());
    // Make leading nonzero positive
    for (int j = 0; j < ncols; ++j) {
        if (kernel[j] != 0) {
            if (kernel[j] < 0)
                for (int k = 0; k < ncols; ++k) kernel[k] = -kernel[k];
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

    // Verify at test points from L3 parametrisation
    std::cout << "  Verification at parametrisation test points:\n";
    auto eval_l3 = [&](int u, int v) -> std::vector<mpz_class> {
        mpz_class U(u), V(v);
        mpz_class x0(0), x1(0), x2(0), x3(0);
        for (const auto& [e, c] : px0) {
            mpz_class term = c;
            for (int i = 0; i < e.first; ++i) term *= U;
            for (int i = 0; i < e.second; ++i) term *= V;
            x0 += term;
        }
        for (const auto& [e, c] : px1) {
            mpz_class term = c;
            for (int i = 0; i < e.first; ++i) term *= U;
            for (int i = 0; i < e.second; ++i) term *= V;
            x1 += term;
        }
        for (const auto& [e, c] : px2) {
            mpz_class term = c;
            for (int i = 0; i < e.first; ++i) term *= U;
            for (int i = 0; i < e.second; ++i) term *= V;
            x2 += term;
        }
        for (const auto& [e, c] : px3) {
            mpz_class term = c;
            for (int i = 0; i < e.first; ++i) term *= U;
            for (int i = 0; i < e.second; ++i) term *= V;
            x3 += term;
        }
        return {x0, x1, x2, x3};
    };

    int test_pts[][2] = {{1,1},{2,1},{1,2},{3,1},{1,3},{2,3},{5,2},{0,3}};
    bool all_ok = true;
    for (auto& pt : test_pts) {
        auto xv = eval_l3(pt[0], pt[1]);
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
        std::cout << "    (u,v)=(" << pt[0] << "," << pt[1]
                  << ") -> F = " << (val == 0 ? "0" : "nonzero")
                  << " [" << (ok ? "OK" : "FAIL") << "]\n";
    }
    std::cout << "  All tests: " << (all_ok ? "PASS" : "FAIL") << "\n";

    std::vector<WPoly> result;
    if (!res.is_zero()) result.push_back(std::move(res));

    time_ms = elapsed_ms(t0);
    std::cout << "  Stage 2 complete: " << result.size()
              << " polynomials in elimination ideal\n";
    return result;
}


/* =====================================================================
 * STAGE 3: Parallel Weighted GCD Normalization (MPI + GMP)
 * ===================================================================== */

static mpz_class weighted_gcd_normalize(WPoly& poly, int rank, int n_procs) {
    if (poly.is_zero()) return mpz_class(1);

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
                  << "  Experiment 6: L3 Three-Stage Parallel Pipeline\n"
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
    if (rank == 0) std::cout << "=== STAGE 1: Setup ===\n";
    stage1_setup(t1_ms);
    if (rank == 0)
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << t1_ms << " ms\n\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // === Stage 2 (rank 0 only) ===
    if (rank == 0) std::cout << "=== STAGE 2: Linear-Algebra Ansatz Elimination ===\n";
    std::vector<WPoly> elim_ideal;
    if (rank == 0) {
        elim_ideal = stage2_elimination(t2_ms);
        std::cout << "  Elimination ideal: " << elim_ideal.size() << " polys\n"
                  << "  Time: " << t2_ms << " ms\n\n";
    }

    // Broadcast results
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
                  << "  Stage 1 (Setup):          " << t1_ms << " ms\n"
                  << "  Stage 2 (Elimination):    " << t2_ms << " ms\n"
                  << "  Stage 3 (Normalization):  " << t3_ms << " ms\n\n";

        for (int i = 0; i < (int)final_ideal.size(); ++i) {
            const auto& p = final_ideal[i];
            std::cout << "Polynomial " << i << ": " << p.nnz() << " terms, wdeg="
                      << p.weighted_degree() << ", w-homog="
                      << (p.is_weighted_homogeneous() ? "YES" : "NO") << "\n";
        }

        // Write CSV
        std::filesystem::create_directories("../../results");
        std::ofstream csv("../../results/exp6_l3_pipeline.csv");
        csv << "stage,time_ms,n_procs,n_threads\n"
            << "setup," << t1_ms << "," << n_procs << "," << n_threads << "\n"
            << "elimination," << t2_ms << "," << n_procs << "," << n_threads << "\n"
            << "normalization," << t3_ms << "," << n_procs << "," << n_threads << "\n"
            << "total," << (t1_ms + t2_ms + t3_ms) << "," << n_procs << "," << n_threads << "\n";
        csv.close();
        std::cout << "\nSaved -> ../../results/exp6_l3_pipeline.csv\n";

        // Write polynomial to file
        {
            std::ofstream pf("../../results/l3_polynomial.txt");
            pf << "# L3 defining polynomial in P(2,4,6,10)\n"
               << "# Weighted degree: 80\n"
               << "# Number of terms: " << final_ideal[0].nnz() << "\n"
               << "# Format: coefficient x0^a x1^b x2^c x3^d\n\n";

            const char* xn[] = {"x0", "x1", "x2", "x3"};
            for (const auto& [m, c] : final_ideal[0].terms) {
                pf << c;
                for (int i = VAR_X0; i <= VAR_X3; ++i)
                    if (m[i] > 0) {
                        pf << "*" << xn[i - VAR_X0];
                        if (m[i] > 1) pf << "^" << m[i];
                    }
                pf << "\n";
            }
            pf.close();
            std::cout << "Saved polynomial -> ../../results/l3_polynomial.txt\n";

            // Also write monomial-only summary (without huge coefficients)
            std::ofstream ms("../../results/l3_monomials.txt");
            ms << "# L3 monomial structure: 318 terms of weighted degree 80\n"
               << "# Format: a b c d  |coeff_digits|\n";
            for (const auto& [m, c] : final_ideal[0].terms) {
                ms << m[VAR_X0] << " " << m[VAR_X1] << " "
                   << m[VAR_X2] << " " << m[VAR_X3] << "  "
                   << c.get_str().size() << "\n";
            }
            ms.close();
            std::cout << "Saved monomial summary -> ../../results/l3_monomials.txt\n";
        }
    }

    MPI_Finalize();
    return 0;
}
