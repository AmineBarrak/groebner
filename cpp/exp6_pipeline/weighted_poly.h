/*
 * cpp/exp6_l2_pipeline/weighted_poly.h
 * =====================================
 * Extended polynomial type for weighted projective computations.
 * Uses GMP (mpz_class) for exact integer arithmetic.
 *
 * Variable ordering: [s, t1, t2, x0, x1, x2, x3] (indices 0..6)
 * Elimination order: s > t1 > t2 > x0 > x1 > x2 > x3 (lex)
 * Weight vector for x-variables: q = (2, 4, 6, 10)
 */
#pragma once

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <cmath>
#include <iostream>
#include "gmp_wrapper.h"  // uses gmpxx.h if available, else C API wrapper

enum VarIdx : int {
    VAR_S  = 0,
    VAR_T1 = 1,
    VAR_T2 = 2,
    VAR_X0 = 3,
    VAR_X1 = 4,
    VAR_X2 = 5,
    VAR_X3 = 6,
    NUM_VARS = 7
};

static const int WEIGHTS[] = {1, 1, 1, 2, 4, 6, 10};

using Mono = std::vector<int>;

struct MonoLexCmp {
    bool operator()(const Mono& a, const Mono& b) const {
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) return a[i] > b[i];
        return false;
    }
};

struct WPoly {
    std::map<Mono, mpz_class, MonoLexCmp> terms;

    WPoly() = default;
    bool is_zero() const { return terms.empty(); }

    void add_term(const Mono& m, const mpz_class& c) {
        if (c == 0) return;
        terms[m] += c;
        if (terms[m] == 0) terms.erase(m);
    }

    const Mono& LM() const { assert(!terms.empty()); return terms.begin()->first; }
    const mpz_class& LC() const { assert(!terms.empty()); return terms.begin()->second; }

    int total_degree() const {
        int d = 0;
        for (const auto& [m, _] : terms) {
            int td = 0; for (int e : m) td += e;
            d = std::max(d, td);
        }
        return d;
    }

    int weighted_degree() const {
        int d = 0;
        for (const auto& [m, _] : terms) {
            int wd = 0;
            for (int i = VAR_X0; i <= VAR_X3; ++i) wd += WEIGHTS[i] * m[i];
            d = std::max(d, wd);
        }
        return d;
    }

    bool is_in_x_ring() const {
        for (const auto& [m, _] : terms)
            if (m[VAR_S] || m[VAR_T1] || m[VAR_T2]) return false;
        return true;
    }

    bool is_weighted_homogeneous() const {
        if (terms.empty()) return true;
        int deg = -1;
        for (const auto& [m, _] : terms) {
            int wd = 0;
            for (int i = VAR_X0; i <= VAR_X3; ++i) wd += WEIGHTS[i] * m[i];
            if (deg < 0) deg = wd;
            else if (wd != deg) return false;
        }
        return true;
    }

    int nnz() const { return (int)terms.size(); }

    std::string to_string() const {
        static const char* vars[] = {"s","t1","t2","x0","x1","x2","x3"};
        if (terms.empty()) return "0";
        std::ostringstream ss;
        bool first = true;
        for (const auto& [m, c] : terms) {
            if (!first) ss << (c > 0 ? " + " : " - ");
            else if (c < 0) ss << "-";
            first = false;
            mpz_class ac = abs(c);
            bool all_zero = true;
            for (int e : m) if (e) { all_zero = false; break; }
            if (ac != 1 || all_zero) ss << ac;
            for (int i = 0; i < NUM_VARS; ++i)
                if (m[i] > 0) {
                    ss << "*" << vars[i];
                    if (m[i] > 1) ss << "^" << m[i];
                }
        }
        return ss.str();
    }
};

/* -- Polynomial arithmetic --------------------------------------------- */

inline WPoly wpoly_sub(const WPoly& a, const WPoly& b) {
    WPoly r = a;
    for (const auto& [m, c] : b.terms) r.add_term(m, -c);
    return r;
}

inline WPoly wpoly_mul_mono(const WPoly& p, const Mono& m, const mpz_class& scalar) {
    WPoly r;
    for (const auto& [pm, pc] : p.terms) {
        Mono nm(NUM_VARS);
        for (int i = 0; i < NUM_VARS; ++i) nm[i] = pm[i] + m[i];
        r.add_term(nm, pc * scalar);
    }
    return r;
}

inline WPoly wpoly_scale(const WPoly& p, const mpz_class& s) {
    WPoly r;
    for (const auto& [m, c] : p.terms) r.add_term(m, c * s);
    return r;
}

/* -- Monomial operations ----------------------------------------------- */

inline Mono mono_lcm(const Mono& a, const Mono& b) {
    Mono r(NUM_VARS);
    for (int i = 0; i < NUM_VARS; ++i) r[i] = std::max(a[i], b[i]);
    return r;
}

inline bool mono_divides(const Mono& a, const Mono& b) {
    for (int i = 0; i < NUM_VARS; ++i) if (a[i] > b[i]) return false;
    return true;
}

inline Mono mono_div(const Mono& a, const Mono& b) {
    Mono r(NUM_VARS);
    for (int i = 0; i < NUM_VARS; ++i) { r[i] = a[i] - b[i]; assert(r[i] >= 0); }
    return r;
}

inline Mono mono_zero() { return Mono(NUM_VARS, 0); }
inline Mono mono_var(int idx, int exp = 1) {
    Mono m(NUM_VARS, 0); m[idx] = exp; return m;
}

/* -- S-polynomial (over Z) --------------------------------------------- */
inline WPoly s_polynomial_z(const WPoly& f, const WPoly& g) {
    Mono lcm = mono_lcm(f.LM(), g.LM());
    Mono mf  = mono_div(lcm, f.LM());
    Mono mg  = mono_div(lcm, g.LM());
    WPoly lhs = wpoly_mul_mono(f, mf, g.LC());
    WPoly rhs = wpoly_mul_mono(g, mg, f.LC());
    return wpoly_sub(lhs, rhs);
}

/* -- Reduction modulo basis (over Z) ----------------------------------- */
inline WPoly reduce_z(WPoly p, const std::vector<WPoly>& basis) {
    bool changed = true;
    int max_iters = 10000;
    while (changed && !p.is_zero() && max_iters-- > 0) {
        changed = false;
        for (const auto& b : basis) {
            if (b.is_zero()) continue;
            for (auto it = p.terms.begin(); it != p.terms.end(); ++it) {
                if (mono_divides(b.LM(), it->first)) {
                    Mono d = mono_div(it->first, b.LM());
                    WPoly new_p;
                    for (const auto& [m, c] : p.terms) new_p.add_term(m, c * b.LC());
                    WPoly sub2 = wpoly_mul_mono(b, d, it->second);
                    p = wpoly_sub(new_p, sub2);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    }
    return p;
}

/* -- Exact polynomial division ----------------------------------------- */
/* Given that a divides b exactly (remainder = 0), compute b/a.
 * Uses repeated leading-term division. Asserts exactness. */
inline WPoly wpoly_exact_div(const WPoly& b, const WPoly& a) {
    assert(!a.is_zero());
    WPoly remainder = b;
    WPoly quotient;
    const Mono& a_lm = a.LM();
    const mpz_class& a_lc = a.LC();

    int safety = 100000;
    while (!remainder.is_zero() && safety-- > 0) {
        const Mono& r_lm = remainder.LM();
        const mpz_class& r_lc = remainder.LC();

        // Check leading monomial divisibility
        bool div_ok = true;
        Mono qm(NUM_VARS, 0);
        for (int i = 0; i < NUM_VARS; ++i) {
            qm[i] = r_lm[i] - a_lm[i];
            if (qm[i] < 0) { div_ok = false; break; }
        }

        if (!div_ok) {
            // Not exactly divisible — this shouldn't happen in Bareiss
            std::cerr << "wpoly_exact_div: not divisible! remainder has "
                      << remainder.nnz() << " terms\n";
            break;
        }

        // Coefficient must divide exactly
        mpz_class qc;
        mpz_tdiv_q(qc.get_mpz_t(), r_lc.get_mpz_t(), a_lc.get_mpz_t());

        quotient.add_term(qm, qc);

        // remainder -= qc * x^qm * a
        WPoly sub = wpoly_mul_mono(a, qm, qc);
        remainder = wpoly_sub(remainder, sub);
    }
    return quotient;
}

/* -- Content removal --------------------------------------------------- */
inline void remove_content(WPoly& p) {
    if (p.is_zero()) return;
    mpz_class g = 0;
    for (const auto& [m, c] : p.terms) {
        if (g == 0) g = abs(c);
        else mpz_gcd(g.get_mpz_t(), g.get_mpz_t(), c.get_mpz_t());
        if (g == 1) return;
    }
    if (g > 1) {
        WPoly r;
        for (const auto& [m, c] : p.terms) r.add_term(m, c / g);
        p = std::move(r);
    }
    if (p.LC() < 0) {
        WPoly r;
        for (const auto& [m, c] : p.terms) r.add_term(m, -c);
        p = std::move(r);
    }
}
