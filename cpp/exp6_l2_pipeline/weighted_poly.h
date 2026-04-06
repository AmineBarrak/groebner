/*
 * cpp/exp6_l2_pipeline/weighted_poly.h
 * =====================================
 * Extended polynomial type for weighted projective computations.
 * Uses GMP (mpz_class) for exact integer arithmetic — essential for
 * the large coefficients arising in F4 elimination.
 *
 * Variable ordering for the L2 example:
 *   [s, t1, t2, x0, x1, x2, x3]   (7 variables, indices 0..6)
 *
 * Elimination order: s > t1 > t2 > x0 > x1 > x2 > x3  (lex)
 *
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
#include <gmpxx.h>

/* ── Variable indices ──────────────────────────────────────────────────── */
enum VarIdx : int {
    VAR_S  = 0,   // homogenization variable
    VAR_T1 = 1,   // parameter t1
    VAR_T2 = 2,   // parameter t2
    VAR_X0 = 3,   // x0  (weight 2)
    VAR_X1 = 4,   // x1  (weight 4)
    VAR_X2 = 5,   // x2  (weight 6)
    VAR_X3 = 6,   // x3  (weight 10)
    NUM_VARS = 7
};

static const int WEIGHTS[] = {1, 1, 1, 2, 4, 6, 10};
// s, t1, t2 get weight 1 for the homogenization grading

/* ── Monomial (exponent vector, length NUM_VARS) ──────────────────────── */
using Mono = std::vector<int>;

struct MonoLexCmp {
    // Lex order: s > t1 > t2 > x0 > x1 > x2 > x3
    bool operator()(const Mono& a, const Mono& b) const {
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return a[i] > b[i];
        }
        return false;
    }
};

/* ── Polynomial with GMP integer coefficients ─────────────────────────── */
struct WPoly {
    std::map<Mono, mpz_class, MonoLexCmp> terms;

    WPoly() = default;

    bool is_zero() const { return terms.empty(); }

    void add_term(const Mono& m, const mpz_class& c) {
        if (c == 0) return;
        terms[m] += c;
        if (terms[m] == 0) terms.erase(m);
    }

    // Leading monomial (lex order)
    const Mono& LM() const {
        assert(!terms.empty());
        return terms.begin()->first;
    }
    const mpz_class& LC() const {
        assert(!terms.empty());
        return terms.begin()->second;
    }

    // Total degree
    int total_degree() const {
        int d = 0;
        for (const auto& [m, _] : terms) {
            int td = 0; for (int e : m) td += e;
            d = std::max(d, td);
        }
        return d;
    }

    // Weighted degree w.r.t. q = (2,4,6,10) for x-vars only
    int weighted_degree() const {
        int d = 0;
        for (const auto& [m, _] : terms) {
            int wd = 0;
            for (int i = VAR_X0; i <= VAR_X3; ++i)
                wd += WEIGHTS[i] * m[i];
            d = std::max(d, wd);
        }
        return d;
    }

    // Is the polynomial free of variables s, t1, t2?
    bool is_in_x_ring() const {
        for (const auto& [m, _] : terms) {
            if (m[VAR_S] != 0 || m[VAR_T1] != 0 || m[VAR_T2] != 0)
                return false;
        }
        return true;
    }

    // Is the polynomial weighted homogeneous in (x0,..,x3) with weights (2,4,6,10)?
    bool is_weighted_homogeneous() const {
        if (terms.empty()) return true;
        int deg = -1;
        for (const auto& [m, _] : terms) {
            int wd = 0;
            for (int i = VAR_X0; i <= VAR_X3; ++i)
                wd += WEIGHTS[i] * m[i];
            if (deg < 0) deg = wd;
            else if (wd != deg) return false;
        }
        return true;
    }

    int nnz() const { return (int)terms.size(); }

    std::string to_string(const char* vars[] = nullptr) const {
        static const char* default_vars[] = {"s","t1","t2","x0","x1","x2","x3"};
        if (!vars) vars = default_vars;
        if (terms.empty()) return "0";
        std::ostringstream ss;
        bool first = true;
        for (const auto& [m, c] : terms) {
            if (!first) { ss << (c > 0 ? " + " : " - "); }
            else if (c < 0) { ss << "-"; }
            first = false;
            mpz_class ac = abs(c);
            bool all_zero = true;
            for (int e : m) if (e) { all_zero = false; break; }
            if (ac != 1 || all_zero) ss << ac;
            for (int i = 0; i < NUM_VARS; ++i) {
                if (m[i] > 0) {
                    ss << "*" << vars[i];
                    if (m[i] > 1) ss << "^" << m[i];
                }
            }
        }
        return ss.str();
    }
};

/* ── Polynomial arithmetic ────────────────────────────────────────────── */

inline WPoly wpoly_add(const WPoly& a, const WPoly& b) {
    WPoly r = a;
    for (const auto& [m, c] : b.terms) r.add_term(m, c);
    return r;
}

inline WPoly wpoly_sub(const WPoly& a, const WPoly& b) {
    WPoly r = a;
    for (const auto& [m, c] : b.terms) r.add_term(m, -c);
    return r;
}

inline WPoly wpoly_mul(const WPoly& a, const WPoly& b) {
    WPoly r;
    for (const auto& [ma, ca] : a.terms) {
        for (const auto& [mb, cb] : b.terms) {
            Mono mn(NUM_VARS);
            for (int i = 0; i < NUM_VARS; ++i) mn[i] = ma[i] + mb[i];
            r.add_term(mn, ca * cb);
        }
    }
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

/* ── Monomial operations ──────────────────────────────────────────────── */

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

/* ── S-polynomial (over Z, clearing leading coefficient) ──────────────── */
inline WPoly s_polynomial_z(const WPoly& f, const WPoly& g) {
    Mono lcm = mono_lcm(f.LM(), g.LM());
    Mono mf  = mono_div(lcm, f.LM());
    Mono mg  = mono_div(lcm, g.LM());
    // S(f,g) = LC(g)*x^mf*f - LC(f)*x^mg*g  (integer variant, no division)
    WPoly lhs = wpoly_mul_mono(f, mf, g.LC());
    WPoly rhs = wpoly_mul_mono(g, mg, f.LC());
    return wpoly_sub(lhs, rhs);
}

/* ── Reduction of p modulo basis (over Z) ─────────────────────────────── */
inline WPoly reduce_z(WPoly p, const std::vector<WPoly>& basis) {
    bool changed = true;
    int max_iters = 10000;
    while (changed && !p.is_zero() && max_iters-- > 0) {
        changed = false;
        for (const auto& b : basis) {
            if (b.is_zero()) continue;
            // Find any term of p divisible by LM(b)
            for (auto it = p.terms.begin(); it != p.terms.end(); ++it) {
                if (mono_divides(b.LM(), it->first)) {
                    Mono d = mono_div(it->first, b.LM());
                    // p = LC(b)*p - LC(p_term)*x^d*b
                    WPoly new_p = wpoly_scale(p, b.LC());
                    WPoly sub   = wpoly_mul_mono(b, d, it->second * b.LC() / b.LC());
                    // Actually: p <- b.LC()*p - p_coef*x^d*b
                    new_p = WPoly();
                    for (const auto& [m, c] : p.terms)
                        new_p.add_term(m, c * b.LC());
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

/* ── Content removal (divide all coefficients by their GCD) ───────────── */
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
        for (const auto& [m, c] : p.terms)
            r.add_term(m, c / g);
        p = std::move(r);
    }
    // Make leading coefficient positive
    if (p.LC() < 0) {
        WPoly r;
        for (const auto& [m, c] : p.terms) r.add_term(m, -c);
        p = std::move(r);
    }
}
