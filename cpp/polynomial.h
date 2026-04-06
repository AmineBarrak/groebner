/*
 * cpp/polynomial.h
 * ================
 * Lightweight polynomial type over Q for Gröbner basis experiments.
 * Monomial = std::vector<int> (exponent vector).
 * Coefficient = double (sufficient for profiling; use mpq_class for production).
 *
 * Shared by exp3_1_openmp and exp3_2_cuda.
 */
#pragma once
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <cassert>


/* ── Monomial (exponent vector) ─────────────────────────────────────────── */
using Monomial = std::vector<int>;

struct MonomialCmp {
    // Lex order
    bool operator()(const Monomial& a, const Monomial& b) const {
        assert(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return a[i] > b[i];   // descending = highest first
        }
        return false;
    }
};

/* ── Polynomial over Q (double coefficients for speed) ──────────────────── */
struct Poly {
    int n_vars;
    std::map<Monomial, double, MonomialCmp> terms;  // monomial → coefficient

    explicit Poly(int nv = 0) : n_vars(nv) {}

    bool is_zero() const { return terms.empty(); }

    void add_term(const Monomial& m, double c) {
        if (std::abs(c) < 1e-12) return;
        terms[m] += c;
        if (std::abs(terms[m]) < 1e-12) terms.erase(m);
    }

    // Leading monomial (first key in lex-descending order)
    const Monomial& LM() const {
        if (terms.empty()) throw std::runtime_error("LM of zero polynomial");
        return terms.begin()->first;
    }
    double LC() const {
        if (terms.empty()) throw std::runtime_error("LC of zero polynomial");
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

    // Number of non-zero terms
    int nnz() const { return (int)terms.size(); }

    std::string to_string() const {
        if (terms.empty()) return "0";
        std::ostringstream ss;
        bool first = true;
        for (const auto& [m, c] : terms) {
            if (!first) ss << (c > 0 ? " + " : " - ");
            else if (c < 0) ss << "-";
            first = false;
            double ac = std::abs(c);
            if (std::abs(ac - 1.0) > 1e-10) ss << ac;
            for (size_t i = 0; i < m.size(); ++i)
                if (m[i] > 0) ss << "x" << i << (m[i]>1 ? "^"+std::to_string(m[i]) : "");
        }
        return ss.str();
    }
};

/* ── Polynomial arithmetic ────────────────────────────────────────────────── */

inline Poly poly_sub(const Poly& a, const Poly& b) {
    Poly r(a.n_vars);
    r.terms = a.terms;
    for (const auto& [m, c] : b.terms) r.add_term(m, -c);
    return r;
}

inline Poly poly_mul_monomial(const Poly& p, const Monomial& m, double scalar) {
    Poly r(p.n_vars);
    for (const auto& [pm, pc] : p.terms) {
        Monomial nm(p.n_vars);
        for (int i = 0; i < p.n_vars; ++i) nm[i] = pm[i] + m[i];
        r.add_term(nm, pc * scalar);
    }
    return r;
}

/* ── LCM of two monomials ──────────────────────────────────────────────────── */
inline Monomial lcm_monomial(const Monomial& a, const Monomial& b) {
    assert(a.size() == b.size());
    Monomial r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = std::max(a[i], b[i]);
    return r;
}

inline bool divides(const Monomial& a, const Monomial& b) {
    for (size_t i = 0; i < a.size(); ++i) if (a[i] > b[i]) return false;
    return true;
}

inline Monomial mono_div(const Monomial& a, const Monomial& b) {
    Monomial r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

/* ── S-polynomial ─────────────────────────────────────────────────────────── */
inline Poly s_polynomial(const Poly& f, const Poly& g) {
    Monomial lcm = lcm_monomial(f.LM(), g.LM());
    Monomial mf  = mono_div(lcm, f.LM());
    Monomial mg  = mono_div(lcm, g.LM());
    Poly lhs = poly_mul_monomial(f, mf,  1.0 / f.LC());
    Poly rhs = poly_mul_monomial(g, mg,  1.0 / g.LC());
    return poly_sub(lhs, rhs);
}

/* ── Reduction of p modulo basis ───────────────────────────────────────────── */
inline Poly reduce(Poly p, const std::vector<Poly>& basis) {
    bool changed = true;
    while (changed && !p.is_zero()) {
        changed = false;
        for (const auto& b : basis) {
            if (b.is_zero()) continue;
            if (divides(b.LM(), p.LM())) {
                Monomial d  = mono_div(p.LM(), b.LM());
                double   sc = p.LC() / b.LC();
                Poly sub    = poly_mul_monomial(b, d, sc);
                p           = poly_sub(p, sub);
                changed     = true;
                break;
            }
        }
    }
    return p;
}

/* ── Build parametrization generators (x_i - t^{i+1}) ─────────────────────── */
// var ordering: [alpha, t, x0, x1, ..., x_{n-1}]
// alpha=0, t=1, x_i = 2+i
inline std::vector<Poly> make_monomial_curve_gens(int n) {
    // After homogenization: alpha^{i+1} * x_i - t^{i+1}
    // 2+n vars total
    int n_vars = 2 + n;
    std::vector<Poly> gens;
    for (int i = 0; i < n; ++i) {
        Poly f(n_vars);
        // alpha^{i+1} * x_i
        {
            Monomial m(n_vars, 0);
            m[0] = i+1;      // alpha
            m[2+i] = 1;      // x_i
            f.add_term(m, 1.0);
        }
        // - t^{i+1}
        {
            Monomial m(n_vars, 0);
            m[1] = i+1;      // t
            f.add_term(m, -1.0);
        }
        gens.push_back(f);
    }
    return gens;
}
