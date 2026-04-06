#!/usr/bin/env python3
"""
SymPy baseline for L2 elimination in P(2,4,6,10).
Computes the elimination ideal using sympy.groebner with lex order.
This is the sequential baseline the paper compares against.
"""
import time
import sys
from sympy import symbols, groebner, expand, Poly, ZZ

print("=" * 60)
print("SymPy Baseline: L2 Elimination in P(2,4,6,10)")
print("=" * 60)
print(f"Python {sys.version}")

t1, t2, x0, x1, x2, x3 = symbols('t1 t2 x0 x1 x2 x3')

# Parametrisation equations: x_i - phi_i(t1,t2) = 0
F0 = x0 - (-120 - 8*t1)
F1 = x1 - (t1**2 - 126*t1 + 12*t2 + 405)
F2 = x2 - (-3*t1**3 + 53*t1**2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985)
inner = -t1**2 - 18*t1 + 4*t2 + 27
F3 = x3 - (-2 * inner**2)

gens = [F0, F1, F2, F3]

print(f"\nGenerators: {len(gens)} polynomials in Z[t1,t2,x0,x1,x2,x3]")
for i, g in enumerate(gens):
    p = Poly(expand(g), t1, t2, x0, x1, x2, x3, domain='ZZ')
    print(f"  F{i}: {len(p.monoms())} terms, total deg {p.total_degree()}")

# --- Method 1: Groebner basis with lex order (elimination) ---
print(f"\n{'='*60}")
print("Method 1: sympy.groebner with lex order (t1 > t2 > x0 > x1 > x2 > x3)")
print(f"{'='*60}")

t_start = time.perf_counter()
G = groebner(gens, t1, t2, x0, x1, x2, x3, order='lex', domain='ZZ')
t_lex = time.perf_counter() - t_start

print(f"Time: {t_lex*1000:.3f} ms")
print(f"Basis size: {len(G.polys)}")

# Extract polynomials in elimination ideal (only x0..x3)
elim_polys = []
for p in G.polys:
    poly = Poly(p, t1, t2, x0, x1, x2, x3, domain='ZZ')
    degs_t = [poly.degree(t1), poly.degree(t2)]
    if degs_t[0] == 0 and degs_t[1] == 0:
        elim_polys.append(p)

print(f"Polynomials in elimination ideal (no t1,t2): {len(elim_polys)}")
for i, p in enumerate(elim_polys):
    poly = Poly(p, x0, x1, x2, x3, domain='ZZ')
    print(f"  G{i}: {len(poly.monoms())} terms, total deg {poly.total_degree()}")

    # Check weighted homogeneity
    weights = {x0: 2, x1: 4, x2: 6, x3: 10}
    wdegs = set()
    for monom in poly.monoms():
        wd = monom[0]*2 + monom[1]*4 + monom[2]*6 + monom[3]*10
        wdegs.add(wd)
    is_wh = len(wdegs) == 1
    print(f"      Weighted homogeneous: {'YES' if is_wh else 'NO'}, "
          f"weighted degrees: {sorted(wdegs)}")

# --- Method 2: Substitution + groebner ---
print(f"\n{'='*60}")
print("Method 2: Substitute t1,t2 then groebner on {G2,G3}")
print(f"{'='*60}")

t_start = time.perf_counter()

# From F0: t1 = -(x0+120)/8
t1_val = -(x0 + 120) / 8
# From F1: t2 = (x1 - t1^2 + 126*t1 - 405)/12
t2_val = (x1 - t1_val**2 + 126*t1_val - 405) / 12

G2_expr = expand(F2.subs(t1, t1_val).subs(t2, t2_val))
G3_expr = expand(F3.subs(t1, t1_val).subs(t2, t2_val))

t_sub = time.perf_counter() - t_start

print(f"Substitution time: {t_sub*1000:.3f} ms")

G2p = Poly(G2_expr, x0, x1, x2, x3, domain='QQ')
G3p = Poly(G3_expr, x0, x1, x2, x3, domain='QQ')
print(f"G2: {len(G2p.monoms())} terms")
print(f"G3: {len(G3p.monoms())} terms")

# Check weighted homogeneity of G2, G3
for name, poly in [("G2", G2p), ("G3", G3p)]:
    wdegs = set()
    for monom in poly.monoms():
        wd = monom[0]*2 + monom[1]*4 + monom[2]*6 + monom[3]*10
        wdegs.add(wd)
    is_wh = len(wdegs) == 1
    print(f"  {name} weighted homogeneous: {'YES' if is_wh else 'NO'}, "
          f"weighted degrees: {sorted(wdegs)}")

# --- Method 3: Direct ansatz (what our pipeline does) ---
print(f"\n{'='*60}")
print("Method 3: Linear-algebra ansatz (same as C++ pipeline)")
print(f"{'='*60}")

from collections import defaultdict
from functools import reduce
from math import gcd

def pmul(p, q):
    r = defaultdict(int)
    for (a1,a2), c1 in p.items():
        for (b1,b2), c2 in q.items():
            r[(a1+b1, a2+b2)] += c1 * c2
    return {k:v for k,v in r.items() if v}

def pscale(p, c):
    if c == 0: return {}
    return {k:v*c for k,v in p.items() if v*c}

px0 = {(0,0): -120, (1,0): -8}
px1 = {(2,0): 1, (1,0): -126, (0,1): 12, (0,0): 405}
px2 = {(3,0): -3, (2,0): 53, (1,1): -20, (1,0): 2583, (0,1): -12, (0,0): -14985}
pinner = {(2,0): -1, (1,0): -18, (0,1): 4, (0,0): 27}
px3 = pscale(pmul(pinner, pinner), -2)

W = (2, 4, 6, 10)
D = 30

t_start = time.perf_counter()

monoms = []
for d in range(D // W[3] + 1):
    for c in range((D - W[3]*d) // W[2] + 1):
        for b in range((D - W[3]*d - W[2]*c) // W[1] + 1):
            rem = D - W[3]*d - W[2]*c - W[1]*b
            if rem >= 0 and rem % W[0] == 0:
                a = rem // W[0]
                monoms.append((a, b, c, d))

def precompute(base, n):
    pows = [{(0,0): 1}]
    for i in range(n): pows.append(pmul(pows[-1], base))
    return pows

max_a = max(m[0] for m in monoms)
max_b = max(m[1] for m in monoms)
max_c = max(m[2] for m in monoms)
max_d = max(m[3] for m in monoms)

x0p = precompute(px0, max_a)
x1p = precompute(px1, max_b)
x2p = precompute(px2, max_c)
x3p = precompute(px3, max_d)

all_t = set()
cols = []
for idx, (a,b,c,d) in enumerate(monoms):
    p = pmul(pmul(x0p[a], x1p[b]), pmul(x2p[c], x3p[d]))
    cols.append(p)
    all_t.update(p.keys())

all_t = sorted(all_t)
t_idx = {m:i for i,m in enumerate(all_t)}
nrows = len(all_t)
ncols = len(monoms)

from sympy import Matrix, Rational, ilcm

M_rows = [[0]*ncols for _ in range(nrows)]
for j, col in enumerate(cols):
    for monom, val in col.items():
        M_rows[t_idx[monom]][j] = val

M = Matrix(M_rows)
ker = M.nullspace()

t_ansatz = time.perf_counter() - t_start

print(f"Time: {t_ansatz*1000:.3f} ms")
print(f"Matrix: {nrows} x {ncols}")
print(f"Nullspace dim: {len(ker)}")

if ker:
    v = ker[0]
    denoms = [Rational(v[i]).q for i in range(len(v))]
    L = reduce(ilcm, denoms)
    int_v = [int(v[i] * L) for i in range(len(v))]
    nonzero = [abs(x) for x in int_v if x != 0]
    g = reduce(gcd, nonzero)
    int_v = [x // g for x in int_v]
    nterms = sum(1 for x in int_v if x != 0)
    print(f"Result: {nterms} terms, weighted degree {D}")

    # Check weighted homogeneity
    wdegs = set()
    for j, (a,b,c,d) in enumerate(monoms):
        if int_v[j] != 0:
            wdegs.add(W[0]*a + W[1]*b + W[2]*c + W[3]*d)
    print(f"Weighted homogeneous: {'YES' if len(wdegs)==1 else 'NO'}")

# --- Summary ---
print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'Method':<45} {'Time (ms)':>12} {'W-Homog?':>10}")
print("-" * 67)
print(f"{'SymPy groebner (lex elimination)':<45} {t_lex*1000:>12.3f} {'see above':>10}")
print(f"{'SymPy substitution':<45} {t_sub*1000:>12.3f} {'NO':>10}")
print(f"{'Python ansatz (sequential)':<45} {t_ansatz*1000:>12.3f} {'YES':>10}")
print(f"{'C++ pipeline on V100 (4 MPI, 8 OMP)':<45} {'22.828':>12} {'YES':>10}")
print(f"\nKey: only the ansatz/pipeline methods guarantee weighted homogeneity.")
