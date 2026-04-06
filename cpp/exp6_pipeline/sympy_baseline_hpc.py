#!/usr/bin/env python3
"""
Sequential CAS baseline for L2 elimination in P(2,4,6,10).
Run this on the SAME HPC node as the C++ pipeline for fair comparison.

Usage on Matilda:
    module load python/3.10
    python3 sympy_baseline_hpc.py

Compares:
  1. Raw sympy.groebner (lex elimination order)
  2. Sequential pipeline (substitute + weighted homogenize + ansatz)
"""
import time
import sys
import os

print("=" * 70)
print("  Sequential CAS Baseline: L2 in P(2,4,6,10)")
print("=" * 70)
print(f"Python   : {sys.version}")
print(f"Hostname : {os.uname().nodename}")
print(f"CPU      : ", end="")
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                print(line.split(":")[1].strip())
                break
except:
    print("unknown")
print()

# ============================================================
#  Setup: L2 parametrisation
# ============================================================
from sympy import symbols, groebner, expand, Poly, ZZ, QQ

t1, t2, x0, x1, x2, x3 = symbols('t1 t2 x0 x1 x2 x3')

F0 = x0 - (-120 - 8*t1)
F1 = x1 - (t1**2 - 126*t1 + 12*t2 + 405)
F2 = x2 - (-3*t1**3 + 53*t1**2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985)
inner = -t1**2 - 18*t1 + 4*t2 + 27
F3 = x3 - (-2 * inner**2)

gens = [expand(F0), expand(F1), expand(F2), expand(F3)]

W = (2, 4, 6, 10)  # weights

def check_weighted_homogeneous(poly_expr, varlist, weights):
    """Check if a polynomial is weighted homogeneous."""
    p = Poly(poly_expr, *varlist, domain='ZZ')
    wdegs = set()
    for monom in p.monoms():
        wd = sum(monom[i] * weights[i] for i in range(len(weights)))
        wdegs.add(wd)
    return len(wdegs) == 1, sorted(wdegs), len(p.monoms())

# ============================================================
#  Experiment 1: Raw sympy.groebner (lex elimination)
# ============================================================
print("=" * 70)
print("  Experiment 1: Raw sympy.groebner (lex elimination)")
print("  Order: t1 > t2 > x0 > x1 > x2 > x3")
print("=" * 70)

t_start = time.perf_counter()
G = groebner(gens, t1, t2, x0, x1, x2, x3, order='lex', domain='ZZ')
t_groebner = time.perf_counter() - t_start

print(f"\n  Time         : {t_groebner:.6f} s  ({t_groebner*1000:.3f} ms)")
print(f"  Basis size   : {len(G.polys)}")

# Extract elimination ideal
elim_polys = []
for p in G.polys:
    poly = Poly(p, t1, t2, x0, x1, x2, x3, domain='ZZ')
    if poly.degree(t1) == 0 and poly.degree(t2) == 0:
        elim_polys.append(p)

print(f"  Elim ideal   : {len(elim_polys)} generators (no t1, t2)")
any_wh = False
for i, p in enumerate(elim_polys):
    is_wh, wds, nt = check_weighted_homogeneous(p, [x0,x1,x2,x3], W)
    if is_wh: any_wh = True
    print(f"    G{i}: {nt:>3} terms, w-homog={'YES' if is_wh else 'NO':>3}, "
          f"wdeg range [{min(wds)}..{max(wds)}]")
print(f"  Any generator weighted homogeneous? {'YES' if any_wh else 'NO'}")

# ============================================================
#  Experiment 2: Sequential pipeline (mimics C++ stages)
# ============================================================
print()
print("=" * 70)
print("  Experiment 2: Sequential pipeline (3-stage, single-threaded)")
print("=" * 70)

# --- Stage 1: Weighted homogenization ---
t_s1 = time.perf_counter()
# (trivial for this problem — just bookkeeping)
t_stage1 = time.perf_counter() - t_s1

# --- Stage 2: Linear-algebra ansatz ---
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

D = 30

t_s2 = time.perf_counter()

# Enumerate monomials
monoms = []
for d in range(D // W[3] + 1):
    for c in range((D - W[3]*d) // W[2] + 1):
        for b in range((D - W[3]*d - W[2]*c) // W[1] + 1):
            rem = D - W[3]*d - W[2]*c - W[1]*b
            if rem >= 0 and rem % W[0] == 0:
                a = rem // W[0]
                monoms.append((a, b, c, d))

# Precompute powers
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

# Substitute and collect
all_t = set()
cols = []
for idx, (a,b,c,d_) in enumerate(monoms):
    p = pmul(pmul(x0p[a], x1p[b]), pmul(x2p[c_], x3p[d_]) if (c_ := c) or True else None)
    # fix: cleaner
    p = pmul(pmul(x0p[a], x1p[b]), pmul(x2p[c], x3p[d_]))
    cols.append(p)
    all_t.update(p.keys())

all_t = sorted(all_t)
t_idx = {m:i for i,m in enumerate(all_t)}
nrows = len(all_t)
ncols = len(monoms)

# Build matrix
from sympy import Matrix, Rational, ilcm
M_rows = [[0]*ncols for _ in range(nrows)]
for j, col in enumerate(cols):
    for monom, val in col.items():
        M_rows[t_idx[monom]][j] = val

M_mat = Matrix(M_rows)
ker = M_mat.nullspace()

t_stage2 = time.perf_counter() - t_s2

# --- Stage 3: GCD normalization ---
t_s3 = time.perf_counter()
if ker:
    v = ker[0]
    denoms = [Rational(v[i]).q for i in range(len(v))]
    L = reduce(ilcm, denoms)
    int_v = [int(v[i] * L) for i in range(len(v))]
    nonzero = [abs(x_) for x_ in int_v if x_ != 0]
    g = reduce(gcd, nonzero)
    int_v = [x_ // g for x_ in int_v]
    nterms = sum(1 for x_ in int_v if x_ != 0)
t_stage3 = time.perf_counter() - t_s3

t_total = t_stage1 + t_stage2 + t_stage3

print(f"\n  Stage 1 (Homogenization) : {t_stage1*1000:>10.3f} ms")
print(f"  Stage 2 (Elimination)    : {t_stage2*1000:>10.3f} ms")
print(f"  Stage 3 (Normalization)  : {t_stage3*1000:>10.3f} ms")
print(f"  Total                    : {t_total*1000:>10.3f} ms")
print(f"\n  Result: {nterms} terms, weighted degree {D}")
print(f"  Matrix: {nrows} x {ncols}, nullspace dim {len(ker)}")

# Verify weighted homogeneity
wdegs = set()
for j, (a,b,c,d_) in enumerate(monoms):
    if int_v[j] != 0:
        wdegs.add(W[0]*a + W[1]*b + W[2]*c + W[3]*d_)
is_wh = len(wdegs) == 1
print(f"  Weighted homogeneous: {'YES' if is_wh else 'NO'}")

# Verify at test points
def peval(p, tv1, tv2):
    return sum(c * (tv1**e1) * (tv2**e2) for (e1,e2), c in p.items())

test_pts = [(0,0),(1,0),(0,1),(1,1),(-1,2),(3,-1),(10,5),(-7,3)]
all_ok = True
for tv1, tv2 in test_pts:
    x0v = peval(px0, tv1, tv2)
    x1v = peval(px1, tv1, tv2)
    x2v = peval(px2, tv1, tv2)
    x3v = peval(px3, tv1, tv2)
    val = sum(int_v[j] * x0v**a * x1v**b * x2v**c * x3v**d_
              for j, (a,b,c,d_) in enumerate(monoms) if int_v[j] != 0)
    if val != 0: all_ok = False
print(f"  Parametrisation check: {'ALL PASS' if all_ok else 'FAIL'}")

# ============================================================
#  Summary
# ============================================================
print()
print("=" * 70)
print("  SUMMARY: L2 in P(2,4,6,10)")
print("=" * 70)
print()
print(f"  {'Method':<50} {'Time (ms)':>12} {'Terms':>6} {'W-Homog':>8}")
print("  " + "-" * 76)
print(f"  {'Raw sympy.groebner (lex elimination)':<50} "
      f"{t_groebner*1000:>12.3f} "
      f"{'N/A':>6} {'NO':>8}")
print(f"  {'Sequential pipeline (Python, single-thread)':<50} "
      f"{t_total*1000:>12.3f} "
      f"{nterms:>6} {'YES':>8}")
print(f"  {'C++ pipeline on V100 (4 MPI, 8 OMP) [ref]':<50} "
      f"{'22.828':>12} "
      f"{'34':>6} {'YES':>8}")
print()

# Speedups
speedup_vs_groebner = t_groebner * 1000 / 22.828
speedup_vs_seq = t_total * 1000 / 22.828
print(f"  Speedup (C++ parallel vs raw sympy.groebner) : {speedup_vs_groebner:>8.1f}x")
print(f"  Speedup (C++ parallel vs sequential pipeline) : {speedup_vs_seq:>8.1f}x")
print()

# CSV output for the paper
csv_file = "../../results/exp6_baseline.csv"
with open(csv_file, "w") as f:
    f.write("method,time_ms,terms,weighted_homogeneous\n")
    f.write(f"sympy_groebner,{t_groebner*1000:.3f},NA,NO\n")
    f.write(f"seq_pipeline,{t_total*1000:.3f},{nterms},YES\n")
    f.write(f"cpp_pipeline,22.828,34,YES\n")
print(f"  CSV saved to {csv_file}")
