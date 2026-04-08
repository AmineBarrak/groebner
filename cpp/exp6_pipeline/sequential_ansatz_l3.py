#!/usr/bin/env python3
"""
Sequential Python Ansatz for L3 in P(2,4,6,10)
================================================
Runs ONLY the linear-algebra ansatz (no SymPy groebner).
This gives us the sequential baseline timing for comparison
with the C++ parallel pipeline.

L3 Parametrisation (parameters u,v):
  x0 = 2v(4u^2 - 12uv + 3v^2 + 252u - 54v - 405)
  x1 = 4v(u^4*v - 24u^4 - ...)
  x2 = 4v(2u^6*v^2 - 8u^5*v^3 + ...)
  x3 = -16v^2(v-27)(4u^3 - u^2*v - 18uv + 4v^2 + 27v)^3

Weights: q = (2, 4, 6, 10), target degree D = 80
Expected: 521 monomials, 4374 rows, rank 520, nullity 1 => 318 terms
"""
import time
import sys
import os
import csv
from collections import defaultdict
from functools import reduce
from math import gcd

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# --- Polynomial arithmetic in Z[u,v] ---
def pmul(p, q):
    r = defaultdict(int)
    for (a1,a2), c1 in p.items():
        for (b1,b2), c2 in q.items():
            r[(a1+b1, a2+b2)] += c1 * c2
    return {k:v for k,v in r.items() if v}

def padd(p, q):
    r = defaultdict(int)
    for k,v in p.items(): r[k] += v
    for k,v in q.items(): r[k] += v
    return {k:v for k,v in r.items() if v}

def pscale(p, c):
    if c == 0: return {}
    return {k:v*c for k,v in p.items() if v*c}

def ppow(p, n):
    if n == 0: return {(0,0): 1}
    r = {(0,0): 1}
    for _ in range(n): r = pmul(r, p)
    return r

def peval(p, u, v):
    return sum(c * (u**e1) * (v**e2) for (e1,e2), c in p.items())

U = {(1,0): 1}; V = {(0,1): 1}
def const(c): return {} if c == 0 else {(0,0): c}

print("="*70)
print("  Sequential Python Ansatz - L3 in P(2,4,6,10)")
print("="*70)
print(f"  Python: {sys.version}")
print()

# --- Build L3 parametrisation ---
print("  Building L3 parametrisation...")
t_start = time.time()

inner_x0 = padd(padd(padd(padd(padd(
    pscale(ppow(U,2), 4), pscale(pmul(U,V), -12)),
    pscale(ppow(V,2), 3)), pscale(U, 252)), pscale(V, -54)), const(-405))
px0 = pscale(pmul(V, inner_x0), 2)

terms_x1 = [(1,4,1),(-24,4,0),(-66,3,1),(9,2,2),(1188,3,0),(297,2,1),
            (138,1,2),(-36,0,3),(-8424,1,1),(945,0,2),(14580,0,1)]
inner_x1 = {}
for coeff, eu, ev in terms_x1:
    inner_x1 = padd(inner_x1, pscale(pmul(ppow(U,eu), ppow(V,ev)), coeff))
px1 = pscale(pmul(V, inner_x1), 4)

terms_x2 = [
    (2,6,2),(-8,5,3),(2,4,4),(-40,6,1),(106,5,2),(495,4,3),(-204,3,4),
    (18,2,5),(-144,6,0),(1476,5,1),(-18756,4,2),(4280,3,3),(-1038,2,4),
    (564,1,5),(-72,0,6),(160704,4,1),(4464,3,2),(75024,2,3),(-33480,1,4),
    (3186,0,5),(-104004,3,1),(-1353996,2,2),(315252,1,3),(-4032,0,4),
    (3669786,1,2),(-622323,0,3),(-2821230,0,2)]
inner_x2 = {}
for coeff, eu, ev in terms_x2:
    inner_x2 = padd(inner_x2, pscale(pmul(ppow(U,eu), ppow(V,ev)), coeff))
px2 = pscale(pmul(V, inner_x2), 4)

v_minus_27 = padd(V, const(-27))
cubic = padd(padd(padd(padd(
    pscale(ppow(U,3), 4), pscale(pmul(ppow(U,2), V), -1)),
    pscale(pmul(U,V), -18)), pscale(ppow(V,2), 4)), pscale(V, 27))
px3 = pscale(pmul(pmul(ppow(V,2), v_minus_27), ppow(cubic, 3)), -16)

W = (2, 4, 6, 10)
D = 80

print(f"  Parametrisation degrees: x0={max(e1+e2 for e1,e2 in px0.keys())}, "
      f"x1={max(e1+e2 for e1,e2 in px1.keys())}, "
      f"x2={max(e1+e2 for e1,e2 in px2.keys())}, "
      f"x3={max(e1+e2 for e1,e2 in px3.keys())}")
print(f"  Target weighted degree: {D}")
print()

# --- Enumerate monomials ---
monoms = []
for dd in range(D // W[3] + 1):
    for c in range((D - W[3]*dd) // W[2] + 1):
        for b in range((D - W[3]*dd - W[2]*c) // W[1] + 1):
            rem = D - W[3]*dd - W[2]*c - W[1]*b
            if rem >= 0 and rem % W[0] == 0:
                a = rem // W[0]
                monoms.append((a, b, c, dd))

ncols = len(monoms)
max_a = max(m[0] for m in monoms)
max_b = max(m[1] for m in monoms)
max_c = max(m[2] for m in monoms)
max_d = max(m[3] for m in monoms)
print(f"  {ncols} monomials of weighted degree {D}")
print(f"  Max powers: x0^{max_a} x1^{max_b} x2^{max_c} x3^{max_d}")

# --- Precompute powers ---
def precompute(base, n):
    pows = [{(0,0): 1}]
    for i in range(n):
        pows.append(pmul(pows[-1], base))
    return pows

t0 = time.time()
print(f"  Precomputing x0 powers (up to {max_a})...")
x0p = precompute(px0, max_a)
print(f"    x0 done ({time.time()-t0:.1f}s)")
print(f"  Precomputing x1 powers (up to {max_b})...")
x1p = precompute(px1, max_b)
print(f"    x1 done ({time.time()-t0:.1f}s)")
print(f"  Precomputing x2 powers (up to {max_c})...")
x2p = precompute(px2, max_c)
print(f"    x2 done ({time.time()-t0:.1f}s)")
print(f"  Precomputing x3 powers (up to {max_d})...")
x3p = precompute(px3, max_d)
t_powers = time.time() - t0
print(f"    x3 done ({t_powers:.1f}s)")
print(f"  All powers precomputed in {t_powers:.1f}s")
print()

# --- Substitute monomials ---
print(f"  Substituting {ncols} monomials...")
t_sub0 = time.time()
all_t = set()
cols = []
for idx, (a, b, c, dd) in enumerate(monoms):
    p = pmul(pmul(x0p[a], x1p[b]), pmul(x2p[c], x3p[dd]))
    cols.append(p)
    all_t.update(p.keys())
    if (idx+1) % 50 == 0:
        elapsed = time.time() - t_sub0
        print(f"    {idx+1}/{ncols} monomials ({len(p)} terms each, {elapsed:.1f}s)")

all_t = sorted(all_t)
t_idx = {m:i for i,m in enumerate(all_t)}
nrows = len(all_t)
t_sub = time.time() - t_sub0
print(f"  Substitution done in {t_sub:.1f}s")
print(f"  Matrix: {nrows} x {ncols}")
print()

# --- Build matrix ---
print(f"  Building matrix...")
t_build0 = time.time()
M = [[0]*ncols for _ in range(nrows)]
for j, col in enumerate(cols):
    for monom, val in col.items():
        M[t_idx[monom]][j] = val
t_build = time.time() - t_build0
print(f"  Matrix built in {t_build:.1f}s")

# --- Fraction-free Gaussian elimination with GCD ---
print(f"  Gaussian elimination (fraction-free with GCD content removal)...")
t_gauss0 = time.time()
pivot_col = [-1]*nrows
cur_row = 0
for col in range(ncols):
    if cur_row >= nrows: break
    pr = -1
    for r in range(cur_row, nrows):
        if M[r][col] != 0:
            pr = r; break
    if pr < 0: continue
    if pr != cur_row:
        M[pr], M[cur_row] = M[cur_row], M[pr]
    pivot_col[cur_row] = col
    piv = M[cur_row][col]
    for r in range(nrows):
        if r == cur_row or M[r][col] == 0: continue
        factor = M[r][col]
        for c2 in range(ncols):
            M[r][c2] = M[r][c2] * piv - factor * M[cur_row][c2]
        # GCD content removal
        g = 0
        for c2 in range(ncols):
            if M[r][c2] != 0:
                g = gcd(g, abs(M[r][c2]))
        if g > 1:
            for c2 in range(ncols):
                M[r][c2] //= g
    cur_row += 1
    if cur_row % 10 == 0:
        elapsed = time.time() - t_gauss0
        print(f"    pivot {cur_row}/{ncols} ({elapsed:.1f}s)")

rank = cur_row
nullity = ncols - rank
t_gauss = time.time() - t_gauss0
print(f"  Gauss done in {t_gauss:.1f}s")
print(f"  Rank: {rank}, Nullity: {nullity}")

t_total_s = time.time() - t0
t_total_ms = t_total_s * 1000
print()

# --- Extract kernel vector ---
nterms = -1
verified = False
if nullity == 1:
    print(f"  Extracting kernel vector...")
    t_kern0 = time.time()
    pivot_cols_set = set()
    for r in range(rank):
        pivot_cols_set.add(pivot_col[r])
    free_cols = [c for c in range(ncols) if c not in pivot_cols_set]
    fc = free_cols[0]

    scale = 1
    for r in range(rank):
        scale *= M[r][pivot_col[r]]

    vec = [0]*ncols
    vec[fc] = scale
    for r in range(rank-1, -1, -1):
        pc = pivot_col[r]
        rhs = -M[r][fc] * scale
        for c in range(ncols):
            if c != pc and c != fc and vec[c] != 0:
                rhs -= M[r][c] * vec[c]
        diag = M[r][pc]
        vec[pc] = rhs // diag

    nonzero = [abs(x) for x in vec if x != 0]
    if nonzero:
        g = reduce(gcd, nonzero)
        vec = [x // g for x in vec]
        if vec[next(i for i,x in enumerate(vec) if x != 0)] < 0:
            vec = [-x for x in vec]

    nterms = sum(1 for x in vec if x != 0)
    t_kern = time.time() - t_kern0
    print(f"  Kernel extraction: {t_kern:.1f}s")
    print(f"  Polynomial: {nterms} terms")

    # Verify at test points
    print(f"  Verifying at test points...")
    test_points = [(1,1),(2,1),(1,2),(3,1),(0,3),(5,2),(1,7),(3,4)]
    all_ok = True
    for uv, vv in test_points:
        x0v = peval(px0, uv, vv)
        x1v = peval(px1, uv, vv)
        x2v = peval(px2, uv, vv)
        x3v = peval(px3, uv, vv)
        val = sum(vec[j] * x0v**a * x1v**b * x2v**c * x3v**dd
                  for j, (a,b,c,dd) in enumerate(monoms) if vec[j] != 0)
        if val != 0:
            all_ok = False
            print(f"    FAIL at ({uv},{vv}): val={val}")
    verified = all_ok
    print(f"  Verification: {'PASS' if all_ok else 'FAIL'}")
else:
    print(f"  WARNING: nullity={nullity}, expected 1")

print()

# --- Timing Summary ---
print("="*70)
print("  TIMING SUMMARY")
print("="*70)
print(f"  Power precomputation: {t_powers:12.1f}s")
print(f"  Monomial substitution: {t_sub:11.1f}s")
print(f"  Matrix construction:   {t_build:11.1f}s")
print(f"  Gaussian elimination:  {t_gauss:11.1f}s")
print(f"  ----------------------------------------")
print(f"  TOTAL:                 {t_total_s:11.1f}s  ({t_total_ms:.1f} ms)")
print()
print(f"  Result: {nterms} terms, degree {D}, verified={verified}")
print()

# --- Compare with C++ pipeline ---
print("--- Comparison with C++ Pipeline ---")
cpp_csv = "results/exp6_l3_pipeline.csv"
if not os.path.exists(cpp_csv):
    cpp_csv = "../../results/exp6_l3_pipeline.csv"
if os.path.exists(cpp_csv):
    with open(cpp_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('stage') == 'total':
                t_cpp = float(row['time_ms'])
                speedup = t_total_ms / t_cpp if t_cpp > 0 else 0
                print(f"  C++ pipeline:       {t_cpp:12.1f} ms ({t_cpp/1000:.1f}s)")
                print(f"  Sequential Python:  {t_total_ms:12.1f} ms ({t_total_s:.1f}s)")
                print(f"  Speedup (C++/Py):   {speedup:12.1f}x")
else:
    print(f"  C++ results not found")
    print(f"  Sequential Python:  {t_total_ms:.1f} ms ({t_total_s:.1f}s)")

# --- Save results ---
os.makedirs("results", exist_ok=True)
with open("results/exp6_l3_sequential.csv", "w") as f:
    f.write("stage,time_ms\n")
    f.write(f"powers,{t_powers*1000:.1f}\n")
    f.write(f"substitution,{t_sub*1000:.1f}\n")
    f.write(f"matrix_build,{t_build*1000:.1f}\n")
    f.write(f"gaussian_elim,{t_gauss*1000:.1f}\n")
    f.write(f"total,{t_total_ms:.1f}\n")
print(f"\nSaved -> results/exp6_l3_sequential.csv")

# Also save summary
with open("results/exp6_l3_baseline_summary.csv", "w") as f:
    f.write("method,time_ms,nterms,degree,status\n")
    f.write(f"sympy_groebner,{24*3600*1000},-1,{D},TIMEOUT_24h\n")
    f.write(f"modular_crt,-1,-1,{D},FAILED_500_primes\n")
    f.write(f"sequential_python,{t_total_ms:.1f},{nterms},{D},{'OK' if verified else 'FAIL'}\n")
print(f"Saved -> results/exp6_l3_baseline_summary.csv")
