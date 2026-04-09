#!/usr/bin/env python3
"""
SymPy Baseline for L2 in P(2,4,6,10)
=====================================
Compares two approaches:
  1. SymPy groebner() on the full ideal (should be fast for L2)
  2. Sequential Python ansatz (linear-algebra over Z)

L2 Parametrisation (parameters t1,t2):
  x0 = -120 - 8*t1
  x1 = t1^2 - 126*t1 + 12*t2 + 405
  x2 = -3*t1^3 + 53*t1^2 - 20*t1*t2 + 2583*t1 - 12*t2 - 14985
  x3 = -2*(-t1^2 - 18*t1 + 4*t2 + 27)^2

Weights: q = (2, 4, 6, 10), target degree D = 30
Expected: 47 monomials total, 34 terms in the resulting polynomial (nullity=1)
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

TIMEOUT = 7200  # 2 hours for SymPy groebner (though L2 should be much faster)

# --- Polynomial arithmetic in Z[t1,t2] ---
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

def peval(p, t1, t2):
    return sum(c * (t1**e1) * (t2**e2) for (e1,e2), c in p.items())

T1 = {(1,0): 1}; T2 = {(0,1): 1}
def const(c): return {} if c == 0 else {(0,0): c}

# --- Build L2 parametrisation ---
px0 = padd(const(-120), pscale(T1, -8))

px1 = padd(padd(padd(
    ppow(T1, 2), pscale(T1, -126)),
    pscale(T2, 12)), const(405))

inner_x2 = padd(padd(padd(
    pscale(ppow(T1, 3), -3), pscale(ppow(T1, 2), 53)),
    pscale(pmul(T1, T2), -20)), padd(padd(
    pscale(T1, 2583), pscale(T2, -12)), const(-14985)))
px2 = inner_x2

inner_x3_base = padd(padd(padd(
    pscale(ppow(T1, 2), -1), pscale(T1, -18)),
    pscale(T2, 4)), const(27))
px3 = pscale(ppow(inner_x3_base, 2), -2)

W = (2, 4, 6, 10)
D = 30

print("="*70)
print("  L2 Baseline Comparison in P(2,4,6,10)")
print("="*70)
print(f"  Parametrisation degrees: x0={max(e1+e2 for e1,e2 in px0.keys())}, "
      f"x1={max(e1+e2 for e1,e2 in px1.keys())}, "
      f"x2={max(e1+e2 for e1,e2 in px2.keys())}, "
      f"x3={max(e1+e2 for e1,e2 in px3.keys())}")
print(f"  Target weighted degree: {D}")
print()

results = {}

# =========================================
# Method 1: SymPy groebner
# =========================================
print("--- Method 1: SymPy groebner() ---")
try:
    import sympy
    print(f"  SymPy version: {sympy.__version__}")

    from sympy import symbols, groebner, Rational
    x0, x1, x2, x3, t1_sym, t2_sym = symbols('x0 x1 x2 x3 t1 t2')

    # Build parametrisation symbolically
    px0_sym = -120 - 8*t1_sym
    px1_sym = t1_sym**2 - 126*t1_sym + 12*t2_sym + 405
    px2_sym = -3*t1_sym**3 + 53*t1_sym**2 - 20*t1_sym*t2_sym + 2583*t1_sym - 12*t2_sym - 14985

    inner_x3_sym = -t1_sym**2 - 18*t1_sym + 4*t2_sym + 27
    px3_sym = -2 * inner_x3_sym**2

    # Generators: x_i - param_i(t1,t2)
    gens_sym = [x0 - px0_sym, x1 - px1_sym, x2 - px2_sym, x3 - px3_sym]

    print(f"  Computing groebner basis with lex order (t1,t2 > x0,x1,x2,x3)...")

    t0 = time.time()
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("SymPy groebner timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)

        gb = groebner(gens_sym, t1_sym, t2_sym, x0, x1, x2, x3, order='lex')

        signal.alarm(0)
        t_sympy = (time.time() - t0) * 1000

        # Find polynomials only in x0..x3
        elim_polys = [p for p in gb.polys if not p.has(t1_sym) and not p.has(t2_sym)]
        print(f"  Time: {t_sympy:.1f} ms")
        print(f"  Basis size: {len(gb.polys)}, eliminated: {len(elim_polys)}")
        if elim_polys:
            for p in elim_polys:
                nterms = len(p.as_dict())
                print(f"    -> {nterms} terms")
        results['sympy_groebner'] = t_sympy
    except TimeoutError:
        t_sympy = TIMEOUT * 1000
        print(f"  TIMEOUT after {TIMEOUT}s")
        results['sympy_groebner'] = t_sympy
    except MemoryError:
        print(f"  OUT OF MEMORY")
        results['sympy_groebner'] = -1
    except Exception as e:
        print(f"  ERROR: {e}")
        results['sympy_groebner'] = -1
except ImportError:
    print("  SymPy not available")
    results['sympy_groebner'] = -1

print()

# =========================================
# Method 2: Sequential Python Ansatz
# =========================================
print("--- Method 2: Sequential Python Ansatz ---")

monoms = []
for dd in range(D // W[3] + 1):
    for c in range((D - W[3]*dd) // W[2] + 1):
        for b in range((D - W[3]*dd - W[2]*c) // W[1] + 1):
            rem = D - W[3]*dd - W[2]*c - W[1]*b
            if rem >= 0 and rem % W[0] == 0:
                a = rem // W[0]
                monoms.append((a, b, c, dd))

ncols = len(monoms)
print(f"  {ncols} monomials of weighted degree {D}")

# Precompute powers
def precompute(base, n):
    pows = [{(0,0): 1}]
    for i in range(n):
        pows.append(pmul(pows[-1], base))
    return pows

max_a = max(m[0] for m in monoms)
max_b = max(m[1] for m in monoms)
max_c = max(m[2] for m in monoms)
max_d = max(m[3] for m in monoms)
print(f"  Max powers: x0^{max_a} x1^{max_b} x2^{max_c} x3^{max_d}")

t0 = time.time()

print(f"  Precomputing powers...")
x0p = precompute(px0, max_a)
x1p = precompute(px1, max_b)
x2p = precompute(px2, max_c)
x3p = precompute(px3, max_d)
t_powers = time.time() - t0
print(f"  Powers done in {t_powers:.1f}s")

# Substitute
print(f"  Substituting...")
t_sub0 = time.time()
all_t = set()
cols = []
for idx, (a, b, c, dd) in enumerate(monoms):
    p = pmul(pmul(x0p[a], x1p[b]), pmul(x2p[c], x3p[dd]))
    cols.append(p)
    all_t.update(p.keys())
    if (idx+1) % 10 == 0:
        print(f"    {idx+1}/{ncols} ({len(p)} terms)")

all_t = sorted(all_t)
t_idx = {m:i for i,m in enumerate(all_t)}
nrows = len(all_t)
t_sub = time.time() - t_sub0
print(f"  Substitution done in {t_sub:.1f}s")
print(f"  Matrix: {nrows} x {ncols}")

# Build matrix
M = [[0]*ncols for _ in range(nrows)]
for j, col in enumerate(cols):
    for monom, val in col.items():
        M[t_idx[monom]][j] = val

# Gaussian elimination
print(f"  Gaussian elimination...")
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
        g = 0
        for c2 in range(ncols):
            if M[r][c2] != 0:
                g = gcd(g, abs(M[r][c2]))
        if g > 1:
            for c2 in range(ncols):
                M[r][c2] //= g
    cur_row += 1
    if cur_row % 10 == 0:
        print(f"    pivot {cur_row}/{ncols} ({time.time()-t_gauss0:.1f}s)")

rank = cur_row
nullity = ncols - rank
t_gauss = time.time() - t_gauss0
print(f"  Gauss done in {t_gauss:.1f}s, rank={rank}, nullity={nullity}")

t_total = (time.time() - t0) * 1000
print(f"  Total sequential time: {t_total:.1f} ms")
results['sequential_python'] = t_total

if nullity == 1:
    # Extract kernel
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
    print(f"  Polynomial: {nterms} terms")

    # Verify
    test_points = [(1,1),(2,1),(1,2),(3,1),(0,3)]
    all_ok = True
    for t1v, t2v in test_points:
        x0v = peval(px0, t1v, t2v)
        x1v = peval(px1, t1v, t2v)
        x2v = peval(px2, t1v, t2v)
        x3v = peval(px3, t1v, t2v)
        val = sum(vec[j] * x0v**a * x1v**b * x2v**c * x3v**dd
                  for j, (a,b,c,dd) in enumerate(monoms) if vec[j] != 0)
        if val != 0:
            all_ok = False
    print(f"  Verification: {'PASS' if all_ok else 'FAIL'}")

print()

# =========================================
# Summary
# =========================================
print("="*70)
print("  SUMMARY: L2 in P(2,4,6,10), weighted degree 30")
print("="*70)
for method, t in results.items():
    if t < 0:
        print(f"  {method:25s}: N/A")
    elif t >= TIMEOUT * 1000:
        print(f"  {method:25s}: TIMEOUT (>{TIMEOUT}s)")
    else:
        print(f"  {method:25s}: {t:12.1f} ms")

# Save CSV
os.makedirs("results", exist_ok=True)
with open("results/exp6_l2_baseline.csv", "w") as f:
    f.write("method,time_ms,nterms,degree\n")
    for method, t in results.items():
        f.write(f"{method},{t},{nterms if 'nterms' in dir() else -1},{D}\n")
print(f"\nSaved -> results/exp6_l2_baseline.csv")
