#!/usr/bin/env python3
"""
SymPy Baseline for L3 in P(2,4,6,10)
=====================================
Compares three approaches:
  1. SymPy groebner() on the full ideal (may timeout)
  2. Sequential Python ansatz (linear-algebra over Z)
  3. Reference: C++ pipeline timing from CSV

L3 Parametrisation (parameters u,v):
  x0 = 2v(4u^2 - 12uv + 3v^2 + 252u - 54v - 405)
  x1 = 4v(u^4*v - 24u^4 - 66u^3*v + 9u^2*v^2 + 1188u^3 + 297u^2*v
          + 138uv^2 - 36v^3 - 8424uv + 945v^2 + 14580v)
  x2 = 4v(2u^6*v^2 - 8u^5*v^3 + ... - 2821230v^2)
  x3 = -16v^2(v-27)(4u^3 - u^2*v - 18uv + 4v^2 + 27v)^3

Weights: q = (2, 4, 6, 10)
Expected: degree-80 weighted homogeneous polynomial, 318 terms
"""
import time
import sys
import os
import csv
from collections import defaultdict
from functools import reduce
from math import gcd

TIMEOUT = 7200  # 2 hours for SymPy groebner (L3 is much harder)

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

# --- Build L3 parametrisation ---
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

print("="*70)
print("  L3 Baseline Comparison in P(2,4,6,10)")
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
    x0, x1, x2, x3, u_sym, v_sym = symbols('x0 x1 x2 x3 u v')

    # Build parametrisation symbolically
    inner_x0_sym = 4*u_sym**2 - 12*u_sym*v_sym + 3*v_sym**2 + 252*u_sym - 54*v_sym - 405
    px0_sym = 2*v_sym * inner_x0_sym

    px1_sym = 4*v_sym * (u_sym**4*v_sym - 24*u_sym**4 - 66*u_sym**3*v_sym
              + 9*u_sym**2*v_sym**2 + 1188*u_sym**3 + 297*u_sym**2*v_sym
              + 138*u_sym*v_sym**2 - 36*v_sym**3 - 8424*u_sym*v_sym
              + 945*v_sym**2 + 14580*v_sym)

    # x2 inner
    x2_inner = (2*u_sym**6*v_sym**2 - 8*u_sym**5*v_sym**3 + 2*u_sym**4*v_sym**4
               - 40*u_sym**6*v_sym + 106*u_sym**5*v_sym**2 + 495*u_sym**4*v_sym**3
               - 204*u_sym**3*v_sym**4 + 18*u_sym**2*v_sym**5 - 144*u_sym**6
               + 1476*u_sym**5*v_sym - 18756*u_sym**4*v_sym**2 + 4280*u_sym**3*v_sym**3
               - 1038*u_sym**2*v_sym**4 + 564*u_sym*v_sym**5 - 72*v_sym**6
               + 160704*u_sym**4*v_sym + 4464*u_sym**3*v_sym**2 + 75024*u_sym**2*v_sym**3
               - 33480*u_sym*v_sym**4 + 3186*v_sym**5 - 104004*u_sym**3*v_sym
               - 1353996*u_sym**2*v_sym**2 + 315252*u_sym*v_sym**3 - 4032*v_sym**4
               + 3669786*u_sym*v_sym**2 - 622323*v_sym**3 - 2821230*v_sym**2)
    px2_sym = 4*v_sym * x2_inner

    cubic_sym = 4*u_sym**3 - u_sym**2*v_sym - 18*u_sym*v_sym + 4*v_sym**2 + 27*v_sym
    px3_sym = -16*v_sym**2 * (v_sym - 27) * cubic_sym**3

    # Generators: x_i - param_i(u,v)
    gens_sym = [x0 - px0_sym, x1 - px1_sym, x2 - px2_sym, x3 - px3_sym]

    print(f"  Computing groebner basis with lex order (u,v > x0,x1,x2,x3)...")
    print(f"  WARNING: L3 has high parametric degree, this may take very long or run OOM")

    t0 = time.time()
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("SymPy groebner timed out")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)

        gb = groebner(gens_sym, u_sym, v_sym, x0, x1, x2, x3, order='lex')

        signal.alarm(0)
        t_sympy = (time.time() - t0) * 1000

        # Find polynomials only in x0..x3
        elim_polys = [p for p in gb.polys if not p.has(u_sym) and not p.has(v_sym)]
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
# Method 2: Sequential Python ansatz
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
    if (idx+1) % 50 == 0:
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
    if cur_row % 50 == 0:
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
    for uv, vv in test_points:
        x0v = peval(px0, uv, vv)
        x1v = peval(px1, uv, vv)
        x2v = peval(px2, uv, vv)
        x3v = peval(px3, uv, vv)
        val = sum(vec[j] * x0v**a * x1v**b * x2v**c * x3v**dd
                  for j, (a,b,c,dd) in enumerate(monoms) if vec[j] != 0)
        if val != 0:
            all_ok = False
    print(f"  Verification: {'PASS' if all_ok else 'FAIL'}")

print()

# =========================================
# Method 3: C++ Pipeline (from CSV)
# =========================================
print("--- Method 3: C++ Pipeline (reference) ---")
cpp_csv = "../../results/exp6_l3_pipeline.csv"
if os.path.exists(cpp_csv):
    with open(cpp_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['stage'] == 'total':
                t_cpp = float(row['time_ms'])
                print(f"  C++ pipeline total: {t_cpp:.1f} ms")
                results['cpp_pipeline'] = t_cpp
else:
    print(f"  No C++ results found at {cpp_csv}")
    print(f"  Run the C++ pipeline first: mpirun ./build/l3_pipeline")
    results['cpp_pipeline'] = -1

print()

# =========================================
# Summary
# =========================================
print("="*70)
print("  SUMMARY: L3 in P(2,4,6,10), weighted degree 80")
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
with open("results/exp6_l3_baseline.csv", "w") as f:
    f.write("method,time_ms,nterms,degree\n")
    for method, t in results.items():
        f.write(f"{method},{t},{nterms if 'nterms' in dir() else -1},{D}\n")
print(f"\nSaved -> results/exp6_l3_baseline.csv")
