#!/usr/bin/env python3
"""
Compute the L3 defining polynomial in P(2,4,6,10) at degree 80.
Uses modular arithmetic + CRT with many primes.
Optimized for HPC execution.
"""
from collections import defaultdict
from functools import reduce
from math import gcd
import time, random, sys, json

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

U = {(1,0): 1}; V = {(0,1): 1}
def const(c): return {} if c == 0 else {(0,0): c}

# --- L3 Parametrisation ---
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

# Convert to fast evaluation form: list of (coeff, eu, ev)
def poly_to_list(p):
    return [(c, e1, e2) for (e1, e2), c in p.items()]

px0_list = poly_to_list(px0)
px1_list = poly_to_list(px1)
px2_list = poly_to_list(px2)
px3_list = poly_to_list(px3)

def fast_eval_mod(plist, u, v, P):
    """Evaluate polynomial at (u,v) mod P."""
    s = 0
    for c, e1, e2 in plist:
        s = (s + c * pow(u, e1, P) * pow(v, e2, P)) % P
    return s

W = (2, 4, 6, 10)
D = 80

# Enumerate monomials of weighted degree 80
monoms = []
for d in range(D // W[3] + 1):
    for c in range((D - W[3]*d) // W[2] + 1):
        for b in range((D - W[3]*d - W[2]*c) // W[1] + 1):
            rem = D - W[3]*d - W[2]*c - W[1]*b
            if rem >= 0 and rem % W[0] == 0:
                a = rem // W[0]
                monoms.append((a, b, c, d))

ncols = len(monoms)
print(f"D={D}: {ncols} monomials")

# Generate many large primes
# Using primes slightly below 2^62 for fast arithmetic
def is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i+2) == 0: return False
        i += 6
    return True

# Pre-selected large primes (30-bit range for fast modular arithmetic)
# We need enough to cover potentially very large coefficients
PRIMES = []
p = 1000000007
while len(PRIMES) < 500:
    if is_prime(p):
        PRIMES.append(p)
    p += 2
    if p > 1100000000:
        p = 900000000
        while not is_prime(p):
            p += 1

print(f"Generated {len(PRIMES)} primes")
print(f"Bits per prime: ~30, total bits with 500 primes: ~{30*500}")

def kernel_mod_prime(P, seed):
    """Find the 1-d kernel vector mod prime P."""
    rng = random.Random(seed)
    n_pts = ncols + 50

    # Build evaluation matrix mod P
    rows = []
    for _ in range(n_pts):
        u = rng.randint(1, P-1)
        v = rng.randint(1, P-1)
        x0v = fast_eval_mod(px0_list, u, v, P)
        x1v = fast_eval_mod(px1_list, u, v, P)
        x2v = fast_eval_mod(px2_list, u, v, P)
        x3v = fast_eval_mod(px3_list, u, v, P)
        row = []
        for (a, b, c, dd) in monoms:
            val = pow(x0v, a, P) * pow(x1v, b, P) % P
            val = val * pow(x2v, c, P) % P * pow(x3v, dd, P) % P
            row.append(val)
        rows.append(row)

    # Gaussian elimination mod P (RREF)
    M = [list(r) for r in rows]
    nrows = len(M)
    pivot_col_for_row = [-1] * nrows
    pivot_row = 0
    pivot_cols = []

    for col in range(ncols):
        pr = -1
        for r in range(pivot_row, nrows):
            if M[r][col] % P != 0:
                pr = r; break
        if pr < 0: continue
        M[pr], M[pivot_row] = M[pivot_row], M[pr]
        inv_piv = pow(M[pivot_row][col], P-2, P)
        for c2 in range(ncols):
            M[pivot_row][c2] = M[pivot_row][c2] * inv_piv % P
        for r in range(nrows):
            if r == pivot_row or M[r][col] % P == 0: continue
            factor = M[r][col]
            for c2 in range(ncols):
                M[r][c2] = (M[r][c2] - factor * M[pivot_row][c2]) % P
        pivot_col_for_row[pivot_row] = col
        pivot_cols.append(col)
        pivot_row += 1

    rank = pivot_row
    nullity = ncols - rank
    if nullity != 1:
        return None, rank, nullity

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(ncols) if c not in pivot_set]
    fc = free_cols[0]

    vec = [0] * ncols
    vec[fc] = 1
    for r in range(rank - 1, -1, -1):
        pc = pivot_col_for_row[r]
        val = 0
        for c2 in range(ncols):
            if c2 != pc:
                val = (val + M[r][c2] * vec[c2]) % P
        vec[pc] = (-val) % P

    return vec, rank, nullity

def crt2(r1, m1, r2, m2):
    """CRT for two congruences."""
    g = gcd(m1, m2)
    if (r2 - r1) % g != 0:
        raise ValueError("No CRT solution")
    lcm = m1 // g * m2
    # Extended GCD to find inverse
    def extended_gcd(a, b):
        if a == 0: return b, 0, 1
        g, x, y = extended_gcd(b % a, a)
        return g, y - (b // a) * x, x
    _, inv, _ = extended_gcd(m1 // g, m2 // g)
    inv = inv % (m2 // g)
    x = (r1 + m1 * ((r2 - r1) // g * inv % (m2 // g))) % lcm
    return x, lcm

def balanced_mod(x, m):
    x = x % m
    if x > m // 2:
        x -= m
    return x

# --- Main computation ---
print(f"\nComputing L3 kernel vector using modular CRT...")
print(f"{'='*70}")

all_vecs = []
all_primes_used = []
combined_mod = 1
combined_vec = None
prev_final = None
converged = False

for i, P in enumerate(PRIMES):
    t0 = time.time()
    vec, rank, nullity = kernel_mod_prime(P, seed=54321 + i * 7919)
    dt = time.time() - t0

    if vec is None:
        print(f"  Prime {i+1} ({P}): rank={rank}, null={nullity} - BAD PRIME, skip")
        continue

    # Normalize: ensure free_col (last column = 520) has value 1
    fc = 520  # x3^8 column
    if vec[fc] != 1:
        inv = pow(vec[fc], P-2, P)
        vec = [v * inv % P for v in vec]

    nz = sum(1 for v in vec if v != 0)

    # Incremental CRT
    if combined_vec is None:
        combined_vec = list(vec)
        combined_mod = P
    else:
        new_vec = [0] * ncols
        for j in range(ncols):
            r, m = crt2(combined_vec[j], combined_mod, vec[j], P)
            new_vec[j] = r
        combined_vec = new_vec
        combined_mod = combined_mod * P

    all_primes_used.append(P)

    # Check convergence every 10 primes
    if len(all_primes_used) % 10 == 0:
        test_vec = [balanced_mod(combined_vec[j], combined_mod) for j in range(ncols)]
        nonzero = [abs(x) for x in test_vec if x != 0]
        if nonzero:
            g = reduce(gcd, nonzero)
            test_vec_norm = [x // g for x in test_vec]
        else:
            test_vec_norm = test_vec

        max_c = max(abs(x) for x in test_vec_norm) if any(test_vec_norm) else 0
        nz_count = sum(1 for x in test_vec_norm if x != 0)

        import math
        log2_max = math.log2(max_c) if max_c > 0 else 0
        log2_mod = math.log2(combined_mod)

        print(f"  After {len(all_primes_used)} primes: {nz_count} nonzero, "
              f"max|c|/gcd ~ 2^{log2_max:.0f}, mod ~ 2^{log2_mod:.0f}  "
              f"({dt:.2f}s/prime)")

        # Check if converged: coefficients stable between iterations
        if prev_final is not None and test_vec_norm == prev_final:
            print(f"  --> CONVERGED! Coefficients stable.")
            converged = True
            break
        prev_final = test_vec_norm

        if log2_max < log2_mod * 0.3:
            print(f"  --> Coefficients look converged (well within mod range)")
            converged = True
            break
    elif len(all_primes_used) <= 5 or len(all_primes_used) % 50 == 0:
        print(f"  Prime {len(all_primes_used)}/{len(PRIMES)} ({P}): ok ({dt:.2f}s)")

if not converged:
    print(f"\nWARNING: May not have converged after {len(all_primes_used)} primes!")
    print(f"Try running with more primes or using symbolic computation.")

# Final reconstruction
print(f"\nFinal CRT reconstruction with {len(all_primes_used)} primes...")
final_vec = [balanced_mod(combined_vec[j], combined_mod) for j in range(ncols)]

# Normalize
nonzero_coeffs = [abs(x) for x in final_vec if x != 0]
if nonzero_coeffs:
    g = reduce(gcd, nonzero_coeffs)
    final_vec = [x // g for x in final_vec]
    for j in range(ncols):
        if final_vec[j] != 0:
            if final_vec[j] < 0:
                final_vec = [-x for x in final_vec]
            break

nterms = sum(1 for x in final_vec if x != 0)
max_coeff = max(abs(x) for x in final_vec if x != 0) if nterms > 0 else 0

print(f"\nL3 polynomial: {nterms} terms, max |coeff| = {max_coeff}")

# Verify at test points
print(f"\nVerification at test points:")
def peval_exact(p, u, v):
    return sum(c * (u**e1) * (v**e2) for (e1,e2), c in p.items())

test_points = [(1,1),(2,1),(1,2),(3,1),(1,3),(2,3),(5,2),(0,3),(3,-1),(7,4)]
all_ok = True
for uv, vv in test_points:
    x0v = peval_exact(px0, uv, vv)
    x1v = peval_exact(px1, uv, vv)
    x2v = peval_exact(px2, uv, vv)
    x3v = peval_exact(px3, uv, vv)
    val = sum(final_vec[j] * x0v**a * x1v**b * x2v**c * x3v**dd
              for j, (a,b,c,dd) in enumerate(monoms) if final_vec[j] != 0)
    status = "PASS" if val == 0 else "FAIL"
    if val != 0:
        all_ok = False
    print(f"  (u,v)=({uv},{vv}): {status}")

print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")

# Save result
result = {
    'D': D,
    'W': W,
    'nterms': nterms,
    'max_coeff': str(max_coeff),
    'primes_used': len(all_primes_used),
    'converged': converged,
    'verified': all_ok,
    'terms': []
}
for j, (a,b,c,dd) in enumerate(monoms):
    if final_vec[j] != 0:
        result['terms'].append({
            'a': a, 'b': b, 'c': c, 'd': dd,
            'coeff': str(final_vec[j])
        })

with open('results/l3_polynomial.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to results/l3_polynomial.json")

# Print polynomial
print(f"\n{'='*70}")
print(f"F(x0,x1,x2,x3) = ")
for j, (a,b,c,dd) in enumerate(monoms):
    if final_vec[j] == 0:
        continue
    coeff = final_vec[j]
    parts = []
    if a > 0: parts.append(f"x0^{a}" if a > 1 else "x0")
    if b > 0: parts.append(f"x1^{b}" if b > 1 else "x1")
    if c > 0: parts.append(f"x2^{c}" if c > 1 else "x2")
    if dd > 0: parts.append(f"x3^{dd}" if dd > 1 else "x3")
    mono = "*".join(parts) if parts else "1"
    if coeff > 0:
        print(f"  + {coeff}*{mono}")
    else:
        print(f"  - {-coeff}*{mono}")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Weighted projective space: P(2,4,6,10)")
print(f"  Locus: L3")
print(f"  Weighted degree: {D}")
print(f"  Number of terms: {nterms}")
print(f"  Number of primes used: {len(all_primes_used)}")
print(f"  Converged: {converged}")
print(f"  Verified: {all_ok}")
vars_used = set()
for j, (a,b,c,dd) in enumerate(monoms):
    if final_vec[j] != 0:
        if a > 0: vars_used.add('x0')
        if b > 0: vars_used.add('x1')
        if c > 0: vars_used.add('x2')
        if dd > 0: vars_used.add('x3')
print(f"  Variables: {sorted(vars_used)}")
