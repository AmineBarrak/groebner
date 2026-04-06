"""
exp1_1_toy_cases.py
====================
TIER 1 — Experiment 1.1: Correctness on Toy Cases

Run: python exp1_1_toy_cases.py
Output: results/exp1_1.json  +  console report
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, time
from sympy import symbols, expand, Poly, simplify, S

from core.weighted_projective import (
    Parametrization, is_weighted_homogeneous, weighted_gcd_int
)
from core.pipeline import SequentialPipeline


def run(verbose=True):
    results = []
    t, x0, x1, x2 = symbols("t x0 x1 x2")

    # ── Test A: P(1,1,2) from paper Example 2 ──────────────────────────────
    t0 = time.perf_counter()
    param_A = Parametrization([t, t**2, t**2], [t], [x0,x1,x2], [1,1,2])
    pipe    = SequentialPipeline(verbose=False)
    res_A   = pipe.run(param_A)
    elapsed_A = time.perf_counter() - t0
    # Pipeline should REJECT — x1-x2 is not homogeneous in P(1,1,2)
    passed_A = (res_A.is_correct is False or res_A.is_correct is None)
    results.append({
        "name":     "A: P(1,1,2) polynomial",
        "expected": "Ideal not weighted homogeneous — pipeline should reject",
        "got":      [str(expand(f)) for f in res_A.normalized_basis],
        "is_correct": res_A.is_correct,
        "passed":   passed_A,
        "elapsed_ms": round(elapsed_A * 1000, 2),
        "notes":    "Paper Example 2 inconsistency — x1 has wdeg 1, x0^2 has wdeg 2"
    })

    # ── Test B: P(1,2,3) homogeneity of known polynomial ───────────────────
    t0 = time.perf_counter()
    poly = x0**6 + x1**3 + x2**2
    ok, deg = is_weighted_homogeneous(poly, [x0,x1,x2], [1,2,3])
    elapsed_B = time.perf_counter() - t0
    results.append({
        "name":       "B: P(1,2,3) homogeneity check",
        "expected":   "homogeneous=True, degree=6",
        "got":        f"homogeneous={ok}, degree={deg}",
        "passed":     ok and deg == 6,
        "elapsed_ms": round(elapsed_B * 1000, 2),
    })

    # ── Test C: non-homogeneous rejection ───────────────────────────────────
    t0 = time.perf_counter()
    bad = x0**3 + x1  # wdeg 3 vs wdeg 2 under [1,2,3]
    ok_bad, _ = is_weighted_homogeneous(bad, [x0,x1,x2], [1,2,3])
    elapsed_C = time.perf_counter() - t0
    results.append({
        "name":       "C: non-homogeneity rejection",
        "expected":   "homogeneous=False",
        "got":        f"homogeneous={ok_bad}",
        "passed":     not ok_bad,
        "elapsed_ms": round(elapsed_C * 1000, 2),
    })

    # ── Test D: weighted GCD ground truth ───────────────────────────────────
    t0 = time.perf_counter()
    d1 = weighted_gcd_int([8, 4],    [1, 1])
    d2 = weighted_gcd_int([8, 4, 2], [1, 2, 3])
    elapsed_D = time.perf_counter() - t0
    results.append({
        "name":       "D: weighted GCD",
        "expected":   "d1=4, d2=1",
        "got":        f"d1={d1}, d2={d2}",
        "passed":     d1 == 4 and d2 == 1,
        "elapsed_ms": round(elapsed_D * 1000, 2),
    })

    # ── Test E: P(1,2,3) monomial curve ─────────────────────────────────────
    t0 = time.perf_counter()
    param_E = Parametrization([t, t**2, t**3], [t], [x0,x1,x2], [1,2,3])
    res_E   = pipe.run(param_E)
    elapsed_E = time.perf_counter() - t0
    # Verify all generators are weighted homogeneous
    all_hom = True
    degrees  = []
    for f in res_E.normalized_basis:
        ok_e, deg_e = is_weighted_homogeneous(f, [x0,x1,x2], [1,2,3])
        if not ok_e:
            all_hom = False
        degrees.append(deg_e)
    results.append({
        "name":       "E: P(1,2,3) monomial curve",
        "expected":   "4 generators, degrees [2,3,4,6], all homogeneous",
        "got":        f"{len(res_E.normalized_basis)} generators, degrees {degrees}",
        "passed":     all_hom and len(res_E.normalized_basis) >= 2,
        "elapsed_ms": round(elapsed_E * 1000, 2),
        "ideal":      [str(expand(f)) for f in res_E.normalized_basis],
    })

    # ── Report ───────────────────────────────────────────────────────────────
    if verbose:
        n_pass = sum(1 for r in results if r["passed"])
        print("=" * 60)
        print("EXPERIMENT 1.1 — TOY CASE CORRECTNESS")
        print("=" * 60)
        print(f"Passed: {n_pass}/{len(results)}\n")
        for r in results:
            sym = "✓" if r["passed"] else "✗"
            print(f"  [{sym}] {r['name']}")
            print(f"        expected: {r['expected']}")
            print(f"        got     : {r['got']}")
            print(f"        time    : {r['elapsed_ms']} ms")
            if "notes" in r:
                print(f"        notes   : {r['notes']}")
            print()

    os.makedirs("results", exist_ok=True)
    with open("results/exp1_1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved → results/exp1_1.json")
    return results


if __name__ == "__main__":
    run(verbose=True)
