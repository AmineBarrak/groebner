"""
exp1_2_homogeneity_check.py
============================
TIER 1 — Experiment 1.2: Weighted Homogeneity Auto-Checker

Run: python exp1_2_homogeneity_check.py
Output: results/exp1_2.json  +  console report
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
from sympy import symbols, expand

from core.weighted_projective import Parametrization, is_weighted_homogeneous
from core.pipeline import SequentialPipeline


def check_ideal(basis, coord_vars, weights):
    report = {
        "total": len(basis), "homogeneous": 0,
        "non_homogeneous": 0, "degrees": [], "failures": []
    }
    for i, f in enumerate(basis):
        ok, deg = is_weighted_homogeneous(f, coord_vars, weights)
        if ok:
            report["homogeneous"] += 1
            report["degrees"].append(deg)
        else:
            report["non_homogeneous"] += 1
            report["failures"].append({"index": i, "poly": str(expand(f))})
    report["all_pass"] = report["non_homogeneous"] == 0
    return report


def run(verbose=True):
    t, x0, x1, x2 = symbols("t x0 x1 x2")
    pipe = SequentialPipeline(verbose=False)

    test_cases = [
        {
            "name":  "P(1,1,2) — x0=t, x1=t², x2=t²",
            "param": Parametrization([t, t**2, t**2], [t], [x0,x1,x2], [1,1,2]),
        },
        {
            "name":  "P(1,2,3) — x0=t, x1=t², x2=t³",
            "param": Parametrization([t, t**2, t**3], [t], [x0,x1,x2], [1,2,3]),
        },
        {
            "name":  "P(1,1,1) — twisted",
            "param": Parametrization([t, t, t**2 - t], [t], [x0,x1,x2], [1,1,1]),
        },
        {
            "name":  "P(2,3,5) — x0=t³, x1=t², x2=t",
            "param": Parametrization([t**3, t**2, t], [t], [x0,x1,x2], [2,3,5]),
        },
    ]

    results = []
    for tc in test_cases:
        try:
            res    = pipe.run(tc["param"])
            report = check_ideal(res.normalized_basis,
                                  list(tc["param"].coord_vars),
                                  tc["param"].weights)
            report["name"]       = tc["name"]
            report["passed"]     = report["all_pass"]
            report["basis_size"] = len(res.normalized_basis)
        except Exception as e:
            report = {"name": tc["name"], "passed": False,
                      "error": str(e), "all_pass": False,
                      "total": 0, "homogeneous": 0,
                      "non_homogeneous": 0, "degrees": [], "failures": []}
        results.append(report)

    if verbose:
        print("=" * 60)
        print("EXPERIMENT 1.2 — HOMOGENEITY AUTO-CHECKER")
        print("=" * 60)
        n_pass = sum(1 for r in results if r.get("passed"))
        print(f"Passed: {n_pass}/{len(results)}\n")
        for r in results:
            sym = "✓" if r.get("passed") else "✗"
            print(f"  [{sym}] {r['name']}")
            if r.get("error"):
                print(f"       ERROR: {r['error']}")
            else:
                print(f"       Generators: {r['total']} | "
                      f"Homogeneous: {r['homogeneous']} | "
                      f"Degrees: {r['degrees']}")
                for fail in r.get("failures", []):
                    print(f"       FAIL gen[{fail['index']}]: {fail['poly']}")
            print()

    os.makedirs("results", exist_ok=True)
    with open("results/exp1_2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved → results/exp1_2.json")
    return results


if __name__ == "__main__":
    run(verbose=True)
