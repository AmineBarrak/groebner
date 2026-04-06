"""
Experiment 5 — End-to-End Integrated Pipeline Benchmark
========================================================
Measures the *complete* three-stage pipeline (homogenization → F4 →
normalisation) under sequential and parallel execution, reporting:
  - per-stage wall-clock times
  - total wall-clock time
  - speedup S(p) = T(1) / T(p)  and  efficiency E(p) = S(p) / p
  - correctness: all generators verified weighted homogeneous

This closes the gap identified in the review: previous experiments
benchmarked individual stages in isolation but never measured the
integrated pipeline end-to-end with parallelism enabled.

Usage:
    python exp5_integrated_benchmark.py
"""
import sys, os, time, csv, json
sys.path.insert(0, os.path.dirname(__file__))

from core.weighted_projective import Parametrization, is_weighted_homogeneous
from core.pipeline import WeightedGroebnerPipeline
from sympy import symbols, Symbol


# ── Test cases ──────────────────────────────────────────────────────────────

def monomial_curve(n):
    """ξ_i = t^{i+1} in P(1, 2, ..., n+1)."""
    t = Symbol('t')
    weights = list(range(1, n + 2))
    x = symbols(' '.join(f'x{i}' for i in range(n + 1)))
    gens = [t ** (i + 1) for i in range(n + 1)]
    return Parametrization(list(gens), [t], list(x), weights)


def prime_multiplier_curve(n):
    """ξ_i = p_{2i}·t^{i+1} + (-1)^i·p_{2i+1} with small primes."""
    t = Symbol('t')
    primes = [3, 5, 7, 7, 5, 3, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    weights = list(range(1, n + 2))
    x = symbols(' '.join(f'x{i}' for i in range(n + 1)))
    gens = []
    for i in range(n + 1):
        p2i = primes[2 * i % len(primes)]
        p2i1 = primes[(2 * i + 1) % len(primes)]
        gens.append(p2i * t ** (i + 1) + ((-1) ** i) * p2i1)
    return Parametrization(list(gens), [t], list(x), weights)


# ── Benchmark runner ────────────────────────────────────────────────────────

def benchmark(param, p_values, n_reps=2):
    """Run pipeline at each processor count, return list of result dicts."""
    rows = []
    for p in p_values:
        use_par = (p > 1)
        times = []
        stage_times_agg = {"homogenization": [], "elimination": [], "normalization": []}
        basis_size = None
        is_correct = None

        for rep in range(n_reps):
            pipe = WeightedGroebnerPipeline(
                n_processors=p, use_parallel=use_par, verbose=False
            )
            r = pipe.run(param)
            times.append(r.total_time)
            for stage in stage_times_agg:
                stage_times_agg[stage].append(r.stage_times.get(stage, 0))
            basis_size = len(r.normalized_basis)
            is_correct = r.is_correct

        avg_total = sum(times) / len(times)
        row = {
            "n": param.n,
            "m": param.m,
            "weights": str(param.weights),
            "p": p,
            "t_hom": sum(stage_times_agg["homogenization"]) / n_reps,
            "t_elim": sum(stage_times_agg["elimination"]) / n_reps,
            "t_norm": sum(stage_times_agg["normalization"]) / n_reps,
            "t_total": avg_total,
            "basis_size": basis_size,
            "correct": is_correct,
        }
        rows.append(row)
    return rows


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    p_values = [1, 2, 4]
    all_rows = []

    print("=" * 72)
    print("Experiment 5 — End-to-End Integrated Pipeline Benchmark")
    print("=" * 72)

    # Monomial curve family
    for n in [2, 3, 4, 5, 6, 7]:
        param = monomial_curve(n)
        print(f"\n--- Monomial curve  n={n}  P({','.join(map(str,param.weights))}) ---")
        rows = benchmark(param, p_values, n_reps=2)
        t1 = [r for r in rows if r["p"] == 1][0]["t_total"]
        for r in rows:
            s = t1 / r["t_total"] if r["t_total"] > 0 else 0
            e = s / r["p"]
            r["speedup"] = round(s, 3)
            r["efficiency"] = round(e, 3)
            r["family"] = "monomial"
            print(f"  p={r['p']:2d}  total={r['t_total']:.4f}s  "
                  f"hom={r['t_hom']:.4f}  elim={r['t_elim']:.4f}  "
                  f"norm={r['t_norm']:.4f}  "
                  f"S={r['speedup']:.2f}x  E={r['efficiency']:.2f}  "
                  f"|G|={r['basis_size']}  ok={r['correct']}")
        all_rows.extend(rows)

    # Prime-multiplier family (tests GCD normalization impact)
    for n in [2, 3, 4, 5]:
        param = prime_multiplier_curve(n)
        print(f"\n--- Prime-mult curve  n={n}  P({','.join(map(str,param.weights))}) ---")
        rows = benchmark(param, p_values, n_reps=2)
        t1 = [r for r in rows if r["p"] == 1][0]["t_total"]
        for r in rows:
            s = t1 / r["t_total"] if r["t_total"] > 0 else 0
            e = s / r["p"]
            r["speedup"] = round(s, 3)
            r["efficiency"] = round(e, 3)
            r["family"] = "prime_mult"
            print(f"  p={r['p']:2d}  total={r['t_total']:.4f}s  "
                  f"hom={r['t_hom']:.4f}  elim={r['t_elim']:.4f}  "
                  f"norm={r['t_norm']:.4f}  "
                  f"S={r['speedup']:.2f}x  E={r['efficiency']:.2f}  "
                  f"|G|={r['basis_size']}  ok={r['correct']}")
        all_rows.extend(rows)

    # Write CSV
    csv_path = os.path.join(out_dir, "exp5_integrated.csv")
    fields = ["family", "n", "m", "weights", "p",
              "t_hom", "t_elim", "t_norm", "t_total",
              "basis_size", "correct", "speedup", "efficiency"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nResults written to {csv_path}")

    # Summary
    print("\n" + "=" * 72)
    print("Summary — Key findings:")
    print("=" * 72)
    mono_rows = [r for r in all_rows if r["family"] == "monomial"]
    for n in sorted(set(r["n"] for r in mono_rows)):
        n_rows = [r for r in mono_rows if r["n"] == n]
        s2 = [r for r in n_rows if r["p"] == 2]
        s4 = [r for r in n_rows if r["p"] == 4]
        if s2 and s4:
            print(f"  n={n}: S(2)={s2[0]['speedup']:.2f}x  S(4)={s4[0]['speedup']:.2f}x  "
                  f"correct={s4[0]['correct']}")


if __name__ == "__main__":
    main()
