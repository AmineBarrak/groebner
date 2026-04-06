"""
exp3_1_hom_speedup.py
======================
TIER 3 — Experiment 3.1: Homogenization Stage Speedup

Uses Python multiprocessing. On your 32 GB / RTX machine, the break-even
point (where real computation > spawn overhead) is around n=30 generators.
Results for n < 30 will show IPC overhead; n >= 30 will show real speedup.

Run: python exp3_1_hom_speedup.py
Output: results/exp3_1.csv  +  results/exp3_1_speedup.png

For a true OpenMP measurement on your machine use the C++ version:
  cd cpp/exp3_1_openmp && cmake -B build && cmake --build build
  ./build/hom_speedup
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import Symbol

from core.weighted_projective import Parametrization
from core.homogenization import homogenize_ideal, parallel_homogenize_ideal


def make_param(n):
    t = Symbol("t")
    cvars = [Symbol(f"x{i}") for i in range(n)]
    return Parametrization(
        [t**(i+1) for i in range(n)], [t], cvars, list(range(1, n+1))
    )


def run(n_list=None, p_list=None, n_repeats=4, verbose=True):
    if n_list is None:
        # For your 32 GB machine: n=10..60 gives a good range
        # Below 20: IPC dominates. Above 30: real computation starts to dominate.
        n_list = [10, 20, 30, 40, 50, 60]
    if p_list is None:
        # Adapt to your CPU core count (n_cores - 1 is safe)
        import multiprocessing
        max_p = multiprocessing.cpu_count() - 1
        p_list = [1, 2, 4, min(8, max_p)]
    p_list = sorted(set(p_list))

    records   = []
    baselines = {}

    for n in n_list:
        param = make_param(n)
        for p in p_list:
            times = []
            for _ in range(n_repeats):
                if p == 1:
                    t0 = time.perf_counter()
                    homogenize_ideal(param)
                    times.append(time.perf_counter() - t0)
                else:
                    t0 = time.perf_counter()
                    parallel_homogenize_ideal(param, n_processors=p)
                    times.append(time.perf_counter() - t0)

            avg_t = sum(times) / len(times)
            if p == 1:
                baselines[n] = avg_t

            speedup    = baselines.get(n, avg_t) / max(avg_t, 1e-9)
            efficiency = speedup / p

            records.append({
                "n_generators": n,
                "n_processors": p,
                "avg_time_s":   round(avg_t, 5),
                "speedup":      round(speedup, 3),
                "efficiency":   round(efficiency, 3),
            })
            if verbose:
                print(f"  n={n:3d}  p={p:2d}  "
                      f"t={avg_t:.4f}s  S={speedup:.2f}×  E={efficiency:.2f}")

    df = pd.DataFrame(records)

    if verbose:
        print("\n" + "=" * 60)
        print("SPEEDUP TABLE  S(p) = T(1)/T(p)")
        print("=" * 60)
        pivot = df.pivot(index="n_generators", columns="n_processors",
                         values="speedup")
        print(pivot.round(2).to_string())
        print("\nNOTE: for n < ~30 Python IPC (~300ms) dominates.")
        print("      C++ OpenMP version removes this overhead entirely.")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 3.1 — Homogenization Speedup\n"
                 "(Python multiprocessing — IPC dominates for n < 30)",
                 fontsize=11)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_list)))

    for n, col in zip(n_list, colors):
        sub = df[df["n_generators"] == n].sort_values("n_processors")
        axes[0].plot(sub["n_processors"], sub["speedup"], "o-",
                     color=col, label=f"n={n}", lw=2, ms=5)
        axes[1].plot(sub["n_processors"], sub["efficiency"], "s-",
                     color=col, label=f"n={n}", lw=2, ms=5)

    for ax in axes:
        ax.axhline(y=1.0, color="k", ls="--", alpha=0.35)
        ax.set_xlabel("Processors (p)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Speedup S(p)")
    axes[0].set_title("Speedup")
    axes[1].set_ylabel("Efficiency E(p) = S(p)/p")
    axes[1].set_title("Parallel Efficiency")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/exp3_1_speedup.png", dpi=150, bbox_inches="tight")
    plt.close()
    df.to_csv("results/exp3_1.csv", index=False)
    print("Saved → results/exp3_1.csv  +  results/exp3_1_speedup.png")
    return df


if __name__ == "__main__":
    run(verbose=True)
