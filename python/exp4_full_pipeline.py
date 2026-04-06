"""
exp4_full_pipeline.py
======================
TIER 4 — Experiments 4.1, 4.2, 4.3

4.1  Full pipeline scaling  (n=2..6,  p=1,2,4)
4.2  vs raw sympy.groebner  (correctness + timing)
4.3  Genus-2 curve  P(1,2,3,4,5)

Run: python exp4_full_pipeline.py
Output: results/exp4_*.csv  +  results/exp4_full_pipeline.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import symbols, Symbol, groebner as sym_groebner, expand

from core.weighted_projective import Parametrization, is_weighted_homogeneous
from core.pipeline import WeightedGroebnerPipeline, SequentialPipeline


# ── Helper: monomial curve ─────────────────────────────────────────────────

def monomial_curve(n):
    t = Symbol("t")
    cv = [Symbol(f"x{i}") for i in range(n)]
    return Parametrization([t**(i+1) for i in range(n)], [t], cv,
                            list(range(1, n+1)))


# ═══════════════════════════════════════════════════════════════════
#  4.1  Full pipeline scaling
# ═══════════════════════════════════════════════════════════════════

def run_4_1(n_range=None, p_list=None, n_repeats=2, verbose=True):
    if n_range is None: n_range = [2, 3, 4, 5, 6]
    if p_list  is None: p_list  = [1, 2, 4]
    baselines  = {}
    records    = []

    for n in n_range:
        param = monomial_curve(n)
        for p in p_list:
            times = []
            for _ in range(n_repeats):
                pipe = WeightedGroebnerPipeline(
                    n_processors=p, use_parallel=(p > 1), verbose=False)
                res  = pipe.run(param)
                times.append(res.total_time)

            avg_t = sum(times) / len(times)
            if p == 1:
                baselines[n] = avg_t

            speedup    = baselines.get(n, avg_t) / max(avg_t, 1e-9)
            efficiency = speedup / p
            records.append({
                "n_vars":       n,
                "n_processors": p,
                "avg_time_s":   round(avg_t, 5),
                "speedup":      round(speedup, 3),
                "efficiency":   round(efficiency, 3),
            })
            if verbose:
                print(f"  4.1  n={n}  p={p:2d}  t={avg_t:.4f}s  "
                      f"S={speedup:.2f}×  E={efficiency:.2f}")

    df = pd.DataFrame(records)
    if verbose:
        print("\nSpeedup table:")
        print(df.pivot(index="n_vars", columns="n_processors",
                        values="speedup").round(2).to_string())
    df.to_csv("results/exp4_1.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════
#  4.2  Comparison with raw sympy.groebner
# ═══════════════════════════════════════════════════════════════════

def run_4_2(n_range=None, p_best=4, n_repeats=2, verbose=True):
    if n_range is None: n_range = [2, 3, 4]
    records = []

    for n in n_range:
        param = monomial_curve(n)
        cv    = list(param.coord_vars)
        t_sym = Symbol("t")

        ideal_gens = [xi - gi for xi, gi in zip(param.coord_vars,
                                                  param.generators)]
        all_vars   = [t_sym] + list(param.coord_vars)

        # Raw sympy
        times_raw = []
        raw_basis  = None
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            G  = sym_groebner(ideal_gens, *all_vars, order="lex")
            times_raw.append(time.perf_counter() - t0)
            raw_basis = list(G)
        t_raw = sum(times_raw) / len(times_raw)

        # Check if raw output is weighted homogeneous
        raw_hom = all(
            is_weighted_homogeneous(f, cv, param.weights)[0]
            for f in raw_basis if f != 0
        )

        # Sequential pipeline
        times_seq = []
        for _ in range(n_repeats):
            res = SequentialPipeline(verbose=False).run(param)
            times_seq.append(res.total_time)
        t_seq = sum(times_seq) / len(times_seq)

        # Parallel pipeline
        times_par = []
        for _ in range(n_repeats):
            res = WeightedGroebnerPipeline(n_processors=p_best,
                                           use_parallel=True,
                                           verbose=False).run(param)
            times_par.append(res.total_time)
        t_par = sum(times_par) / len(times_par)

        records.append({
            "n_vars":              n,
            "t_raw_sympy":         round(t_raw, 5),
            "t_seq_pipeline":      round(t_seq, 5),
            "t_par_pipeline":      round(t_par, 5),
            "raw_is_weighted_hom": raw_hom,
            "speedup_vs_raw":      round(t_raw / max(t_par, 1e-9), 3),
            "p":                   p_best,
        })
        if verbose:
            print(f"  4.2  n={n}  raw={t_raw:.4f}s  seq={t_seq:.4f}s  "
                  f"par={t_par:.4f}s  raw_hom={raw_hom}")

    df = pd.DataFrame(records)
    df.to_csv("results/exp4_2.csv", index=False)
    return df


# ═══════════════════════════════════════════════════════════════════
#  4.3  Genus-2 curve  P(1,2,3,4,5)
# ═══════════════════════════════════════════════════════════════════

def run_4_3(p_list=None, verbose=True):
    if p_list is None: p_list = [1, 2, 4]
    t = Symbol("t")
    x0,x1,x2,x3,x4 = symbols("x0 x1 x2 x3 x4")

    param = Parametrization(
        generators=[t, t**2, t**3, t**4, t**5],
        param_vars=[t],
        coord_vars=[x0,x1,x2,x3,x4],
        weights=[1,2,3,4,5],
    )
    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4.3 — GENUS-2 CURVE  P(1,2,3,4,5)")
        print("=" * 60)
        print(param)

    result = {"space": "P(1,2,3,4,5)"}

    for p in p_list:
        pipe = WeightedGroebnerPipeline(
            n_processors=p, use_parallel=(p > 1), verbose=False)
        t0  = time.perf_counter()
        res = pipe.run(param)
        elapsed = time.perf_counter() - t0

        result[f"p{p}_time"]       = round(elapsed, 4)
        result[f"p{p}_basis_size"] = len(res.normalized_basis)
        result[f"p{p}_correct"]    = res.is_correct

        if verbose:
            print(f"\n  p={p}  time={elapsed:.4f}s  "
                  f"|J|={len(res.normalized_basis)}  correct={res.is_correct}")
            for f in res.normalized_basis[:5]:
                ok, deg = is_weighted_homogeneous(f, list(param.coord_vars),
                                                   param.weights)
                print(f"    [deg {deg}]  {expand(f)}  hom={ok}")
            if len(res.normalized_basis) > 5:
                print(f"    ... ({len(res.normalized_basis)-5} more)")

    with open("results/exp4_3_genus2.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ═══════════════════════════════════════════════════════════════════
#  Combined plot
# ═══════════════════════════════════════════════════════════════════

def plot_tier4(df_41, df_42, out="results/exp4_full_pipeline.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Tier 4 — End-to-End Pipeline Validation", fontsize=12)

    # 4.1 heatmap
    ax = axes[0]
    try:
        pivot = df_41.pivot(index="n_vars", columns="n_processors",
                             values="speedup")
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                        vmin=0, vmax=max(pivot.values.max(), 4))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"p={p}" for p in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"n={n}" for n in pivot.index])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i,j]:.2f}×",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("4.1 Speedup S(n,p)")
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center", transform=ax.transAxes)
        ax.set_title("4.1 Speedup")

    # 4.2 bar comparison
    ax2 = axes[1]
    if len(df_42) > 0:
        x = np.arange(len(df_42))
        w = 0.28
        ax2.bar(x - w, df_42["t_raw_sympy"],    w, label="Raw sympy",        color="#e15759")
        ax2.bar(x,     df_42["t_seq_pipeline"],  w, label="Seq pipeline",     color="#4e79a7")
        ax2.bar(x + w, df_42["t_par_pipeline"],  w,
                label=f"Par pipeline (p={df_42['p'].iloc[0]})", color="#59a14f")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"n={n}" for n in df_42["n_vars"]])
        ax2.set_ylabel("Time (s)")
        ax2.set_yscale("log")
        ax2.set_title("4.2 Pipeline vs raw sympy")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis="y")

    # 4.1 efficiency
    ax3 = axes[2]
    for p, grp in df_41.groupby("n_processors"):
        grp = grp.sort_values("n_vars")
        ax3.plot(grp["n_vars"], grp["efficiency"], "o-", label=f"p={p}", lw=2)
    ax3.axhline(y=1.0, color="k", ls="--", alpha=0.4)
    ax3.set_xlabel("n (variables)")
    ax3.set_ylabel("Efficiency E(p) = S(p)/p")
    ax3.set_title("4.1 Efficiency vs n")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("\n--- Experiment 4.1: Full pipeline scaling ---")
    df_41 = run_4_1(n_range=[2,3,4,5,6], p_list=[1,2,4],
                     n_repeats=2, verbose=True)

    print("\n--- Experiment 4.2: vs raw sympy ---")
    df_42 = run_4_2(n_range=[2,3,4], p_best=4, n_repeats=2, verbose=True)

    print("\n--- Experiment 4.3: Genus-2 curve ---")
    g2 = run_4_3(p_list=[1,2,4], verbose=True)

    plot_tier4(df_41, df_42)
    print("\nAll Tier-4 results saved to results/")
