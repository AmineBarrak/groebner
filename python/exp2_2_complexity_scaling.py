"""
exp2_2_complexity_scaling.py  (FINAL)
======================================
TIER 2 — Experiment 2.2: Complexity Scaling

Two families:
  monomial — x_i = t^{i+1}, always ±1 coefficients (structured)
  prime    — x_i = p_i * t^{i+1} + q_i, small primes (3,5,7)
             produces 5-20 bit coefficients without computation blowup

Run: python exp2_2_complexity_scaling.py
Output: results/exp2_2.csv  +  results/exp2_2_scaling.png
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
from core.pipeline import SequentialPipeline


# ── Small primes — kept intentionally modest so n=8 stays tractable ───────
_PRIMES = [3, 5, 7, 7, 5, 3, 5, 7, 3, 5]


def monomial_curve(n_vars):
    """x_i = t^{i+1}, weights=[1..n]. Always ±1 coefficients."""
    t = Symbol("t")
    cvars = [Symbol(f"x{i}") for i in range(n_vars)]
    return Parametrization(
        [t**(i+1) for i in range(n_vars)],
        [t], cvars, list(range(1, n_vars + 1))
    )


def prime_curve(n_vars):
    """
    x_i = p_{2i} * t^{i+1} + sign * p_{2i+1}
    Small prime multipliers (3,5,7) force multi-bit coefficients in
    the ideal without causing computation blowup. Tractable to n=8.
    """
    t = Symbol("t")
    cvars = [Symbol(f"x{i}") for i in range(n_vars)]
    gens = []
    for i in range(n_vars):
        p_mul  = _PRIMES[(2 * i)     % len(_PRIMES)]
        p_off  = _PRIMES[(2 * i + 1) % len(_PRIMES)]
        sign   = 1 if i % 2 == 0 else -1
        gens.append(p_mul * t**(i+1) + sign * p_off)
    return Parametrization(gens, [t], cvars, list(range(1, n_vars + 1)))


def run(n_range_monomial=None, n_range_dense=None,
        n_repeats=2, verbose=True):

    if n_range_monomial is None:
        n_range_monomial = [2, 3, 4, 5, 6, 7, 8]
    if n_range_dense is None:
        n_range_dense = [2, 3, 4, 5, 6, 7, 8]

    pipe    = SequentialPipeline(verbose=False)
    records = []

    def measure(param, family_label, n):
        times, basis_sizes, max_bits = [], [], []
        for _ in range(n_repeats):
            t0  = time.perf_counter()
            res = pipe.run(param)
            times.append(time.perf_counter() - t0)
            basis_sizes.append(len(res.normalized_basis))
            from sympy import Poly
            bits = []
            for f in res.normalized_basis:
                try:
                    for c in Poly(f, *param.coord_vars).coeffs():
                        bits.append(abs(int(c)).bit_length())
                except Exception:
                    pass
            max_bits.append(max(bits) if bits else 0)

        avg = lambda lst: sum(lst) / len(lst)
        rec = {
            "family":         family_label,
            "n_vars":         n,
            "avg_time_s":     round(avg(times), 5),
            "avg_basis_size": round(avg(basis_sizes), 1),
            "max_coeff_bits": round(avg(max_bits), 1),
            "weights":        str(param.weights),
        }
        records.append(rec)
        if verbose:
            print(f"  [{family_label:<8}]  n={n}  "
                  f"time={avg(times):.4f}s  "
                  f"|G|={avg(basis_sizes):.0f}  "
                  f"bits={avg(max_bits):.0f}")
        return rec

    if verbose:
        print("\n--- Monomial curve family (structured, +-1 coefficients) ---")
    for n in n_range_monomial:
        try:
            measure(monomial_curve(n), "monomial", n)
        except Exception as e:
            print(f"  [monomial n={n}] SKIPPED: {e}")

    if verbose:
        print("\n--- Prime-multiplier family (unstructured, large coefficients) ---")
    for n in n_range_dense:
        try:
            measure(prime_curve(n), "prime", n)
        except Exception as e:
            print(f"  [prime n={n}] SKIPPED: {e}")

    df = pd.DataFrame(records)

    if verbose:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2.2 — COMPLEXITY SCALING")
        print("=" * 70)
        for fam in df["family"].unique():
            sub = df[df["family"] == fam]
            print(f"\n  Family: {fam}")
            print(sub[["n_vars", "avg_time_s", "avg_basis_size",
                        "max_coeff_bits"]].to_string(index=False))

        print("\nCoefficient bit comparison (same n):")
        for n in [3, 4, 5, 6, 7, 8]:
            m = df[(df["family"] == "monomial") & (df["n_vars"] == n)]
            p = df[(df["family"] == "prime")    & (df["n_vars"] == n)]
            if not m.empty and not p.empty:
                bm = m["max_coeff_bits"].values[0]
                bp = p["max_coeff_bits"].values[0]
                print(f"  n={n}:  monomial={bm:.0f}b  prime={bp:.0f}b  "
                      f"ratio={bp/max(bm, 1):.0f}x")

        print("\nRuntime growth (monomial):")
        mono = df[df["family"] == "monomial"].sort_values("n_vars")
        ts   = mono["avg_time_s"].tolist()
        ns   = mono["n_vars"].tolist()
        for i in range(1, len(ts)):
            if ts[i-1] > 0:
                print(f"  t(n={ns[i]}) / t(n={ns[i-1]}) = "
                      f"{ts[i]/ts[i-1]:.2f}x")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Experiment 2.2 — Complexity Scaling: "
        "Monomial vs Prime-Multiplier Family",
        fontsize=12
    )

    colors  = {"monomial": "#4e79a7", "prime": "#e15759"}
    markers = {"monomial": "o",       "prime": "s"}

    for ax_idx, (col, ylabel, title, use_log) in enumerate([
        ("avg_time_s",     "Time (s)",
         "Runtime vs n",   True),
        ("avg_basis_size", "Generators |G|",
         "Basis size vs n", False),
        ("max_coeff_bits", "Max coefficient bits",
         "Coefficient growth vs n\n"
         "(prime family triggers GCD bottleneck)", False),
    ]):
        ax = axes[ax_idx]
        for fam in df["family"].unique():
            sub = df[df["family"] == fam].sort_values("n_vars")
            ax.plot(sub["n_vars"], sub[col],
                    f"{markers[fam]}-", color=colors[fam],
                    label=fam, lw=2, ms=7)
        ax.set_xlabel("n (variables)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        if use_log and (df[col] > 0).any():
            ax.set_yscale("log")

    # Annotate the coefficient gap at n=6 if both families reached it
    try:
        bm = df[(df["family"] == "monomial") &
                (df["n_vars"] == 6)]["max_coeff_bits"].values[0]
        bp = df[(df["family"] == "prime") &
                (df["n_vars"] == 6)]["max_coeff_bits"].values[0]
        if bp > bm + 5:
            axes[2].annotate(
                f"prime: {bp:.0f}b",
                xy=(6, bp), xytext=(6.15, bp * 0.82),
                arrowprops=dict(arrowstyle="->", color="#e15759"),
                fontsize=8, color="#e15759"
            )
            axes[2].annotate(
                f"monomial: {bm:.0f}b",
                xy=(6, bm), xytext=(6.15, bm + 2),
                arrowprops=dict(arrowstyle="->", color="#4e79a7"),
                fontsize=8, color="#4e79a7"
            )
    except Exception:
        pass

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/exp2_2_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    df.to_csv("results/exp2_2.csv", index=False)
    print("\nSaved -> results/exp2_2.csv  +  results/exp2_2_scaling.png")
    return df


if __name__ == "__main__":
    run(
        n_range_monomial=[2, 3, 4, 5, 6, 7, 8],
        n_range_dense   =[2, 3, 4, 5, 6, 7, 8],
        n_repeats=2,
        verbose=True
    )