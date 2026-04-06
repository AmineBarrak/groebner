"""
exp3_4_gcd_scaling.py
======================
TIER 3 — Experiment 3.4: Weighted GCD Factorization Scaling

Valid result — measures the factorization explosion.
Extended to 256-bit on your 32 GB machine (256-bit may take ~30 min).

Run: python exp3_4_gcd_scaling.py
Output: results/exp3_4.csv  +  results/exp3_4_gcd_scaling.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.weighted_projective import weighted_gcd_int


def run(bit_sizes=None, n_coeffs=8, n_repeats=5, verbose=True):
    if bit_sizes is None:
        # 256-bit will be slow (~minutes). Remove if you only want fast results.
        bit_sizes = [8, 16, 32, 64, 128, 256]

    random.seed(42)
    weights = [1] * n_coeffs
    records = []

    for bits in bit_sizes:
        coeffs = [random.randint(2**(bits-1), 2**bits) for _ in range(n_coeffs)]
        times  = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            weighted_gcd_int(coeffs, weights)
            times.append(time.perf_counter() - t0)

        avg_us  = sum(times) / len(times) * 1e6
        records.append({
            "coeff_bits":    bits,
            "avg_time_us":   round(avg_us, 2),
            "avg_time_s":    round(avg_us / 1e6, 6),
        })
        if verbose:
            print(f"  {bits:4d} bits → {avg_us:>12.2f} μs")

    df = pd.DataFrame(records)
    # Add growth factor column
    base = df.loc[df["coeff_bits"] == 16, "avg_time_us"].values
    if len(base) > 0:
        df["growth_vs_16bit"] = (df["avg_time_us"] / base[0]).round(1)
    else:
        df["growth_vs_16bit"] = None

    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3.4 — GCD FACTORIZATION SCALING")
        print("=" * 60)
        print(df.to_string(index=False))
        peak = df["avg_time_us"].max()
        print(f"\nPeak time: {peak:.0f} μs  "
              f"({peak/1e6:.2f} s) at {df.loc[df['avg_time_us'].idxmax(),'coeff_bits']} bits")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 3.4 — Weighted GCD Factorization Scaling", fontsize=12)

    ax = axes[0]
    ax.semilogy(df["coeff_bits"], df["avg_time_us"], "o-",
                color="#e15759", lw=2.5, ms=7)
    ax.set_xlabel("Coefficient bit size")
    ax.set_ylabel("Avg factorization time (μs)")
    ax.set_title("Time vs bit size (log scale)")
    ax.fill_between(df["coeff_bits"], df["avg_time_us"],
                    alpha=0.12, color="#e15759")
    ax.grid(alpha=0.3)

    # Mark the 128-bit explosion
    if 128 in df["coeff_bits"].values:
        y128 = df.loc[df["coeff_bits"]==128, "avg_time_us"].values[0]
        ax.annotate("factorization\nexplosion", xy=(128, y128),
                    xytext=(90, y128*0.15),
                    arrowprops=dict(arrowstyle="->", color="#f28e2b"),
                    fontsize=9, color="#f28e2b")

    ax2 = axes[1]
    if "growth_vs_16bit" in df.columns and df["growth_vs_16bit"].notna().any():
        sub = df[df["growth_vs_16bit"].notna()]
        ax2.bar(sub["coeff_bits"].astype(str), sub["growth_vs_16bit"],
                color="#4e79a7", alpha=0.8)
        ax2.set_xlabel("Bit size")
        ax2.set_ylabel("Growth vs 16-bit")
        ax2.set_title("Growth factor vs 16-bit baseline")
        ax2.set_yscale("log")
        ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/exp3_4_gcd_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    df.to_csv("results/exp3_4.csv", index=False)
    print("Saved → results/exp3_4.csv  +  results/exp3_4_gcd_scaling.png")
    return df


if __name__ == "__main__":
    run(verbose=True)
