"""
python/plot_cpp_results.py
===========================
Reads CSV outputs produced by the C++ binaries and generates
publication-ready figures.

Run AFTER the C++ experiments have completed:
  python plot_cpp_results.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def plot_exp3_1_cpp():
    """Homogenization speedup — OpenMP results."""
    csv_path = os.path.join(RESULTS, "exp3_1_cpp.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found — run ./build/hom_speedup first")
        return

    df = pd.read_csv(csv_path)
    n_list = sorted(df["n_generators"].unique())
    p_list = sorted(df["n_processors"].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_list)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment 3.1 — Homogenization Speedup  (C++ / OpenMP)",
                 fontsize=12)

    for n, col in zip(n_list, colors):
        sub = df[df["n_generators"] == n].sort_values("n_processors")
        axes[0].plot(sub["n_processors"], sub["speedup"],
                     "o-", color=col, label=f"n={n}", lw=2, ms=5)
        axes[1].plot(sub["n_processors"], sub["efficiency"],
                     "s-", color=col, label=f"n={n}", lw=2, ms=5)

    # Ideal linear speedup
    axes[0].plot(p_list, p_list, "k--", alpha=0.35, label="Ideal")
    axes[0].set_xlabel("Processors (p)")
    axes[0].set_ylabel("Speedup  S(p) = T(1)/T(p)")
    axes[0].set_title("Speedup")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(y=1.0, color="k", ls="--", alpha=0.35)
    axes[1].set_xlabel("Processors (p)")
    axes[1].set_ylabel("Efficiency  E(p) = S(p)/p")
    axes[1].set_title("Parallel Efficiency")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "exp3_1_cpp_speedup.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_exp3_2_cuda():
    """F4 elimination speedup — CUDA results."""
    csv_path = os.path.join(RESULTS, "exp3_2_cuda.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found — run ./build/f4_cuda first")
        return

    df = pd.read_csv(csv_path)
    df = df[df["gpu_gauss_ms"] > 0].copy()   # drop rows where GPU was skipped
    if df.empty:
        print("  [SKIP] No valid GPU rows in exp3_2_cuda.csv")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Experiment 3.2 — F4 Elimination  (C++ / CUDA + cuSPARSE)",
                 fontsize=12)

    n_vals = sorted(df["n_vars"].unique())

    # Left: matrix reduction time CPU vs GPU
    ax = axes[0]
    sub4 = df[df["cpu_threads"] == 4].sort_values("n_vars")
    if not sub4.empty:
        ax.semilogy(sub4["n_vars"], sub4["cpu_gauss_ms"],
                    "o-", color="#e15759", lw=2, ms=6, label="CPU Gauss")
        ax.semilogy(sub4["n_vars"], sub4["gpu_gauss_ms"],
                    "s-", color="#4e79a7", lw=2, ms=6, label="GPU (cuSPARSE)")
    ax.set_xlabel("n (variables)")
    ax.set_ylabel("Matrix reduction time (ms)")
    ax.set_title("CPU vs GPU: matrix reduction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Centre: GPU speedup over CPU
    ax2 = axes[1]
    sub4 = df[df["cpu_threads"] == 4].sort_values("n_vars")
    if not sub4.empty:
        ax2.plot(sub4["n_vars"], sub4["gpu_speedup"],
                 "D-", color="#59a14f", lw=2.5, ms=7)
    ax2.axhline(y=1.0, color="k", ls="--", alpha=0.4, label="Break-even")
    ax2.set_xlabel("n (variables)")
    ax2.set_ylabel("GPU speedup  T_cpu / T_gpu")
    ax2.set_title("GPU speedup over CPU")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Right: matrix size (rows × cols)
    ax3 = axes[2]
    sub1 = df[df["cpu_threads"] == 1].sort_values("n_vars")
    if not sub1.empty:
        ax3.plot(sub1["n_vars"], sub1["mat_rows"],
                 "o-", color="#f28e2b", lw=2, ms=6, label="Rows")
        ax3.plot(sub1["n_vars"], sub1["mat_cols"],
                 "s-", color="#7c5cbf", lw=2, ms=6, label="Cols")
    ax3.set_xlabel("n (variables)")
    ax3.set_ylabel("Matrix dimension")
    ax3.set_title("Macaulay matrix size vs n")
    ax3.set_yscale("log")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "exp3_2_cuda_speedup.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_combined_parallel():
    """
    One combined figure for the paper:
    Left panel  : OpenMP speedup (exp3_1_cpp.csv)
    Centre panel: GPU speedup    (exp3_2_cuda.csv)
    Right panel : GCD scaling    (exp3_4.csv)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Tier 3 — Parallel Performance  (C++ / OpenMP + CUDA)",
                 fontsize=12)

    # ── Left: OpenMP speedup ──────────────────────────────────────────────
    ax = axes[0]
    csv = os.path.join(RESULTS, "exp3_1_cpp.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        n_list = sorted(df["n_generators"].unique())
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_list)))
        p_list = sorted(df["n_processors"].unique())
        for n, col in zip(n_list, colors):
            sub = df[df["n_generators"] == n].sort_values("n_processors")
            ax.plot(sub["n_processors"], sub["speedup"],
                    "o-", color=col, label=f"n={n}", lw=1.8, ms=4)
        ax.plot(p_list, p_list, "k--", alpha=0.35, lw=1, label="Ideal")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "Run ./build/hom_speedup\nto generate data",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Processors (p)")
    ax.set_ylabel("Speedup")
    ax.set_title("3.1  Homogenization (OpenMP)")
    ax.grid(alpha=0.3)

    # ── Centre: GPU speedup ───────────────────────────────────────────────
    ax2 = axes[1]
    csv2 = os.path.join(RESULTS, "exp3_2_cuda.csv")
    if os.path.exists(csv2):
        df2 = pd.read_csv(csv2)
        df2 = df2[df2["gpu_gauss_ms"] > 0]
        if not df2.empty:
            sub = df2[df2["cpu_threads"] == 4].sort_values("n_vars")
            ax2.plot(sub["n_vars"], sub["gpu_speedup"],
                     "D-", color="#59a14f", lw=2.5, ms=7)
            ax2.axhline(y=1.0, color="k", ls="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "Run ./build/f4_cuda\nto generate data",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=9)
    ax2.set_xlabel("n (variables)")
    ax2.set_ylabel("GPU speedup  T_cpu / T_gpu")
    ax2.set_title("3.2  F4 Elimination (CUDA)")
    ax2.grid(alpha=0.3)

    # ── Right: GCD scaling ────────────────────────────────────────────────
    ax3 = axes[2]
    csv3 = os.path.join(RESULTS, "exp3_4.csv")
    if os.path.exists(csv3):
        df3 = pd.read_csv(csv3)
        ax3.semilogy(df3["coeff_bits"], df3["avg_time_us"],
                     "o-", color="#e15759", lw=2.5, ms=7, label="Sequential")
        # Projected p=4 parallel (ideal)
        ax3.semilogy(df3["coeff_bits"], df3["avg_time_us"] / 4,
                     "--", color="#4e79a7", lw=1.5, alpha=0.7, label="Ideal p=4")
        if 128 in df3["coeff_bits"].values:
            y128 = df3.loc[df3["coeff_bits"] == 128, "avg_time_us"].values[0]
            ax3.annotate("128-bit\nexplosion",
                         xy=(128, y128), xytext=(85, y128 * 0.1),
                         arrowprops=dict(arrowstyle="->", color="#f28e2b"),
                         fontsize=8, color="#f28e2b")
        ax3.legend(fontsize=9)
    ax3.set_xlabel("Coefficient bit size")
    ax3.set_ylabel("Factorization time (μs)")
    ax3.set_title("3.4  GCD Scaling  (Python)")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS, "exp3_combined_cpp.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


if __name__ == "__main__":
    print("Plotting C++ experiment results...")
    plot_exp3_1_cpp()
    plot_exp3_2_cuda()
    plot_combined_parallel()
    print("Done.")
