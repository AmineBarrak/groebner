"""
exp2_1_phase_timing.py
=======================
TIER 2 — Experiment 2.1: Phase-wise Timing Breakdown (Sequential)

Run: python exp2_1_phase_timing.py
Output: results/exp2_1.csv  +  results/exp2_1_breakdown.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols

from core.weighted_projective import Parametrization
from core.pipeline import SequentialPipeline


def build_test_cases():
    t, t1, t2 = symbols("t t1 t2")
    x0, x1, x2, x3 = symbols("x0 x1 x2 x3")
    return [
        {"label": "P(1,1,2) n=2 m=1",
         "param": Parametrization([t, t**2, t**2], [t], [x0,x1,x2], [1,1,2])},
        {"label": "P(1,2,3) n=2 m=1",
         "param": Parametrization([t, t**2, t**3], [t], [x0,x1,x2], [1,2,3])},
        {"label": "P(1,1,2) n=2 m=2",
         "param": Parametrization([t1, t1**2, t1*t2], [t1,t2], [x0,x1,x2], [1,1,2])},
        {"label": "P(1,2,3,4) n=3 m=1",
         "param": Parametrization([t, t**2, t**3, t**4], [t], [x0,x1,x2,x3], [1,2,3,4])},
        {"label": "P(1,2,3,4) n=3 m=2",
         "param": Parametrization([t1, t1**2, t1*t2, t2**2], [t1,t2],
                                   [x0,x1,x2,x3], [1,2,3,4])},
    ]


def run(n_repeats=3, verbose=True):
    pipe = SequentialPipeline(verbose=False)
    records = []

    for tc in build_test_cases():
        times_h, times_e, times_n, times_tot = [], [], [], []
        for _ in range(n_repeats):
            res = pipe.run(tc["param"])
            times_h.append(res.stage_times.get("homogenization", 0))
            times_e.append(res.stage_times.get("elimination", 0))
            times_n.append(res.stage_times.get("normalization", 0))
            times_tot.append(res.total_time)

        def avg(lst): return sum(lst) / len(lst)
        t_h = avg(times_h); t_e = avg(times_e)
        t_n = avg(times_n); t_t = avg(times_tot)

        records.append({
            "label":    tc["label"],
            "t_hom":    round(t_h, 5),
            "t_elim":   round(t_e, 5),
            "t_norm":   round(t_n, 5),
            "t_total":  round(t_t, 5),
            "pct_hom":  round(100 * t_h / max(t_t, 1e-9), 1),
            "pct_elim": round(100 * t_e / max(t_t, 1e-9), 1),
            "pct_norm": round(100 * t_n / max(t_t, 1e-9), 1),
        })

    df = pd.DataFrame(records)

    if verbose:
        print("=" * 70)
        print("EXPERIMENT 2.1 — PHASE-WISE TIMING BREAKDOWN")
        print("=" * 70)
        print(df[["label","t_hom","t_elim","t_norm","t_total",
                  "pct_hom","pct_elim","pct_norm"]].to_string(index=False))
        print()
        dom = df.loc[df["t_total"].idxmax()]
        print(f"Hardest case: {dom['label']}")
        for ph, pct in sorted(
            {"Homogenization": dom["pct_hom"],
             "Elimination":    dom["pct_elim"],
             "Normalization":  dom["pct_norm"]}.items(),
            key=lambda x: -x[1]
        ):
            bar = "█" * int(pct / 2)
            print(f"  {ph:<18} {pct:5.1f}%  {bar}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 2.1 — Phase-wise Timing Breakdown (Sequential)",
                 fontsize=12)
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    w = 0.55
    COLORS = {"hom": "#4e79a7", "elim": "#f28e2b", "norm": "#59a14f"}

    for ax_idx, (ykey, ylabel, title) in enumerate([
        (("t_hom","t_elim","t_norm"), "Time (s)", "Absolute stage times"),
        (("pct_hom","pct_elim","pct_norm"), "Percentage (%)", "Stage time fractions"),
    ]):
        ax = axes[ax_idx]
        if "t_" in ykey[0]:
            ax.bar(x, df["t_hom"],  w, label="Homogenization",   color=COLORS["hom"])
            ax.bar(x, df["t_elim"], w, bottom=df["t_hom"],
                   label="F4 Elimination",  color=COLORS["elim"])
            ax.bar(x, df["t_norm"], w,
                   bottom=df["t_hom"] + df["t_elim"],
                   label="GCD Normalization", color=COLORS["norm"])
        else:
            ax.bar(x, df["pct_hom"],  w, label="Homogenization",   color=COLORS["hom"])
            ax.bar(x, df["pct_elim"], w, bottom=df["pct_hom"],
                   label="F4 Elimination",  color=COLORS["elim"])
            ax.bar(x, df["pct_norm"], w,
                   bottom=df["pct_hom"] + df["pct_elim"],
                   label="GCD Normalization", color=COLORS["norm"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/exp2_1_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    df.to_csv("results/exp2_1.csv", index=False)
    print("Saved → results/exp2_1.csv  +  results/exp2_1_breakdown.png")
    return df


if __name__ == "__main__":
    run(n_repeats=3, verbose=True)
