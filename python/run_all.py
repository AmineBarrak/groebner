"""
python/run_all.py
==================
Master runner for all Python experiments.

Usage:
  # All tiers:
  python run_all.py

  # Single tier:
  python run_all.py --tier 1
  python run_all.py --tier 2
  python run_all.py --tier 3
  python run_all.py --tier 4

  # Quick mode (smaller n, fewer repeats):
  python run_all.py --quick

  # Slow: extend n to 8 for Tier 2 (takes ~10-30 min for n=7,8)
  python run_all.py --tier 2 --full
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import traceback
import time

os.makedirs("results", exist_ok=True)


def run_tier1(quick=False):
    print("\n" + "▓" * 60)
    print("  TIER 1 — CORRECTNESS VALIDATION")
    print("▓" * 60)
    from exp1_1_toy_cases      import run as run11
    from exp1_2_homogeneity_check import run as run12
    run11(verbose=True)
    run12(verbose=True)


def run_tier2(quick=False, full=False):
    print("\n" + "▓" * 60)
    print("  TIER 2 — SEQUENTIAL BASELINE PROFILING")
    print("▓" * 60)
    from exp2_1_phase_timing     import run as run21
    import exp2_2_complexity_scaling as _m22; run22 = _m22.run

    n_reps = 2 if quick else 3
    run21(n_repeats=n_reps, verbose=True)

    if full:
        n_mono  = [2, 3, 4, 5, 6, 7, 8]
        n_dense = [2, 3, 4, 5]
        print("\nFull mode: monomial n up to 8, dense n up to 5")
    elif quick:
        n_mono  = [2, 3, 4]
        n_dense = [2, 3]
    else:
        n_mono  = [2, 3, 4, 5, 6]
        n_dense = [2, 3, 4]

    run22(n_range_monomial=n_mono, n_range_dense=n_dense,
          n_repeats=2, verbose=True)


def run_tier3(quick=False):
    print("\n" + "▓" * 60)
    print("  TIER 3 — PARALLEL PERFORMANCE (Python prototype)")
    print("▓" * 60)
    from exp3_1_hom_speedup  import run as run31
    from exp3_4_gcd_scaling  import run as run34
    from exp3_5_load_imbalance import run as run35

    import multiprocessing
    max_p  = multiprocessing.cpu_count() - 1
    p_list = [1, 2, 4] if quick else [1, 2, 4, min(8, max_p)]
    n_list = [10, 20, 30] if quick else [10, 20, 30, 40, 60]

    run31(n_list=n_list, p_list=p_list, n_repeats=3, verbose=True)

    bit_sizes = [8, 16, 32, 64, 128] if quick else [8, 16, 32, 64, 128, 256]
    run34(bit_sizes=bit_sizes, n_repeats=3, verbose=True)

    p_imb = [2, 4] if quick else [2, 4, min(8, max_p)]
    run35(p_list=p_imb, n_repeats=2, verbose=True)


def run_tier4(quick=False):
    print("\n" + "▓" * 60)
    print("  TIER 4 — END-TO-END SCALING & APPLICATION")
    print("▓" * 60)
    from exp4_full_pipeline import run_4_1, run_4_2, run_4_3, plot_tier4

    n_range = [2, 3, 4] if quick else [2, 3, 4, 5, 6]
    p_list  = [1, 2, 4]
    n_reps  = 1 if quick else 2

    df41 = run_4_1(n_range=n_range, p_list=p_list,
                    n_repeats=n_reps, verbose=True)
    df42 = run_4_2(n_range=n_range[:3], p_best=4,
                    n_repeats=n_reps, verbose=True)
    run_4_3(p_list=[1, 2, 4], verbose=True)
    plot_tier4(df41, df42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier",  type=int, default=0,
                        help="Run only tier N (1-4). 0 = all.")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sizes / fewer repeats.")
    parser.add_argument("--full",  action="store_true",
                        help="Tier 2 only: extend n to 8 (slow).")
    args = parser.parse_args()

    tiers = [args.tier] if args.tier in [1,2,3,4] else [1,2,3,4]
    t_start = time.perf_counter()

    for tier in tiers:
        try:
            if   tier == 1: run_tier1(args.quick)
            elif tier == 2: run_tier2(args.quick, args.full)
            elif tier == 3: run_tier3(args.quick)
            elif tier == 4: run_tier4(args.quick)
        except Exception as e:
            print(f"\n  ERROR in Tier {tier}: {e}")
            traceback.print_exc()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"All done in {elapsed:.1f}s")
    print(f"Results saved to results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()