# Weighted Projective Gröbner Basis — Experiment Suite

Full source code for the experimental validation of the parallel pipeline
described in Shaska (2026). Every experiment has its own file and can be
run independently.

---

## Machine requirements

Your machine:  32 GB RAM · 4 GHz CPU · NVIDIA RTX 40xx (6 GB VRAM)

| Experiment | Needs GPU | Min RAM | Est. time |
|---|---|---|---|
| Tier 1 (1.1, 1.2) | No | 2 GB | < 1 min |
| Tier 2 (2.1) | No | 4 GB | 2–5 min |
| Tier 2 (2.2, n≤6) | No | 4 GB | 5–10 min |
| Tier 2 (2.2, n=7,8) | No | 8 GB | 30–90 min |
| Tier 3 Python (3.1, 3.4, 3.5) | No | 4 GB | 5–15 min |
| Tier 3 C++ OpenMP (3.1) | No | 4 GB | 2–5 min |
| Tier 3 CUDA (3.2) | **Yes** | 8 GB | 5–20 min |
| Tier 4 (4.1, 4.2, 4.3) | No | 8 GB | 10–20 min |

---

## Project structure

```
groebner_experiments/
├── requirements.txt
├── results/                    ← all outputs land here
│
├── python/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── weighted_projective.py
│   │   ├── homogenization.py
│   │   ├── groebner.py
│   │   ├── weighted_gcd.py
│   │   └── pipeline.py
│   │
│   ├── exp1_1_toy_cases.py         ← Tier 1
│   ├── exp1_2_homogeneity_check.py ← Tier 1
│   ├── exp2_1_phase_timing.py      ← Tier 2
│   ├── exp2_2_complexity_scaling.py← Tier 2
│   ├── exp3_1_hom_speedup.py       ← Tier 3 Python
│   ├── exp3_4_gcd_scaling.py       ← Tier 3 Python
│   ├── exp3_5_load_imbalance.py    ← Tier 3 Python
│   ├── exp4_full_pipeline.py       ← Tier 4
│   ├── plot_cpp_results.py         ← plots for C++ outputs
│   └── run_all.py                  ← master runner
│
└── cpp/
    ├── polynomial.h                ← shared polynomial type
    ├── exp3_1_openmp/
    │   ├── CMakeLists.txt
    │   └── hom_speedup.cpp         ← Exp 3.1 OpenMP
    └── exp3_2_cuda/
        ├── CMakeLists.txt
        └── f4_cuda.cu              ← Exp 3.2 CUDA + cuSPARSE
```

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Verify CUDA toolkit (for Exp 3.2 only)
nvcc --version        # should show ≥ 11.8
nvidia-smi            # confirm RTX 40xx is visible
```

---

## Running Python experiments

```bash
cd python

# All tiers (recommended first run, ~30 min on your machine)
python run_all.py

# Quick sanity check (small sizes, fast)
python run_all.py --quick

# One tier at a time
python run_all.py --tier 1
python run_all.py --tier 2
python run_all.py --tier 3
python run_all.py --tier 4

# Tier 2 extended to n=8 (slow — 30-90 min)
python run_all.py --tier 2 --full

# Individual experiments
python exp1_1_toy_cases.py
python exp1_2_homogeneity_check.py
python exp2_1_phase_timing.py
python exp2_2_complexity_scaling.py
python exp3_1_hom_speedup.py
python exp3_4_gcd_scaling.py
python exp3_5_load_imbalance.py
python exp4_full_pipeline.py
```

All outputs (CSV + PNG) go to `results/`.

---

## Running C++ experiments

### Experiment 3.1 — OpenMP homogenization speedup

```bash
# Install OpenMP (usually already present with g++)
sudo apt-get install libomp-dev    # Ubuntu/Debian

cd cpp/exp3_1_openmp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

# Run (uses all available cores by default)
./build/hom_speedup

# Outputs: ../../results/exp3_1_cpp.csv
```

Expected output on your 4 GHz machine with 8+ cores:
```
n=320  p=1   time ~800 ms
n=320  p=4   time ~220 ms   S ≈ 3.6×
n=320  p=8   time ~115 ms   S ≈ 7.0×
```

### Experiment 3.2 — CUDA F4 elimination speedup

```bash
# Prerequisites
sudo apt-get install nvidia-cuda-toolkit    # or use your CUDA install path

cd cpp/exp3_2_cuda
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

# Run
./build/f4_cuda

# Outputs: ../../results/exp3_2_cuda.csv
```

Expected output on RTX 40xx:
```
n=6  cpu_gauss ~400 ms  gpu_gauss ~12 ms  speedup ~33×
n=7  cpu_gauss ~3s      gpu_gauss ~80 ms  speedup ~37×
n=8  cpu_gauss ~25s     gpu_gauss ~500 ms speedup ~50×
```

Note: for n ≥ 8, the dense matrix may exceed 4 GB. The code
automatically skips the GPU step and prints a warning. Use n ≤ 7
for GPU benchmarking on the 6 GB VRAM card.

### Generate combined figure from C++ results

```bash
cd python
python plot_cpp_results.py
# Outputs: results/exp3_1_cpp_speedup.png
#          results/exp3_2_cuda_speedup.png
#          results/exp3_combined_cpp.png
```

---

## Output files

| File | Experiment | Description |
|---|---|---|
| `results/exp1_1.json` | 1.1 | Toy case pass/fail |
| `results/exp1_2.json` | 1.2 | Homogeneity check |
| `results/exp2_1.csv` + `.png` | 2.1 | Phase timing breakdown |
| `results/exp2_2.csv` + `.png` | 2.2 | Complexity scaling |
| `results/exp3_1.csv` + `.png` | 3.1 Python | Hom. speedup (IPC) |
| `results/exp3_1_cpp.csv` + `.png` | 3.1 C++ | Hom. speedup (OpenMP) |
| `results/exp3_2_cuda.csv` + `.png` | 3.2 C++ | F4 speedup (CUDA) |
| `results/exp3_4.csv` + `.png` | 3.4 | GCD scaling |
| `results/exp3_5.csv` + `.png` | 3.5 | Load imbalance |
| `results/exp4_1.csv` | 4.1 | Full pipeline scaling |
| `results/exp4_2.csv` | 4.2 | vs raw sympy |
| `results/exp4_3_genus2.json` | 4.3 | Genus-2 curve |
| `results/exp4_full_pipeline.png` | 4.1-4.3 | Combined figure |
| `results/exp3_combined_cpp.png` | 3.1+3.2+3.4 | Combined C++ figure |

---

## Which results are valid now vs need running

| Experiment | Status |
|---|---|
| 1.1, 1.2 | **Valid** — mathematical facts, hardware-independent |
| 2.1 | **Valid** (ratios) — re-run on your machine for absolute times |
| 2.2 n≤5 | **Valid** — re-run to confirm |
| 2.2 n=6..8 | **Run now** — not yet measured |
| 3.1 Python | Re-run — shows IPC overhead, not real speedup |
| 3.1 C++ | **Run now** — first real OpenMP speedup measurement |
| 3.2 CUDA | **Run now** — GPU speedup, requires RTX 40xx |
| 3.4 | **Valid** — re-run to extend to 256-bit |
| 3.5 | Re-run — new dynamic scheduling implementation |
| 4.1, 4.2 | Re-run on your machine |
| 4.3 | **Valid** (correctness) — re-run for updated timing |
