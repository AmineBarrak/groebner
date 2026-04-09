# Publication-Quality Plots for Groebner Basis Computation Paper

## Overview
A comprehensive Python script (`generate_paper_plots.py`) that generates five publication-ready plots comparing Groebner basis computation methods and analyzing computational scaling.

## Script Details
- **Location**: `/sessions/exciting-great-ride/mnt/groebner_experiments/groebner/cpp/exp6_pipeline/generate_paper_plots.py`
- **Output**: `figures/` subdirectory with 10 files (5 plots × 2 formats)
- **Formats**: PDF (vector) and PNG (raster) at 300 DPI
- **Size**: ~11 KB script, ~712 KB total output
- **Executable**: Yes (#!/usr/bin/env python3 shebang)

## Generated Plots

### Plot 1: L₃ Pivot Rate (`l3_pivot_rate.pdf/png`)
**Purpose**: Track computational progress through 521 Gaussian elimination pivots

**Features**:
- X-axis: Pivot number (0-521)
- Y-axis: Cumulative time in seconds
- Blue line: Actual measured progress (27 data points from pivots 10-270)
- Green dashed line: Reference completion time (C++ pipeline at 14,926s)
- Gray dashed line: Target milestone (521 total pivots)
- Black dotted line: Linear extrapolation to estimate total time

**Data Points**: 27 measurements from HPC run showing cumulative time progression

---

### Plot 2: L₃ Pivot Interval (`l3_pivot_interval.pdf/png`)
**Purpose**: Show variability in 10-pivot elimination block computation

**Features**:
- X-axis: Pivot block (10, 20, ..., 270)
- Y-axis: Time per 10-pivot block (seconds)
- Bar chart showing interval timing differences
- Reveals GCD effectiveness variability across elimination blocks
- Demonstrates computational bottlenecks in different problem regions

---

### Plot 3: Baseline Comparison (`baseline_comparison_l3.pdf/png`)
**Purpose**: Compare four different computational approaches for L₃

**Features**:
- Horizontal bar chart with log scale (100-1,000,000 seconds)
- Four methods:
  - **C++ Parallel Pipeline**: 14,926s (green) - fastest
  - **Sequential Python Ansatz**: 170,000s (blue, hatched) - projected estimate
  - **SymPy groebner**: >86,400s (red) - did not finish in 24 hours
  - **Modular CRT (500 primes)**: Failed (gray, hatched) - insufficient moduli
- Text annotations for each bar
- Log scale accommodates >6-order-of-magnitude performance range

---

### Plot 4: L₂ vs L₃ Scaling (`l2_vs_l3_scaling.pdf/png`)
**Purpose**: Demonstrate dramatic scaling from L₂ to L₃

**Two-subplot design**:

**Left subplot - Problem Dimensions (Linear scale)**:
- Parametric degree: L₂=4 → L₃=12 (3×)
- Monomials: L₂=47 → L₃=521 (11×)
- Matrix rows: L₂≈130 → L₃=4,374 (34×)
- Polynomial terms: L₂=34 → L₃=318 (9×)

**Right subplot - Computational Cost (Log scale)**:
- C++ pipeline time: L₂=0.0228s → L₃=14,926s (653,000× increase)
- Demonstrates why method improvements are critical at L₃

**Features**:
- Grouped bar charts comparing L₂ (blue) vs L₃ (orange)
- Value labels on all bars
- Log scale on right for timing comparison

---

### Plot 5: Coefficient Size (`coefficient_size.pdf/png`)
**Purpose**: Explain why modular CRT approach fails for larger problems

**Features**:
- X-axis: Three problem examples
- Y-axis: Maximum coefficient bit size (log scale)
- Three bars:
  - P(1,2,3) toy: ~7 bits
  - L₂ in P(2,4,6,10): ~50 bits
  - L₃ in P(2,4,6,10): 14,649 bits
- Yellow annotation box: "Why modular CRT fails for L₃"
- Demonstrates >200× increase in coefficient magnitude

---

## Technical Specifications

### Style
- **Font Family**: Serif (DejaVu Serif)
- **Font Size**: 11pt base, 12pt labels, 13pt titles
- **Grid**: None (clean academic appearance)
- **Spines**: Top and right spines removed
- **Color Scheme**: Professional palette (green, blue, orange, red)
- **Mathematical Notation**: LaTeX-style rendering ($\mathcal{L}_3$, $\mathbb{P}$)

### Output
- **DPI**: 300 for publication quality
- **Format**: 
  - PDF: Vector graphics for crisp reproduction
  - PNG: Raster for web/preview (1397×822 to 4160×1810 pixels)
- **File Sizes**: 15-31 KB (PDF), 39-184 KB (PNG)

### Dependencies
```
matplotlib>=3.10.8
numpy
```

## Usage

### Run the script
```bash
python3 generate_paper_plots.py
```

### Or execute directly
```bash
./generate_paper_plots.py
```

### Output
- Creates `figures/` subdirectory if not present
- Generates 10 files: 5 PDFs + 5 PNGs
- Prints status messages for each saved file

## Key Design Decisions

1. **Dual Format Output**: PDF for publication, PNG for web preview
2. **Log Scales**: Used where data spans 3+ orders of magnitude
3. **Hatching Patterns**: Distinguish projected vs. actual data, failed vs. completed
4. **Subplots**: L₂/L₃ scaling uses separate linear and log axes for clarity
5. **Annotations**: Direct labels on bars for easy reading
6. **Extrapolation**: Linear projection estimates total time for incomplete run

## Integration Notes

- Script sets matplotlib backend to 'Agg' for headless operation
- Garbage collection enabled after each plot to manage memory
- Clean figure closure after saving prevents memory leaks
- DPI output set to 300 for publication standards
