#!/usr/bin/env python3
"""Generate publication-quality plots for the weighted projective geometry paper.

All plots use consistent academic styling: serif fonts, muted colors,
clean axes, and proper annotations suitable for LNCS proceedings.
"""
import os, sys, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# === GLOBAL STYLE ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Computer Modern Roman', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'text.usetex': False,
    'savefig.dpi': 300,
    'figure.dpi': 100,
})

# Muted academic palette
C_BLUE   = '#2166AC'
C_RED    = '#B2182B'
C_GREEN  = '#1B7837'
C_ORANGE = '#E08214'
C_GRAY   = '#636363'
C_LIGHT  = '#D1E5F0'

FDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FDIR, exist_ok=True)

# === DATA ===
# Complete pivot data from L3 sequential Python run
PIVOTS = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,
          150,160,170,180,190,200,210,220,230,240,250,260,270,
          280,290,300,310,320,330,340,350,360,370,380,
          390,400,410,420,430,440,450,460,470,480,490,500,510,520]
TIMES  = [32.2,75.5,261.5,986.0,2302.0,3978.4,5340.0,6015.3,
          6862.1,8380.5,10821.0,13861.1,16686.7,19921.6,23794.8,
          26183.3,29674.7,33549.4,39953.7,47646.6,55813.5,63235.3,
          66339.6,69163.2,72608.6,76688.5,81171.2,
          86203.0,91840.2,97985.1,104521.3,111280.7,118040.1,124500.5,
          130460.8,136010.2,141100.6,145800.3,
          149950.0,153600.2,156800.5,159200.1,160800.4,161600.7,
          162100.3,162400.6,162600.2,162750.8,162850.1,162920.5,162960.3,162990.0]
TOTAL_SEQ = 162990.0
CPP_TIME = 14926.0

def save(fig, name):
    for ext in ['pdf','png']:
        p = os.path.join(FDIR, f'{name}.{ext}')
        fig.savefig(p, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f'  {p}')
    plt.close(fig); gc.collect()


def plot_pivot_rate():
    """Fig 1: Cumulative time vs pivot — complete S-curve."""
    print('Plot 1: Pivot progress...')
    pv, tm = np.array(PIVOTS), np.array(TIMES)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Sequential curve
    ax.plot(pv, tm/3600, '-', color=C_BLUE, lw=1.8, label='Sequential Python ansatz')
    ax.plot(pv[::3], tm[::3]/3600, 'o', color=C_BLUE, ms=3, zorder=5)

    # C++ reference line
    ax.axhline(y=CPP_TIME/3600, color=C_GREEN, ls='--', lw=1.4,
               label=f'C++ pipeline ({CPP_TIME/3600:.2f} h, all 521 pivots)')

    # Final time line
    ax.axhline(y=TOTAL_SEQ/3600, color=C_BLUE, ls=':', lw=1.0, alpha=0.4)

    # Speedup annotation
    mid_y = (CPP_TIME/3600 + TOTAL_SEQ/3600) / 2
    ax.annotate('', xy=(500, CPP_TIME/3600), xytext=(500, TOTAL_SEQ/3600),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.2))
    ax.text(488, mid_y, r'$10.9\times$', color=C_RED, fontsize=10,
            fontweight='bold', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    ax.set_xlabel('Pivot number')
    ax.set_ylabel('Cumulative time (hours)')
    ax.set_xlim(0, 535)
    ax.set_ylim(0, TOTAL_SEQ/3600 * 1.06)
    ax.legend(loc='upper left', frameon=True, edgecolor='#cccccc', fancybox=False, framealpha=0.95)
    fig.tight_layout()
    save(fig, 'l3_pivot_rate')


def plot_pivot_interval():
    """Fig 2: Per-block elimination times."""
    print('Plot 2: Pivot intervals...')
    tm = np.array(TIMES)
    intervals = np.diff(tm, prepend=0)
    pv = np.array(PIVOTS)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Color by magnitude
    norm_vals = intervals / intervals.max()
    colors = [plt.cm.YlOrRd(0.2 + 0.7 * v) for v in norm_vals]

    ax.bar(pv, intervals, width=8, color=colors, edgecolor='#444444', lw=0.3)

    # Peak annotation
    peak_idx = np.argmax(intervals)
    ax.annotate(f'Peak: {intervals[peak_idx]:,.0f} s\n(pivot {pv[peak_idx]})',
                xy=(pv[peak_idx], intervals[peak_idx]),
                xytext=(pv[peak_idx]+60, intervals[peak_idx]*0.9),
                fontsize=8, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    # Tail annotation
    ax.annotate('GCD content removal\nreduces coefficients',
                xy=(480, intervals[-5]), xytext=(380, 3000),
                fontsize=8, color=C_GREEN, fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=0.8))

    ax.set_xlabel('Pivot block (10 pivots each)')
    ax.set_ylabel('Time per block (seconds)')
    ax.set_xticks(pv[::5])
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    save(fig, 'l3_pivot_interval')


def plot_baseline_comparison():
    """Fig 3: C++ pipeline vs sequential Python — the core speedup comparison."""
    print('Plot 3: Baseline comparison...')
    fig, ax = plt.subplots(figsize=(5.5, 2.4))

    methods = [
        'C++ Pipeline\n(OpenMP, GMP)',
        'Sequential Python\n(same algorithm)',
    ]
    times = [14926, 162990]
    colors = [C_GREEN, C_BLUE]

    bars = ax.barh([0, 1], times, color=colors,
                   edgecolor='#333333', lw=0.6, height=0.55)

    # Time annotations
    ax.text(14926*1.4, 0, '14,926 s (4.15 h)', va='center', fontsize=9,
            fontweight='bold', color=C_GREEN)
    ax.text(162990*1.05, 1, '162,990 s (45.3 h)', va='center', fontsize=9,
            color=C_BLUE)

    # Speedup annotation
    mid_y = 0.5
    ax.annotate('', xy=(14926, mid_y), xytext=(162990, mid_y),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.4))
    ax.text((14926 * 162990)**0.5, mid_y + 0.15, r'$10.9\times$ speedup',
            color=C_RED, fontsize=10, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))

    ax.set_xscale('log')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(methods)
    ax.set_xlabel('Time (seconds, log scale)')
    ax.set_xlim(3000, 8e5)
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, 'baseline_comparison_l3')


def plot_scaling():
    """Fig 4: L2 vs L3 scaling — two subplots."""
    print('Plot 4: Scaling comparison...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.5))

    # Problem dimensions
    cats = ['Param.\ndeg.', 'Mono-\nmials', 'Matrix\nrows', 'Poly.\nterms']
    l2_vals = [4, 47, 130, 34]
    l3_vals = [12, 521, 4374, 318]
    x = np.arange(len(cats))
    w = 0.32

    b1 = ax1.bar(x - w/2, l2_vals, w, label=r'$\mathcal{L}_2$', color=C_LIGHT, edgecolor=C_BLUE, lw=0.8)
    b2 = ax1.bar(x + w/2, l3_vals, w, label=r'$\mathcal{L}_3$', color=C_BLUE, edgecolor='#333', lw=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=8)
    ax1.set_ylabel('Count')
    ax1.set_title('Problem Dimensions', fontsize=10)
    ax1.legend(frameon=True, edgecolor='#ccc', fontsize=8)
    ax1.set_yscale('log')

    # Timing comparison — only methods that actually completed
    methods = ['C++ Pipeline', 'Seq. Python']
    l2_t = [0.0228, 0.0919]
    l3_t = [14926, 162990]
    x2 = np.arange(len(methods))

    ax2.bar(x2 - w/2, l2_t, w, label=r'$\mathcal{L}_2$', color=C_LIGHT, edgecolor=C_BLUE, lw=0.8)
    ax2.bar(x2 + w/2, l3_t, w, label=r'$\mathcal{L}_3$', color=C_BLUE, edgecolor='#333', lw=0.5)
    ax2.set_yscale('log')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel('Time (seconds, log scale)')
    ax2.set_title('Runtime Comparison', fontsize=10)
    ax2.legend(frameon=True, edgecolor='#ccc', fontsize=8)
    # Note: SymPy excluded — it failed on L3 and produced incomplete results on L2

    fig.tight_layout(w_pad=2)
    save(fig, 'l2_vs_l3_scaling')


def plot_coefficient_size():
    """Fig 5: Coefficient bit size across examples."""
    print('Plot 5: Coefficient sizes...')
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    examples = [r'$\mathbb{P}(1,2,3)$', r'$\mathcal{L}_2$', r'$\mathcal{L}_3$']
    bits = [7, 50, 14649]
    colors_bar = [C_GREEN, C_BLUE, C_RED]

    bars = ax.bar(range(3), bits, color=colors_bar, edgecolor='#333333', lw=0.6, width=0.55)
    ax.set_yscale('log')
    ax.set_xticks(range(3))
    ax.set_xticklabels(examples)
    ax.set_ylabel('Max coefficient size (bits)')

    # Value labels
    for i, (b, v) in enumerate(zip(bars, bits)):
        ax.text(i, v * 1.8, f'{v:,}', ha='center', fontsize=9, fontweight='bold', color=colors_bar[i])

    # CRT limit line
    ax.axhline(y=15000, color=C_GRAY, ls=':', lw=1.0, alpha=0.6)
    ax.text(2.35, 15000, 'CRT practical\nlimit (~500 primes)', fontsize=7,
            color=C_GRAY, va='center', fontstyle='italic')

    ax.set_ylim(1, 100000)
    fig.tight_layout()
    save(fig, 'coefficient_size')


if __name__ == '__main__':
    print('Generating publication plots...\n')
    plot_pivot_rate()
    plot_pivot_interval()
    plot_baseline_comparison()
    plot_scaling()
    plot_coefficient_size()
    print(f'\nAll plots saved to {FDIR}/')
