#!/usr/bin/env python3
"""Generate publication-quality plots for the Groebner basis paper."""
import os, sys, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'text.usetex': False,
    'savefig.dpi': 300, 'figure.dpi': 100,
})

FDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FDIR, exist_ok=True)

# === DATA ===
PIVOTS = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,
          150,160,170,180,190,200,210,220,230,240,250,260,270]
TIMES  = [32.2,75.5,261.5,986.0,2302.0,3978.4,5340.0,6015.3,
          6862.1,8380.5,10821.0,13861.1,16686.7,19921.6,23794.8,
          26183.3,29674.7,33549.4,39953.7,47646.6,55813.5,63235.3,
          66339.6,69163.2,72608.6,76688.5,81171.2]
CPP_TIME = 14926.0

def save(fig, name):
    for ext in ['pdf','png']:
        p = os.path.join(FDIR, f'{name}.{ext}')
        fig.savefig(p, format=ext, dpi=300, bbox_inches='tight')
        print(f'  Saved {p}')
    plt.close(fig); gc.collect()

def plot1():
    """L3 pivot progress"""
    print('Plot 1: L3 pivot rate...')
    pv, tm = np.array(PIVOTS), np.array(TIMES)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(pv, tm, 'o-', color='#1f77b4', lw=2, ms=4, label='Sequential Python progress')
    ax.axhline(y=CPP_TIME, color='#2ca02c', ls='--', lw=1.8, label=f'C++ pipeline: all 521 pivots in {CPP_TIME:.0f}s')
    ax.axvline(x=521, color='gray', ls=':', lw=1.2, alpha=0.6)
    ax.text(521, max(tm)*0.05, ' 521', color='gray', fontsize=9)
    # extrapolation
    z = np.polyfit(pv[-5:], tm[-5:], 1)
    xp = np.linspace(270, 521, 50)
    ax.plot(xp, np.polyval(z, xp), 'k:', lw=1.2, alpha=0.5, label='Linear extrapolation')
    ax.set_xlabel('Pivot number'); ax.set_ylabel('Cumulative time (s)')
    ax.set_title(r'Sequential Python Ansatz: Pivot Progress for $\mathcal{L}_3$')
    ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)
    ax.set_xlim(0, 550)
    save(fig, 'l3_pivot_rate')

def plot2():
    """Per-block interval times"""
    print('Plot 2: L3 pivot intervals...')
    tm = np.array(TIMES)
    intervals = np.diff(tm, prepend=0)
    pv = np.array(PIVOTS)
    fig, ax = plt.subplots(figsize=(9,5))
    colors = ['#d62728' if v > 5000 else '#ff7f0e' if v > 2000 else '#1f77b4' for v in intervals]
    ax.bar(pv, intervals, width=8, color=colors, edgecolor='black', lw=0.5)
    ax.set_xlabel('Pivot block'); ax.set_ylabel('Time for 10 pivots (s)')
    ax.set_title(r'Per-Block Elimination Time for $\mathcal{L}_3$')
    ax.set_xticks(pv[::3])
    save(fig, 'l3_pivot_interval')

def plot3():
    """Baseline comparison bar chart"""
    print('Plot 3: L3 baseline comparison...')
    fig, ax = plt.subplots(figsize=(9,4.5))
    methods = ['C++ Parallel Pipeline\n(8 threads, GMP, OpenMP)',
               'Sequential Python Ansatz\n(projected)',
               'SymPy groebner\n(> 24h, no output)',
               'Modular CRT\n(500 primes, failed)']
    times = [14926, 170000, 86400, 86400]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#7f7f7f']
    hatches = ['', '///', '', 'xx']
    bars = ax.barh(range(4), times, color=colors, edgecolor='black', lw=0.8)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    # annotations
    ax.text(14926*1.3, 0, f'{14926:,}s (4.15h)', va='center', fontsize=10, fontweight='bold')
    ax.text(170000*1.1, 1, r'~170,000s (~47h, projected)', va='center', fontsize=10)
    ax.text(86400*1.3, 2, 'DNF: killed at 24h', va='center', fontsize=10, color='#d62728')
    ax.text(86400*1.3, 3, 'FAILED: coefficients ~$2^{14649}$', va='center', fontsize=10, color='#7f7f7f')
    ax.set_xscale('log')
    ax.set_yticks(range(4)); ax.set_yticklabels(methods)
    ax.set_xlabel('Time (seconds, log scale)')
    ax.set_title(r'$\mathcal{L}_3$ in $\mathbb{P}(2,4,6,10)$: Method Comparison')
    ax.set_xlim(1000, 1e6)
    fig.tight_layout()
    save(fig, 'baseline_comparison_l3')

def plot4():
    """L2 vs L3 scaling"""
    print('Plot 4: L2 vs L3 scaling...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Problem dimensions
    cats = ['Param.\ndegree', 'Monomials', 'Matrix\nrows', 'Polynomial\nterms']
    l2_vals = [4, 47, 130, 34]
    l3_vals = [12, 521, 4374, 318]
    x = np.arange(len(cats))
    w = 0.35
    ax1.bar(x - w/2, l2_vals, w, label=r'$\mathcal{L}_2$', color='#1f77b4', edgecolor='black', lw=0.5)
    ax1.bar(x + w/2, l3_vals, w, label=r'$\mathcal{L}_3$', color='#ff7f0e', edgecolor='black', lw=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(cats)
    ax1.set_ylabel('Count'); ax1.set_title('Problem Dimensions')
    ax1.legend(frameon=True, edgecolor='black')
    # Timing
    methods = ['C++ Pipeline']
    l2_t = [0.0228]
    l3_t = [14926]
    x2 = np.arange(1)
    ax2.bar(x2 - w/2, l2_t, w, label=r'$\mathcal{L}_2$', color='#1f77b4', edgecolor='black', lw=0.5)
    ax2.bar(x2 + w/2, l3_t, w, label=r'$\mathcal{L}_3$', color='#ff7f0e', edgecolor='black', lw=0.5)
    ax2.set_yscale('log')
    ax2.set_xticks(x2); ax2.set_xticklabels(methods)
    ax2.set_ylabel('Time (seconds, log scale)'); ax2.set_title('Pipeline Runtime')
    ax2.legend(frameon=True, edgecolor='black')
    ax2.text(-0.175, 0.0228*2, '22.8 ms', fontsize=9, ha='center')
    ax2.text(0.175, 14926*2, '14,926 s', fontsize=9, ha='center')
    fig.suptitle(r'Scaling from $\mathcal{L}_2$ to $\mathcal{L}_3$ in $\mathbb{P}(2,4,6,10)$', fontsize=14)
    fig.tight_layout()
    save(fig, 'l2_vs_l3_scaling')

def plot5():
    """Coefficient size comparison"""
    print('Plot 5: Coefficient sizes...')
    fig, ax = plt.subplots(figsize=(7, 5))
    examples = [r'$\mathbb{P}(1,2,3)$'+'\ntoy case',
                r'$\mathcal{L}_2$ in $\mathbb{P}(2,4,6,10)$',
                r'$\mathcal{L}_3$ in $\mathbb{P}(2,4,6,10)$']
    bits = [7, 50, 14649]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    bars = ax.bar(range(3), bits, color=colors, edgecolor='black', lw=0.8)
    ax.set_yscale('log')
    ax.set_xticks(range(3)); ax.set_xticklabels(examples)
    ax.set_ylabel('Max coefficient size (bits, log scale)')
    ax.set_title('Maximum Coefficient Size by Example')
    for i, (b, v) in enumerate(zip(bars, bits)):
        ax.text(i, v*1.5, f'{v:,} bits', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=15000, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(2.4, 15000, 'CRT limit\n(500 primes)', fontsize=8, color='gray', va='center')
    save(fig, 'coefficient_size')

if __name__ == '__main__':
    print('Generating paper plots...')
    plot1(); plot2(); plot3(); plot4(); plot5()
    print(f'\nAll plots saved to {FDIR}/')
