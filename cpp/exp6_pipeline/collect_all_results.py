#!/usr/bin/env python3
"""
Collect all experimental results and generate final paper data.
Run this AFTER all HPC jobs have completed.

Usage:
    python3 collect_all_results.py

It reads .out files and results/ CSVs, prints a summary, and
saves everything to results/paper_summary.csv
"""
import os, sys, re, csv, json
from datetime import timedelta

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("  COLLECTING ALL EXPERIMENTAL RESULTS")
print("=" * 70)
print()

all_results = {}

# ============================================================
# 1. L3 C++ Pipeline
# ============================================================
print("--- L3 C++ Pipeline ---")
l3_pipeline_files = [f for f in os.listdir('.') if f.startswith('l3_pipeline_') and f.endswith('.out')]
for fname in sorted(l3_pipeline_files):
    print(f"  Reading {fname}...")
    with open(fname) as f:
        content = f.read()
    # Look for total time
    m = re.search(r'TOTAL\s+[\d,]+\s+ms\s+\(([\d.]+)s\)', content)
    if not m:
        m = re.search(r'Total.*?:\s*([\d.]+)\s*ms', content)
    if m:
        t = float(m.group(1))
        # Check if it's ms or s
        if t > 100000:  # likely ms
            all_results['l3_cpp_pipeline_ms'] = t
            print(f"    Total: {t:.1f} ms ({t/1000:.1f}s)")
        else:
            all_results['l3_cpp_pipeline_ms'] = t * 1000
            print(f"    Total: {t:.1f}s ({t*1000:.1f} ms)")
    # Look for terms
    m = re.search(r'(\d+)\s+terms', content)
    if m:
        all_results['l3_terms'] = int(m.group(1))
        print(f"    Terms: {m.group(1)}")
    # Look for verification
    if 'PASS' in content:
        all_results['l3_verified'] = True
        print(f"    Verified: PASS")

# Also check results CSV
if os.path.exists(f'{RESULTS_DIR}/exp6_l3_pipeline.csv'):
    print(f"  Reading {RESULTS_DIR}/exp6_l3_pipeline.csv...")
    with open(f'{RESULTS_DIR}/exp6_l3_pipeline.csv') as f:
        for row in csv.DictReader(f):
            if row.get('stage') == 'total':
                all_results['l3_cpp_pipeline_ms'] = float(row['time_ms'])
                print(f"    CSV total: {row['time_ms']} ms")

print()

# ============================================================
# 2. L3 Sequential Python
# ============================================================
print("--- L3 Sequential Python ---")
l3_seq_files = [f for f in os.listdir('.') if f.startswith('l3_sequential_') and f.endswith('.out')]
for fname in sorted(l3_seq_files):
    print(f"  Reading {fname}...")
    with open(fname) as f:
        content = f.read()
    # Look for pivot data
    pivots = re.findall(r'pivot\s+(\d+)/521\s+\(([\d.]+)s\)', content)
    if pivots:
        last_pivot, last_time = int(pivots[-1][0]), float(pivots[-1][1])
        print(f"    Last pivot: {last_pivot}/521 at {last_time:.1f}s")
        all_results['l3_seq_last_pivot'] = last_pivot
        all_results['l3_seq_last_time_s'] = last_time
    # Check if completed
    m = re.search(r'TOTAL:\s+([\d.]+)s', content)
    if m:
        all_results['l3_seq_total_s'] = float(m.group(1))
        print(f"    COMPLETED! Total: {m.group(1)}s")
    m = re.search(r'Polynomial:\s+(\d+)\s+terms', content)
    if m:
        all_results['l3_seq_terms'] = int(m.group(1))
        print(f"    Terms: {m.group(1)}")
    if 'Verification: PASS' in content:
        all_results['l3_seq_verified'] = True
        print(f"    Verified: PASS")

# Check CSV
if os.path.exists(f'{RESULTS_DIR}/exp6_l3_sequential.csv'):
    print(f"  Reading {RESULTS_DIR}/exp6_l3_sequential.csv...")
    with open(f'{RESULTS_DIR}/exp6_l3_sequential.csv') as f:
        for row in csv.DictReader(f):
            if row.get('stage') == 'total':
                all_results['l3_seq_total_ms'] = float(row['time_ms'])
                print(f"    CSV total: {row['time_ms']} ms")

print()

# ============================================================
# 3. L3 SymPy Groebner
# ============================================================
print("--- L3 SymPy Groebner ---")
l3_base_files = [f for f in os.listdir('.') if f.startswith('l3_baseline_') and f.endswith('.out')]
for fname in sorted(l3_base_files):
    print(f"  Reading {fname}...")
    with open(fname) as f:
        content = f.read()
    if 'Running L3 Baseline' in content:
        print(f"    Found L3 baseline output")
        # Check for sympy result
        m = re.search(r'SymPy.*?Time:\s*([\d.]+)\s*ms', content)
        if m:
            all_results['l3_sympy_ms'] = float(m.group(1))
            print(f"    SymPy time: {m.group(1)} ms")
        elif 'TIMEOUT' in content:
            all_results['l3_sympy_status'] = 'TIMEOUT'
            print(f"    SymPy: TIMEOUT")
        else:
            all_results['l3_sympy_status'] = 'NO_OUTPUT'
            print(f"    SymPy: no result (likely killed by time limit)")

# Check err files for time limit
l3_base_err = [f for f in os.listdir('.') if f.startswith('l3_baseline_') and f.endswith('.err')]
for fname in sorted(l3_base_err):
    with open(fname) as f:
        content = f.read()
    if 'TIME LIMIT' in content:
        all_results['l3_sympy_status'] = 'TIMEOUT_24h'
        print(f"  {fname}: killed by TIME LIMIT")

print()

# ============================================================
# 4. L3 Modular CRT
# ============================================================
print("--- L3 Modular CRT ---")
l3_crt_files = [f for f in os.listdir('.') if f.startswith('l3_compute_') and f.endswith('.out')]
for fname in sorted(l3_crt_files):
    print(f"  Reading {fname}...")
    with open(fname) as f:
        content = f.read()
    if 'did NOT converge' in content.lower() or 'not converge' in content.lower():
        all_results['l3_crt_status'] = 'FAILED'
        print(f"    CRT: FAILED (did not converge)")
    m = re.search(r'(\d+)\s+primes', content)
    if m:
        all_results['l3_crt_primes'] = int(m.group(1))
        print(f"    Primes used: {m.group(1)}")
    m = re.search(r'max coeff.*?(\d+)\s*bits', content, re.IGNORECASE)
    if m:
        all_results['l3_crt_bits'] = int(m.group(1))
        print(f"    Max coeff bits: {m.group(1)}")

print()

# ============================================================
# 5. L2 C++ Pipeline
# ============================================================
print("--- L2 C++ Pipeline ---")
l2_pipeline_files = [f for f in os.listdir('.') if f.startswith('l2_pipeline_') and f.endswith('.out')]
if not l2_pipeline_files:
    l2_pipeline_files = [f for f in os.listdir('.') if f.startswith('l2_baseline_') and f.endswith('.out')]
for fname in sorted(l2_pipeline_files):
    print(f"  Reading {fname}...")
    with open(fname) as f:
        content = f.read()
    m = re.search(r'Total.*?:\s*([\d.]+)\s*ms', content)
    if m:
        all_results['l2_cpp_pipeline_ms'] = float(m.group(1))
        print(f"    Total: {m.group(1)} ms")
    m = re.search(r'(\d+)\s+terms', content)
    if m:
        all_results['l2_terms'] = int(m.group(1))
        print(f"    Terms: {m.group(1)}")

# Check CSV
if os.path.exists(f'{RESULTS_DIR}/exp6_l2_pipeline.csv'):
    print(f"  Reading {RESULTS_DIR}/exp6_l2_pipeline.csv...")
    with open(f'{RESULTS_DIR}/exp6_l2_pipeline.csv') as f:
        for row in csv.DictReader(f):
            if row.get('stage') == 'total':
                all_results['l2_cpp_pipeline_ms'] = float(row['time_ms'])
                print(f"    CSV total: {row['time_ms']} ms")

print()

# ============================================================
# 6. L2 Baselines
# ============================================================
print("--- L2 Baselines ---")
l2_base_files = [f for f in os.listdir('.') if f.startswith('l2_baselines_') and f.endswith('.out')]
for fname in sorted(l2_base_files):
    print(f"  Reading {fname}...")
    with open(fname) as f:
        content = f.read()
    # SymPy result
    m = re.search(r'SymPy groebner.*?Time:\s*([\d.]+)\s*ms', content, re.DOTALL)
    if m:
        all_results['l2_sympy_ms'] = float(m.group(1))
        print(f"    SymPy time: {m.group(1)} ms")
    elif 'TIMEOUT' in content and 'SymPy' in content:
        all_results['l2_sympy_status'] = 'TIMEOUT'
        print(f"    SymPy: TIMEOUT")
    # Sequential result
    m = re.search(r'Sequential.*?[Tt]otal.*?:\s*([\d.]+)\s*ms', content, re.DOTALL)
    if not m:
        m = re.search(r'TOTAL:\s+([\d.]+)s', content)
        if m:
            all_results['l2_seq_total_ms'] = float(m.group(1)) * 1000
            print(f"    Sequential total: {float(m.group(1))*1000:.1f} ms")
    else:
        all_results['l2_seq_total_ms'] = float(m.group(1))
        print(f"    Sequential total: {m.group(1)} ms")

# Check CSVs
for csv_name in ['exp6_l2_sequential.csv', 'exp6_l2_baseline.csv', 'exp6_l2_sympy.csv']:
    path = f'{RESULTS_DIR}/{csv_name}'
    if os.path.exists(path):
        print(f"  Reading {path}...")
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(f"    {row}")

print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print()

print("  L2 in P(2,4,6,10) — degree 30, expected 34 terms:")
print(f"    C++ pipeline:     {all_results.get('l2_cpp_pipeline_ms', 'NOT FOUND')} ms")
print(f"    Sequential Python: {all_results.get('l2_seq_total_ms', 'NOT FOUND')} ms")
print(f"    SymPy groebner:   {all_results.get('l2_sympy_ms', all_results.get('l2_sympy_status', 'NOT FOUND'))} ms")
print(f"    Terms:            {all_results.get('l2_terms', 'NOT FOUND')}")
print()

print("  L3 in P(2,4,6,10) — degree 80, expected 318 terms:")
cpp_ms = all_results.get('l3_cpp_pipeline_ms', 'NOT FOUND')
print(f"    C++ pipeline:     {cpp_ms} ms", end='')
if isinstance(cpp_ms, (int, float)):
    print(f" ({cpp_ms/1000:.1f}s, {cpp_ms/3600000:.2f}h)")
else:
    print()

seq_s = all_results.get('l3_seq_total_s')
if seq_s:
    print(f"    Sequential Python: {seq_s:.1f}s ({seq_s/3600:.1f}h) — COMPLETED")
else:
    lp = all_results.get('l3_seq_last_pivot', '?')
    lt = all_results.get('l3_seq_last_time_s', '?')
    print(f"    Sequential Python: pivot {lp}/521 at {lt}s — IN PROGRESS")

print(f"    SymPy groebner:   {all_results.get('l3_sympy_status', 'NOT FOUND')}")
print(f"    Modular CRT:      {all_results.get('l3_crt_status', 'NOT FOUND')}")
print(f"    Terms:            {all_results.get('l3_terms', 'NOT FOUND')}")
print(f"    Verified:         {all_results.get('l3_verified', 'NOT FOUND')}")
print()

# Speedup calculation
if isinstance(cpp_ms, (int, float)) and seq_s:
    speedup = (seq_s * 1000) / cpp_ms
    print(f"  L3 Speedup (C++ vs Sequential Python): {speedup:.1f}x")
elif isinstance(cpp_ms, (int, float)) and all_results.get('l3_seq_last_time_s'):
    lt = all_results['l3_seq_last_time_s']
    lp = all_results['l3_seq_last_pivot']
    # Simple linear extrapolation
    est_total = lt * 521 / lp
    est_speedup = (est_total * 1000) / cpp_ms
    print(f"  L3 Estimated speedup: ~{est_speedup:.1f}x (extrapolated from {lp}/521 pivots)")

print()

# Save all results
summary_path = f'{RESULTS_DIR}/paper_all_results.json'
with open(summary_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"  Saved all results to {summary_path}")

# Save CSV summary
csv_path = f'{RESULTS_DIR}/paper_summary.csv'
with open(csv_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(['example', 'method', 'time_ms', 'time_s', 'terms', 'status'])
    # L2
    w.writerow(['L2', 'cpp_pipeline', all_results.get('l2_cpp_pipeline_ms', ''),
                all_results.get('l2_cpp_pipeline_ms', 0)/1000 if isinstance(all_results.get('l2_cpp_pipeline_ms'), (int,float)) else '',
                all_results.get('l2_terms', ''), 'completed'])
    w.writerow(['L2', 'sequential_python', all_results.get('l2_seq_total_ms', ''), '', '',
                'completed' if all_results.get('l2_seq_total_ms') else 'pending'])
    w.writerow(['L2', 'sympy_groebner', all_results.get('l2_sympy_ms', ''), '', '',
                all_results.get('l2_sympy_status', 'pending')])
    # L3
    w.writerow(['L3', 'cpp_pipeline', all_results.get('l3_cpp_pipeline_ms', ''),
                all_results.get('l3_cpp_pipeline_ms', 0)/1000 if isinstance(all_results.get('l3_cpp_pipeline_ms'), (int,float)) else '',
                all_results.get('l3_terms', ''), 'completed'])
    w.writerow(['L3', 'sequential_python', all_results.get('l3_seq_total_ms', ''),
                all_results.get('l3_seq_total_s', ''),
                all_results.get('l3_seq_terms', ''),
                'completed' if all_results.get('l3_seq_total_s') else 'in_progress'])
    w.writerow(['L3', 'sympy_groebner', '', '', '', all_results.get('l3_sympy_status', '')])
    w.writerow(['L3', 'modular_crt', '', '', '', all_results.get('l3_crt_status', '')])
print(f"  Saved CSV to {csv_path}")
print()
print("  Done!")
