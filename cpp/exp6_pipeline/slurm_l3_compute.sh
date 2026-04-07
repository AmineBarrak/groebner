#!/bin/bash
#SBATCH --job-name=l3_compute
#SBATCH --output=l3_compute_%j.out
#SBATCH --error=l3_compute_%j.err
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# This script computes the L3 polynomial coefficients using modular CRT.
# The result is saved to results/l3_polynomial.json
# Run this BEFORE the baseline or pipeline to verify the degree-80 result.

# Clean old logs (keep current job)
for f in l3_compute_*.out l3_compute_*.err; do
    [[ "$f" == *"${SLURM_JOB_ID}"* ]] && continue
    rm -f "$f"
done

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $(pwd)"
echo ""

# Load Python
for mod in python/3.12 python/3.11 python/3.10 python/3.9 python/3.8 python3; do
    module load "$mod" 2>/dev/null && break
done

echo "Python: $(python3 --version 2>&1)"
echo ""

mkdir -p results

echo "=== Computing L3 Polynomial (Modular CRT) ==="
python3 l3_compute_hpc.py

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
