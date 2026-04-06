#!/bin/bash
#SBATCH --job-name=l2_baseline
#SBATCH --output=l2_baseline_%j.out
#SBATCH --error=l2_baseline_%j.err
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# Clean old logs (keep current job)
for f in l2_baseline_*.out l2_baseline_*.err; do
    [[ "$f" == *"${SLURM_JOB_ID}"* ]] && continue
    rm -f "$f"
done

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $(pwd)"
echo ""

module load python/3.10 2>/dev/null || true
pip install --user sympy 2>/dev/null || true

echo "Python: $(python3 --version 2>&1)"
echo ""

python3 sympy_baseline_hpc.py

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
