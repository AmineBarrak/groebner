#!/bin/bash
#SBATCH --job-name=l3_baseline
#SBATCH --output=l3_baseline_%j.out
#SBATCH --error=l3_baseline_%j.err
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Clean old logs (keep current job)
for f in l3_baseline_*.out l3_baseline_*.err; do
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
pip install --user sympy 2>/dev/null || pip3 install --user sympy 2>/dev/null || true

echo "Python: $(python3 --version 2>&1)"
python3 -c "import sympy; print('SymPy:', sympy.__version__)" 2>&1 || echo "SymPy: NOT FOUND"
echo ""

echo "=== Running L3 Baseline ==="
python3 sympy_baseline_l3.py

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
