#!/bin/bash
#SBATCH --job-name=l3_seqpy
#SBATCH --output=l3_sequential_%j.out
#SBATCH --error=l3_sequential_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general-long

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $HOSTNAME"
echo "Date:      $(date)"
echo "Directory: $(pwd)"
echo ""

module load git
module load gcc/10.2.0

pip install --user sympy 2>/dev/null

echo "Python: $(python3 --version)"
echo ""
echo "=== Running Sequential Python Ansatz for L3 ==="
echo ""

python3 -u sequential_ansatz_l3.py
