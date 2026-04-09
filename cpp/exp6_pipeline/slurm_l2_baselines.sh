#!/bin/bash
#SBATCH --job-name=l2_baselines
#SBATCH --partition=general-long
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=l2_baselines_%j.out
#SBATCH --error=l2_baselines_%j.err

# Load required modules for Matilda cluster
module load git gcc/10.2.0

echo "======================================================================"
echo "  L2 Baselines: SymPy + Sequential Python Ansatz"
echo "======================================================================"
echo "  Partition: $SLURM_JOB_PARTITION"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Start time: $(date)"
echo "======================================================================"
echo

cd /sessions/exciting-great-ride/mnt/groebner_experiments/groebner/cpp/exp6_pipeline

# Run SymPy baseline first (with unbuffered output)
echo "--- Running SymPy Baseline (L2) ---"
python3 -u sympy_baseline_l2.py
SYMPY_EXIT=$?
echo
echo "SymPy baseline exited with code: $SYMPY_EXIT"
echo

# Run Sequential Python Ansatz
echo "--- Running Sequential Python Ansatz (L2) ---"
python3 -u sequential_ansatz_l2.py
SEQUENTIAL_EXIT=$?
echo
echo "Sequential ansatz exited with code: $SEQUENTIAL_EXIT"
echo

echo "======================================================================"
echo "  Completion: $(date)"
echo "======================================================================"
exit $(( $SYMPY_EXIT + $SEQUENTIAL_EXIT ))
