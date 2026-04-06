#!/bin/bash
#====================================================================
# Slurm Job: L2 Pipeline -- CPU-only fallback
# Use if all GPU nodes are busy. Stage 2 uses CPU row reduction.
# Standard compute nodes: 40 cores, 192GB RAM
#====================================================================
#SBATCH --job-name=l2_cpu
#SBATCH --output=l2_cpu_%j.out
#SBATCH --error=l2_cpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00

module purge
module load gcc/13.1.0
module load CUDA/12.1
module load OpenMPI/4.1.1c
module load cmake-gcc/3.31.7
module load gmp/6.1.2

cd ${SLURM_SUBMIT_DIR}

echo "=== CPU-only run (no GPU requested) ==="
echo "Node: $(hostname), Date: $(date)"
echo ""

if [ ! -f build/l2_pipeline ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j8
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mpirun ./build/l2_pipeline

echo "=== Done: $(date) ==="
