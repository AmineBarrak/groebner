#!/bin/bash
#====================================================================
# Slurm Job Script: L2 Pipeline — CPU-only fallback
#
# Use this if GPU nodes are fully occupied. The code has a CPU
# fallback for the row-reduction in Stage 2 (slower, but works).
# Stages 1 and 3 are CPU-only anyway.
#
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
module load CMake
module load OpenMPI/4.1.1c
module load GCC
module load GMP

cd ${SLURM_SUBMIT_DIR}

echo "=== CPU-only run (no GPU) ==="
echo "Node: $(hostname)"
echo ""

# Build without CUDA (need to adjust CMakeLists — or just let
# CUDA fail gracefully; the code detects dev_count == 0)
if [ ! -f build/l2_pipeline ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j8
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
mpirun ./build/l2_pipeline

echo "=== Done ==="
