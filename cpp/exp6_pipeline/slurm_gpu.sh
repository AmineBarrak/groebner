#!/bin/bash
#====================================================================
# Slurm Job: L2 Pipeline (GPU + MPI + OpenMP) on Matilda HPC
#
#   Stage 1 (OpenMP):  CPU cores on the GPU node       [no GPU]
#   Stage 2 (CUDA):    1x Tesla V100 16GB               [GPU]
#   Stage 3 (MPI+GMP): MPI ranks across CPU cores       [no GPU]
#
# GPU nodes: 4x V100 16GB, 48 cores, 192GB RAM each
#====================================================================
#SBATCH --job-name=l2_pipeline
#SBATCH --output=l2_pipeline_%j.out
#SBATCH --error=l2_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:1

# -- Load exact Matilda modules -----------------------------------------
module purge
module load gcc/13.1.0
module load CUDA/12.1
module load OpenMPI/4.1.1c
module load cmake-gcc/3.31.7
module load gmp/6.1.2

# -- Environment --------------------------------------------------------
cd ${SLURM_SUBMIT_DIR}

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $(pwd)"
echo ""
nvidia-smi
echo ""
echo "MPI:    $(which mpirun 2>/dev/null || echo 'not found')"
echo "GCC:    $(gcc --version 2>/dev/null | head -1)"
echo "NVCC:   $(nvcc --version 2>/dev/null | tail -1)"
echo "CMake:  $(cmake --version 2>/dev/null | head -1)"
echo ""

# -- Build --------------------------------------------------------------
if [ ! -f build/l2_pipeline ]; then
    echo "=== Building ==="
    cmake -B build \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j8
    echo ""
fi

# -- Run ----------------------------------------------------------------
echo "=== Running L2 Pipeline ==="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Per Matilda docs: use mpirun WITHOUT -np flag with OpenMPI+Slurm
mpirun ./build/l2_pipeline

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
