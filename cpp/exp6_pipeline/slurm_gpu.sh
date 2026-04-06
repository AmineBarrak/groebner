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

# -- Load Matilda modules -----------------------------------------------
# OpenMPI/4.1.1-cuda12.1 forces gcc/10.2.0 (its build dependency).
# gcc 10.2 is fine: supports C++17 and is compatible with CUDA 12.1.
# Load order matters: CUDA/12.1 first, then the CUDA-aware OpenMPI.
module purge
module load CUDA/12.1
module load OpenMPI/4.1.1-cuda12.1
module load cmake-gcc/3.31.7
module load gmp/6.1.2

echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "nvcc path: $(which nvcc 2>/dev/null)"

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

# -- Verify modules loaded correctly -------------------------------------
NVCC_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
if [[ "$NVCC_VER" != 12.* ]]; then
    echo "ERROR: Expected CUDA 12.x but got nvcc version: $NVCC_VER"
    echo "Module loading failed. Check module dependencies."
    exit 1
fi

# -- Build (clean rebuild to avoid stale cmake cache) -------------------
echo "=== Building ==="
rm -rf build
cmake -B build \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_HOST_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
echo ""

if [ ! -f build/l2_pipeline ]; then
    echo "ERROR: Build failed, executable not found."
    exit 1
fi

# -- Run ----------------------------------------------------------------
echo "=== Running L2 Pipeline ==="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Per Matilda docs: use mpirun WITHOUT -np flag with OpenMPI+Slurm
mpirun ./build/l2_pipeline

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
