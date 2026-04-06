#!/bin/bash
#====================================================================
# Slurm Job Script: L2 Pipeline (GPU + MPI + OpenMP)
#
# Runs the full three-stage pipeline on a GPU node.
#   Stage 1 (OpenMP):      uses CPU cores on the GPU node
#   Stage 2 (CUDA):        uses 1 V100 GPU
#   Stage 3 (MPI + GMP):   uses MPI ranks across CPU cores
#
# GPU nodes: 4x V100 16GB, 48 cores, 192GB RAM each
# Available: hpc-gpu-p01, hpc-gpu-p02, hpc-gpu-p03
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

# ── Load modules ───────────────────────────────────────────────────
# Adjust module names to what's available on Matilda.
# Run 'module avail' to see exact versions.
module purge
module load CMake
module load CUDA
module load OpenMPI/4.1.1c
module load GCC
module load GMP              # if available as module; see notes below

# ── Enter working directory ────────────────────────────────────────
cd ${SLURM_SUBMIT_DIR}

# ── Report environment ─────────────────────────────────────────────
echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $(pwd)"
echo ""
nvidia-smi
echo ""
echo "MPI:       $(which mpirun)"
echo "GCC:       $(gcc --version | head -1)"
echo "CUDA:      $(nvcc --version | tail -1)"
echo ""

# ── Build (if not already built) ──────────────────────────────────
if [ ! -f build/l2_pipeline ]; then
    echo "=== Building ==="
    cmake -B build \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j8
    echo ""
fi

# ── Run ────────────────────────────────────────────────────────────
echo "=== Running L2 Pipeline ==="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Use mpirun WITHOUT -np flag (OpenMPI+Slurm handles it via --ntasks)
mpirun ./build/l2_pipeline

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
