#!/bin/bash
#SBATCH --job-name=l3_pipeline
#SBATCH --output=l3_pipeline_%j.out
#SBATCH --error=l3_pipeline_%j.err
#SBATCH --partition=general-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# Clean old logs (keep current job)
for f in l3_pipeline_*.out l3_pipeline_*.err; do
    [[ "$f" == *"${SLURM_JOB_ID}"* ]] && continue
    rm -f "$f"
done

echo "=== Job Info ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Directory: $(pwd)"
echo ""

# Load modules
module load gcc/10.2.0 2>/dev/null || module load gcc 2>/dev/null
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null
module load openmpi/4.1.1 2>/dev/null || module load openmpi 2>/dev/null
module load cmake 2>/dev/null

echo "GCC:    $(gcc --version 2>&1 | head -1)"
echo "nvcc:   $(nvcc --version 2>&1 | tail -1)"
echo "MPI:    $(mpirun --version 2>&1 | head -1)"
echo ""

# Build L3 pipeline
echo "=== Building L3 Pipeline ==="
# Check if CMakeLists has L3 target, otherwise build directly
if grep -q "l3_pipeline" CMakeLists.txt 2>/dev/null; then
    cmake -B build -DL3=ON && cmake --build build -j8
else
    # Direct build
    mkdir -p build
    nvcc -O2 -std=c++17 -arch=sm_70 \
        -Xcompiler "-fopenmp" \
        l3_pipeline.cu \
        -lgmp -lgmpxx \
        $(pkg-config --libs --cflags mpi 2>/dev/null || echo "-lmpi") \
        -o build/l3_pipeline
fi
echo ""

if [ ! -f build/l3_pipeline ]; then
    echo "BUILD FAILED"
    exit 1
fi

echo "=== Running L3 Pipeline ==="
export OMP_NUM_THREADS=8
mpirun -np 1 ./build/l3_pipeline

echo ""
echo "=== Job Complete ==="
echo "End: $(date)"
