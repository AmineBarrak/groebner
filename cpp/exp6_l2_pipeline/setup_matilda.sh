#!/bin/bash
#====================================================================
# setup_matilda.sh — First-time setup for Matilda HPC
#
# Run this interactively on the login node to verify environment
# and install GMP if it's not available as a module.
#
# Usage:  bash setup_matilda.sh
#====================================================================

echo "=== Checking Matilda Environment ==="
echo ""

# 1. Check available modules
echo "--- Available modules matching our needs ---"
echo ""
echo "CMake:"
module avail CMake 2>&1 | grep -i cmake || echo "  (not found as module)"
echo ""
echo "CUDA:"
module avail CUDA 2>&1 | grep -i cuda || echo "  (not found as module)"
echo ""
echo "OpenMPI:"
module avail OpenMPI 2>&1 | grep -i openmpi || echo "  (not found as module)"
echo ""
echo "GCC:"
module avail GCC 2>&1 | grep -i gcc || echo "  (not found as module)"
echo ""
echo "GMP:"
module avail GMP 2>&1 | grep -i gmp || echo "  (not found as module — will need local install)"
echo ""

# 2. Check if GMP is already available system-wide
echo "--- Checking for GMP installation ---"
if ldconfig -p 2>/dev/null | grep -q libgmp; then
    echo "GMP found in system libraries"
elif [ -f /usr/include/gmpxx.h ]; then
    echo "GMP headers found at /usr/include/gmpxx.h"
elif pkg-config --exists gmp 2>/dev/null; then
    echo "GMP found via pkg-config"
else
    echo "GMP not found system-wide."
    echo ""
    echo "To install GMP locally:"
    echo "  mkdir -p ~/local/src && cd ~/local/src"
    echo "  wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz"
    echo "  tar xf gmp-6.3.0.tar.xz && cd gmp-6.3.0"
    echo "  ./configure --prefix=\$HOME/local --enable-cxx"
    echo "  make -j8 && make install"
    echo ""
    echo "Then add to your .bashrc:"
    echo "  export LD_LIBRARY_PATH=\$HOME/local/lib:\$LD_LIBRARY_PATH"
    echo "  export CPATH=\$HOME/local/include:\$CPATH"
    echo "  export CMAKE_PREFIX_PATH=\$HOME/local:\$CMAKE_PREFIX_PATH"
fi

echo ""

# 3. Check GPU node availability
echo "--- GPU node availability ---"
sinfo -p general-lo -N -o "%N %G %T" 2>/dev/null | grep gpu || echo "(run on login node to check)"
echo ""
echo "Current GPU jobs:"
squeue --format="%.8i %.12j %.8u %.8T %.8M %.4D %R" | grep gpu || echo "  (none)"
echo ""

# 4. Summary
echo "=== Next Steps ==="
echo "1. Fix module names in slurm_gpu.sh to match what's available above"
echo "2. If GMP needs local install, follow the instructions above"
echo "3. Push code to GitHub, clone on Matilda, then:"
echo "     cd groebner_experiments/cpp/exp6_l2_pipeline"
echo "     sbatch slurm_gpu.sh"
echo "4. Monitor with: squeue -u \$USER"
echo "5. Check output: cat l2_pipeline_<jobid>.out"
