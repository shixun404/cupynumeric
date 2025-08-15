#!/bin/bash

# --- Core Software Paths ---
export HPC_SDK_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3
export CONDA_ENV_DIR="/project/coreai_devtech_all/shixunw/miniconda3/envs/legate"
export MPI_HOME="$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi"
export UCX_HOME="$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx"
export UCC_HOME="$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucc"
export HDF5_ROOT="/root/hdf5-1.14.3/hdf5" # Note: Running from /root is unusual

# --- Executable Path (PATH) ---
# Order is important: Conda, MPI, and HPC compilers are prepended to the existing PATH
export PATH="$CONDA_ENV_DIR/bin:$MPI_HOME/bin:$HPC_SDK_ROOT/compilers/bin:$PATH"

# --- Library Path (LD_LIBRARY_PATH) ---
# Start fresh and add paths in a clean order
export LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib"
export LD_LIBRARY_PATH="$MPI_HOME/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$UCX_HOME/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$UCC_HOME/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HPC_SDK_ROOT/compilers/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HPC_SDK_ROOT/math_libs/12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:$LD_LIBRARY_PATH"

# --- UCX/MPI Runtime Configuration ---
export UCX_TLS="rc,dc,sm,self,cuda_copy"
export UCX_IB_MLX5_DEVX="n"
export UCX_NET_DEVICES="mlx5_0:1"
export UCX_LOG_LEVEL="debug" # Set to "warn" or "error" for normal runs
export OMPI_MCA_pml="ucx"   # Explicitly tell OpenMPI to use the UCX PML

# --- Compiler and Build Tool Settings ---
export MPI_C_COMPILER="mpicc"
export MPI_CXX_COMPILER="mpicxx"
export CMAKE_PREFIX_PATH="$HDF5_ROOT:$CMAKE_PREFIX_PATH"
export CMAKE_PREFIX_PATH="/usr/local/cuda-12:$CMAKE_PREFIX_PATH"

# --- Application-Specific Settings ---
export LEGATE_TEST=1
# Avoid running as root if possible
# export OMPI_ALLOW_RUN_AS_ROOT=1
# export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# --- Run the application ---
echo "--- Starting Legate Application ---"
legate --gpus 8 --fbmem 50000 /project/coreai_devtech_all/shixunw/cupynumeric.internal/test_fancy_indexing.py