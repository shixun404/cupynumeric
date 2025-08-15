#!/bin/bash

# --- Core Software Paths ---
export CONDA_ENV_DIR="/project/coreai_devtech_all/shixunw/miniconda3/envs/legate"
export PATH="$CONDA_ENV_DIR/bin:$PATH"
export HOME=/project/coreai_devtech_all/shixunw;
export MPI_HOME=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi;
export HPC_SDK_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3;
export OPAL_PREFIX=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi;
export UCX_HOME=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx
export UCX_MODULE_DIR=$UCX_HOME/lib/ucx


export UCX_LIB_DIR="$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx/lib"
export MPI_LIB_DIR="$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/lib"
export LD_LIBRARY_PATH="$UCX_HOME/lib:$MPI_HOME/lib"
export HDF5_INCLUDE_DIRS=/root/hdf5-1.14.3/hdf5/include;
export HDF5_LIBRARIES=/root/hdf5-1.14.3/hdf5/lib;
export HDF5_ROOT=/root/hdf5-1.14.3/hdf5;
export HDF5_DIR=/root/hdf5-1.14.3/config/cmake;
export CMAKE_PREFIX_PATH=$HDF5_ROOT:$CMAKE_PREFIX_PATH;


export LD_LIBRARY_PATH=$HDF5_ROOT/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucc/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/latest/ucc/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/latest/ucx/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/compilers/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/compilers/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH;
export LD_PRELOAD="$UCX_LIB_DIR/libucs.so.0:$UCX_LIB_DIR/libucp.so.0:$UCX_LIB_DIR/libucm.so.0:$MPI_LIB_DIR/libmpi.so"


export PATH=$HPC_SDK_ROOT/compilers/bin:$PATH;
export PATH=$MPI_HOME/bin:$PATH;

export MPI_C_COMPILER=mpicc;
export MPI_CXX_COMPILER=mpicxx;
export ucc_DIR=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucc/lib/cmake/ucc;
#export UCX_TLS=rc,dc,sm,self,cuda_copy
# export UCX_IB_GPU_DIRECT_RDMA=no
export UCX_LOG_LEVEL=debug;
#export OMPI_MCA_pml_ucx_tls="sm,cuda_copy,ib"
# export UCX_NET_DEVICES=mlx5_0:1;
#export UCX_NET_DEVICES=mlx5_0:1;


export CMAKE_PREFIX_PATH=/usr/local/cuda-12:$CMAKE_PREFIX_PATH;
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux::$CMAKE_PREFIX_PATH;
#export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate_1/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate_1/lib/libibverbs:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH;
export CMAKE_PREFIX_PATH=$HDF5_ROOT:$CMAKE_PREFIX_PATH;
export UCX_IB_PREFER_NEAREST_DEVICE=0;
export PATH=$HPC_SDK_ROOT/compilers/bin:$PATH;
export PATH=$MPI_HOME/bin:$PATH;
export MPI_C_COMPILER=mpicc;
export MPI_CXX_COMPILER=mpicxx
export ucc_DIR=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucc/lib/cmake/ucc;
export CUDNN_PATH=$HPC_SDK_ROOT/math_libs/12.8/targets/sbsa-linux;
export CUDA_PATH=$HPC_SDSK_ROOT;
export CPATH=$CUDNN_PATH/include:$CPATH;
export RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=1;
export UCX_LOG_LEVEL=debug
export LEGATE_TEST=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate_1/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_ENV_DIR/lib"
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate_1/lib/libibverbs:$LD_LIBRARY_PATH;
export OMPI_MCA_btl_base_verbose=100
# NEW: Explicitly tell the UCX CUDA transport to avoid gdrcopy
#export UCX_CUDA_TLS=ipc,cma
# Disable the VFS feature to prevent non-fatal startup warnings
# export UCX_VFS_ENABLE=n
export UCX_IB_MLX5_DEVX=n
# --- Run the application ---
echo "--- Starting Legate Application ---"
legate --gpus 8 --fbmem 50000 /project/coreai_devtech_all/shixunw/cupynumeric.internal/test_fancy_indexing.py
