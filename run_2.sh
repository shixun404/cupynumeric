#!/bin/bash

# --- Core Software Paths ---
__conda_setup="$('/project/coreai_devtech_all/shixunw/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
   eval "$__conda_setup"
else
   if [ -f "/project/coreai_devtech_all/shixunw/miniconda3/etc/profile.d/conda.sh" ]; then
       . "/project/coreai_devtech_all/shixunw/miniconda3/etc/profile.d/conda.sh"
   else
       export PATH="/project/coreai_devtech_all/shixunw/miniconda3/bin:$PATH"
   fi
fi
unset __conda_setup
conda activate legate

export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export GASNET_AM_CREDITS_PP=16
export GASNET_IBV_PORTS=mlx5_0+mlx5_3+mlx5_4+mlx5_5+mlx5_6+mlx5_9+mlx5_10+mlx5_11

# NUMAS_PER_NODE=2
# RAM_PER_NUMA=950000
# GPUS_PER_NODE=8
# CORES_PER_NUMA=56
# FB_PER_GPU=76000
# CPU_SLOTS=" 0-13,112-125  14-27,126-139  28-41,140-153  42-55,154-167  56-69,168-181  70-83,182-195  84-97,196-209  98-111,210-223"
# GPU_SLOTS="            0              1              2              3              4              5              6               7"
# MEM_SLOTS="            0              0              0              0              1              1              1               1"
# NIC_SLOTS="       mlx5_0         mlx5_3         mlx5_4         mlx5_5         mlx5_6         mlx5_9        mlx5_10         mlx5_11"

# export LEGATE_DEBUG=1
# export REALM_BACKTRACE=1
# export LEGION_BACKTRACE=1

export OMPI_MCA_btl_base_verbose=100  # 最详细的 BTL 调试
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
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucx/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/compilers/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HPC_SDK_ROOT/compilers/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH;
# export LD_PRELOAD="$UCX_LIB_DIR/libucs.so.0:$UCX_LIB_DIR/libucp.so.0:$UCX_LIB_DIR/libucm.so.0:$MPI_LIB_DIR/libmpi.so"
export PATH=$HPC_SDK_ROOT/compilers/bin:$PATH;
export PATH=$MPI_HOME/bin:$PATH;

export MPI_C_COMPILER=mpicc;
export MPI_CXX_COMPILER=mpicxx;
export ucc_DIR=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucc/lib/cmake/ucc;
export UCX_LOG_LEVEL=error;
export UCX_MODULE_LOG_LEVEL=error;



export UCX_IB_TIMEOUT=22s
export UCX_TCP_KEEPALIVE=1

export CMAKE_PREFIX_PATH=/usr/local/cuda-12:$CMAKE_PREFIX_PATH;
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux::$CMAKE_PREFIX_PATH;
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate/lib:$LD_LIBRARY_PATH;
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate/lib/libibverbs:$LD_LIBRARY_PATH;
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
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export UCX_VFS_ENABLE=n
# --- Run the application ---
echo "--- Starting Legate Application ---"
# legate --gpus 8 --fbmem 10000 --launcher none /project/coreai_devtech_all/shixunw/cupynumeric.internal/test_fancy_indexing.py
legate --logdir /project/coreai_devtech_all/shixunw/result --launcher none  --cpus 1  --gpus 8 --fbmem 50000 --verbose --log-to-file --nodes 4 --ranks-per-node 1 /project/coreai_devtech_all/shixunw/cupynumeric.internal/test_fancy_indexing.py
# nsys profile -t cuda,mpi,ucx,nvtx /project/coreai_devtech_all/shixunw/osu-micro-benchmarks-7.5.1/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall
# /project/coreai_devtech_all/shixunw/osu-micro-benchmarks-7.5.1/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall