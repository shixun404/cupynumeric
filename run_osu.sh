#!/bin/bash
#export OMPI_MCA_pml=ucx
#export OMPI_MCA_coll=^hcoll
#which mpirun
export HOME=/project/coreai_devtech_all/shixunw;
export MPI_HOME=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi;
export HPC_SDK_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3;
export OPAL_PREFIX=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi;

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


export PATH=$HPC_SDK_ROOT/compilers/bin:$PATH;
export PATH=$MPI_HOME/bin:$PATH;

export MPI_C_COMPILER=mpicc;
export MPI_CXX_COMPILER=mpicxx;
export ucc_DIR=$HPC_SDK_ROOT/comm_libs/12.8/hpcx/hpcx-2.22.1/ucc/lib/cmake/ucc; 

#export UCX_TLS=cma,cuda,cuda_copy,cuda_ipc,mm,posix,self,shm,sm,sysv,tcp
# export UCX_TLS=self,tls,rc,shm,cuda,tcp,cuda_copy,cuda_ipc,gdr_copy
export UCX_LOG_LEVEL=error;
# export UCX_NET_DEVICES=mlx5_0:1;
#export UCX_NET_DEVICES=mlx5_0:1;


export UCX_TCP_KEEPALIVE=1;
export UCX_IB_TIMEOUT=20s;
export UCX_RNDV_THRESH=0;
export CMAKE_PREFIX_PATH=/usr/local/cuda-12:$CMAKE_PREFIX_PATH;
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/12.8/targets/x86_64-linux::$CMAKE_PREFIX_PATH;
export UCX_MEMTYPE_CACHE=n;
export UCX_RNDV_SCHEME=put_zcopy;
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/legate_1/lib:$LD_LIBRARY_PATH;
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
# 设置虚拟内存为无限制
ulimit -v unlimited

# 设置数据段大小为无限制  
ulimit -d unlimited

# 设置所有内存相关限制为无限制
ulimit -m unlimited
ulimit -v unlimited
ulimit -d unlimited
ulimit -s unlimited  # 栈大小
# which mpirun
#which mpiexec
#which mpicc
# nsys profile -o /project/coreai_devtech_all/shixunw/nsys_result/report_%q{SLURM_PROCID} --force-overwrite true  -t cuda,nvtx,mpi /project/coreai_devtech_all/shixunw/cupynumeric-intern-project/build/global_shuffle_bidirectional_2D 1000000000 8 1 1
#/project/coreai_devtech_all/shixunw/osu-micro-benchmarks-7.5.1/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall
nsys profile -o /project/coreai_devtech_all/shixunw/nsys_result/report_%q{SLURM_PROCID} -t cuda,mpi,ucx --force-overwrite true /project/coreai_devtech_all/shixunw/osu-micro-benchmarks-7.5.1/libexec/osu-micro-benchmarks/xccl/collective/osu_xccl_alltoall -m 2:2^30  -x 1 -i 1
# nsys profile -o /project/coreai_devtech_all/shixunw/nsys_result/report_%q{SLURM_PROCID} -t cuda,nvtx,mpi /project/coreai_devtech_all/shixunw/test_shuffle.py
# /project/coreai_devtech_all/shixunw/cupynumeric-intern-project/mpi_hello_world
# /project/coreai_devtech_all/shixunw/cupynumeric-intern-project/nccl_hello_world/nccl_hello_world
#ibdev2netdev

