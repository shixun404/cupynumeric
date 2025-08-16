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

# Minimal UCX configuration for testing
export UCX_LOG_LEVEL=error
export UCX_TLS=tcp,self
export UCX_VFS_ENABLE=n
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# --- Run the minimal test ---
echo "--- Starting Minimal Legate Test ---"
legate --gpus 1 --fbmem 10000 /project/coreai_devtech_all/shixunw/cupynumeric.internal/test_minimal.py
