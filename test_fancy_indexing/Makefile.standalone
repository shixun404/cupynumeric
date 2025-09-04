# Makefile for standalone fancy indexing
CXX = nvcc
MPICC = mpicc

# Compiler flags
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_70 -lineinfo
NVCC_FLAGS += -Xcompiler -fopenmp
NVCC_FLAGS += -DWITH_NVTX

# Include paths
CUDA_PATH ?= /usr/local/cuda
NCCL_PATH ?= /usr/local/nccl
MPI_PATH ?= /usr/lib/x86_64-linux-gnu/openmpi

INCLUDES = -I$(CUDA_PATH)/include
INCLUDES += -I$(NCCL_PATH)/include
INCLUDES += -I$(MPI_PATH)/include

# Library paths and libraries
LIBS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcurand -lnvToolsExt
LIBS += -L$(NCCL_PATH)/lib -lnccl
LIBS += -lmpi

# Target
TARGET = fancy_indexing_standalone
SOURCE = fancy_indexing_standalone_complete.cu

# Build rules
$(TARGET): $(SOURCE)
	$(CXX) $(NVCC_FLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

# Clean
clean:
	rm -f $(TARGET)

# Run with nsys
profile: $(TARGET)
	nsys profile --trace=cuda,nvtx,mpi --output=fancy_indexing_profile mpirun -np 2 ./$(TARGET)

# Run with multiple nodes (adjust as needed)
profile_multinode: $(TARGET)
	nsys profile --trace=cuda,nvtx,mpi --output=fancy_indexing_profile_multinode \
		mpirun --hostfile hostfile -np 4 ./$(TARGET)

# Simple test run
test: $(TARGET)
	mpirun -np 2 ./$(TARGET)

.PHONY: clean profile profile_multinode test
