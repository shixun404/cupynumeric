cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GlobalShuffle LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target H100 architecture (SM 90):
set(CMAKE_CUDA_ARCHITECTURES 80 90)
# Add “--extended-lambda” to your compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Find MPI
find_package(MPI REQUIRED)

# If you need to locate NCCL manually, adjust these lines:
find_library(NCCL_LIBRARY nccl HINTS /usr/lib /usr/local/lib)

add_executable(fancy_indexing_standalone_complete fancy_indexing_standalone_complete.cu)

target_link_libraries(fancy_indexing_standalone_complete
    PRIVATE
    MPI::MPI_CXX          # MPI
    ${NCCL_LIBRARY}       # NCCL
    cuda                  # CUDA runtime
)