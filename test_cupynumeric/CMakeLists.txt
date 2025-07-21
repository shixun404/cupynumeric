# Increase minimum version for better CUDA support
cmake_minimum_required(VERSION 3.18)

# Enable both C++ and CUDA as project languages
project(MyProject CXX CUDA)

# Set C++ and CUDA language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find the CUDA Toolkit to get access to libraries like cuRAND
find_package(CUDAToolkit REQUIRED)

# --- Manual Thrust Configuration ---
# We still need this from the previous step
set(THRUST_INCLUDE_PATH "/home/shixunw/miniconda3/envs/legate_1/include")
if(NOT EXISTS "${THRUST_INCLUDE_PATH}/thrust")
    message(FATAL_ERROR "Thrust headers not found in specified path: ${THRUST_INCLUDE_PATH}")
endif()

# --- Build Targets ---

# Target 1: Pure C++ executable
add_executable(test_bijection test_bijection.cpp)
target_include_directories(test_bijection PRIVATE ${THRUST_INCLUDE_PATH})

# Target 2: C++/CUDA executable
add_executable(test_bijection_inverse test_bijection_inverse.cpp)

# Tell CMake to compile this specific .cpp file with the CUDA compiler
set_source_files_properties(test_bijection_inverse.cpp PROPERTIES LANGUAGE CUDA)

# Link the CUDA target to Thrust and the necessary CUDA libraries (cuRAND)
target_include_directories(test_bijection_inverse PRIVATE ${THRUST_INCLUDE_PATH})
target_link_libraries(test_bijection_inverse PRIVATE CUDA::curand)