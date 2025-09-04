/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// CUDA and system includes
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <curand_kernel.h>
#include <nvtx3/nvToolsExt.h>

// Thrust includes
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// STD includes
#include <random>
#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

// CUDA STD includes
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

// Error checking macros
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(call) do { \
    ncclResult_t error = call; \
    if (error != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d - %s\n", __FILE__, __LINE__, ncclGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CHECK_MPI(call) do { \
    int error = call; \
    if (error != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Simple Point and Rect structures to replace legate ones
template<int DIM>
struct Point {
    int64_t coords[DIM];
    
    __host__ __device__ int64_t& operator[](int i) { return coords[i]; }
    __host__ __device__ const int64_t& operator[](int i) const { return coords[i]; }
};

template<int DIM>
struct Rect {
    Point<DIM> lo, hi;
};

// Memory management helper
template<typename T>
class DeviceBuffer {
private:
    T* ptr_;
    size_t size_;
    
public:
    DeviceBuffer(size_t size) : size_(size) {
        CHECK_CUDA(cudaMalloc(&ptr_, size * sizeof(T)));
    }
    
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    
    // Move constructor
    DeviceBuffer(DeviceBuffer&& other) : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    // Move assignment
    DeviceBuffer& operator=(DeviceBuffer&& other) {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    T* ptr() { return ptr_; }
    const T* ptr() const { return ptr_; }
    size_t size() const { return size_; }
};

// Compute send histogram kernel for 1D (vectors)
template<int DIM_input>
__global__ void compute_send_histogram(const Point<DIM_input>* indices, 
                                      unsigned int* send_histo, 
                                      Rect<DIM_input>* rect_buf, 
                                      int num_ranks, 
                                      int indices_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < indices_len){
      int target_rank = 0;
      int is_in_rect = 1;
      Point<DIM_input> point = indices[idx];
      for(; target_rank < num_ranks; target_rank++){
        is_in_rect = 1;
        for(int d = 0; d < DIM_input; d++){
          if(rect_buf[target_rank].lo[d] > point[d] || rect_buf[target_rank].hi[d] < point[d]){
            is_in_rect = 0;
            break;
          }
        }
        if(is_in_rect){
          break;
        }
      }
      atomicAdd(&send_histo[target_rank], is_in_rect);
    }
}

// Pack send data kernel for 2D (vectors)
template<typename DataType, int DIM_input>
__global__ void pack_send_data_kernel(const DataType* data, Point<DIM_input>* indices, size_t request_count,
                                          DataType* send_data, Rect<DIM_input> input_rect) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
      Point<DIM_input> point = indices[idx];
      int data_idx = 0;
      for(int d = 0; d < DIM_input - 1; d++){
        data_idx += (point[d] - input_rect.lo[d]) * (input_rect.hi[d + 1] - input_rect.lo[d + 1] + 1);
      }
      data_idx += (point[DIM_input - 1] - input_rect.lo[DIM_input - 1]);
      send_data[idx] = data[data_idx];
    }
}

template<typename DataType, int DIM_input>
__global__ void unpack_recv_data_kernel(DataType* data, unsigned int* request_indices, size_t request_count,
                                          const DataType* recv_data) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
        data[request_indices[idx]] = recv_data[idx];
    }
}

template<int DIM_input>
__global__ void pack_request_indices_kernel(const Point<DIM_input>* indices, size_t vector_count,
                                          Point<DIM_input>* send_indices, unsigned int* send_offsets, unsigned int* request_indices,
                                          unsigned int* counters, Rect<DIM_input>* rect_buf, int num_ranks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_count) {
        int target_rank = 0;
        int is_in_rect = 1;
        Point<DIM_input> point = indices[idx];
        for(; target_rank < num_ranks; target_rank++){
          is_in_rect = 1;
          for(int d = 0; d < DIM_input; d++){
            if(rect_buf[target_rank].lo[d] > point[d] || rect_buf[target_rank].hi[d] < point[d]){
              is_in_rect = 0;
              break;
            }
          }
          if(is_in_rect){
            break;
          }
        }
        size_t pos = atomicAdd(&counters[target_rank], 1);
        size_t offset = send_offsets[target_rank] + pos;
        send_indices[offset] = indices[idx];
        request_indices[offset] = idx;
    }
}

template<int DIM>
size_t get_volume(Rect<DIM> rect){
  size_t volume = 1;
  for(int i = 0; i < DIM; i++){
    volume *= rect.hi[i] - rect.lo[i] + 1;
  }
  return volume;
}

// NCCL barrier implementation
inline void nccl_barrier(ncclComm_t nccl_comm, cudaStream_t stream) {
    int dummy = 0;
    CHECK_NCCL(ncclAllReduce(&dummy, &dummy, 1, ncclInt32, ncclSum, nccl_comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

/*
 * Implements fancy indexing using 3-round NCCL All2All communication:
 * 
 * Round 1: Exchange request size histograms
 *   - Each rank computes how many indices it wants from each other rank
 *   - All2All exchange of histogram counts to determine communication pattern
 * 
 * Round 2: Exchange request indices  
 *   - Send the actual indices that each rank needs from other ranks
 *   - Receive indices that other ranks need from this rank
 * 
 * Round 3: Exchange actual data
 *   - Send the requested data elements to other ranks
 *   - Receive the data elements this rank requested from other ranks
 *   - Unpack received data into final output array
 */
template<typename DataType, int DIM_input, int DIM_output>
void global_all2all(
  const DataType* input_ptr, 
  const Point<DIM_input>* index_ptr, 
  DataType* output_ptr, 
  const Rect<DIM_input> input_rect,
  const Rect<DIM_output> index_rect,
  const Rect<DIM_output> output_rect,
  int rank_id, 
  int num_ranks,
  ncclComm_t nccl_comm, 
  cudaStream_t stream
) {
  nvtxRangePushA("global_all2all");
  
  // Print current rank's GPU device ID and properties
  int device_id;
  CHECK_CUDA(cudaGetDevice(&device_id));
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
  printf("Rank %d is using GPU device %d (%s)\n", rank_id, device_id, prop.name);
  
  // Add barrier at the beginning
  nccl_barrier(nccl_comm, stream);

  size_t local_input_count = get_volume<DIM_input>(input_rect);
  size_t local_index_count = get_volume<DIM_output>(index_rect);
  size_t local_output_count = get_volume<DIM_output>(output_rect);
  
  size_t num_requests = local_index_count;
  
  // ===== Round 0: Exchange rects =====
  DeviceBuffer<int64_t> global_rects(num_ranks * DIM_input * 2);
  DeviceBuffer<int64_t> input_rect_device(DIM_input * 2);
  
  // Initialize buffers
  CHECK_CUDA(cudaMemsetAsync(global_rects.ptr(), 0, num_ranks * DIM_input * 2 * sizeof(int64_t), stream));
  CHECK_CUDA(cudaMemsetAsync(input_rect_device.ptr(), 0, DIM_input * 2 * sizeof(int64_t), stream));
  
  CHECK_CUDA(cudaMemcpy((Rect<DIM_input>*)input_rect_device.ptr(), &input_rect, sizeof(input_rect), cudaMemcpyHostToDevice));

  // ===== Round 1: Exchange request size histograms =====
  DeviceBuffer<unsigned int> round1_send_histo(num_ranks);
  DeviceBuffer<unsigned int> round1_send_offsets(num_ranks);
  DeviceBuffer<unsigned int> round1_recv_histo(num_ranks);
  DeviceBuffer<unsigned int> round1_recv_offsets(num_ranks);
  DeviceBuffer<unsigned int> packing_counters(num_ranks);
  
  // Initialize Round 1 buffers
  CHECK_CUDA(cudaMemsetAsync(round1_send_histo.ptr(), 0, num_ranks * sizeof(unsigned int), stream));
  CHECK_CUDA(cudaMemsetAsync(round1_send_offsets.ptr(), 0, num_ranks * sizeof(unsigned int), stream));
  CHECK_CUDA(cudaMemsetAsync(round1_recv_histo.ptr(), 0, num_ranks * sizeof(unsigned int), stream));
  CHECK_CUDA(cudaMemsetAsync(round1_recv_offsets.ptr(), 0, num_ranks * sizeof(unsigned int), stream));
  CHECK_CUDA(cudaMemsetAsync(packing_counters.ptr(), 0, num_ranks * sizeof(unsigned int), stream));

  // ===== Round 2: Exchange request indices =====
  DeviceBuffer<int64_t> round2_send_indices(num_requests * DIM_input);
  DeviceBuffer<unsigned int> round2_request_positions(num_requests);
  
  // Initialize Round 2 buffers
  CHECK_CUDA(cudaMemsetAsync(round2_send_indices.ptr(), 0, num_requests * DIM_input * sizeof(int64_t), stream));
  CHECK_CUDA(cudaMemsetAsync(round2_request_positions.ptr(), 0, num_requests * sizeof(unsigned int), stream));
  
  // ===== Round 3: Exchange actual data =====
  DeviceBuffer<DataType> round3_recv_data(num_requests);
  
  // Initialize Round 3 buffers
  CHECK_CUDA(cudaMemsetAsync(round3_recv_data.ptr(), 0, num_requests * sizeof(DataType), stream));

  const size_t block_size = 256;
  const size_t grid_size = (local_index_count + block_size - 1) / block_size;
  
  // Ensure all initialization is complete
  CHECK_CUDA(cudaStreamSynchronize(stream));
  
  // ===== Round 0: Exchange rects =====
  nvtxRangePushA("Exchange rects");
  CHECK_NCCL(ncclAllGather((void*)input_rect_device.ptr(), 
                         (void*)global_rects.ptr(), 
                         sizeof(input_rect), 
                         ncclInt8, 
                         nccl_comm, 
                         stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // Barrier after Round 0
  nccl_barrier(nccl_comm, stream);
  
  // ===== Round 1: Compute request size histogram =====
  nvtxRangePushA("Compute request size histogram");
  compute_send_histogram<DIM_input><<<grid_size, block_size, 0, stream>>>(
    index_ptr, 
    round1_send_histo.ptr(), 
    (Rect<DIM_input>*)(global_rects.ptr()), 
    num_ranks, local_index_count);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  nvtxRangePushA("Exclusive scan");
  thrust::exclusive_scan(thrust::cuda::par.on(stream), 
                        (unsigned int*)round1_send_histo.ptr(), 
                        ((unsigned int*)round1_send_histo.ptr()) + num_ranks, 
                        round1_send_offsets.ptr());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();

  // ===== Round 1: All2All exchange request size histograms =====
  nvtxRangePushA("All2All exchange request size histograms");
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
    CHECK_NCCL(ncclRecv((void*)&round1_recv_histo.ptr()[i], 1, ncclUint32, i, nccl_comm, stream));  
    CHECK_NCCL(ncclSend((void*)&round1_send_histo.ptr()[i], 1, ncclUint32, i, nccl_comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // Barrier after Round 1
  nccl_barrier(nccl_comm, stream);
  
  nvtxRangePushA("Reduce");
  size_t total_indices_to_receive = thrust::reduce(thrust::cuda::par.on(stream), 
    round1_recv_histo.ptr(), round1_recv_histo.ptr() + num_ranks);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // Pack request indices by target rank
  nvtxRangePushA("Pack request indices");
  pack_request_indices_kernel<DIM_input><<<grid_size, block_size, 0, stream>>>(
    index_ptr, num_requests,  
    (Point<DIM_input>*)(round2_send_indices.ptr()), 
    round1_send_offsets.ptr(),
    round2_request_positions.ptr(), 
    packing_counters.ptr(), 
    (Rect<DIM_input>*)(global_rects.ptr()), num_ranks);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  nvtxRangePushA("Exclusive scan");
  DeviceBuffer<Point<DIM_input>> round2_recv_indices(total_indices_to_receive);
  CHECK_CUDA(cudaMemsetAsync(round2_recv_indices.ptr(), 0, 
    total_indices_to_receive * sizeof(Point<DIM_input>), stream));
  
  thrust::exclusive_scan(thrust::cuda::par.on(stream), 
    round1_recv_histo.ptr(), round1_recv_histo.ptr() + num_ranks, round1_recv_offsets.ptr());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // ===== Round 2: All2All exchange request indices =====
  nvtxRangePushA("All2All exchange request indices");
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
      unsigned int indices_to_send_to_rank_i, send_offset_for_rank_i, indices_to_recv_from_rank_i, recv_offset_for_rank_i;
      CHECK_CUDA(cudaMemcpy(&indices_to_send_to_rank_i, &round1_send_histo.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&send_offset_for_rank_i, &round1_send_offsets.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&indices_to_recv_from_rank_i, &round1_recv_histo.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&recv_offset_for_rank_i, &round1_recv_offsets.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      
      if (indices_to_send_to_rank_i > 0) {
          CHECK_NCCL(ncclSend((void*)(&round2_send_indices.ptr()[send_offset_for_rank_i * DIM_input]),
            indices_to_send_to_rank_i * sizeof(Point<DIM_input>), ncclInt8, i, nccl_comm, stream));
      }
      
      if (indices_to_recv_from_rank_i > 0) {
          CHECK_NCCL(ncclRecv(&round2_recv_indices.ptr()[recv_offset_for_rank_i * DIM_input],
                  indices_to_recv_from_rank_i * sizeof(Point<DIM_input>), ncclInt8, i, nccl_comm, stream));
      }
  }
  CHECK_NCCL(ncclGroupEnd());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // Barrier after Round 2
  nccl_barrier(nccl_comm, stream);
  
  nvtxRangePushA("Pack send data");
  DeviceBuffer<DataType> round3_send_data(total_indices_to_receive);
  CHECK_CUDA(cudaMemsetAsync(round3_send_data.ptr(), 0, 
    total_indices_to_receive * sizeof(DataType), stream));
  
  pack_send_data_kernel<DataType, DIM_input><<<grid_size, block_size, 0, stream>>>(
    input_ptr, (Point<DIM_input>*)round2_recv_indices.ptr(),
    total_indices_to_receive, round3_send_data.ptr(), input_rect);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // ===== Round 3: All2All exchange actual data =====
  nvtxRangePushA("All2All exchange actual data");
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
      unsigned int data_to_send_to_rank_i, send_offset_for_rank_i, data_to_recv_from_rank_i, recv_offset_for_rank_i;
      CHECK_CUDA(cudaMemcpy(&data_to_send_to_rank_i, &round1_send_histo.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&send_offset_for_rank_i, &round1_send_offsets.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&data_to_recv_from_rank_i, &round1_recv_histo.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(&recv_offset_for_rank_i, &round1_recv_offsets.ptr()[i], sizeof(unsigned int), cudaMemcpyDeviceToHost));

      if (data_to_recv_from_rank_i > 0) {
        CHECK_NCCL(ncclSend(&round3_send_data.ptr()[recv_offset_for_rank_i],
        data_to_recv_from_rank_i * sizeof(DataType), ncclInt8, i, nccl_comm, stream));
      }
      if (data_to_send_to_rank_i > 0) {
        CHECK_NCCL(ncclRecv(&round3_recv_data.ptr()[send_offset_for_rank_i],
        data_to_send_to_rank_i * sizeof(DataType), ncclInt8, i, nccl_comm, stream));
      }   
  }
  CHECK_NCCL(ncclGroupEnd());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // ===== Final step: Unpack received data to output =====
  nvtxRangePushA("unpack received data");
  unpack_recv_data_kernel<DataType, DIM_output><<<grid_size, block_size, 0, stream>>>(
    output_ptr, round2_request_positions.ptr(),
    num_requests, round3_recv_data.ptr());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  nvtxRangePop();
  
  // Final barrier
  nccl_barrier(nccl_comm, stream);
  
  nvtxRangePop(); // global_all2all
}

// Test data generation
template<typename T>
void generate_test_data(T* data, size_t size, int rank) {
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(rank * 1000 + i);
    }
}

template<int DIM>
void generate_test_indices(Point<DIM>* indices, size_t count, Rect<DIM> global_rect, int seed) {
    std::mt19937 gen(seed);
    for (size_t i = 0; i < count; i++) {
        for (int d = 0; d < DIM; d++) {
            std::uniform_int_distribution<int64_t> dis(global_rect.lo[d], global_rect.hi[d]);
            indices[i][d] = dis(gen);
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    CHECK_MPI(MPI_Init(&argc, &argv));
    
    int rank, size;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    
    printf("Rank %d of %d started\n", rank, size);
    
    // Set CUDA device
    int num_devices;
    CHECK_CUDA(cudaGetDeviceCount(&num_devices));
    int device = rank % num_devices;
    CHECK_CUDA(cudaSetDevice(device));
    
    // Initialize NCCL
    ncclUniqueId nccl_id;
    if (rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    CHECK_MPI(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
    
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));
    
    // Create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // Test parameters
    const int DIM_INPUT = 2;
    const int DIM_OUTPUT = 1;
    const size_t ELEMENTS_PER_RANK = 60000;
    const size_t INDICES_PER_RANK = 200000;
    
    // Define global and local rectangles
    Rect<DIM_INPUT> global_input_rect;
    global_input_rect.lo[0] = 0; global_input_rect.lo[1] = 0;
    global_input_rect.hi[0] = size * 10 - 1; global_input_rect.hi[1] = 100 - 1;
    
    Rect<DIM_INPUT> local_input_rect;
    local_input_rect.lo[0] = rank * 10; local_input_rect.lo[1] = 0;
    local_input_rect.hi[0] = (rank + 1) * 10 - 1; local_input_rect.hi[1] = 100 - 1;
    
    Rect<DIM_OUTPUT> index_rect;
    index_rect.lo[0] = 0;
    index_rect.hi[0] = INDICES_PER_RANK - 1;
    
    Rect<DIM_OUTPUT> output_rect = index_rect;
    
    // Allocate host data
    std::vector<float> host_input(ELEMENTS_PER_RANK);
    std::vector<Point<DIM_INPUT>> host_indices(INDICES_PER_RANK);
    std::vector<float> host_output(INDICES_PER_RANK);
    
    // Generate test data
    generate_test_data(host_input.data(), ELEMENTS_PER_RANK, rank);
    generate_test_indices(host_indices.data(), INDICES_PER_RANK, global_input_rect, rank + 42);
    
    // Allocate device memory
    DeviceBuffer<float> device_input(ELEMENTS_PER_RANK);
    DeviceBuffer<Point<DIM_INPUT>> device_indices(INDICES_PER_RANK);
    DeviceBuffer<float> device_output(INDICES_PER_RANK);
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(device_input.ptr(), host_input.data(), 
                         ELEMENTS_PER_RANK * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_indices.ptr(), host_indices.data(), 
                         INDICES_PER_RANK * sizeof(Point<DIM_INPUT>), cudaMemcpyHostToDevice));
    
    // Synchronize all ranks before starting
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    
    if (rank == 0) {
        printf("Starting fancy indexing test with %d ranks...\n", size);
    }
    
    // Run multiple iterations for profiling
    const int NUM_ITERATIONS = 3;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        if (rank == 0) {
            printf("Iteration %d/%d\n", iter + 1, NUM_ITERATIONS);
        }
        
        nvtxRangePushA("Iteration");
        
        // Run the fancy indexing
        global_all2all<float, DIM_INPUT, DIM_OUTPUT>(
            device_input.ptr(),
            device_indices.ptr(),
            device_output.ptr(),
            local_input_rect,
            index_rect,
            output_rect,
            rank,
            size,
            nccl_comm,
            stream
        );
        
        nvtxRangePop();
        
        // Synchronize between iterations
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    }
    
    // Copy result back to host for verification
    CHECK_CUDA(cudaMemcpy(host_output.data(), device_output.ptr(), 
                         INDICES_PER_RANK * sizeof(float), cudaMemcpyDeviceToHost));
    
    if (rank == 0) {
        printf("Fancy indexing test completed successfully!\n");
        printf("Sample results: ");
        for (int i = 0; i < std::min(10, (int)INDICES_PER_RANK); i++) {
            printf("%.1f ", host_output[i]);
        }
        printf("\n");
    }
    
    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(nccl_comm));
    CHECK_MPI(MPI_Finalize());
    
    return 0;
}
