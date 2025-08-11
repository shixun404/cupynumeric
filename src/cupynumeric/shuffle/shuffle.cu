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

#include "cupynumeric/shuffle/shuffle.h"
#include "cupynumeric/utilities/thrust_allocator.h"
#include "cupynumeric/utilities/thrust_util.h"
#include "cupynumeric/cuda_help.h"
#include "cupynumeric/runtime.h"
#include "cupynumeric/pitches.h"

// 
// CUDA and system includes
#include <cuda_runtime.h>
// #include <nccl.h>
// #include <mpi.h>
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

// CUDA STD includes
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

namespace cupynumeric {

using namespace legate;
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// CUDA Kernels (from global_shuffle.cu)
/////////////////////////////////////////////////////////////////////////////////////////////////

// Global Shuffle Implementation for Multi-Node Multi-GPU
// Single file implementation under 500 lines

// 1. Kernel to initialize a cuRAND state for each thread


/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


class feistel_bijection {
public:
    using index_type = std::uint64_t;
    
    template <class URBG>
    __host__ __device__ feistel_bijection(std::uint64_t m, URBG&& g) {
        std::uint64_t total_bits = get_cipher_bits(m);
        left_side_bits = total_bits / 2;
        left_side_mask = (1ull << left_side_bits) - 1;
        right_side_bits = total_bits - left_side_bits;
        right_side_mask = (1ull << right_side_bits) - 1;
        
        thrust::uniform_int_distribution<std::uint32_t> dist;
        for (std::uint32_t i = 0; i < num_rounds; i++) {
            key[i] = dist(g);
        }
    }
    
    __host__ __device__ std::uint64_t nearest_power_of_two() const {
        return 1ull << (left_side_bits + right_side_bits);
    }
    
    __host__ __device__ std::uint64_t operator()(const std::uint64_t val) const {
        std::uint32_t state[2] = {static_cast<std::uint32_t>(val >> right_side_bits),
                                  static_cast<std::uint32_t>(val & right_side_mask)};
        for (std::uint32_t i = 0; i < num_rounds; i++) {
            std::uint32_t hi, lo;
            constexpr std::uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
            mulhilo(M0, state[0], hi, lo);
            lo = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
            state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
            state[1] = lo & right_side_mask;
        }
        return (static_cast<std::uint64_t>(state[0]) << right_side_bits) | static_cast<std::uint64_t>(state[1]);
    }

    __host__ __device__ std::uint64_t inverse(const std::uint64_t val) const {
        std::uint32_t state[2] = {static_cast<std::uint32_t>(val >> right_side_bits),
                                  static_cast<std::uint32_t>(val & right_side_mask)};
        for (std::uint32_t i = num_rounds; i != 0;) {
          i -= 1;
          std::uint32_t hi, lo, b;
          b = state[1] & (std::uint32_t)(right_side_bits - left_side_bits);
          constexpr std::uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
          constexpr std::uint64_t M_inv = UINT64_C(0xF5F365BD212BDF9B);
          
          mullo(M_inv, (state[1] >> (right_side_bits - left_side_bits)) & left_side_mask, lo);
          lo = lo & left_side_mask;
          mulhi(M0, lo, hi);
          
          state[1] = ((b << left_side_bits) | (((hi ^ key[i]) ^ state[0]) & left_side_mask)) & right_side_mask;
          state[0] = lo;
          
        }
        return (static_cast<std::uint64_t>(state[0]) << right_side_bits) | static_cast<std::uint64_t>(state[1]);
      }

private:
    static __host__ __device__ void mulhilo(std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo) {
        std::uint64_t product = a * b;
        hi = static_cast<std::uint32_t>(product >> 32);
        lo = static_cast<std::uint32_t>(product);
    }

    static __host__ __device__ void mullo(std::uint64_t a, std::uint64_t b, std::uint32_t& lo) {
        std::uint64_t product = a * b;
        lo = static_cast<std::uint32_t>(product);
      }
      
    static  __host__ __device__ void mulhi(std::uint64_t a, std::uint64_t b, std::uint32_t& hi) {
        std::uint64_t product = a * b;
        hi = static_cast<std::uint32_t>(product >> 32);
      }
    
    static __host__ __device__ std::uint64_t get_cipher_bits(std::uint64_t m) {
        if (m <= 16) return 4;
        std::uint64_t i = 0;
        m--;
        while (m != 0) { i++; m >>= 1; }
        return i;
    }
    
    static constexpr std::uint32_t num_rounds = 24;
    std::uint64_t right_side_bits, left_side_bits, right_side_mask, left_side_mask;
    std::uint32_t key[num_rounds];
};

template <class IndexType>
class random_bijection {
private:
    feistel_bijection bijection;
    IndexType n;

public:
    using index_type = IndexType;
    
    template <class URBG>
    __host__ __device__ random_bijection(IndexType n, URBG&& g) : bijection(n, g), n(n) {}
    
    __host__ __device__ IndexType operator()(IndexType i) const {
        auto upcast_i = static_cast<std::uint64_t>(i);
        auto upcast_n = static_cast<std::uint64_t>(n);
        
        if (upcast_i >= upcast_n) return upcast_i;
        
        do {
            upcast_i = bijection(upcast_i);
        } while (upcast_i >= upcast_n);
        return static_cast<IndexType>(upcast_i);
    }

    __host__ __device__ IndexType inverse(IndexType i) const {
        auto upcast_i = static_cast<std::uint64_t>(i);
        auto upcast_n = static_cast<std::uint64_t>(n);
        
        if (upcast_i >= upcast_n) return upcast_i;
        
        do {
            upcast_i = bijection.inverse(upcast_i);
        } while (upcast_i >= upcast_n);
        return static_cast<IndexType>(upcast_i);
    }
    __host__ __device__ IndexType size() const { return n; }
};

// Apply bijection kernel
template<typename IndexType>
__global__ void apply_bijection_kernel(IndexType* indices, size_t count, 
                                       random_bijection<IndexType> bijection) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        indices[idx] = bijection(indices[idx]);
    }
}

// Apply inverse bijection kernel
template<typename IndexType>
__global__ void apply_inverse_bijection_kernel(IndexType* indices, size_t count, 
                                       random_bijection<IndexType> bijection) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        indices[idx] = bijection.inverse(indices[idx]);
    }
}

// Compute send histogram kernel for 2D (vectors)
template<typename IndexType>
__global__ void compute_send_histogram_2d_kernel(const IndexType* indices, 
                                                 unsigned int* send_histo, size_t local_vector_count, size_t total_gpus) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < local_vector_count) {
        size_t target_gpu = indices[idx] / local_vector_count;
        
        
        atomicAdd(&send_histo[target_gpu], 1);
        
        
    }
}

// Pack send data kernel for 2D (vectors)
template<typename DataType, typename IndexType>
__global__ void pack_send_data_2d_kernel(const DataType* data, const IndexType* indices, size_t vector_count,
                                          DataType* send_data, IndexType* send_indices,
                                          unsigned int* send_offsets, unsigned int* counters,
                                          size_t local_vector_count, size_t total_gpus, size_t vector_length) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_count) {
        size_t target_gpu = indices[idx] / local_vector_count;
        
        size_t pos = atomicAdd(&counters[target_gpu], 1);
        size_t offset = send_offsets[target_gpu] + pos;
        
        // Copy entire vector
        for (size_t v = 0; v < vector_length; v++) {
            send_data[offset * vector_length + v] = data[idx * vector_length + v];
        }
        send_indices[offset] = indices[idx];
    
    }
}

// Unpack received data kernel for 2D (vectors)
template<typename DataType, typename IndexType>
__global__ void unpack_recv_data_2d_kernel(const DataType* recv_data, const IndexType* recv_indices,
                                           size_t vector_count, DataType* output, size_t local_vector_count, 
                                           size_t gpu_vector_offset, size_t vector_length) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_count) {
        IndexType global_vector_idx = recv_indices[idx];
        size_t local_vector_idx = global_vector_idx - gpu_vector_offset;
        for (size_t v = 0; v < vector_length; v++) {
            output[local_vector_idx * vector_length + v] = recv_data[idx * vector_length + v];
        }
    }
}


// Main global shuffle function for 2D data
template<typename DataType>
void global_shuffle_bidirectional(
  DataType* local_data, 
  size_t vector_length,
  size_t local_count, 
  size_t global_n,
  int rank_id, 
  int num_ranks,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
) {

    nvtxRangePush("global_shuffle_bidirectional");
    
    size_t total_gpus = num_ranks;
    size_t global_gpu_id = rank_id;
    
    // Calculate vector counts
    size_t local_vector_count = local_count / vector_length;
    size_t global_vector_count = global_n / vector_length;
    size_t gpu_vector_offset = rank_id * local_vector_count;
    // printf("-----Shuffle Info------\n");
    // printf("rank_id:             %d\n", rank_id);
    // printf("num_ranks:           %zu\n", num_ranks);
    // printf("total_gpus:          %zu\n", total_gpus);
    // printf("global_gpu_id:       %zu\n", global_gpu_id);
    // printf("vector_length:       %zu\n", vector_length);
    // printf("local_count:         %zu\n", local_count);
    // printf("global_n:            %zu\n", global_n);
    // printf("local_vector_count:  %zu\n", local_vector_count);
    // printf("global_vector_count: %zu\n", global_vector_count);
    // printf("gpu_vector_offset:   %zu\n", gpu_vector_offset);
    // printf("--------------------------------\n");
    const size_t block_size = 256;
    const size_t grid_size = (local_vector_count + block_size - 1) / block_size;
    
    // Step 3: Initialize index array with global vector IDs
    // printf("Rank %d, Step 3, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    nvtxRangePush("Initialize indices");
    thrust::device_vector<uint64_t> indices(local_vector_count);
    thrust::sequence(indices.begin(), indices.end(), gpu_vector_offset);
    nvtxRangePop();

    // // Debug first few indices
    //   for(int i = 0; i < std::min(5, (int)local_vector_count); i++) {
    //     uint64_t value = indices[i];  // This forces a device-to-host copy
    //     printf("Rank %d, Before bijection indices[%d]: %zu\n", rank_id, i, value);
    //   }
    //   printf("sizeof(DataType): %zu\n", sizeof(DataType));


    // Step 4: Apply Feistel bijection to vector indices
    // printf("Rank %d, Step 4, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    
    thrust::default_random_engine rng(42);
    random_bijection<uint64_t> bijection(global_vector_count, rng);

    apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
      thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);
        
    
    
      // // Debug first few indices
      // for(int i = 0; i < std::min(5, (int)local_vector_count); i++) {
      //   uint64_t value = indices[i];  // This forces a device-to-host copy
      //   printf("Rank %d, After bijection indices[%d]: %zu\n", rank_id, i, value);
      // }
      // printf("sizeof(DataType): %zu\n", sizeof(DataType));

    // Step 5: Compute send histogram for vectors
    // printf("Rank %d, Step 5, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    thrust::device_vector<unsigned int> send_histo(total_gpus, 0);
    thrust::device_vector<unsigned int> counters(total_gpus, 0);
    cudaStreamSynchronize(stream);


    // printf("grid_size: %zu, block_size: %zu\n", grid_size, block_size);
    compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(send_histo.data()), local_vector_count, total_gpus);
    cudaStreamSynchronize(stream);
    // for(int i = 0; i < total_gpus; i++) {
    //   unsigned int value = send_histo[i];
    //   printf("Rank %d, send_histo[%d]: %u\n", rank_id, i, value);
    // }
    
    // Compute send offsets
    thrust::device_vector<unsigned int> send_offsets(total_gpus);
    thrust::exclusive_scan(send_histo.begin(), send_histo.end(), send_offsets.begin());
    

    thrust::host_vector<unsigned int> h_send_offsets(total_gpus);
    thrust::copy_n(send_offsets.begin(), total_gpus, h_send_offsets.begin());

    // Pack send data (vectors)
    size_t total_send_vectors = thrust::reduce(send_histo.begin(), send_histo.end());
    thrust::device_vector<DataType> send_data(total_send_vectors * vector_length);
    thrust::device_vector<uint64_t> send_indices(total_send_vectors);
    cudaStreamSynchronize(stream);
    thrust::fill(counters.begin(), counters.end(), 0);
    pack_send_data_2d_kernel<<<grid_size, block_size, 0, stream>>>(
        local_data, thrust::raw_pointer_cast(indices.data()), local_vector_count,
        thrust::raw_pointer_cast(send_data.data()), thrust::raw_pointer_cast(send_indices.data()),
        thrust::raw_pointer_cast(send_offsets.data()), thrust::raw_pointer_cast(counters.data()),
        local_vector_count, total_gpus, vector_length);
    

    // Step 6: All2all exchange histograms
    // printf("Rank %d, Step 6, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    thrust::host_vector<unsigned int> h_recv_histo(total_gpus);
    thrust::device_vector<unsigned int> recv_histo(total_gpus);
    thrust::host_vector<unsigned int> h_send_histo = send_histo;


    thrust::copy_n(send_histo.begin(), total_gpus, h_send_histo.begin());


    
    // ///////////////////////////////////////////////////
    // Method 2: Use Feistel backward to calculate receive histogram (2D)
    nvtxRangePush("Compute receive histogram (bidirectional)");
    thrust::device_vector<uint64_t> recv_indices(local_vector_count);
    thrust::sequence(recv_indices.begin(), recv_indices.end(), gpu_vector_offset);
    
    thrust::default_random_engine rng_1(42);
    random_bijection<uint64_t> bijection_1(global_vector_count, rng_1);
    cudaStreamSynchronize(stream);
    apply_inverse_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(recv_indices.data()), local_vector_count, bijection_1);

        
      // // Debug first few indices
      // for(int i = 0; i < std::min(5, (int)local_vector_count); i++) {
      //   uint64_t value = recv_indices[i];  // This forces a device-to-host copy
      //   printf("Rank %d, After inverse bijection indices[%d]: %zu\n", rank_id, i, value);
      // }
      // printf("sizeof(DataType): %zu\n", sizeof(DataType));
    
    // Compute histogram from inverse indices
    thrust::fill(recv_histo.begin(), recv_histo.end(), 0);
    cudaStreamSynchronize(stream);
    compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(recv_indices.data()),
    thrust::raw_pointer_cast(recv_histo.data()), local_vector_count, total_gpus);
    cudaStreamSynchronize(stream);
    thrust::copy_n(recv_histo.begin(), total_gpus, h_recv_histo.begin());
    nvtxRangePop();

    
    // Step 7: Create recv buffers for vectors
    // printf("Rank %d, Step 7, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    
    size_t total_recv_vectors = thrust::reduce(recv_histo.begin(), recv_histo.end());
    thrust::device_vector<DataType> recv_data(total_recv_vectors * vector_length);
    
    thrust::device_vector<size_t> recv_offsets(total_gpus);
    thrust::host_vector<size_t> h_recv_offsets(total_gpus);
    
    thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
    thrust::copy_n(recv_offsets.begin(), total_gpus, h_recv_offsets.begin());

    // Step 8: All2all distribute data
    // printf("Rank %d, Step 8, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    nvtxRangePush("NCCL_ALL2ALL");
    CHECK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < total_gpus; ++i) {
        if (h_send_histo[i] > 0) {
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i] * vector_length,
                    h_send_histo[i] * vector_length, ncclInt64, i, *nccl_comm, stream));
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices.data()) + h_send_offsets[i],
                    h_send_histo[i], ncclUint64, i, *nccl_comm, stream));
        }
        
        if (h_recv_histo[i] > 0) {
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i] * vector_length,
                    h_recv_histo[i] * vector_length, ncclInt64, i, *nccl_comm, stream));
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
                    h_recv_histo[i], ncclUint64, i, *nccl_comm, stream));
        }
    }
    CHECK_NCCL(ncclGroupEnd());
    nvtxRangePop();
    // Step 9: Unpack to final positions (vectors)
    // printf("Rank %d, Step 9, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    // printf("total_recv_vectors: %zu\n", total_recv_vectors);
    
    // for (int i = 0; i < total_gpus; ++i) {
    //   printf("----------Rank %d, GPU %d------------------\n", rank_id, i);
    //   printf("Rank %d, send_histo[%d]: %u\n", rank_id, i,h_send_histo[i]);
    //   printf("Rank %d, recv_histo[%d]: %u\n", rank_id, i,h_recv_histo[i]);
    //   // for(int vec_id = 0; vec_id < h_recv_histo[i]; vec_id++) {
    //   //   uint64_t value1 = recv_indices[vec_id];
    //   //   printf("Rank %d, recv_indices[%d]: %zu\n", rank_id, vec_id, value1);
    //   // }
    // }
    // printf("Rank %d, gpu_vector_offset: %zu, total_recv_vectors: %zu \n", rank_id, gpu_vector_offset, total_recv_vectors);
    // printf("--------------------------------\n");
    const size_t unpack_grid_size = (total_recv_vectors + block_size - 1) / block_size;
    cudaStreamSynchronize(stream);
    unpack_recv_data_2d_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
        total_recv_vectors, local_data, local_vector_count, gpu_vector_offset, vector_length);
    
    cudaStreamSynchronize(stream);
    nvtxRangePop(); // End of global_shuffle_bidirectional
    // printf("Rank %d, End of global_shuffle_bidirectional, File: %s:%d\n", rank_id, __FILE__, __LINE__);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template <Type::Code CODE, int32_t DIM>
struct ShuffleImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(TaskContext& context,
    const legate::PhysicalStore& input_output_array,
    const size_t vector_count,
    const size_t vector_length,
    const bool is_index_space,
    const size_t local_rank,
    const size_t num_ranks,
    const size_t num_shuffle_ranks,
    const std::vector<comm::Communicator>& comms)
  {
    // CPU shuffle not yet implemented
    assert(false && "CPU shuffle not yet implemented");
  }
};

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
template <Type::Code CODE, int32_t DIM>
struct ShuffleImplBody<VariantKind::OMP, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input_output_array,
                  const size_t vector_count,
                  const size_t vector_length,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const size_t num_shuffle_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    // OMP shuffle not yet implemented
    assert(false && "OMP shuffle not yet implemented");
  }
};
#endif

template <Type::Code CODE, int32_t DIM>
struct ShuffleImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(TaskContext& context,
    const legate::PhysicalStore& input_output_array,
    const size_t vector_count,
    const size_t vector_length,
    const bool is_index_space,
    const size_t rank,
    const size_t num_ranks,
    const size_t num_shuffle_ranks,
    const std::vector<comm::Communicator>& comms)
  {
    auto rect = input_output_array.shape<DIM>();
    auto input_output = input_output_array.read_write_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input_output.accessor.is_dense_row_major(rect));
    
    auto stream = get_cached_stream();

    bool need_distributed_shuffle = (num_ranks > 1) && is_index_space;

    assert(num_ranks == num_shuffle_ranks);
    // For local shuffle (single node or within a node)
    if (!need_distributed_shuffle) {
    
          // TODO: Implement key-sort local shuffle
          assert(false && "Key-sort local shuffle not yet implemented");
    
    } else {
      // Handle distributed shuffle
      
      // Calculate distributed shuffle parameters
      size_t local_count = vector_count * vector_length / num_ranks; // each rank has local_count vectors
      size_t global_n = vector_count * vector_length;
      
      
      VAL* data_ptr = input_output.ptr(rect.lo);
      
      // printf("------Parameters - Rank %zu------\n"
      //       "vector_length: %zu\n"
      //       "local_count:   %zu\n"
      //       "global_n:      %zu\n"
      //       "rank:          %zu\n"
      //       "num_ranks:     %zu\n"
      //       "--------------------------------\n", rank, vector_length, local_count, global_n, rank, num_ranks);
      cudaEvent_t start1, stop1;
      cudaEventCreate(&start1);
      cudaEventCreate(&stop1);
      cudaEventRecord(start1, stream);
      global_shuffle_bidirectional<VAL>(
          data_ptr,
          vector_length,
          local_count,
          global_n,
          rank,
          num_ranks,
          comms[0].get<ncclComm_t*>(),
          stream
      );
      cudaEventRecord(stop1, stream);
      cudaEventSynchronize(stop1);
      float time1;
      cudaEventElapsedTime(&time1, start1, stop1);
      
      cudaEventDestroy(start1);
      cudaEventDestroy(stop1);

      // std::cout << "[Rank " << world_rank << "] global_shuffle_bidirectional time: " << time2 << " ms" << std::endl;
      printf("Rank %d, time: %f ms\n", rank, time1);


     
    }

    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <VariantKind KIND>
struct ShuffleImpl {
  template <Type::Code CODE, int DIM>
  void operator()(ShuffleArgs& args, TaskContext& context, 
    std::vector<comm::Communicator> comms) const
  {
    using VAL = type_of<CODE>;
    auto rect = args.input_output.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);
    // printf("ShuffleImpl: volume: %zu, __FILE__: %s, __LINE__: %d\n", volume, __FILE__, __LINE__);
    // printf("Datatype: %s\n", typeid(VAL).name());
    // printf("MIN_GPU_CHUNK_DEFAULT: %zu\n", MIN_GPU_CHUNK_DEFAULT);
    if (volume == 0) {
      return;
    }

    ShuffleImplBody<KIND, CODE, DIM>()(
        context,
        args.input_output,
        args.vector_count,
        args.vector_length,
        args.is_index_space,
        args.local_rank,
        args.num_ranks,
        args.num_shuffle_ranks,
        comms
    );
  }
};

static int get_rank(Domain domain, DomainPoint index_point)
{
  int domain_index = 0;
  auto hi          = domain.hi();
  auto lo          = domain.lo();
  for (int i = 0; i < domain.get_dim(); ++i) {
    if (i > 0) {
      domain_index *= hi[i] - lo[i] + 1;
    }
    domain_index += index_point[i];
  }
  return domain_index;
}

template <VariantKind KIND>
static void shuffle_template(TaskContext& context)
{
  // Extract arguments from TaskContext
  auto input_output = context.input(0);
  auto shape_span   = context.scalar(0).values<int64_t>();
  size_t d1 = shape_span[0];
  size_t d2 = 1;
  for (size_t i = 1; i < shape_span.size(); ++i) d2 *= shape_span[i];

  auto domain           = context.get_launch_domain();
  size_t local_rank     = get_rank(domain, context.get_task_index());
  size_t num_ranks      = domain.get_volume();
  size_t num_shuffle_ranks = domain.hi()[0] - domain.lo()[0] + 1;
  // printf("shape_span: %zu, %zu\n", d1, d2);
  // printf("local_rank: %zu, num_ranks: %zu, num_shuffle_ranks: %zu\n", local_rank, num_ranks, num_shuffle_ranks);
  
  ShuffleArgs args{
    input_output,
    d1,
    d2,
    !context.is_single_task(),
    local_rank,
    num_ranks,
    num_shuffle_ranks
  };
  
  double_dispatch(
    args.input_output.dim(), args.input_output.code(), ShuffleImpl<KIND>{}, args, context, context.communicators());
}

/*static*/ void ShuffleTask::gpu_variant(TaskContext context)
{
  shuffle_template<VariantKind::GPU>(context);
}

/*static*/ void ShuffleTask::cpu_variant(TaskContext context)
{
  // CPU variant not implemented yet - shuffle operations will assert false
  shuffle_template<VariantKind::CPU>(context);
}

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
/*static*/ void ShuffleTask::omp_variant(TaskContext context)
{
  // OMP variant not implemented yet - shuffle operations will assert false  
  shuffle_template<VariantKind::OMP>(context);
}
#endif

namespace  // unnamed
{
static const auto cupynumeric_reg_task_ = []() -> char {
  ShuffleTask::register_variants(); 
  return 0;
}();
}  // namespace

}  // namespace cupynumeric 