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

namespace {

template <typename T>
legate::Buffer<T> create_non_empty_buffer(
  size_t size, legate::Memory::Kind kind = legate::Memory::Kind::NO_MEMKIND)
{
  return legate::create_buffer<T>(std::max(size, size_t{1}), kind, alignof(T));
}

}

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

// __global__
// void init_curand_states(curandState *states, unsigned long seed, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         curand_init(seed, idx, 0, &states[idx]);
//     }
// }

// // 2. Device functor that uses cuRAND
// struct CurandFunctor {
//     curandState* states;
//     int n;
//     int gpu_rank;
//     __device__ int operator()() {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x;
//         if (idx < n) {
//             // return curand(&states[idx]) % 1000;
//             return idx + gpu_rank * n + (gpu_rank + 1) * 10000;
//         }
//         return 0;
//     }
// };



// class feistel_bijection {
// public:
//     using index_type = std::uint64_t;
    
//     template <class URBG>
//     __host__ __device__ feistel_bijection(std::uint64_t m, URBG&& g) {
//         std::uint64_t total_bits = get_cipher_bits(m);
//         left_side_bits = total_bits / 2;
//         left_side_mask = (1ull << left_side_bits) - 1;
//         right_side_bits = total_bits - left_side_bits;
//         right_side_mask = (1ull << right_side_bits) - 1;
        
//         thrust::uniform_int_distribution<std::uint32_t> dist;
//         for (std::uint32_t i = 0; i < num_rounds; i++) {
//             key[i] = dist(g);
//         }
//     }
    
//     __host__ __device__ std::uint64_t nearest_power_of_two() const {
//         return 1ull << (left_side_bits + right_side_bits);
//     }
    
//     __host__ __device__ std::uint64_t operator()(const std::uint64_t val) const {
//         std::uint32_t state[2] = {static_cast<std::uint32_t>(val >> right_side_bits),
//                                   static_cast<std::uint32_t>(val & right_side_mask)};
//         for (std::uint32_t i = 0; i < num_rounds; i++) {
//             std::uint32_t hi, lo;
//             constexpr std::uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
//             mulhilo(M0, state[0], hi, lo);
//             lo = (lo << (right_side_bits - left_side_bits)) | state[1] >> left_side_bits;
//             state[0] = ((hi ^ key[i]) ^ state[1]) & left_side_mask;
//             state[1] = lo & right_side_mask;
//         }
//         return (static_cast<std::uint64_t>(state[0]) << right_side_bits) | static_cast<std::uint64_t>(state[1]);
//     }

//     __host__ __device__ std::uint64_t inverse(const std::uint64_t val) const {
//         std::uint32_t state[2] = {static_cast<std::uint32_t>(val >> right_side_bits),
//                                   static_cast<std::uint32_t>(val & right_side_mask)};
//         for (std::uint32_t i = num_rounds; i != 0;) {
//           i -= 1;
//           std::uint32_t hi, lo, b;
//           b = state[1] & (std::uint32_t)(right_side_bits - left_side_bits);
//           constexpr std::uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
//           constexpr std::uint64_t M_inv = UINT64_C(0xF5F365BD212BDF9B);
          
//           mullo(M_inv, (state[1] >> (right_side_bits - left_side_bits)) & left_side_mask, lo);
//           lo = lo & left_side_mask;
//           mulhi(M0, lo, hi);
          
//           state[1] = ((b << left_side_bits) | (((hi ^ key[i]) ^ state[0]) & left_side_mask)) & right_side_mask;
//           state[0] = lo;
          
//         }
//         return (static_cast<std::uint64_t>(state[0]) << right_side_bits) | static_cast<std::uint64_t>(state[1]);
//       }

// private:
//     static __host__ __device__ void mulhilo(std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo) {
//         std::uint64_t product = a * b;
//         hi = static_cast<std::uint32_t>(product >> 32);
//         lo = static_cast<std::uint32_t>(product);
//     }

//     static __host__ __device__ void mullo(std::uint64_t a, std::uint64_t b, std::uint32_t& lo) {
//         std::uint64_t product = a * b;
//         lo = static_cast<std::uint32_t>(product);
//       }
      
//     static  __host__ __device__ void mulhi(std::uint64_t a, std::uint64_t b, std::uint32_t& hi) {
//         std::uint64_t product = a * b;
//         hi = static_cast<std::uint32_t>(product >> 32);
//       }
    
//     static __host__ __device__ std::uint64_t get_cipher_bits(std::uint64_t m) {
//         if (m <= 16) return 4;
//         std::uint64_t i = 0;
//         m--;
//         while (m != 0) { i++; m >>= 1; }
//         return i;
//     }
    
//     static constexpr std::uint32_t num_rounds = 24;
//     std::uint64_t right_side_bits, left_side_bits, right_side_mask, left_side_mask;
//     std::uint32_t key[num_rounds];
// };

// template <class IndexType>
// class random_bijection {
// private:
//     feistel_bijection bijection;
//     IndexType n;

// public:
//     using index_type = IndexType;
    
//     template <class URBG>
//     __host__ __device__ random_bijection(IndexType n, URBG&& g) : bijection(n, g), n(n) {}
    
//     __host__ __device__ IndexType operator()(IndexType i) const {
//         auto upcast_i = static_cast<std::uint64_t>(i);
//         auto upcast_n = static_cast<std::uint64_t>(n);
        
//         if (upcast_i >= upcast_n) return upcast_i;
        
//         do {
//             upcast_i = bijection(upcast_i);
//         } while (upcast_i >= upcast_n);
//         return static_cast<IndexType>(upcast_i);
//     }

//     __host__ __device__ IndexType inverse(IndexType i) const {
//         auto upcast_i = static_cast<std::uint64_t>(i);
//         auto upcast_n = static_cast<std::uint64_t>(n);
        
//         if (upcast_i >= upcast_n) return upcast_i;
        
//         do {
//             upcast_i = bijection.inverse(upcast_i);
//         } while (upcast_i >= upcast_n);
//         return static_cast<IndexType>(upcast_i);
//     }

    
    
//     __host__ __device__ IndexType size() const { return n; }
// };

// // Apply bijection kernel
// template<typename IndexType>
// __global__ void apply_bijection_kernel(IndexType* indices, size_t count, 
//                                        random_bijection<IndexType> bijection) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < count) {
//         indices[idx] = bijection(indices[idx]);
//     }
// }

// // Apply inverse bijection kernel
// template<typename IndexType>
// __global__ void apply_inverse_bijection_kernel(IndexType* indices, size_t count, 
//                                        random_bijection<IndexType> bijection) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < count) {
//         indices[idx] = bijection.inverse(indices[idx]);
//     }
// }

// // Compute send histogram kernel for 2D (vectors)
// template<typename IndexType>
// __global__ void compute_send_histogram_2d_kernel(const IndexType* indices, size_t vector_count,
//                                                  unsigned int* send_histo, size_t local_vector_count, size_t total_gpus) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vector_count) {
//         size_t target_gpu = indices[idx] / local_vector_count;
//         if (target_gpu < total_gpus) {
//             atomicAdd(&send_histo[target_gpu], 1);
//         }
//     }
// }

// // Pack send data kernel for 2D (vectors)
// template<typename DataType, typename IndexType>
// __global__ void pack_send_data_2d_kernel(const DataType* data, const IndexType* indices, size_t vector_count,
//                                           DataType* send_data, IndexType* send_indices,
//                                           unsigned int* send_offsets, unsigned int* counters,
//                                           size_t local_vector_count, size_t total_gpus, size_t vector_length) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vector_count) {
//         size_t target_gpu = indices[idx] / local_vector_count;
//         if (target_gpu < total_gpus) {
//             size_t pos = atomicAdd(&counters[target_gpu], 1);
//             size_t offset = send_offsets[target_gpu] + pos;
            
//             // Copy entire vector
//             for (size_t v = 0; v < vector_length; v++) {
//                 send_data[offset * vector_length + v] = data[idx * vector_length + v];
//             }
//             send_indices[offset] = indices[idx];
//         }
//     }
// }

// // Unpack received data kernel for 2D (vectors)
// template<typename DataType, typename IndexType>
// __global__ void unpack_recv_data_2d_kernel(const DataType* recv_data, const IndexType* recv_indices,
//                                            size_t vector_count, DataType* output, size_t local_vector_count, 
//                                            size_t gpu_vector_offset, size_t vector_length) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vector_count) {
//         IndexType global_vector_idx = recv_indices[idx];
//         size_t local_vector_idx = global_vector_idx - gpu_vector_offset;
//         if (local_vector_idx < local_vector_count) {
//             // Copy entire vector
//             for (size_t v = 0; v < vector_length; v++) {
//                 output[local_vector_idx * vector_length + v] = recv_data[idx * vector_length + v];
//             }
//         }
//     }
// }

// // Kernel to scatter data based on permuted indices for single rank
// template<typename DataType>
// __global__ void single_rank_scatter_kernel(const DataType* input_data, const uint64_t* indices, 
//                                           size_t count, DataType* output_data) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < count) {
//         uint64_t target_idx = indices[idx];
//         if (target_idx < count) {
//             output_data[target_idx] = input_data[idx];
//         }
//     }
// }

// // Main global shuffle function for 2D data
// template<typename DataType>
// void global_shuffle_bidirectional(DataType* local_data, size_t local_count, size_t global_n,
//                    size_t p1_nodes, size_t p2_gpus_per_node, int node_rank, int gpu_rank,
//                    ncclComm_t nccl_comm, cudaStream_t stream, size_t vector_length) {
    
//     nvtxRangePush("global_shuffle_bidirectional");
    
//     size_t total_gpus = p1_nodes * p2_gpus_per_node;
//     size_t global_gpu_id = node_rank * p2_gpus_per_node + gpu_rank;
    
//     // Calculate vector counts
//     size_t local_vector_count = local_count / vector_length;
//     size_t global_vector_count = global_n / vector_length;
//     size_t gpu_vector_offset = global_gpu_id * local_vector_count;
    
//     const size_t block_size = 256;
//     const size_t grid_size = (local_vector_count + block_size - 1) / block_size;
    
//     // Step 3: Initialize index array with global vector IDs
//     nvtxRangePush("Initialize indices");
//     thrust::device_vector<uint64_t> indices(local_vector_count);
//     thrust::sequence(indices.begin(), indices.end(), gpu_vector_offset);
//     nvtxRangePop();
    
//     // Print the first 5 vector indices before Feistel bijection
//     {
//         thrust::host_vector<uint64_t> h_indices_before(std::min<size_t>(5, local_vector_count));
//         thrust::copy_n(indices.begin(), std::min<size_t>(5, local_vector_count), h_indices_before.begin());
//         // printf("rank %d, First 5 vector indices before Feistel bijection: ", gpu_rank);
//         for (int i = 0; i < std::min<size_t>(5, local_vector_count); ++i) {
//             // printf("%zu ", h_indices_before[i]);
//         }
//         // printf("\n");
//     }

//     // Step 4: Apply Feistel bijection to vector indices
//     thrust::default_random_engine rng(42);
//     random_bijection<uint64_t> bijection(global_vector_count, rng);

//     apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
//         thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);

//     // Print the first 5 vector indices after Feistel bijection
//     {
//         cudaStreamSynchronize(stream); // Ensure kernel is done
//         thrust::host_vector<uint64_t> h_indices_after(std::min<size_t>(5, local_vector_count));
//         thrust::copy_n(indices.begin(), std::min<size_t>(5, local_vector_count), h_indices_after.begin());
//         // printf("rank %d, First 5 vector indices after Feistel bijection: ", gpu_rank);
//         for (int i = 0; i < std::min<size_t>(5, local_vector_count); ++i) {
//             // printf("%zu ", h_indices_after[i]);
//         }
//         // printf("\n");
//     }
//     // Step 5: Compute send histogram for vectors
//     thrust::device_vector<unsigned int> send_histo(total_gpus, 0);
//     thrust::device_vector<unsigned int> counters(total_gpus, 0);
    
//     compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
//         thrust::raw_pointer_cast(indices.data()), local_vector_count,
//         thrust::raw_pointer_cast(send_histo.data()), local_vector_count, total_gpus);
    
//     // Compute send offsets
//     thrust::device_vector<unsigned int> send_offsets(total_gpus);
//     thrust::exclusive_scan(send_histo.begin(), send_histo.end(), send_offsets.begin());
    

//     thrust::host_vector<unsigned int> h_send_offsets(total_gpus);
//     thrust::copy_n(send_offsets.begin(), total_gpus, h_send_offsets.begin());
//     {
//         cudaStreamSynchronize(stream); // Ensure kernel is done
//     // Print the send_offsets (exclusive_scan result) to check correctness
//     // printf("rank %d, send_offsets (exclusive_scan result): ", gpu_rank);
//     for (size_t i = 0; i < h_send_offsets.size(); ++i) {
//         // printf("%u ", h_send_offsets[i]);
//     }

//     // printf("\n");
//     }

//     // Pack send data (vectors)
//     size_t total_send_vectors = thrust::reduce(send_histo.begin(), send_histo.end());
//     thrust::device_vector<DataType> send_data(total_send_vectors * vector_length);
//     thrust::device_vector<uint64_t> send_indices(total_send_vectors);
    
//     thrust::fill(counters.begin(), counters.end(), 0);
//     pack_send_data_2d_kernel<<<grid_size, block_size, 0, stream>>>(
//         local_data, thrust::raw_pointer_cast(indices.data()), local_vector_count,
//         thrust::raw_pointer_cast(send_data.data()), thrust::raw_pointer_cast(send_indices.data()),
//         thrust::raw_pointer_cast(send_offsets.data()), thrust::raw_pointer_cast(counters.data()),
//         local_vector_count, total_gpus, vector_length);
    
//     // printf("total_send_vectors: %zu\n", total_send_vectors);
//     // Step 6: All2all exchange histograms
    
//     thrust::host_vector<unsigned int> h_recv_histo(total_gpus);
//     thrust::device_vector<unsigned int> recv_histo(total_gpus);
//     thrust::host_vector<unsigned int> h_send_histo = send_histo;


//     thrust::copy_n(send_histo.begin(), total_gpus, h_send_histo.begin());
//     // for (int i = 0; i < total_gpus; ++i) {
//     //     // printf("before all2all: send_histo[%d]: %u, recv_histo[%d]: %u\n", i, h_send_histo[i], i, h_recv_histo[i]);
//     // }


    
//     // ///////////////////////////////////////////////////
//     // Method 2: Use Feistel backward to calculate receive histogram (2D)
//     nvtxRangePush("Compute receive histogram (bidirectional)");
//     thrust::device_vector<uint64_t> recv_indices(local_vector_count);
//     thrust::sequence(recv_indices.begin(), recv_indices.end(), gpu_vector_offset);
    
//     thrust::default_random_engine rng_1(42);
//     random_bijection<uint64_t> bijection_1(global_vector_count, rng_1);
    
//     apply_inverse_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
//     thrust::raw_pointer_cast(recv_indices.data()), local_vector_count, bijection_1);
    
//     // Compute histogram from inverse indices
//     thrust::fill(recv_histo.begin(), recv_histo.end(), 0);
//     compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
//     thrust::raw_pointer_cast(recv_indices.data()), local_vector_count,
//     thrust::raw_pointer_cast(recv_histo.data()), local_vector_count, total_gpus);
//     thrust::copy_n(recv_histo.begin(), total_gpus, h_recv_histo.begin());
//     nvtxRangePop();

    
//     // Step 7: Create recv buffers for vectors
//     size_t total_recv_vectors = thrust::reduce(recv_histo.begin(), recv_histo.end());
//     thrust::device_vector<DataType> recv_data(total_recv_vectors * vector_length);
    
//     thrust::device_vector<size_t> recv_offsets(total_gpus);
//     thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
    

//     thrust::host_vector<size_t> h_recv_offsets(total_gpus);
//     thrust::copy_n(recv_offsets.begin(), total_gpus, h_recv_offsets.begin());



//     thrust::host_vector<DataType> h_send_data(total_send_vectors * vector_length);
//     thrust::host_vector<uint64_t> h_send_indices(total_send_vectors);
//     thrust::copy_n(send_data.begin(), total_send_vectors * vector_length, h_send_data.begin());
//     thrust::copy_n(send_indices.begin(), total_send_vectors, h_send_indices.begin());
//     for (int i = 0; i < total_recv_vectors; ++i) {
//         // printf("gpu_rank: %d, send_data[%d]: %d, send_indices[%d]: %zu\n", gpu_rank, i, h_send_data[i], i, h_send_indices[i]);
//     }

//     for (int i = 0; i < total_gpus; ++i) {
//         // printf("gpu_rank: %d, send_offsets[%d]: %zu, recv_offsets[%d]: %zu, h_send_histo[%d]: %u, h_recv_histo[%d]: %u\n", gpu_rank, i, h_send_offsets[i], i, h_recv_offsets[i], i, h_send_histo[i], i, h_recv_histo[i]);
//     }

//     // Step 8: All2all distribute data
//     CHECK_NCCL(ncclGroupStart());
//     for (size_t i = 0; i < total_gpus; ++i) {
//         if (h_send_histo[i] > 0) {
//             CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i] * vector_length,
//                     h_send_histo[i] * vector_length, ncclInt, i, nccl_comm, stream));
//             CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices.data()) + h_send_offsets[i],
//                     h_send_histo[i], ncclUint64, i, nccl_comm, stream));
//         }
        
//         if (h_recv_histo[i] > 0) {
//             CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i] * vector_length,
//                     h_recv_histo[i] * vector_length, ncclInt, i, nccl_comm, stream));
//             CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
//                     h_recv_histo[i], ncclUint64, i, nccl_comm, stream));
//         }
//     }
//     CHECK_NCCL(ncclGroupEnd());

//     thrust::host_vector<DataType> h_recv_data(total_recv_vectors * vector_length);
//     thrust::host_vector<uint64_t> h_recv_indices(total_recv_vectors);
//     thrust::copy_n(recv_data.begin(), total_recv_vectors * vector_length, h_recv_data.begin());
//     thrust::copy_n(recv_indices.begin(), total_recv_vectors, h_recv_indices.begin());
//     for (int i = 0; i < total_recv_vectors; ++i) {
//         // printf("gpu_rank: %d, recv_data[%d]: %d, recv_indices[%d]: %zu\n", gpu_rank, i, h_recv_data[i], i, h_recv_indices[i]);
//     }
//     // Step 9: Unpack to final positions (vectors)
//     const size_t unpack_grid_size = (total_recv_vectors + block_size - 1) / block_size;
//     unpack_recv_data_2d_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
//         thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
//         total_recv_vectors, local_data, local_vector_count, gpu_vector_offset, vector_length);
    
//     cudaStreamSynchronize(stream);
//     nvtxRangePop(); // End of global_shuffle_bidirectional
// }

// // Main global shuffle function for 2D data
// template<typename DataType>
// void global_shuffle(DataType* local_data, size_t local_count, size_t global_n,
//                    size_t p1_nodes, size_t p2_gpus_per_node, int node_rank, int gpu_rank,
//                    ncclComm_t nccl_comm, cudaStream_t stream, size_t vector_length) {
    
//     nvtxRangePush("global_shuffle");
//     size_t total_gpus = p1_nodes * p2_gpus_per_node;
//     size_t global_gpu_id = node_rank * p2_gpus_per_node + gpu_rank;
    
//     // Calculate vector counts for 2D
//     size_t local_vector_count = local_count / vector_length;
//     size_t global_vector_count = global_n / vector_length;
//     size_t gpu_vector_offset = global_gpu_id * local_vector_count;
    
//     const size_t block_size = 256;
//     const size_t grid_size = (local_vector_count + block_size - 1) / block_size;
    
//     // Step 3: Initialize index array with global vector IDs
//     thrust::device_vector<uint64_t> indices(local_vector_count);
//     thrust::sequence(indices.begin(), indices.end(), gpu_vector_offset);
    
//     // // Step 4: Apply Feistel bijection
//     // thrust::default_random_engine rng(42);
//     // thrust::detail::random_bijection<uint64_t> bijection(global_n, rng);
    
//     // apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
//     //     thrust::raw_pointer_cast(indices.data()), local_count, bijection);
//     // Print the first 5 vector indices before Feistel bijection
//     {
//         thrust::host_vector<uint64_t> h_indices_before(std::min<size_t>(5, local_vector_count));
//         thrust::copy_n(indices.begin(), std::min<size_t>(5, local_vector_count), h_indices_before.begin());
//         // printf("rank %d, First 5 vector indices before Feistel bijection: ", gpu_rank);
//         for (int i = 0; i < std::min<size_t>(5, local_vector_count); ++i) {
//             // printf("%zu ", h_indices_before[i]);
//         }
//         // printf("\n");
//     }

//     // Step 4: Apply Feistel bijection to vector indices
//     thrust::default_random_engine rng(42);
//     random_bijection<uint64_t> bijection(global_vector_count, rng);

//     apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
//         thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);

//     // Print the first 5 vector indices after Feistel bijection
//     {
//         cudaStreamSynchronize(stream); // Ensure kernel is done
//         thrust::host_vector<uint64_t> h_indices_after(std::min<size_t>(5, local_vector_count));
//         thrust::copy_n(indices.begin(), std::min<size_t>(5, local_vector_count), h_indices_after.begin());
//         // printf("rank %d, First 5 vector indices after Feistel bijection: ", gpu_rank);
//         for (int i = 0; i < std::min<size_t>(5, local_vector_count); ++i) {
//             // printf("%zu ", h_indices_after[i]);
//         }
//         // printf("\n");
//     }
//     // Step 5: Compute send histogram
//     thrust::device_vector<unsigned int> send_histo(total_gpus, 0);
//     thrust::device_vector<unsigned int> counters(total_gpus, 0);
    
//     compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
//         thrust::raw_pointer_cast(indices.data()), local_vector_count,
//         thrust::raw_pointer_cast(send_histo.data()), local_vector_count, total_gpus);
    
//     // Compute send offsets
//     thrust::device_vector<unsigned int> send_offsets(total_gpus);
//     thrust::exclusive_scan(send_histo.begin(), send_histo.end(), send_offsets.begin());
    

//     thrust::host_vector<unsigned int> h_send_offsets(total_gpus);
//     thrust::copy_n(send_offsets.begin(), total_gpus, h_send_offsets.begin());
//     {
//         cudaStreamSynchronize(stream); // Ensure kernel is done
//     // Print the send_offsets (exclusive_scan result) to check correctness
//     // printf("rank %d, send_offsets (exclusive_scan result): ", gpu_rank);
//     for (size_t i = 0; i < h_send_offsets.size(); ++i) {
//         // printf("%u ", h_send_offsets[i]);
//     }

//     // printf("\n");
//     }

//     // Pack send data (vectors)
//     size_t total_send_vectors = thrust::reduce(send_histo.begin(), send_histo.end());
//     thrust::device_vector<DataType> send_data(total_send_vectors * vector_length);
//     thrust::device_vector<uint64_t> send_indices(total_send_vectors);
    
//     thrust::fill(counters.begin(), counters.end(), 0);
//     pack_send_data_2d_kernel<<<grid_size, block_size, 0, stream>>>(
//         local_data, thrust::raw_pointer_cast(indices.data()), local_vector_count,
//         thrust::raw_pointer_cast(send_data.data()), thrust::raw_pointer_cast(send_indices.data()),
//         thrust::raw_pointer_cast(send_offsets.data()), thrust::raw_pointer_cast(counters.data()),
//         local_vector_count, total_gpus, vector_length);
    
//     // printf("total_send: %zu\n", total_send);
//     // Step 6: All2all exchange histograms
    
//     thrust::host_vector<unsigned int> h_recv_histo(total_gpus);
//     thrust::device_vector<unsigned int> recv_histo(total_gpus);
//     thrust::host_vector<unsigned int> h_send_histo = send_histo;


//     thrust::copy_n(send_histo.begin(), total_gpus, h_send_histo.begin());
//     for (int i = 0; i < total_gpus; ++i) {
//         // printf("before all2all: send_histo[%d]: %u, recv_histo[%d]: %u\n", i, h_send_histo[i], i, h_recv_histo[i]);
//     }
    
//     CHECK_NCCL(ncclGroupStart());
//     for (size_t i = 0; i < total_gpus; ++i) {
//         CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_histo.data()) + i, 1, ncclUint32, i, nccl_comm, stream));
//         CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_histo.data()) + i, 1, ncclUint32, i, nccl_comm, stream));
//     }
//     CHECK_NCCL(ncclGroupEnd());
    

//     thrust::copy_n(recv_histo.begin(), total_gpus, h_recv_histo.begin());
//     for (int i = 0; i < total_gpus; ++i) {
//         // printf("gpu_rank: %d, after all2all: send_histo[%d]: %u, recv_histo[%d]: %u\n", gpu_rank, i, h_send_histo[i], i, h_recv_histo[i]);
//     }
    
//     // Step 7: Create recv buffers for vectors
//     size_t total_recv_vectors = thrust::reduce(recv_histo.begin(), recv_histo.end());
//     // printf("gpu_rank: %d, total_recv_vectors: %zu\n", gpu_rank, total_recv_vectors);
//     thrust::device_vector<DataType> recv_data(total_recv_vectors * vector_length);
//     thrust::device_vector<uint64_t> recv_indices(total_recv_vectors);
    
//     thrust::device_vector<size_t> recv_offsets(total_gpus);
//     thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
    

//     thrust::host_vector<size_t> h_recv_offsets(total_gpus);
//     thrust::copy_n(recv_offsets.begin(), total_gpus, h_recv_offsets.begin());



//     thrust::host_vector<DataType> h_send_data(total_send_vectors * vector_length);
//     thrust::host_vector<uint64_t> h_send_indices(total_send_vectors);
//     thrust::copy_n(send_data.begin(), total_send_vectors * vector_length, h_send_data.begin());
//     thrust::copy_n(send_indices.begin(), total_send_vectors, h_send_indices.begin());
//     for (int i = 0; i < total_recv_vectors; ++i) {
//         // printf("gpu_rank: %d, send_data[%d]: %d, send_indices[%d]: %zu\n", gpu_rank, i, h_send_data[i], i, h_send_indices[i]);
//     }

//     for (int i = 0; i < total_gpus; ++i) {
//         // printf("gpu_rank: %d, send_offsets[%d]: %zu, recv_offsets[%d]: %zu, h_send_histo[%d]: %u, h_recv_histo[%d]: %u\n", gpu_rank, i, h_send_offsets[i], i, h_recv_offsets[i], i, h_send_histo[i], i, h_recv_histo[i]);
//     }

//     // Step 8: All2all distribute data
//     CHECK_NCCL(ncclGroupStart());
//     for (size_t i = 0; i < total_gpus; ++i) {
//         if (h_send_histo[i] > 0) {
//             CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i] * vector_length,
//                     h_send_histo[i] * vector_length, ncclInt, i, nccl_comm, stream));
//             CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices.data()) + h_send_offsets[i],
//                     h_send_histo[i], ncclUint64, i, nccl_comm, stream));
//         }
        
//         if (h_recv_histo[i] > 0) {
//             CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i] * vector_length,
//                     h_recv_histo[i] * vector_length, ncclInt, i, nccl_comm, stream));
//             CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
//                     h_recv_histo[i], ncclUint64, i, nccl_comm, stream));
//         }
//     }
//     CHECK_NCCL(ncclGroupEnd());

//     thrust::host_vector<DataType> h_recv_data(total_recv_vectors * vector_length);
//     thrust::host_vector<uint64_t> h_recv_indices(total_recv_vectors);
//     thrust::copy_n(recv_data.begin(), total_recv_vectors * vector_length, h_recv_data.begin());
//     thrust::copy_n(recv_indices.begin(), total_recv_vectors, h_recv_indices.begin());
//     for (int i = 0; i < total_recv_vectors; ++i) {
//         // printf("gpu_rank: %d, recv_data[%d]: %d, recv_indices[%d]: %zu\n", gpu_rank, i, h_recv_data[i], i, h_recv_indices[i]);
//     }
//     // Step 9: Unpack to final positions (vectors)
//     const size_t unpack_grid_size = (total_recv_vectors + block_size - 1) / block_size;
//     unpack_recv_data_2d_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
//         thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
//         total_recv_vectors, local_data, local_vector_count, gpu_vector_offset, vector_length);
    
//     cudaStreamSynchronize(stream);
//     nvtxRangePop(); // End of global_shuffle
// }



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
struct ShuffleImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(TaskContext& context,
                  const legate::PhysicalStore& input_output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t vector_length,
                  const ShuffleMethod method,
                  const DistributedShuffleMethod distributed_method,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const size_t num_shuffle_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input_output = input_output_array.read_write_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input_output.accessor.is_dense_row_major(rect));

    auto stream = get_cached_stream();

    bool need_distributed_shuffle = (num_ranks > 1) && is_index_space;
    
    if (volume == 0) {
      return;
    }

    // For local shuffle (single node or within a node)
    if (!need_distributed_shuffle) {
      // Handle local shuffle methods
      switch (method) {
        case ShuffleMethod::FISHER_YATES:
          // TODO: Implement Fisher-Yates local shuffle
          assert(false && "Fisher-Yates local shuffle not yet implemented");
          break;
        case ShuffleMethod::FEISTEL:
        case ShuffleMethod::FEISTEL_BIDIRECTIONAL:
          // TODO: Implement Feistel local shuffle
          assert(false && "Feistel local shuffle not yet implemented");
          break;
        case ShuffleMethod::KEY_SORT:
        default:
          // TODO: Implement key-sort local shuffle
          assert(false && "Key-sort local shuffle not yet implemented");
          break;
      }
    } else {
      // Handle distributed shuffle
      assert(is_index_space);
      
      // Calculate distributed shuffle parameters
      size_t p2_gpus_per_node = num_shuffle_ranks;
      size_t total_gpus = num_ranks;
      size_t local_count = volume;
      size_t global_n = local_count * total_gpus;
      
      // Determine node and device ranks
      int node_rank = local_rank / p2_gpus_per_node;
      int device_id = local_rank % p2_gpus_per_node;
      
      VAL* data_ptr = input_output.ptr(rect.lo);
      
      switch (distributed_method) {
        case DistributedShuffleMethod::FEISTEL_BIDIRECTIONAL:
        assert(false && "Feistel bidirectional not yet implemented");
        //   global_shuffle_bidirectional<VAL>(
        //       data_ptr,
        //       local_count,
        //       global_n,
        //       total_gpus / p2_gpus_per_node,
        //       p2_gpus_per_node,
        //       node_rank,
        //       device_id,
        //       comms[0].get<ncclComm_t*>(),
        //       stream,
        //       vector_length
        //   );
          break;
        case DistributedShuffleMethod::FEISTEL_ALL2ALL:
        assert(false && "Feistel all2all not yet implemented");
        //   global_shuffle<VAL>(
        //       data_ptr,
        //       local_count,
        //       global_n,
        //       total_gpus / p2_gpus_per_node,
        //       p2_gpus_per_node,
        //       node_rank,
        //       device_id,
        //       comms[0].get<ncclComm_t*>(),
        //       stream,
        //       vector_length
        //   );
          break;
        case DistributedShuffleMethod::FISHER_YATES_GLOBAL:
        assert(false && "Fisher-Yates local shuffle not yet implemented");
          break;

      }
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
    
    if (volume == 0) {
      return;
    }

    ShuffleImplBody<KIND, CODE, DIM>{}(
        context,
        args.input_output,
        pitches,
        rect,
        volume,
        args.first_axis_size,
        args.method,
        DistributedShuffleMethod::FEISTEL_BIDIRECTIONAL, // Default method
        args.is_index_space,
        args.local_rank,
        args.num_ranks,
        args.num_shuffle_ranks,
        comms
        {} // TODO: Get communicators from context
    );
  }
};

template <VariantKind KIND>
static void shuffle_template(TaskContext context)
{
  // Extract arguments from TaskContext
  auto input_output = context.inputs()[0];
  auto method = static_cast<ShuffleMethod>(context.scalars()[0].value<int32_t>());
  auto volume = static_cast<size_t>(context.scalars()[1].value<int64_t>());
  auto first_axis_size = static_cast<size_t>(context.scalars()[2].value<int64_t>());
  auto is_index_space = static_cast<bool>(context.scalars()[3].value<bool>());
  auto local_rank = static_cast<size_t>(context.scalars()[4].value<int64_t>());
  auto num_ranks = static_cast<size_t>(context.scalars()[5].value<int64_t>());
  auto num_shuffle_ranks = static_cast<size_t>(context.scalars()[6].value<int64_t>());

  ShuffleArgs args{
    input_output,
    method,
    volume,
    first_axis_size,
    is_index_space,
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


}  // namespace cupynumeric 