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
/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

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
template<int DIM_input>
__global__ void compute_send_histogram(const uint64_t* indices, 
                                                 unsigned int* send_histo, 
                                                 legate::Rect<DIM_input>* rect_buf, 
                                                 int* shuffle_ranks,
                                                 size_t local_volume,
                                                 int rank_id,
                                                 size_t num_ranks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < local_volume) {
      int target_rank = 0;
      int is_in_rect = 1;
      legate::Point<DIM_input> point;
      int data_idx = idx;
      for(int d = DIM_input - 1; d > 0; d--){
        point[d] = rect_buf[rank_id].lo[d] + (data_idx % (rect_buf[rank_id].hi[d] - rect_buf[rank_id].lo[d] + 1));
        data_idx /= (rect_buf[target_rank].hi[d] - rect_buf[target_rank].lo[d] + 1);
      }
      point[0] = indices[(data_idx % (rect_buf[rank_id].hi[0] - rect_buf[rank_id].lo[0] + 1))];
      

      for(; target_rank < num_ranks; target_rank++){
        is_in_rect = 1;
        for(int d = 0; d < DIM_input; d++){
          if(rect_buf[target_rank].lo[d] > point[d] || rect_buf[target_rank].hi[d] < point[d]){
            is_in_rect = 0;
            break;
          }
        }
        if(is_in_rect){
          atomicAdd(&send_histo[target_rank], 1);
          break;
        }
      }
    }
}

// // Pack send data kernel for 2D (vectors)
// template<typename DataType, int DIM_input>
// __global__ void pack_send_data_2d_kernel(const DataType* data, const IndexType* indices, legate::Rect<DIM_input>* rect_buf, size_t local_volume,
//                                           DataType* send_data, IndexType* send_indices,
//                                           unsigned int* send_offsets, unsigned int* counters,
//                                           size_t num_ranks, int rank_id) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < local_volume) {
//       int target_rank = 0;
//       int is_in_rect = 1;
      
//       legate::Point<DIM_input> point;
//       int data_idx = idx;
//       for(int d = DIM_input - 1; d > 0; d--){
//         point[d] = rect_buf[rank_id].lo[d] + (data_idx % (rect_buf[rank_id].hi[d] - rect_buf[rank_id].lo[d] + 1));
//         data_idx /= (rect_buf[target_rank].hi[d] - rect_buf[target_rank].lo[d] + 1);
//       }
//       point[0] = indices[(data_idx % (rect_buf[rank_id].hi[d] - rect_buf[rank_id].lo[d] + 1))];
      
//       for(; target_rank < num_ranks; target_rank++){
//         is_in_rect = 1;
//         for(int d = 0; d < DIM_input; d++){
//           if(rect_buf[target_rank].lo[d] > point[d] || rect_buf[target_rank].hi[d] < point[d]){
//             is_in_rect = 0;
//             break;
//           }
//         }
//         if(is_in_rect){
//           break;
//         }
//       }
//       size_t pos = atomicAdd(&counters[target_rank], 1);
//       size_t offset = send_offsets[target_rank] + pos;
//       send_data[idx] = data[idx];
//       send_indices[offset] = indices[idx];
    
//     }
// }

// // Unpack received data kernel for 2D (vectors)
// template<typename DataType, typename IndexType>
// __global__ void unpack_recv_data_2d_kernel(const DataType* recv_data, const IndexType* recv_indices,
//                                            size_t local_volume, DataType* output,
//                                            size_t gpu_vector_offset, size_t vector_length) {
//     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < vector_count) {
//         IndexType global_vector_idx = recv_indices[idx];
//         legate::Point<DIM_input> point;
//         int data_idx = idx;
//         for(int d = DIM_input - 1; d > 0; d--){
//           point[d] = rect_buf[rank_id].lo[d] + (data_idx % (rect_buf[rank_id].hi[d] - rect_buf[rank_id].lo[d] + 1));
//           data_idx /= (rect_buf[target_rank].hi[d] - rect_buf[target_rank].lo[d] + 1);
//         }
//         point[0] = recv_indices[(data_idx % (rect_buf[rank_id].hi[d] - rect_buf[rank_id].lo[d] + 1))];
      

//         size_t local_vector_idx = global_vector_idx - gpu_vector_offset;
//         for (size_t v = 0; v < vector_length; v++) {
//             output[local_vector_idx * vector_length + v] = recv_data[idx * vector_length + v];
//         }
//     }
// }

template<typename DataType, int DIM>
void shuffle_regular_tiling(
  DataType* local_data, 
  thrust::host_vector<legate::Rect<DIM>> global_rects_host,
  size_t global_vector_count, // First dimension size, global
  int rank, 
  int num_ranks,
  const Domain domain,
  const DomainPoint index_point,
  std::vector<int> shuffle_ranks,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
)
{
  // 0. Initilization
  // First dimension size, local
  size_t local_vector_count = global_rects_host[rank].hi[0] - global_rects_host[rank].lo[0] + 1;
  
  // Kernel parameters
  const size_t block_size = 256;
  const size_t grid_size = (local_vector_count + block_size - 1) / block_size;

  // First dimension rank
  int first_dimension_rank = domain.hi()[0] - domain.lo()[0] + 1;
  
    
  // Global rects device
  auto global_rects = legate::create_buffer<legate::Rect<DIM>>(num_ranks, Memory::Kind::GPU_FB_MEM);
  auto shuffle_ranks_device = legate::create_buffer<int>(first_dimension_rank, Memory::Kind::GPU_FB_MEM);
  
  cudaMemcpy(global_rects.ptr(0), thrust::raw_pointer_cast(global_rects_host.data()), num_ranks * sizeof(legate::Rect<DIM>), cudaMemcpyHostToDevice);
  cudaMemcpy(shuffle_ranks_device.ptr(0), shuffle_ranks.data(), first_dimension_rank * sizeof(int), cudaMemcpyHostToDevice);
  
  
  // Bijection object 
  thrust::device_vector<uint64_t> indices(local_vector_count);
  thrust::host_vector<uint64_t> h_indices(local_vector_count);
  thrust::default_random_engine rng(42);
  random_bijection<uint64_t> bijection(global_vector_count, rng);

  thrust::sequence(indices.begin(), indices.end(), global_rects_host[rank].lo[0]);
  apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);

  // Product of other dimensions
  int vector_length = 1;
  for(int i = 1; i < DIM; i++){
    vector_length *= global_rects_host[rank].hi[i] - global_rects_host[rank].lo[i] + 1;
  }

  // Buffer:
  // Send indices array: first dimension only
  thrust::device_vector<uint64_t> send_indices(local_vector_count);
  thrust::host_vector<uint64_t> h_send_indices(local_vector_count);
  // Recv indices array: first dimension only
  thrust::device_vector<uint64_t> recv_indices(local_vector_count);
  thrust::host_vector<uint64_t> h_recv_indices(local_vector_count);

  // Send data array
  thrust::device_vector<DataType> send_data(local_vector_count * vector_length);
  // Recv data array
  thrust::device_vector<DataType> recv_data(local_vector_count * vector_length);

  // Histogram array
  thrust::device_vector<unsigned int> histogram(first_dimension_rank);
  thrust::host_vector<unsigned int> h_histogram(first_dimension_rank);
  // Exclusive scan array
  thrust::device_vector<unsigned int> exclusive_scan_buf(first_dimension_rank);
  thrust::host_vector<unsigned int> h_exclusive_scan_buf(first_dimension_rank);
  
  // 1. Feistel Bijection
  // Apply to first dimension only
  apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);
  
  // 2. Compute send histogram
  compute_send_histogram<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(indices.data()),
    thrust::raw_pointer_cast(histogram.data()),
    global_rects.ptr(0),
    shuffle_ranks_device.ptr(0),
    local_vector_count,
    rank,
    first_dimension_rank);

    cudaMemcpy(thrust::raw_pointer_cast(h_histogram.data()), thrust::raw_pointer_cast(histogram.data()), first_dimension_rank * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(thrust::raw_pointer_cast(h_indices.data()), thrust::raw_pointer_cast(indices.data()), local_vector_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("rank %d, local_vector_count: %zu\n", rank, local_vector_count);
    for(int i = 0; i < local_vector_count; i++){
      auto index = h_indices[i];
      printf("indices[%d]: %zu\n", i, index);
    }
    for(int i = 0; i < first_dimension_rank; i++){
      printf("histogram[%d]: %u\n", i, h_histogram[i]);
    }

  // 3. Pack send data and send indices

  // 4. Inverse Feistel Bijection
  //    or All2all request histogram
  
  
  // 5. All2All exchange send data and send indices
  //    Indices exchange cannot be avoided since pack is out-of-order due to atomicAdd.

  
  // 6. Unpack received data
  // 

}



template<typename DataType, int DIM>
void shuffle_fancy_indexing(
){
  // Todo
}





template<typename DataType, int DIM>
void global_shuffle_bidirectional(
  DataType* local_data, 
  const legate::Rect<DIM> rect, 
  size_t global_vector_count,
  int rank, 
  int num_ranks,
  const Domain domain,
  const DomainPoint index_point,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
) {

  // 1. Exchange rects
  auto global_rects = create_buffer<int64_t>(num_ranks * DIM * 2, Memory::Kind::GPU_FB_MEM);
  thrust::host_vector<legate::Rect<DIM>> global_rects_host(num_ranks);
  auto rect_device = create_buffer<int64_t>(DIM * 2, Memory::Kind::GPU_FB_MEM);

  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(global_rects.ptr(0), 0, 
  num_ranks * DIM * 2 * sizeof(int64_t), stream));

  cudaMemcpy(rect_device.ptr(0), &rect, sizeof(rect), cudaMemcpyHostToDevice);
  

  cudaStreamSynchronize(stream);
  CHECK_NCCL(ncclAllGather((void*)rect_device.ptr(0), 
  (void*)global_rects.ptr(0), 
  sizeof(int64_t) * 2 * DIM, 
  ncclInt8, 
  *nccl_comm, 
  stream));
  cudaStreamSynchronize(stream);

  cudaMemcpy(thrust::raw_pointer_cast(global_rects_host.data()), thrust::raw_pointer_cast(global_rects.ptr(0)), num_ranks * DIM * 2 * sizeof(int64_t), cudaMemcpyDeviceToHost);

  
  for(int i = 0; i < num_ranks; i++){
    for(int j = 0; j < DIM; j++){
      printf("Rank %d, rect: lo %zu -- hi %zu\n", rank, rect.lo[j], rect.hi[j]);
      printf("Rank %d, global_rects_host[%d][%d]: lo %zu -- hi %zu\n", i, j, global_rects_host[i].lo[j], global_rects_host[i].hi[j]);
    }
  }


  // 2. Verify whether regular tiling
  bool is_regular_tiling = true;
  auto hi = domain.hi();
  auto lo = domain.lo();
  int first_dimension_rank = domain.hi()[0] - domain.lo()[0] + 1;
  std::vector<int> shuffle_ranks(first_dimension_rank);
  
  for(int i = 0; i < hi[0] - lo[0] + 1; i++){
    DomainPoint index_point_i = index_point;
    index_point_i[0] = lo[0] + i;
    int rank_i = get_rank(domain, index_point_i);
    shuffle_ranks[i] = rank_i;
    Rect<DIM> rect_i = global_rects_host[rank_i];
    for(int j = i + 1; j < hi[0] - lo[0] + 1; j++){
      DomainPoint index_point_j = index_point;
      index_point_j[0] = lo[0] + j;
      int rank_j = get_rank(domain, index_point_j);
      Rect<DIM> rect_j = global_rects_host[rank_j];

      for(int d = 1; d < DIM; d++){
        if(rect_i.lo[d] != rect_j.lo[d] || rect_i.hi[d] != rect_j.hi[d]){
          is_regular_tiling = false;
          break;
        }
      }
      if(!is_regular_tiling){
        break;
      }
    }
  }

  if (is_regular_tiling){
    printf("Regular Tiling\n");
    // 3a. Call Regular Tiling
    shuffle_regular_tiling(local_data, global_rects_host, global_vector_count, rank, num_ranks, domain, index_point, shuffle_ranks, nccl_comm, stream);
  } else {
    // 3b. Call Fancy Indexing
    // shuffle_fancy_indexing(local_data, rect, global_vector_count, rank, num_ranks, nccl_comm, stream);
  }

}


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
    const Domain domain,
    const DomainPoint index_point,
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
                  const Domain domain,
                  const DomainPoint index_point,
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
    const Domain domain,
    const DomainPoint index_point,
    const std::vector<comm::Communicator>& comms)
  {
    auto rect = input_output_array.shape<DIM>();
    auto input_output = input_output_array.read_write_accessor<VAL, DIM>(rect);

    auto stream = get_cached_stream();

    bool need_distributed_shuffle = (num_ranks > 1) && is_index_space;

    // assert(num_ranks == num_shuffle_ranks);
    // For local shuffle (single node or within a node)
    if (!need_distributed_shuffle) {
    
          // TODO: Implement key-sort local shuffle
          assert(false && "Key-sort local shuffle not yet implemented");
    
    } else {
      // Handle distributed shuffle
      
      VAL* data_ptr = input_output.ptr(rect.lo);

      global_shuffle_bidirectional<VAL, DIM>(
          data_ptr,
          rect,
          vector_count,
          rank,
          num_ranks,
          domain,
          index_point,
          comms[0].get<ncclComm_t*>(),
          stream
      );
    }
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

template <VariantKind KIND, int DIM>
struct ShuffleImplBody_type {
  template <Type::Code CODE>
  void operator()(ShuffleArgs& args, TaskContext& context, 
    std::vector<comm::Communicator> comms) const
  {
    auto rect = args.input_output.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);
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
      args.domain,
      args.index_point,
      comms
    ); 
  }
};



template <VariantKind KIND>
struct ShuffleImpl {
  template <int DIM>
  void operator()(ShuffleArgs& args, TaskContext& context, 
    std::vector<comm::Communicator> comms) const
  {
    auto input_code = args.input_output.code();
    
  switch (input_code) {
    case legate::Type::Code::INT64:
    ShuffleImplBody_type<KIND, DIM>{}.template operator()<legate::Type::Code::INT64>(
      args,
      context,
      comms
    ); 
      break;
      default:
        assert(false && "Only INT64 data type is supported in this build");
        break;  
  }
  }
};
// template <VariantKind KIND>
// struct ShuffleImpl {
//   template <Type::Code CODE, int DIM>
//   void operator()(ShuffleArgs& args, TaskContext& context, 
//     std::vector<comm::Communicator> comms) const
//   {
//     using VAL = type_of<CODE>;
//     auto rect = args.input_output.shape<DIM>();

//     Pitches<DIM - 1> pitches;
//     size_t volume = pitches.flatten(rect);
//     if (volume == 0) {
//       return;
//     }

//     ShuffleImplBody<KIND, CODE, DIM>()(
//         context,
//         args.input_output,
//         args.vector_count,
//         args.vector_length,
//         args.is_index_space,
//         args.local_rank,
//         args.num_ranks,
//         args.domain,
//         args.index_point,
//         comms
//     );
//   }
// };


template <VariantKind KIND>
static void shuffle_template(TaskContext& context)
{
  // Extract arguments from TaskContext
  auto input_output = context.input(0);
  auto shape_span   = context.scalar(0).values<int64_t>();
  size_t d1 = shape_span[0];
  size_t d2 = 1;
  for (size_t i = 1; i < shape_span.size(); ++i) d2 *= shape_span[i];

  Domain domain           = context.get_launch_domain();
  DomainPoint index_point = context.get_task_index();
  size_t local_rank     = get_rank(domain, context.get_task_index());
  size_t num_ranks      = domain.get_volume();
  size_t num_shuffle_ranks = domain.hi()[0] - domain.lo()[0] + 1;
  
  ShuffleArgs args{
    input_output,
    d1,
    d2,
    !context.is_single_task(),
    local_rank,
    domain,
    index_point,
    num_ranks,
  };
  
  dim_dispatch(
    args.input_output.dim(), ShuffleImpl<KIND>{}, args, context, context.communicators());
  // double_dispatch(
  //   args.input_output.dim(), args.input_output.code(), ShuffleImpl<KIND>{}, args, context, context.communicators());
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