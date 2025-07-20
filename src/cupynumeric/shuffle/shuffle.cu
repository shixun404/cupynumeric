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
#include "cupynumeric/shuffle/bijection.cu"
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>

namespace cupynumeric {

namespace {

template <typename T>
legate::Buffer<T> create_non_empty_buffer(
  size_t size, legate::Memory::Kind kind = legate::Memory::Kind::NO_MEMKIND)
{
  return legate::create_buffer<T>(std::max(size, size_t{1}), kind, alignof(T));
}

inline void check_nccl(ncclResult_t result, const char* file, int line) {
  if (result != ncclSuccess) {
    std::cerr << "NCCL error at " << file << ":" << line 
              << " - " << ncclGetErrorString(result) << std::endl;
    exit(1);
  }
}

}  // namespace

#define CHECK_NCCL(...)                                      \
  do {                                                       \
    ncclResult_t __result__ = (__VA_ARGS__);                 \
    cupynumeric::check_nccl(__result__, __FILE__, __LINE__); \
  } while (false)

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// CUDA Kernels (from global_shuffle.cu)
/////////////////////////////////////////////////////////////////////////////////////////////////


// Apply bijection kernel
template<typename IndexType>
__global__ void apply_bijection_kernel(IndexType* indices, size_t count, 
                                       random_bijection<IndexType> bijection) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    indices[idx] = bijection(indices[idx]);
  }
}

// // Apply inverse bijection kernel
// template<typename IndexType>
// __global__ void apply_inverse_bijection_kernel(IndexType* indices, size_t count, 
//                                                random_bijection<IndexType> bijection) {
//   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (idx < count) {
//     indices[idx] = bijection.inverse(indices[idx]);
//   }
// }

// Generate Fisher-Yates permutation on CPU
template<typename IndexType>
void generate_fisher_yates_permutation_cpu(std::vector<IndexType>& permutation, size_t global_n, unsigned long seed) {
  // Initialize permutation
  permutation.resize(global_n);
  for (size_t i = 0; i < global_n; ++i) {
    permutation[i] = static_cast<IndexType>(i);
  }
  
  // Fisher-Yates shuffle on CPU
  std::mt19937 rng(seed);
  for (size_t i = global_n - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t j = dist(rng);
    std::swap(permutation[i], permutation[j]);
  }
}

// Compute send histogram kernel
template<typename IndexType>
__global__ void compute_send_histogram_kernel(const IndexType* indices, size_t count,
                                              unsigned int* send_histo, size_t local_size, size_t total_ranks) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    size_t target_rank = indices[idx] / local_size;
    if (target_rank < total_ranks) {
      atomicAdd(&send_histo[target_rank], 1);
    }
  }
}

// Pack send data kernel
template<typename DataType, typename IndexType>
__global__ void pack_send_data_kernel(const DataType* data, const IndexType* indices, size_t count,
                                       DataType* send_data, IndexType* send_indices,
                                       unsigned int* send_offsets, unsigned int* counters,
                                       size_t local_size, size_t total_ranks) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    size_t target_rank = indices[idx] / local_size;
    if (target_rank < total_ranks) {
      size_t pos = atomicAdd(&counters[target_rank], 1);
      size_t offset = send_offsets[target_rank] + pos;
      send_data[offset] = data[idx];
      send_indices[offset] = indices[idx];
    }
  }
}

// Unpack received data kernel
template<typename DataType, typename IndexType>
__global__ void unpack_recv_data_kernel(const DataType* recv_data, const IndexType* recv_indices,
                                         size_t count, DataType* output, size_t local_size, size_t rank_offset) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    IndexType global_idx = recv_indices[idx];
    size_t local_idx = global_idx - rank_offset;
    if (local_idx < local_size) {
      output[local_idx] = recv_data[idx];
    }
  }
}

// Scatter data based on permuted indices for single rank
template<typename DataType>
__global__ void single_rank_scatter_kernel(const DataType* input_data, const uint64_t* indices, 
                                          size_t count, DataType* output_data) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    uint64_t target_idx = indices[idx];
    if (target_idx < count) {
      output_data[target_idx] = input_data[idx];
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// Global Shuffle Implementation (adapted from global_shuffle.cu)
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename DataType>
void global_shuffle_distributed(DataType* local_data, size_t local_count, size_t global_n,
                               size_t num_ranks, size_t local_rank,
                               DistributedShuffleMethod method,
                               ncclComm_t nccl_comm, cudaStream_t stream) {
  
  size_t rank_offset = local_rank * local_count;
  const size_t block_size = 256;
  const size_t grid_size = (local_count + block_size - 1) / block_size;
  
  // Common data structures for all methods
  thrust::device_vector<uint64_t> send_indices(local_count);
  thrust::device_vector<unsigned int> send_histo(num_ranks, 0);
  thrust::device_vector<unsigned int> counters(num_ranks, 0);
  
  // Step 1: Calculate send indices based on method
  switch (method) {
    case DistributedShuffleMethod::FEISTEL_ALL2ALL: {
      // Method 1: Feistel forward to calculate send indices
      thrust::sequence(send_indices.begin(), send_indices.end(), rank_offset);
      
      thrust::default_random_engine rng(42);
      random_bijection<uint64_t> bijection(global_n, rng);
      
      apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(send_indices.data()), local_count, bijection);
      
      break;
    }
    
    case DistributedShuffleMethod::FEISTEL_BIDIRECTIONAL: {
      // Method 2: Feistel forward to calculate send indices (same as method 1)
      // thrust::sequence(send_indices.begin(), send_indices.end(), rank_offset);
      
      // thrust::default_random_engine rng(42);
      // random_bijection<uint64_t> bijection(global_n, rng);
      
      // apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
      //   thrust::raw_pointer_cast(send_indices.data()), local_count, bijection);
      
      break;
    }
    
    case DistributedShuffleMethod::FISHER_YATES_GLOBAL: {
      // Method 3: Fisher-Yates global permutation to calculate send indices
      
      // Generate global permutation on CPU
      std::vector<uint64_t> cpu_permutation;
      generate_fisher_yates_permutation_cpu(cpu_permutation, global_n, 42);
      
      // Copy permutation to GPU
      thrust::device_vector<uint64_t> global_permutation(cpu_permutation);
      
      // Extract send indices for this rank
      thrust::host_vector<uint64_t> h_send_indices(local_count);
      for (size_t i = 0; i < local_count; ++i) {
        h_send_indices[i] = cpu_permutation[rank_offset + i];
      }
      thrust::copy(h_send_indices.begin(), h_send_indices.end(), send_indices.begin());
      
      break;
    }
  }
  
  // Step 2: Compute send histogram (same for all methods)
  compute_send_histogram_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(send_indices.data()), local_count,
    thrust::raw_pointer_cast(send_histo.data()), local_count, num_ranks);
  
  // Step 3: Compute send offsets and pack data (same for all methods)
  thrust::device_vector<unsigned int> send_offsets(num_ranks);
  thrust::exclusive_scan(send_histo.begin(), send_histo.end(), send_offsets.begin());
  
  size_t total_send = thrust::reduce(send_histo.begin(), send_histo.end());
  thrust::device_vector<DataType> send_data(total_send);
  thrust::device_vector<uint64_t> send_indices_packed(total_send);
  
  thrust::fill(counters.begin(), counters.end(), 0);
  pack_send_data_kernel<<<grid_size, block_size, 0, stream>>>(
    local_data, thrust::raw_pointer_cast(send_indices.data()), local_count,
    thrust::raw_pointer_cast(send_data.data()), thrust::raw_pointer_cast(send_indices_packed.data()),
    thrust::raw_pointer_cast(send_offsets.data()), thrust::raw_pointer_cast(counters.data()),
    local_count, num_ranks);
  
  // Step 4: Calculate receive histogram based on method
  thrust::device_vector<unsigned int> recv_histo(num_ranks);
  
  switch (method) {
    case DistributedShuffleMethod::FEISTEL_ALL2ALL: {
      // Method 1: Use all2all exchange to get receive histogram
      
      CHECK_NCCL(ncclGroupStart());
      for (size_t i = 0; i < num_ranks; ++i) {
        CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_histo.data()) + i, 1, ncclUint32, i, nccl_comm, stream));
        CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_histo.data()) + i, 1, ncclUint32, i, nccl_comm, stream));
      }
      CHECK_NCCL(ncclGroupEnd());
      
      break;
    }
    
    case DistributedShuffleMethod::FEISTEL_BIDIRECTIONAL: {
      // Method 2: Use Feistel backward to calculate receive histogram
      thrust::device_vector<uint64_t> recv_indices(local_count);
      thrust::sequence(recv_indices.begin(), recv_indices.end(), rank_offset);
      
      thrust::default_random_engine rng(42);
      random_bijection<uint64_t> bijection(global_n, rng);
      
      apply_inverse_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(recv_indices.data()), local_count, bijection);
      
      // Compute histogram from inverse indices
      thrust::fill(recv_histo.begin(), recv_histo.end(), 0);
      compute_send_histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(recv_indices.data()), local_count,
        thrust::raw_pointer_cast(recv_histo.data()), local_count, num_ranks);
      
      break;
    }
    
    case DistributedShuffleMethod::FISHER_YATES_GLOBAL: {
      // Method 3: Use Fisher-Yates global permutation to calculate receive histogram
      
      // We already have the global permutation, now find what we should receive
      std::vector<uint64_t> cpu_permutation;
      generate_fisher_yates_permutation_cpu(cpu_permutation, global_n, 42);
      
      // Count how many elements each rank should send to us
      thrust::host_vector<unsigned int> h_recv_histo(num_ranks, 0);
      for (size_t i = 0; i < global_n; ++i) {
        uint64_t target_pos = cpu_permutation[i];
        if (target_pos >= rank_offset && target_pos < rank_offset + local_count) {
          size_t source_rank = i / local_count;
          if (source_rank < num_ranks) {
            h_recv_histo[source_rank]++;
          }
        }
      }
      thrust::copy(h_recv_histo.begin(), h_recv_histo.end(), recv_histo.begin());
      
      break;
    }
  }
  
  // Step 5: All2all data exchange (same for all methods)
  size_t total_recv = thrust::reduce(recv_histo.begin(), recv_histo.end());
  thrust::device_vector<DataType> recv_data(total_recv);
  thrust::device_vector<uint64_t> recv_indices(total_recv);
  
  thrust::device_vector<size_t> recv_offsets(num_ranks);
  thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
  
  // Copy to host for NCCL
  thrust::host_vector<unsigned int> h_send_histo = send_histo;
  thrust::host_vector<unsigned int> h_recv_histo = recv_histo;
  thrust::host_vector<size_t> h_recv_offsets = recv_offsets;
  thrust::host_vector<unsigned int> h_send_offsets = send_offsets;
  
  // All2all distribute data  
  ncclDataType_t data_type = sizeof(DataType) == 8 ? ncclInt64 : ncclInt32;
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
    if (h_send_histo[i] > 0) {
      CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i],
              h_send_histo[i], data_type, i, nccl_comm, stream));
      CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices_packed.data()) + h_send_offsets[i],
              h_send_histo[i], ncclUint64, i, nccl_comm, stream));
    }
    
    if (h_recv_histo[i] > 0) {
      CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i],
              h_recv_histo[i], data_type, i, nccl_comm, stream));
      CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
              h_recv_histo[i], ncclUint64, i, nccl_comm, stream));
    }
  }
  CHECK_NCCL(ncclGroupEnd());
  
  // Step 6: Unpack to final positions (same for all methods)
  const size_t unpack_grid_size = (total_recv + block_size - 1) / block_size;
  unpack_recv_data_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
    total_recv, local_data, local_count, rank_offset);
  
  cudaStreamSynchronize(stream);
}

// Single rank shuffle function (adapted from global_shuffle.cu)
template<typename DataType>
void single_rank_shuffle_local(DataType* data, size_t count, size_t global_n, cudaStream_t stream) {
  const size_t block_size = 256;
  const size_t grid_size = (count + block_size - 1) / block_size;
  
  // Initialize index array with local IDs
  thrust::device_vector<uint64_t> indices(count);
  thrust::sequence(indices.begin(), indices.end(), 0);
  
  // Apply Feistel bijection
  thrust::default_random_engine rng(42);
  random_bijection<uint64_t> bijection(global_n, rng);
  
  apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(indices.data()), count, bijection);
  
  // Create temporary buffer for shuffled data
  thrust::device_vector<DataType> temp_data(count);
  
  // Rearrange data based on shuffled indices
  single_rank_scatter_kernel<<<grid_size, block_size, 0, stream>>>(
    data, thrust::raw_pointer_cast(indices.data()), count, 
    thrust::raw_pointer_cast(temp_data.data()));
  
  // Copy shuffled data back to original array
  thrust::copy(thrust::cuda::par.on(stream), temp_data.begin(), temp_data.end(), data);
  
  cudaStreamSynchronize(stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// Shuffle Method Implementations
/////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
void shuffle_feistel(T* data, size_t volume, size_t first_axis_size,
                    ThrustAllocator& alloc, cudaStream_t stream) {
  if (volume == 0) return;
  single_rank_shuffle_local(data, volume, volume, stream);
}

template<typename T>
void shuffle_feistel_bidirectional(T* data, size_t volume, size_t first_axis_size,
                                  ThrustAllocator& alloc, cudaStream_t stream) {
  if (volume == 0) return;
  
  auto exec_policy = DEFAULT_POLICY(alloc).on(stream);
  const size_t block_size = 256;
  const size_t grid_size = (volume + block_size - 1) / block_size;
  
  // Forward pass
  thrust::device_vector<uint64_t> forward_indices(volume);
  thrust::sequence(exec_policy, forward_indices.begin(), forward_indices.end());
  
  thrust::default_random_engine rng1(42);
  random_bijection<uint64_t> forward_bijection(volume, rng1);
  
  apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(forward_indices.data()), volume, forward_bijection);
  
  // Backward pass with different seed
  thrust::device_vector<uint64_t> backward_indices(volume);
  thrust::sequence(exec_policy, backward_indices.begin(), backward_indices.end());
  
  thrust::default_random_engine rng2(123);
  random_bijection<uint64_t> backward_bijection(volume, rng2);
  
  apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(backward_indices.data()), volume, backward_bijection);
  
  // Combine both bijections
  thrust::transform(exec_policy, forward_indices.begin(), forward_indices.end(),
                   backward_indices.begin(), forward_indices.begin(),
                   [volume] __device__ (uint64_t a, uint64_t b) { return (a + b) % volume; });
  
  // Create temporary buffer and rearrange
  thrust::device_vector<T> temp_data(volume);
  thrust::copy(exec_policy, data, data + volume, temp_data.begin());
  
  single_rank_scatter_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(temp_data.data()), 
    thrust::raw_pointer_cast(forward_indices.data()), 
    volume, data);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// Main Shuffle Implementation Body
/////////////////////////////////////////////////////////////////////////////////////////////////

template <VariantKind KIND, Type::Code CODE, int32_t DIM>
struct ShuffleImplBody {
  void operator()(TaskContext& context,
                  legate::PhysicalStore& input_output,
                  ShuffleMethod method,
                  size_t volume,
                  size_t first_axis_size,
                  bool is_index_space,
                  size_t local_rank,
                  size_t num_ranks,
                  size_t num_shuffle_ranks,
                  const std::vector<comm::Communicator>& comms);
};

template <Type::Code CODE, int32_t DIM>
struct ShuffleImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;

  void operator()(TaskContext& context,
                  legate::PhysicalStore& input_output,
                  ShuffleMethod method,
                  size_t volume,
                  size_t first_axis_size,
                  bool is_index_space,
                  size_t local_rank,
                  size_t num_ranks,
                  size_t num_shuffle_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto stream = get_cached_stream();
    auto alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    
    // Get data pointer from the store
    auto rect = input_output.shape<DIM>();
    auto accessor = input_output.write_accessor<VAL, DIM>(rect);
    VAL* data_ptr = accessor.ptr(rect.lo);
    
    if (volume == 0) return;
    
    // Check if we need distributed shuffle
    if (is_index_space && !comms.empty()) {
      // Map shuffle method to distributed method
      DistributedShuffleMethod dist_method;
      switch (method) {
        case ShuffleMethod::FEISTEL:
          dist_method = DistributedShuffleMethod::FEISTEL_ALL2ALL;
          break;
        case ShuffleMethod::FEISTEL_BIDIRECTIONAL:
          dist_method = DistributedShuffleMethod::FEISTEL_BIDIRECTIONAL;
          break;
        case ShuffleMethod::FISHER_YATES:
          dist_method = DistributedShuffleMethod::FISHER_YATES_GLOBAL;
          break;
        default:
          dist_method = DistributedShuffleMethod::FEISTEL_ALL2ALL;
          break;
      }
      
      // Use distributed global shuffle
      auto nccl_comm = comms[0].get<ncclComm_t*>();
      global_shuffle_distributed(data_ptr, volume, volume, num_ranks, local_rank, dist_method, *nccl_comm, stream);
    } else {
      // Use local shuffle methods
      switch (method) {
        case ShuffleMethod::KEY_SORT:
          shuffle_key_sort(data_ptr, volume, first_axis_size, alloc, stream);
          break;
        case ShuffleMethod::FISHER_YATES:
          shuffle_fisher_yates(data_ptr, volume, first_axis_size, alloc, stream);
          break;
        case ShuffleMethod::FEISTEL:
          shuffle_feistel(data_ptr, volume, first_axis_size, alloc, stream);
          break;
        case ShuffleMethod::FEISTEL_BIDIRECTIONAL:
          shuffle_feistel_bidirectional(data_ptr, volume, first_axis_size, alloc, stream);
          break;
      }
    }
    
    CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// Template Instantiations
/////////////////////////////////////////////////////////////////////////////////////////////////

template struct ShuffleImplBody<VariantKind::GPU, Type::Code::INT64, 1>;
template struct ShuffleImplBody<VariantKind::GPU, Type::Code::INT64, 2>;
template struct ShuffleImplBody<VariantKind::GPU, Type::Code::INT64, 3>;

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// Task Implementation
/////////////////////////////////////////////////////////////////////////////////////////////////

template <VariantKind KIND>
void shuffle_template(TaskContext& context) {
  // Parse scalar arguments in order they were added
  auto inputs = context.scalars();
  auto method_value = inputs[0].value<int32_t>();
  auto volume = inputs[1].value<int64_t>();
  auto first_axis_size = inputs[2].value<int64_t>();
  auto is_index_space = inputs[3].value<bool>();
  
  ShuffleMethod method = static_cast<ShuffleMethod>(method_value);
  
  // Get the input/output store
  auto& input_output = context.outputs()[0];
  
  // Get communicators if needed
  std::vector<comm::Communicator> comms;
  if (is_index_space) {
    comms = context.communicators();
  }
  
  // Dispatch based on data type and dimensions
  auto shape = input_output.shape();
  auto code = input_output.code();
  
  if (code == Type::Code::INT64) {
    if (shape.size() == 1) {
      ShuffleImplBody<KIND, Type::Code::INT64, 1>{}(
        context, input_output, method, volume, first_axis_size,
        is_index_space, 0, 0, 0, comms);
    } else if (shape.size() == 2) {
      ShuffleImplBody<KIND, Type::Code::INT64, 2>{}(
        context, input_output, method, volume, first_axis_size,
        is_index_space, 0, 0, 0, comms);
    } else if (shape.size() == 3) {
      ShuffleImplBody<KIND, Type::Code::INT64, 3>{}(
        context, input_output, method, volume, first_axis_size,
        is_index_space, 0, 0, 0, comms);
    } else {
      assert(false && "Unsupported number of dimensions for shuffle");
    }
  } else {
    assert(false && "Unsupported data type for shuffle (only INT64 supported currently)");
  }
}

/*static*/ void ShuffleTask::gpu_variant(TaskContext context) {
  shuffle_template<VariantKind::GPU>(context);
}

/*static*/ void ShuffleTask::cpu_variant(TaskContext context) {
  // For CPU, we use a simple Fisher-Yates shuffle implementation
  // Parse arguments same as GPU variant
  auto inputs = context.scalars();
  auto method_value = inputs[0].value<int32_t>();
  auto volume = inputs[1].value<int64_t>();
  auto first_axis_size = inputs[2].value<int64_t>();
  auto is_index_space = inputs[3].value<bool>();
  
  auto& input_output = context.outputs()[0];
  auto shape = input_output.shape();
  auto code = input_output.code();
  
  if (code == Type::Code::INT64 && shape.size() == 1) {
    auto rect = input_output.shape<1>();
    auto accessor = input_output.write_accessor<int64_t, 1>(rect);
    int64_t* data_ptr = accessor.ptr(rect.lo);
    
    // Simple Fisher-Yates shuffle for CPU
    if (volume > 1) {
      std::random_device rd;
      std::mt19937 gen(rd());
      for (size_t i = volume - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dis(0, i);
        size_t j = dis(gen);
        std::swap(data_ptr[i], data_ptr[j]);
      }
    }
  } else {
    // For now, fallback to GPU implementation for complex cases
    shuffle_template<VariantKind::CPU>(context);
  }
}

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
/*static*/ void ShuffleTask::omp_variant(TaskContext context) {
  shuffle_template<VariantKind::OMP>(context);
}
#endif

}  // namespace cupynumeric 