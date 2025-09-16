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

 #include "cupynumeric/all2all/all2all.h"
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
 #include <cstdio>
 #include <random>
 #include <iostream>
 #include <vector>
 
 // CUDA STD includes
 #include <cuda/std/cstdint>
 #include <cuda/std/type_traits>
//  #define LEGATE_MAX_DIM 1
 
 namespace cupynumeric {
 
 using namespace legate;


// Compute send histogram kernel for 1D (vectors)
template<int DIM_input>
__global__ void compute_send_histogram(const legate::Point<DIM_input>* indices, 
                                      unsigned int* send_histo, 
                                      legate::Rect<DIM_input>* rect_buf, 
                                      int num_ranks, 
                                      int indices_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < indices_len){
      int target_rank = 0;
      int is_in_rect = 1;
      legate::Point<DIM_input> point = indices[idx];
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
      // printf("target_rank %d, idx %d, size %ld, point={%ld, %ld}\n", target_rank, idx, sizeof(legate::Point<DIM_input>), point[0], point[1]);
      atomicAdd(&send_histo[target_rank], is_in_rect);
    }
}

// Pack send data kernel for 2D (vectors)
template<typename DataType, int DIM_input>
__global__ void pack_send_data_kernel(const DataType* data, legate::Point<DIM_input>* indices, size_t request_count,
                                          DataType* send_data, legate::Rect<DIM_input> input_rect) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
      legate::Point<DIM_input> point = indices[idx];
      int data_idx = 0;
      for(int d = 0; d < DIM_input - 1; d++){
        data_idx += (point[d] - input_rect.lo[d]);
        data_idx *= (input_rect.hi[d + 1] - input_rect.lo[d + 1] + 1);
      }
      data_idx += (point[DIM_input - 1] - input_rect.lo[DIM_input - 1]);
      send_data[idx] = data[data_idx];
      // printf("idx %d, data_idx %d, send_data[%d] %ld\n", idx, data_idx, idx, send_data[idx]);
    }
}

// Pack send data kernel for 2D (vectors)
template<typename DataType, int DIM_input>
__global__ void pack_send_data_kernel_single_rank(const DataType* data, const legate::Point<DIM_input>* indices, size_t request_count,
                                          DataType* send_data, legate::Rect<DIM_input> input_rect) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
      legate::Point<DIM_input> point = indices[idx];
      int data_idx = 0;
      for(int d = 0; d < DIM_input - 1; d++){
        data_idx += (point[d] - input_rect.lo[d]);
        data_idx *= (input_rect.hi[d + 1] - input_rect.lo[d + 1] + 1);
      }
      data_idx += (point[DIM_input - 1] - input_rect.lo[DIM_input - 1]);
      
      send_data[idx] = data[data_idx];
    }
}

template<typename DataType, int DIM_input>
__global__ void unpack_recv_data_kernel(DataType* data, unsigned int* request_indices, size_t request_count,
                                          const DataType* recv_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
        // printf("request_indices[%d]: %u, recv_data[%d]: %ld\n", idx, request_indices[idx], idx, recv_data[idx]);
        data[request_indices[idx]] = recv_data[idx];
    }
}
 

template<int DIM_input>
__global__ void pack_request_indices_kernel(const legate::Point<DIM_input>* indices, size_t vector_count,
                                          legate::Point<DIM_input>* send_indices, unsigned int* send_offsets, unsigned int* request_indices,
                                          unsigned int* counters, legate::Rect<DIM_input>* rect_buf, int num_ranks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_count) {
        
      // for(int i = 0; i < num_ranks; i++){
        int target_rank = 0;
        int is_in_rect = 1;
        legate::Point<DIM_input> point = indices[idx];
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
        // printf("pack_request_indices_kernel: idx %d, target_rank %d, indices[%d] = {%ld, %ld}\n", idx, target_rank, idx, indices[idx][0], indices[idx][1]);
      // }
    }
}

template<int DIM>
size_t get_volume(auto rect){
  size_t volume = 1;
  for(int i = 0; i < DIM; i++){
    volume *= (rect.hi[i] - rect.lo[i] + 1) > 0 ? (rect.hi[i] - rect.lo[i] + 1) : 0;
  }
  return volume;
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
  const legate::Point<DIM_input>* index_ptr, 
  DataType* output_ptr, 
  const legate::Rect<DIM_input> input_rect,
  const legate::Rect<DIM_output> index_rect,
  const legate::Rect<DIM_output> output_rect,
  int rank_id, 
  int num_ranks,
  // int * ,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
) {
  // Print current rank's GPU device ID and properties
int device_id;
cudaGetDevice(&device_id);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);
// printf("Rank %d is using GPU device %d (%s)\n", rank_id, device_id, prop.name);
cudaDeviceSynchronize();
cudaStreamSynchronize(stream);

  size_t local_input_count = get_volume<DIM_input>(input_rect);
  size_t local_index_count = get_volume<DIM_output>(index_rect);
  size_t local_output_count = get_volume<DIM_output>(output_rect);
  // printf("rank %d, local_index_count: %ld\n", rank_id, local_index_count);

  size_t num_requests = local_index_count;
  // ===== Round 0: Exchange rects =====
  auto global_rects = create_buffer<int64_t>(num_ranks * DIM_input * 2, Memory::Kind::GPU_FB_MEM);
  auto input_rect_device = create_buffer<int64_t>(DIM_input * 2, Memory::Kind::GPU_FB_MEM);
  cudaMemcpy((legate::Rect<DIM_input>*)input_rect_device.ptr(0), &input_rect, sizeof(input_rect), cudaMemcpyHostToDevice);

    // 初始化Round 0缓冲区
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(global_rects.ptr(0), 0, 
    num_ranks * DIM_input * 2 * sizeof(int64_t), stream));


  // ===== Round 1: Exchange request size histograms =====
  auto round1_send_histo = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // How many requests to send to each rank
  auto round1_send_offsets = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // Offsets for packing requests
  auto round1_recv_histo = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // How many requests to receive from each rank
  auto round1_recv_offsets = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // Offsets for unpacking requests
  auto packing_counters = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // Temporary counters for packing
  
  // 初始化Round 1缓冲区
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round1_send_histo.ptr(0), 0, 
    num_ranks * sizeof(unsigned int), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round1_send_offsets.ptr(0), 0, 
    num_ranks * sizeof(unsigned int), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round1_recv_histo.ptr(0), 0, 
    num_ranks * sizeof(unsigned int), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round1_recv_offsets.ptr(0), 0, 
    num_ranks * sizeof(unsigned int), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(packing_counters.ptr(0), 0, 
    num_ranks * sizeof(unsigned int), stream));

  // // ===== Round 2: Exchange request indices =====
  auto round2_send_indices = create_buffer<int64_t>(num_requests * DIM_input, Memory::Kind::GPU_FB_MEM); // Indices to send (packed by target rank)
  auto round2_request_positions = create_buffer<unsigned int>(num_requests, Memory::Kind::GPU_FB_MEM); // Position of each request in output
  
  // // ===== Round 3: Exchange actual data =====
  auto round3_recv_data = create_buffer<DataType>(num_requests, Memory::Kind::GPU_FB_MEM); // Final received data
//   printf("rank %d, DIM_output: %d\n", rank_id, DIM_output);
// for(int j = 0; j < DIM_output; j++){
//   printf("rank %d, index_rect[%d]: %d, %d\n", rank_id, j, index_rect.lo[j], index_rect.hi[j]);
// }
// printf("rank %d, num_ranks: %d, num_requests: %ld\n", rank_id, num_ranks, num_requests);   
  if(num_requests > 0){
  
    // 初始化Round 2缓冲区
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round2_send_indices.ptr(0), 0, 
    num_requests * DIM_input * sizeof(int64_t), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round2_request_positions.ptr(0), 0, 
    num_requests * sizeof(unsigned int), stream));
     // 初始化Round 3缓冲区
     CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round3_recv_data.ptr(0), 0, 
     num_requests * sizeof(DataType), stream));
 
  }
   
  size_t block_size = 256;
  size_t grid_size = (local_index_count + block_size - 1) / block_size;
  
  // ===== Round 0: Exchange rects =====
  cudaStreamSynchronize(stream);
  nvtxRangePushA("Exchange rects");
  CHECK_NCCL(ncclAllGather((void*)input_rect_device.ptr(0), 
                         (void*)global_rects.ptr(0), 
                         sizeof(input_rect), 
                         ncclInt8, 
                         *nccl_comm, 
                         stream));
  cudaStreamSynchronize(stream);
  nvtxRangePop();


// for(int i = 0; i < num_ranks; i++){
//   legate::Rect<DIM_input> rect;
//   cudaMemcpy(&rect, global_rects.ptr(i * DIM_input * 2), sizeof(input_rect), cudaMemcpyDeviceToHost);
//   for(int j = 0; j < DIM_input; j++){
//     printf("rank %d, global_rects[%d][%d]: %d, %d\n", rank_id, i, j, rect.lo[j], rect.hi[j]);
//   }
// }

  
  // ===== Round 1: Compute request size histogram =====
  nvtxRangePushA("Compute request size histogram");
  // Only launch kernel if we have data to process
  if (local_index_count > 0) {
    compute_send_histogram<DIM_input><<<grid_size, block_size, 0, stream>>>(index_ptr, 
                  round1_send_histo.ptr(0), 
                  (legate::Rect<DIM_input>*)(global_rects.ptr(0)), 
                  num_ranks, local_index_count);
  }
  nvtxRangePop();
  cudaStreamSynchronize(stream);
  // for(int i = 0; i < num_ranks; i++){
  //   unsigned int tmp = 0;
  //   cudaMemcpy(&tmp, round1_send_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //   printf("rank %d, round1_send_histo[%d]: %u\n", rank_id, i, tmp);
  // }

  nvtxRangePushA("Exclusive scan");
  cudaStreamSynchronize(stream);
    thrust::exclusive_scan(DEFAULT_POLICY.on(stream), 
                          (unsigned int*)round1_send_histo.ptr(0), 
                          ((unsigned int*)round1_send_histo.ptr(0)) + num_ranks, round1_send_offsets.ptr(0));
  cudaDeviceSynchronize();
  nvtxRangePop();

  

  // ===== Round 1: All2All exchange request size histograms =====
  nvtxRangePushA("All2All exchange request size histograms");
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
    CHECK_NCCL(ncclRecv((void*)round1_recv_histo.ptr(i), 1, ncclUint32, i, *nccl_comm, stream));  
    CHECK_NCCL(ncclSend((void*)round1_send_histo.ptr(i), 1, ncclUint32, i, *nccl_comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  nvtxRangePop();



  nvtxRangePushA("Reduce");
  size_t total_indices_to_receive = thrust::reduce(DEFAULT_POLICY.on(stream), round1_recv_histo.ptr(0), round1_recv_histo.ptr(0) + num_ranks);
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  // Pack request indices by target rank
  nvtxRangePushA("Pack request indices");
  if (local_index_count > 0) {
  pack_request_indices_kernel<DIM_input><<<grid_size, block_size, 0, stream>>>(index_ptr, num_requests,  
    (legate::Point<DIM_input>*)(round2_send_indices.ptr(0)), 
    round1_send_offsets.ptr(0),
    round2_request_positions.ptr(0), 
    packing_counters.ptr(0), 
    (legate::Rect<DIM_input>*)(global_rects.ptr(0)), num_ranks);
  }
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  nvtxRangePushA("Exclusive scan");
  auto round2_recv_indices = create_buffer<legate::Point<DIM_input>>(total_indices_to_receive, Memory::Kind::GPU_FB_MEM);
  if(total_indices_to_receive > 0){
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round2_recv_indices.ptr(0), 0, 
    total_indices_to_receive * sizeof(legate::Point<DIM_input>), stream));
  }
  thrust::exclusive_scan(DEFAULT_POLICY.on(stream), round1_recv_histo.ptr(0), round1_recv_histo.ptr(0) + num_ranks, round1_recv_offsets.ptr(0));
  // for(int i = 0; i < num_ranks; i++){
  //   unsigned int tmp;
  //   cudaMemcpy(&tmp, round1_recv_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //   printf("rank %d, round1_recv_histo[%d]: %u\n", rank_id, i, tmp);
  // }
  // for(int i = 0; i < num_ranks; i++){
  //   unsigned int tmp;
  //   cudaMemcpy(&tmp, round1_recv_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //   printf("rank %d, round1_recv_offsets[%d]: %u\n", rank_id, i, tmp);
  // }
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  // // ===== Round 2: All2All exchange request indices =====
  nvtxRangePushA("All2All exchange request indices");
  CHECK_NCCL(ncclGroupStart());
  fflush(stdout);
  for (int i = 0; i < num_ranks; ++i) {
      unsigned int indices_to_send_to_rank_i, send_offset_for_rank_i, indices_to_recv_from_rank_i, recv_offset_for_rank_i;
      cudaMemcpy(&indices_to_send_to_rank_i, round1_send_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&send_offset_for_rank_i, round1_send_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&indices_to_recv_from_rank_i, round1_recv_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&recv_offset_for_rank_i, round1_recv_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      // printf("All2All exchange request indices: rank %d, indices_to_send_to_rank_i[%d]: %u, send_offset_for_rank_i[%d]: %u, indices_to_recv_from_rank_i[%d]: %u, recv_offset_for_rank_i[%d]: %u\n", rank_id, i, indices_to_send_to_rank_i, i, send_offset_for_rank_i, i, indices_to_recv_from_rank_i, i, recv_offset_for_rank_i);
      if (indices_to_send_to_rank_i > 0) {
          CHECK_NCCL(ncclSend((void*)(round2_send_indices.ptr(send_offset_for_rank_i * DIM_input)),
            indices_to_send_to_rank_i * sizeof(legate::Point<DIM_input>), ncclInt8, i, *nccl_comm, stream));
      }
      
      if (indices_to_recv_from_rank_i > 0) {
          CHECK_NCCL(ncclRecv(round2_recv_indices.ptr(recv_offset_for_rank_i),
                  indices_to_recv_from_rank_i * sizeof(legate::Point<DIM_input>), ncclInt8, i, *nccl_comm, stream));
      }
  }
  CHECK_NCCL(ncclGroupEnd());
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  nvtxRangePushA("Pack send data");
  auto round3_send_data = create_buffer<DataType>(total_indices_to_receive, Memory::Kind::GPU_FB_MEM);
  // CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round3_send_data.ptr(0), 0, 
  //   total_indices_to_receive * sizeof(DataType), stream));
  grid_size = (total_indices_to_receive + block_size - 1) / block_size;
  if (total_indices_to_receive > 0) {
  pack_send_data_kernel<DataType, DIM_input><<<grid_size, block_size, 0, stream>>>(input_ptr, (legate::Point<DIM_input>*)round2_recv_indices.ptr(0),
  total_indices_to_receive, round3_send_data.ptr(0), input_rect);
  }
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  // ===== Round 3: All2All exchange actual data =====
  // printf("rank %d, round3_send_data: %p\n", rank_id, round3_send_data.ptr(0));
  nvtxRangePushA("All2All exchange actual data");
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
      unsigned int data_to_send_to_rank_i, send_offset_for_rank_i, data_to_recv_from_rank_i, recv_offset_for_rank_i;
      cudaMemcpy(&data_to_send_to_rank_i, round1_send_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&send_offset_for_rank_i, round1_send_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&data_to_recv_from_rank_i, round1_recv_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&recv_offset_for_rank_i, round1_recv_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // printf("rank %d recv from rank %d: recv offset %d, recv data %d\n", rank_id, i, recv_offset_for_rank_i, data_to_recv_from_rank_i);
      if (data_to_recv_from_rank_i > 0) {
      CHECK_NCCL(ncclSend(round3_send_data.ptr(recv_offset_for_rank_i),
      data_to_recv_from_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
    }
      if (data_to_send_to_rank_i > 0) {
        CHECK_NCCL(ncclRecv(round3_recv_data.ptr(send_offset_for_rank_i),
        data_to_send_to_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
      }   
  }
  CHECK_NCCL(ncclGroupEnd());
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  // ===== Final step: Unpack received data to output =====
  nvtxRangePushA("unpack received data");
  grid_size = (num_requests + block_size - 1) / block_size;
  if (num_requests > 0) {
  unpack_recv_data_kernel<DataType, DIM_output><<<grid_size, block_size, 0, stream>>>(output_ptr, round2_request_positions.ptr(0),
    num_requests, round3_recv_data.ptr(0));
  }
  cudaStreamSynchronize(stream);
  nvtxRangePop();
}

template<typename DataType, int DIM_input, int DIM_output>
void global_all2all_single_rank(
  const DataType* input_ptr, 
  const legate::Point<DIM_input>* index_ptr, 
  DataType* output_ptr, 
  const legate::Rect<DIM_input> input_rect,
  const legate::Rect<DIM_output> index_rect,
  const legate::Rect<DIM_output> output_rect,
  int rank_id, 
  int num_ranks,
  cudaStream_t stream
) {
cudaDeviceSynchronize();
cudaStreamSynchronize(stream);

  size_t local_input_count = get_volume<DIM_input>(input_rect);
  size_t local_index_count = get_volume<DIM_output>(index_rect);
  size_t local_output_count = get_volume<DIM_output>(output_rect);
  
  size_t num_requests = local_index_count;
  const size_t block_size = 256;
  const size_t grid_size = (local_index_count + block_size - 1) / block_size;
  if(local_index_count > 0){
  pack_send_data_kernel_single_rank<DataType, DIM_input><<<grid_size, block_size, 0, stream>>>(input_ptr, index_ptr,
    local_index_count, output_ptr, input_rect);
  }
}
 
 template <Type::Code CODE, int32_t DIM_input, int32_t DIM_output>
 struct All2AllImplBody<VariantKind::CPU, CODE, DIM_input, DIM_output> {
   using VAL = type_of<CODE>;
 
   void operator()(TaskContext& context,
    const legate::PhysicalStore& input_array,
    const legate::PhysicalStore& index_array,
    const legate::PhysicalStore& output_array,
     const bool is_index_space,
     const size_t rank,
     const size_t num_ranks,
     const std::vector<comm::Communicator>& comms)
   {
     // CPU all2all not yet implemented
     assert(false && "CPU all2all not yet implemented");
   }
 };
 
 #if LEGATE_DEFINED(LEGATE_USE_OPENMP)
 template <Type::Code CODE, int32_t DIM_input, int32_t DIM_output>
 struct All2AllImplBody<VariantKind::OMP, CODE, DIM_input, DIM_output> {
   using VAL = type_of<CODE>;
 
   void operator()(TaskContext& context,
                  const legate::PhysicalStore& input_array,
                  const legate::PhysicalStore& index_array,
                  const legate::PhysicalStore& output_array,
                   const bool is_index_space,
                   const size_t rank,
                   const size_t num_ranks,
                   const std::vector<comm::Communicator>& comms)
   {
     // OMP all2all not yet implemented
     assert(false && "OMP all2all not yet implemented");
   }
 };
 #endif
 
 template <Type::Code CODE, int32_t DIM_input, int32_t DIM_output>
struct All2AllImplBody<VariantKind::GPU, CODE, DIM_input, DIM_output> {
  using VAL = type_of<CODE>;
  
   void operator()(TaskContext& context,
     const legate::PhysicalStore& input_array,
     const legate::PhysicalStore& index_array,
     const legate::PhysicalStore& output_array,
     const bool is_index_space,
     const size_t rank,
     const size_t num_ranks,
     const std::vector<comm::Communicator>& comms)
   {
    using INDEX_VAL = legate::Point<DIM_input>;
    // auto input_dim = args.input.dim();

    
    auto input_rect = input_array.shape<DIM_input>();
    auto index_rect = index_array.shape<DIM_output>();
    auto output_rect = output_array.shape<DIM_output>();
    
    auto input = input_array.read_accessor<VAL, DIM_input>(input_rect);
    auto index = index_array.read_accessor<INDEX_VAL, DIM_output>(index_rect);
    auto output = output_array.read_write_accessor<VAL, DIM_output>(output_rect);
 
     auto stream = context.get_task_stream();
 
     bool need_distributed_all2all = (num_ranks > 1) && is_index_space;
     const VAL* input_ptr = input.ptr(input_rect.lo);
     const INDEX_VAL* index_ptr = index.ptr(index_rect.lo);
     VAL* output_ptr = output.ptr(output_rect.lo);
     // For local all2all (single node or within a node)
     if (!need_distributed_all2all) {
      //  printf("local_all2all\n"); 
           // TODO: Implement key-sort local all2all
           global_all2all_single_rank<VAL, DIM_input, DIM_output>(
               input_ptr,
               index_ptr,
               output_ptr,
               input_rect,
               index_rect,
               output_rect,
               rank,
               num_ranks,
               stream
           );
          //  assert(false && "Key-sort local all2all not yet implemented");
     
     } else {
       // Handle distributed all2all
      //  printf("num_ranks: %d, global_all2all\n", num_ranks);
       
         nvtxRangePushA("global_all2all");
      cudaDeviceSynchronize();
      // printf("rank_id: %d, num_ranks: %d, comms size: %d\n", rank, num_ranks, comms.size());
       global_all2all<VAL, DIM_input, DIM_output>(
           input_ptr,
           index_ptr,
           output_ptr,
           input_rect,
           index_rect,
           output_rect,
           rank,
           num_ranks,
           comms[0].get<ncclComm_t*>(),
           stream
       );
       cudaDeviceSynchronize();
       nvtxRangePop();
     }
     cudaDeviceSynchronize();
     CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
    //  printf("end global_all2all\n");
   }
 };


 template <VariantKind KIND, int DIM_input, int DIM_output>
 struct All2AllImpl_type {
   template <Type::Code CODE>
   void operator()(All2AllArgs& args, TaskContext& context, 
     std::vector<comm::Communicator> comms) const
   {

     auto rect_input = args.input.shape<DIM_input>();
     auto rect_index_array = args.index_array.shape<DIM_output>();
     auto rect_output = args.output.shape<DIM_output>();
    
     Pitches<DIM_input - 1> pitches_input;
     size_t input_volume = pitches_input.flatten(rect_input);
     Pitches<DIM_output - 1> pitches_output;
     size_t index_volume = pitches_output.flatten(rect_index_array);
     size_t output_volume = pitches_output.flatten(rect_output);
     
    //  if (input_volume == 0 && index_volume == 0 && output_volume == 0) {
    //    return;
    //  }

     
     for (int i = 0; i < DIM_input; i++) {  
      auto hi          = rect_input.hi;
     auto lo          = rect_input.lo;
     }

     All2AllImplBody<KIND, CODE, DIM_input, DIM_output>()(
         context,
         args.input,
         args.index_array,
         args.output,
         args.is_index_space,
         args.rank_id,
         args.num_ranks,
         comms
     );
   }
 };

 template <VariantKind KIND>
 struct All2AllImpl {
   template <int DIM_input, int DIM_output>
   void operator()(All2AllArgs& args, TaskContext& context, 
     std::vector<comm::Communicator> comms) const
   {
    type_dispatch(
      args.input.code(), All2AllImpl_type<KIND, DIM_input, DIM_output>{}, args, context, context.communicators());
   }
 };

// template <VariantKind KIND>
// struct All2AllImpl {
//   template <int DIM_input, int DIM_output>
//   void operator()(All2AllArgs& args, TaskContext& context, 
//     std::vector<comm::Communicator> comms) const
//   {
//    // Custom type dispatch to only compile int64 type for faster compilation
//    auto input_code = args.input.code();
//    auto output_code = args.output.code();
//    auto index_code = args.index_array.code();
//   //  printf("input_code: %d, output_code: %d, index_code: %d\n", input_code, output_code, index_code);
//   //  exit(0);
//    switch (input_code) {
//     //  case legate::Type::Code::FLOAT32:
//     //    printf("FLOAT32\n");
//     //    All2AllImpl_type<KIND, DIM_input, DIM_output>{}.template operator()<legate::Type::Code::FLOAT32>(args, context, context.communicators());
//     //    break;
//      case legate::Type::Code::FLOAT64:
//       //  printf("FLOAT64\n");
//        All2AllImpl_type<KIND, DIM_input, DIM_output>{}.template operator()<legate::Type::Code::FLOAT64>(args, context, context.communicators());
//        break;
//      case legate::Type::Code::INT64:
//       //  printf("INT64\n");
//        All2AllImpl_type<KIND, DIM_input, DIM_output>{}.template operator()<legate::Type::Code::INT64>(args, context, context.communicators());
//        break;
//     //  case legate::Type::Code::INT32:
//     //    printf("INT32\n");
//     //    All2AllImpl_type<KIND, DIM_input, DIM_output>{}.template operator()<legate::Type::Code::INT32>(args, context, context.communicators());
//     //    break;
//      default:
//        printf("Unsupported data type code: %d. Only INT64&FLOAT64 is supported for fast compilation.\n", (int)input_code);
//        assert(false && "Only INT64 data type is supported in this build");
//        break;
//    }
//   }
// };
 
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
    //  printf("domain.hi()[%d]: %d, domain.lo()[%d]: %d, index_point[%d]: %d\n", i, hi[i], i, lo[i], i, index_point[i]);
   }
   return domain_index;
 }
 
 template <VariantKind KIND>
 static void all2all_template(TaskContext& context)
 {
   // Extract arguments from TaskContext
   auto input = context.input(0);
   auto index_array = context.input(1);
   auto output = context.output(0);
   
   
   auto domain           = context.get_launch_domain();
   size_t rank_id     = get_rank(domain, context.get_task_index());
   size_t num_ranks      = domain.get_volume();
   
   
   All2AllArgs args{
     input,
     index_array,
     output,
     !context.is_single_task(),
     rank_id,
     num_ranks
   };
  
   auto dim_input = std::max(args.input.dim(), 1);
   auto dim_output = std::max(args.output.dim(), 1);

  // printf("dim_input: %d, dim_output: %d\n", dim_input, dim_output);
  double_dispatch(
    dim_input, dim_output, All2AllImpl<KIND>{}, args, context, context.communicators());
 }
 
 /*static*/ void All2AllTask::gpu_variant(TaskContext context)
 {
   all2all_template<VariantKind::GPU>(context);
 }
 
 /*static*/ void All2AllTask::cpu_variant(TaskContext context)
 {
   // CPU variant not implemented yet - all2all operations will assert false
   all2all_template<VariantKind::CPU>(context);
 }
 
 #if LEGATE_DEFINED(LEGATE_USE_OPENMP)
 /*static*/ void All2AllTask::omp_variant(TaskContext context)
 {
   // OMP variant not implemented yet - all2all operations will assert false  
   all2all_template<VariantKind::OMP>(context);
 }
 #endif
 
 namespace  // unnamed
 {
 static const auto cupynumeric_reg_task_ = []() -> char {
   All2AllTask::register_variants(); 
   return 0;
 }();
 }  // namespace
 
 }  // namespace cupynumeric 