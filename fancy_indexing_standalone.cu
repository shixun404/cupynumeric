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
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
      atomicAdd(&send_histo[target_rank], is_in_rect);
    }
}

// Pack send data kernel for 2D (vectors)
template<typename DataType, int DIM_input>
__global__ void pack_send_data_kernel(const DataType* data, legate::Point<DIM_input>* indices, size_t request_count,
                                          DataType* send_data, legate::Rect<DIM_input> input_rect) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
      legate::Point<DIM_input> point = indices[idx];
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
__global__ void pack_request_indices_kernel(const legate::Point<DIM_input>* indices, size_t vector_count,
                                          legate::Point<DIM_input>* send_indices, unsigned int* send_offsets, unsigned int* request_indices,
                                          unsigned int* counters, legate::Rect<DIM_input>* rect_buf, int num_ranks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
      // }
    }
}

template<int DIM>
size_t get_volume(auto rect){
  size_t volume = 1;
  for(int i = 0; i < DIM; i++){
    volume *= rect.hi[i] - rect.lo[i] + 1;
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
  
  size_t num_requests = local_index_count;
  // ===== Round 0: Exchange rects =====
  auto global_rects = create_buffer<int64_t>(num_ranks * DIM_input * 2, Memory::Kind::GPU_FB_MEM);
  auto input_rect_device = create_buffer<int64_t>(DIM_input * 2, Memory::Kind::GPU_FB_MEM);
  cudaMemcpy((legate::Rect<DIM_input>*)input_rect_device.ptr(0), &input_rect, sizeof(input_rect), cudaMemcpyHostToDevice);

    // 初始化Round 0缓冲区
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(global_rects.ptr(0), 0, 
    num_ranks * DIM_input * 2 * sizeof(int64_t), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(input_rect_device.ptr(0), 0, 
    DIM_input * 2 * sizeof(int64_t), stream));

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
   

    // 初始化Round 2缓冲区
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round2_send_indices.ptr(0), 0, 
    num_requests * DIM_input * sizeof(int64_t), stream));
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round2_request_positions.ptr(0), 0, 
    num_requests * sizeof(unsigned int), stream));
  
  // // ===== Round 3: Exchange actual data =====
  auto round3_recv_data = create_buffer<DataType>(num_requests, Memory::Kind::GPU_FB_MEM); // Final received data

    // 初始化Round 3缓冲区
    CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round3_recv_data.ptr(0), 0, 
    num_requests * sizeof(DataType), stream));

  const size_t block_size = 256;
  const size_t grid_size = (local_index_count + block_size - 1) / block_size;
  
  // ===== Round 0: Exchange rects =====
  // ToDO: Replace with AllGather
  cudaStreamSynchronize(stream);
  nvtxRangePushA("Exchange rects");
  // CHECK_NCCL(ncclGroupStart());
  // for(int i = 0; i < num_ranks; i++){
  //   CHECK_NCCL(ncclSend((void*)input_rect_device.ptr(0), sizeof(input_rect), ncclInt8, i, *nccl_comm, stream));
  //   CHECK_NCCL(ncclRecv((void*)global_rects.ptr(i * DIM_input * 2), sizeof(input_rect), ncclInt8, i, *nccl_comm, stream));
  // }
  // CHECK_NCCL(ncclGroupEnd());
  CHECK_NCCL(ncclAllGather((void*)input_rect_device.ptr(0), 
                         (void*)global_rects.ptr(0), 
                         sizeof(input_rect), 
                         ncclInt8, 
                         *nccl_comm, 
                         stream));
  cudaStreamSynchronize(stream);
  nvtxRangePop();

  // if(rank_id == 0){
  //   for(int i = 0; i < num_ranks; i++){
  //     legate::Rect<DIM_input> rect;
  //     cudaMemcpy(&rect, global_rects.ptr(i * DIM_input * 2), sizeof(input_rect), cudaMemcpyDeviceToHost);
  //     for(int j = 0; j < DIM_input; j++){
  //       printf("rank %d, global_rects[%d][%d]: %d, %d\n", rank_id, i, j, rect.lo[j], rect.hi[j]);
  //     }
  //   }
  // }
  
  // ===== Round 1: Compute request size histogram =====
  nvtxRangePushA("Compute request size histogram");
  compute_send_histogram<DIM_input><<<grid_size, block_size, 0, stream>>>(index_ptr, 
                round1_send_histo.ptr(0), 
                (legate::Rect<DIM_input>*)(global_rects.ptr(0)), 
                num_ranks, local_index_count);
  nvtxRangePop();
  // if(rank_id == 0) {
  // printf("round1_send_histo: ");
  // std::string send_histo_str = "";
  // for(size_t i = 0; i < num_ranks; i++){
  //   // unsigned int tmp = *(round1_send_histo.ptr(i));
  //   unsigned int tmp = 0;
  //   cudaMemcpy(&tmp, round1_send_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //   send_histo_str += std::to_string(tmp) + " ";
  // }
  // printf("%s\n", send_histo_str.c_str());
  // }
  nvtxRangePushA("Exclusive scan");
  cudaStreamSynchronize(stream);
    thrust::exclusive_scan(DEFAULT_POLICY.on(stream), 
                          (unsigned int*)round1_send_histo.ptr(0), 
                          ((unsigned int*)round1_send_histo.ptr(0)) + num_ranks, round1_send_offsets.ptr(0));
  cudaDeviceSynchronize();
  nvtxRangePop();
  // if(rank_id == 0){
  //   for(int i = 0; i < num_ranks; i++){
  //     unsigned int tmp = 0;
  //     cudaMemcpy(&tmp, round1_send_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //     printf("round1_send_offsets[%d]: %d\n", i, tmp);
  //   }
  // }
  

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
  pack_request_indices_kernel<DIM_input><<<grid_size, block_size, 0, stream>>>(index_ptr, num_requests,  
    (legate::Point<DIM_input>*)(round2_send_indices.ptr(0)), 
    round1_send_offsets.ptr(0),
    round2_request_positions.ptr(0), 
    packing_counters.ptr(0), 
    (legate::Rect<DIM_input>*)(global_rects.ptr(0)), num_ranks);
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  nvtxRangePushA("Exclusive scan");
  auto round2_recv_indices = create_buffer<legate::Point<DIM_input>>(total_indices_to_receive, Memory::Kind::GPU_FB_MEM);
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round2_recv_indices.ptr(0), 0, 
    total_indices_to_receive * sizeof(legate::Point<DIM_input>), stream));
  thrust::exclusive_scan(DEFAULT_POLICY.on(stream), round1_recv_histo.ptr(0), round1_recv_histo.ptr(0) + num_ranks, round1_recv_offsets.ptr(0));
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  // // ===== Round 2: All2All exchange request indices =====
  nvtxRangePushA("All2All exchange request indices");
  CHECK_NCCL(ncclGroupStart());
  for (size_t i = 0; i < num_ranks; ++i) {
      unsigned int indices_to_send_to_rank_i, send_offset_for_rank_i, indices_to_recv_from_rank_i, recv_offset_for_rank_i;
      cudaMemcpy(&indices_to_send_to_rank_i, round1_send_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&send_offset_for_rank_i, round1_send_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&indices_to_recv_from_rank_i, round1_recv_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&recv_offset_for_rank_i, round1_recv_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
      if (indices_to_send_to_rank_i > 0) {
          CHECK_NCCL(ncclSend((void*)(round2_send_indices.ptr(send_offset_for_rank_i * DIM_input)),
            indices_to_send_to_rank_i * sizeof(legate::Point<DIM_input>), ncclInt8, i, *nccl_comm, stream));
      }
      
      if (indices_to_recv_from_rank_i > 0) {
          CHECK_NCCL(ncclRecv(round2_recv_indices.ptr(recv_offset_for_rank_i * DIM_input),
                  indices_to_recv_from_rank_i * sizeof(legate::Point<DIM_input>), ncclInt8, i, *nccl_comm, stream));
      }
  }
  CHECK_NCCL(ncclGroupEnd());
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  nvtxRangePushA("Pack send data");
  auto round3_send_data = create_buffer<DataType>(total_indices_to_receive, Memory::Kind::GPU_FB_MEM);
  CUPYNUMERIC_CHECK_CUDA(cudaMemsetAsync(round3_send_data.ptr(0), 0, 
    total_indices_to_receive * sizeof(DataType), stream));
  pack_send_data_kernel<DataType, DIM_input><<<grid_size, block_size, 0, stream>>>(input_ptr, (legate::Point<DIM_input>*)round2_recv_indices.ptr(0),
  total_indices_to_receive, round3_send_data.ptr(0), input_rect);
  cudaStreamSynchronize(stream);
  nvtxRangePop();
  // ===== Round 3: All2All exchange actual data =====
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
  unpack_recv_data_kernel<DataType, DIM_output><<<grid_size, block_size, 0, stream>>>(output_ptr, round2_request_positions.ptr(0),
    num_requests, round3_recv_data.ptr(0));
  cudaStreamSynchronize(stream);
  nvtxRangePop();
}