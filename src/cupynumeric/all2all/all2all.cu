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
printf("Rank %d is using GPU device %d (%s)\n", rank_id, device_id, prop.name);

  size_t local_input_count = get_volume<DIM_input>(input_rect);
  size_t local_index_count = get_volume<DIM_output>(index_rect);
  size_t local_output_count = get_volume<DIM_output>(output_rect);
  
  size_t num_requests = local_index_count;
  // ===== Round 0: Exchange rects =====
  auto global_rects = create_buffer<int64_t>(num_ranks * DIM_input * 2, Memory::Kind::GPU_FB_MEM);
  auto input_rect_device = create_buffer<int64_t>(DIM_input * 2, Memory::Kind::GPU_FB_MEM);
  cudaMemcpy((legate::Rect<DIM_input>*)input_rect_device.ptr(0), &input_rect, sizeof(input_rect), cudaMemcpyHostToDevice);

  // ===== Round 1: Exchange request size histograms =====
  auto round1_send_histo = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // How many requests to send to each rank
  auto round1_send_offsets = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // Offsets for packing requests
  auto round1_recv_histo = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // How many requests to receive from each rank
  auto round1_recv_offsets = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // Offsets for unpacking requests
  auto packing_counters = create_buffer<unsigned int>(num_ranks, Memory::Kind::GPU_FB_MEM); // Temporary counters for packing
  
  // // ===== Round 2: Exchange request indices =====
  auto round2_send_indices = create_buffer<int64_t>(num_requests * DIM_input, Memory::Kind::GPU_FB_MEM); // Indices to send (packed by target rank)
  auto round2_request_positions = create_buffer<unsigned int>(num_requests, Memory::Kind::GPU_FB_MEM); // Position of each request in output
   
  
  // // ===== Round 3: Exchange actual data =====
  auto round3_recv_data = create_buffer<DataType>(num_requests, Memory::Kind::GPU_FB_MEM); // Final received data

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
  
  // pack_send_data_kernel<DataType, DIM_input><<<grid_size, block_size, 0, stream>>>(input_ptr, (legate::Point<DIM_input>*)round2_recv_indices.ptr(0),
  // total_indices_to_receive, round3_send_data.ptr(0), input_rect);
  // cudaStreamSynchronize(stream);
  // nvtxRangePop();
  // // ===== Round 3: All2All exchange actual data =====
  // nvtxRangePushA("All2All exchange actual data");
  // CHECK_NCCL(ncclGroupStart());
  // for (size_t i = 0; i < num_ranks; ++i) {
  //     unsigned int data_to_send_to_rank_i, send_offset_for_rank_i, data_to_recv_from_rank_i, recv_offset_for_rank_i;
  //     cudaMemcpy(&data_to_send_to_rank_i, round1_send_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //     cudaMemcpy(&send_offset_for_rank_i, round1_send_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //     cudaMemcpy(&data_to_recv_from_rank_i, round1_recv_histo.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);
  //     cudaMemcpy(&recv_offset_for_rank_i, round1_recv_offsets.ptr(i), sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //     if (data_to_recv_from_rank_i > 0) {
  //     CHECK_NCCL(ncclSend(round3_send_data.ptr(recv_offset_for_rank_i),
  //     data_to_recv_from_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
  //   }
  //     if (data_to_send_to_rank_i > 0) {
  //       CHECK_NCCL(ncclRecv(round3_recv_data.ptr(send_offset_for_rank_i),
  //       data_to_send_to_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
  //     }   
  // }
  // CHECK_NCCL(ncclGroupEnd());
  // cudaStreamSynchronize(stream);
  // nvtxRangePop();
  // // ===== Final step: Unpack received data to output =====
  // nvtxRangePushA("unpack received data");
  // unpack_recv_data_kernel<DataType, DIM_output><<<grid_size, block_size, 0, stream>>>(output_ptr, round2_request_positions.ptr(0),
  //   num_requests, round3_recv_data.ptr(0));
  // cudaStreamSynchronize(stream);
  // nvtxRangePop();
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
 
     auto stream = get_cached_stream();
 
     bool need_distributed_all2all = (num_ranks > 1) && is_index_space;
   
     // For local all2all (single node or within a node)
     if (!need_distributed_all2all) {
     
           // TODO: Implement key-sort local all2all
           assert(false && "Key-sort local all2all not yet implemented");
     
     } else {
       // Handle distributed all2all
       
       
        const VAL* input_ptr = input.ptr(input_rect.lo);
         const INDEX_VAL* index_ptr = index.ptr(index_rect.lo);
         VAL* output_ptr = output.ptr(output_rect.lo);
         nvtxRangePushA("global_all2all");
      cudaDeviceSynchronize();
       global_all2all<VAL, DIM_input, DIM_output>(
           input_ptr,
           index_ptr,
           output_ptr,
           input_rect,
           index_rect,
           output_rect,
           rank,
           num_ranks,
          //  input_dim,
           comms[0].get<ncclComm_t*>(),
           stream
       );
       cudaDeviceSynchronize();
       nvtxRangePop();
     }
 
     CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
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
     
     if (input_volume == 0 && index_volume == 0 && output_volume == 0) {
       return;
     }
    //  printf("DIM_input: %d, DIM_output: %d\n", DIM_input, DIM_output);
    //  auto input_shape_span = context.scalar(0).values<int64_t>();
    //  for (size_t i = 0; i < input_shape_span.size(); ++i) {
    //   printf("input shape_span[%d]: %d\n", i, int(input_shape_span[i]));
    //  }

    //  auto index_shape_span = context.scalar(1).values<int64_t>();
    //  for (size_t i = 0; i < index_shape_span.size(); ++i) {
    //   printf("index shape_span[%d]: %d\n", i, int(index_shape_span[i]));
    //  }
     
     for (int i = 0; i < DIM_input; i++) {  
      auto hi          = rect_input.hi;
     auto lo          = rect_input.lo;
      // printf("num_ranks %d, rank %d, rect_input.hi()[%d]: %d, rect_input.lo()[%d]: %d\n", args.num_ranks, args.rank_id, i, hi[i], i, lo[i]);
     }
    //  for (int i = 0; i < DIM_output; i++) {
    //   auto hi          = rect_output.hi;
    //   auto lo          = rect_output.lo;
    //   printf("rect_output.hi()[%d]: %d, rect_output.lo()[%d]: %d\n", i, hi[i], i, lo[i]);
    //  }
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

//  template <VariantKind KIND>
//  struct All2AllImpl {
//    template <int DIM_input, int DIM_output>
//    void operator()(All2AllArgs& args, TaskContext& context, 
//      std::vector<comm::Communicator> comms) const
//    {
//     type_dispatch(
//       args.input.code(), All2AllImpl_type<KIND, DIM_input, DIM_output>{}, args, context, context.communicators());
//    }
//  };

template <VariantKind KIND>
struct All2AllImpl {
  template <int DIM_input, int DIM_output>
  void operator()(All2AllArgs& args, TaskContext& context, 
    std::vector<comm::Communicator> comms) const
  {
   // Custom type dispatch to only compile int64 type for faster compilation
   auto input_code = args.input.code();
   switch (input_code) {
     case legate::Type::Code::FLOAT32:
       All2AllImpl_type<KIND, DIM_input, DIM_output>{}.template operator()<legate::Type::Code::FLOAT32>(args, context, context.communicators());
       break;
     default:
       printf("Unsupported data type code: %d. Only INT64 is supported for fast compilation.\n", (int)input_code);
       assert(false && "Only INT64 data type is supported in this build");
       break;
   }
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
  
   auto dim_input = args.input.dim();
   auto dim_output = args.output.dim();

  
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