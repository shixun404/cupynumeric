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
        
      for(int i = 0; i < num_ranks; i++){
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
      }
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
  
  size_t local_input_count = get_volume<DIM_input>(input_rect);
  size_t local_index_count = get_volume<DIM_output>(index_rect);
  size_t local_output_count = get_volume<DIM_output>(output_rect);
  
  size_t num_requests = local_index_count;

  // ===== Round 0: Exchange rects =====
  thrust::device_vector<int64_t> global_rects(num_ranks * DIM_input * 2, 0);

  // ===== Round 1: Exchange request size histograms =====
  thrust::device_vector<unsigned int> round1_send_histo(num_ranks, 0);  // How many requests to send to each rank
  thrust::device_vector<unsigned int> round1_send_offsets(num_ranks, 0); // Offsets for packing requests
  thrust::device_vector<unsigned int> round1_recv_histo(num_ranks, 0);  // How many requests to receive from each rank
  thrust::device_vector<unsigned int> round1_recv_offsets(num_ranks, 0); // Offsets for unpacking requests
  thrust::device_vector<unsigned int> packing_counters(num_ranks, 0);   // Temporary counters for packing
  
  // ===== Round 2: Exchange request indices =====
  thrust::device_vector<int64_t> round2_send_indices(num_requests * DIM_input, 0);    // Indices to send (packed by target rank)
  thrust::device_vector<unsigned int> round2_request_positions(num_requests, 0); // Position of each request in output
  
  // ===== Round 3: Exchange actual data =====
  thrust::device_vector<DataType> round3_recv_data(num_requests, 0);  // Final received data

  const size_t block_size = 256;
  const size_t grid_size = (local_index_count + block_size - 1) / block_size;
  
  // ===== Round 0: Exchange rects =====
  CHECK_NCCL(ncclGroupStart());
  for(int i = 0; i < num_ranks; i++){
    CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(&input_rect), sizeof(input_rect), ncclInt8, i, *nccl_comm, stream));
    CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(global_rects.data() + i * DIM_input * 2), sizeof(input_rect), ncclInt8, i, *nccl_comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  cudaStreamSynchronize(stream);

  // if(rank_id == 0){
    for(int i = 0; i < num_ranks; i++){
      legate::Rect<DIM_input> rect;
      cudaMemcpy(&rect, thrust::raw_pointer_cast(global_rects.data() + i * DIM_input * 2), sizeof(input_rect), cudaMemcpyDeviceToHost);
      for(int j = 0; j < DIM_input; j++){
        printf("rank %d, global_rects[%d][%d]: %d, %d\n", rank_id, i, j, rect.lo[j], rect.hi[j]);
      }
    }
  // }
  
  // // ===== Round 1: Compute request size histogram =====
  // compute_send_histogram<DIM_input><<<grid_size, block_size, 0, stream>>>(index_ptr, 
  //               thrust::raw_pointer_cast(round1_send_histo.data()), 
  //               (legate::Rect<DIM_input>*)thrust::raw_pointer_cast(global_rects.data()), 
  //               num_ranks, local_index_count);
  
  // printf("round1_send_histo: ");
  // for(size_t i = 0; i < num_ranks; i++){
  //   unsigned int tmp = round1_send_histo[i];
  //   printf("%u ", tmp);
  // }
  // printf("\n");
  
  // thrust::exclusive_scan(round1_send_histo.begin(), round1_send_histo.end(), round1_send_offsets.begin());
  // cudaStreamSynchronize(stream);
  
  // // Pack request indices by target rank
  // pack_request_indices_kernel<DIM_input><<<grid_size, block_size, 0, stream>>>(index_ptr, num_requests,  
  //   (legate::Point<DIM_input>*)thrust::raw_pointer_cast(round2_send_indices.data()), 
  //   thrust::raw_pointer_cast(round1_send_offsets.data()),
  //   thrust::raw_pointer_cast(round2_request_positions.data()), 
  //   thrust::raw_pointer_cast(packing_counters.data()), 
  //   (legate::Rect<DIM_input>*)thrust::raw_pointer_cast(global_rects.data()), num_ranks);
  // cudaStreamSynchronize(stream);

  // // ===== Round 1: All2All exchange request size histograms =====
  // CHECK_NCCL(ncclGroupStart());
    
  // for (size_t i = 0; i < num_ranks; ++i) {
  //     CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(round1_send_histo.data()) + i, 1, ncclUint32, i, *nccl_comm, stream));
  //     CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(round1_recv_histo.data()) + i, 1, ncclUint32, i, *nccl_comm, stream));
  // }
  // CHECK_NCCL(ncclGroupEnd());
  // size_t total_indices_to_receive = thrust::reduce(round1_recv_histo.begin(), round1_recv_histo.end());
    
  // thrust::device_vector<legate::Point<DIM_input>> round2_recv_indices(total_indices_to_receive);
  // thrust::exclusive_scan(round1_recv_histo.begin(), round1_recv_histo.end(), round1_recv_offsets.begin());

  // cudaStreamSynchronize(stream);
  // // // ===== Round 2: All2All exchange request indices =====
  // CHECK_NCCL(ncclGroupStart());
  // for (size_t i = 0; i < num_ranks; ++i) {
  //     unsigned int indices_to_send_to_rank_i = round1_send_histo[i];
  //     unsigned int send_offset_for_rank_i = round1_send_offsets[i];
  //     unsigned int indices_to_recv_from_rank_i = round1_recv_histo[i];
  //     unsigned int recv_offset_for_rank_i = round1_recv_offsets[i];
  //     if (indices_to_send_to_rank_i > 0) {
  //         CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(round2_send_indices.data()) + send_offset_for_rank_i * DIM_input,
  //           indices_to_send_to_rank_i * sizeof(legate::Point<DIM_input>), ncclInt8, i, *nccl_comm, stream));
  //     }
      
  //     if (indices_to_recv_from_rank_i > 0) {
  //         CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(round2_recv_indices.data()) + recv_offset_for_rank_i * DIM_input,
  //                 indices_to_recv_from_rank_i * sizeof(legate::Point<DIM_input>), ncclInt8, i, *nccl_comm, stream));
  //     }
  // }
  // CHECK_NCCL(ncclGroupEnd());
  // cudaStreamSynchronize(stream);

  // thrust::device_vector<DataType> round3_send_data(total_indices_to_receive);
  
  // pack_send_data_kernel<DataType, DIM_input><<<grid_size, block_size, 0, stream>>>(input_ptr, (legate::Point<DIM_input>*)thrust::raw_pointer_cast(round2_recv_indices.data()),
  // total_indices_to_receive, thrust::raw_pointer_cast(round3_send_data.data()), input_rect);
  // cudaStreamSynchronize(stream);

  // // ===== Round 3: All2All exchange actual data =====
  // CHECK_NCCL(ncclGroupStart());
  // for (size_t i = 0; i < num_ranks; ++i) {
  //     unsigned int data_to_send_to_rank_i = round1_send_histo[i];
  //     unsigned int send_offset_for_rank_i = round1_send_offsets[i];
  //     unsigned int data_to_recv_from_rank_i = round1_recv_histo[i];
  //     unsigned int recv_offset_for_rank_i = round1_recv_offsets[i];
  //     if (data_to_recv_from_rank_i > 0) {
  //     CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(round3_send_data.data()) + recv_offset_for_rank_i,
  //     data_to_recv_from_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
  //   }
  //     if (data_to_send_to_rank_i > 0) {
  //       CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(round3_recv_data.data()) + send_offset_for_rank_i,
  //       data_to_send_to_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
  //     }   
  // }
  // CHECK_NCCL(ncclGroupEnd());
  // cudaStreamSynchronize(stream);
  
  // // ===== Final step: Unpack received data to output =====
  // unpack_recv_data_kernel<DataType, DIM_output><<<grid_size, block_size, 0, stream>>>(output_ptr, thrust::raw_pointer_cast(round2_request_positions.data()),
  //   num_requests, thrust::raw_pointer_cast(round3_recv_data.data()));
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
      printf("rank %d, rect_input.hi()[%d]: %d, rect_input.lo()[%d]: %d\n", args.rank_id, i, hi[i], i, lo[i]);
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