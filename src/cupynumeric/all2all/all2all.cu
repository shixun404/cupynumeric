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
 
 namespace cupynumeric {
 
 using namespace legate;

//  static int get_index(Domain domain, DomainPoint index_point)
//  {
//    int domain_index = 0;
//    auto hi          = domain.hi();
//    auto lo          = domain.lo();
//    for (int i = 0; i < domain.get_dim(); ++i) {
//      if (i > 0) {
//        domain_index *= hi[i] - lo[i] + 1;
//      }
//      domain_index += index_point[i];
//    }
//    return domain_index;
//  }

// Compute send histogram kernel for 2D (vectors)
template<typename IndexType>
__global__ void compute_send_histogram(const IndexType* indices, 
                                                 unsigned int* send_histo, size_t local_vector_count, size_t local_input_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < local_vector_count) {
        size_t target_gpu = indices[idx] / local_input_count;
        // printf("idx: %d, indices[idx]: %d, target_gpu: %d\n", (int)idx, (int)indices[idx], (int)target_gpu);
        atomicAdd(&send_histo[target_gpu], 1);
    }
}

// Pack send data kernel for 2D (vectors)
template<typename DataType, typename IndexType>
__global__ void pack_send_data_kernel(const DataType* data, const IndexType* indices, size_t request_count,
                                          DataType* send_data, int rank_id, int local_input_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
        send_data[idx] = data[indices[idx] - rank_id * local_input_count];
    }
}

template<typename DataType, typename IndexType>
__global__ void unpack_recv_data_kernel(DataType* data, const IndexType* request_indices, size_t request_count,
                                          const DataType* recv_data) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < request_count) {
        // printf("idx: %d, request_indices[idx]: %d\n", (int)idx, (int)request_indices[idx]);
        data[request_indices[idx]] = recv_data[idx];
    }
}
 

template<typename IndexType>
__global__ void pack_request_indices_kernel(const IndexType* indices, size_t vector_count,
                                          IndexType* send_indices, unsigned int* send_offsets, unsigned int* request_indices,
                                          unsigned int* counters, size_t local_input_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vector_count) {
        size_t target_gpu = indices[idx] / local_input_count;
        size_t pos = atomicAdd(&counters[target_gpu], 1);
        size_t offset = send_offsets[target_gpu] + pos;
        send_indices[offset] = indices[idx];
        request_indices[offset] = idx;
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
template<typename DataType, typename IndexType, int DIM>
void global_all2all(
  const DataType* input_ptr, 
  const IndexType* index_ptr, 
  DataType* output_ptr, 
  auto input_rect,
  auto index_rect,
  auto output_rect,
  int rank_id, 
  int num_ranks,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
) {
  
  size_t local_input_count = get_volume<DIM>(input_rect);
  size_t local_index_count = get_volume<DIM>(index_rect);
  size_t local_output_count = get_volume<DIM>(output_rect);
  printf("local_input_count: %zu\n", local_input_count);
  printf("local_index_count: %zu\n", local_index_count);
  printf("local_output_count: %zu\n", local_output_count);
  
  size_t num_requests = local_index_count;

  // ===== Round 1: Exchange request size histograms =====
  thrust::device_vector<unsigned int> round1_send_histo(num_ranks, 0);  // How many requests to send to each rank
  thrust::device_vector<unsigned int> round1_send_offsets(num_ranks, 0); // Offsets for packing requests
  thrust::device_vector<unsigned int> round1_recv_histo(num_ranks, 0);  // How many requests to receive from each rank
  thrust::device_vector<unsigned int> round1_recv_offsets(num_ranks, 0); // Offsets for unpacking requests
  thrust::device_vector<unsigned int> packing_counters(num_ranks, 0);   // Temporary counters for packing
  
  // ===== Round 2: Exchange request indices =====
  thrust::device_vector<IndexType> round2_send_indices(num_requests, 0);    // Indices to send (packed by target rank)
  thrust::device_vector<unsigned int> round2_request_positions(num_requests, 0); // Position of each request in output
  
  // ===== Round 3: Exchange actual data =====
  thrust::device_vector<DataType> round3_recv_data(num_requests, 0);  // Final received data

  const size_t block_size = 256;
  const size_t grid_size = (local_index_count + block_size - 1) / block_size;
  
  // ===== Round 1: Compute request size histogram =====
  compute_send_histogram<<<grid_size, block_size>>>(index_ptr, thrust::raw_pointer_cast(round1_send_histo.data()), local_index_count, local_input_count);

  printf("round1_send_histo: ");
  for(size_t i = 0; i < num_ranks; i++){
    unsigned int tmp = round1_send_histo[i];
    printf("%u ", tmp);
  }
  printf("\n");
  
  thrust::exclusive_scan(round1_send_histo.begin(), round1_send_histo.end(), round1_send_offsets.begin());

  printf("round1_send_offsets: ");
  for(size_t i = 0; i < num_ranks; i++){
    unsigned int tmp = round1_send_offsets[i];
    printf("%u ", tmp);
  }
  printf("\n");
  
  
  // Pack request indices by target rank
  pack_request_indices_kernel<<<grid_size, block_size>>>(index_ptr, num_requests,  
    thrust::raw_pointer_cast(round2_send_indices.data()), thrust::raw_pointer_cast(round1_send_offsets.data()),
    thrust::raw_pointer_cast(round2_request_positions.data()), thrust::raw_pointer_cast(packing_counters.data()), local_input_count);

    printf("round2_send_indices: ");
    for(size_t i = 0; i < num_requests; i++){
      unsigned int tmp = round2_send_indices[i];
      printf("%u ", tmp);
    }
    printf("\n");
    
    

    // ===== Round 1: All2All exchange request size histograms =====
    CHECK_NCCL(ncclGroupStart());
    
    for (size_t i = 0; i < num_ranks; ++i) {
        CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(round1_send_histo.data()) + i, 1, ncclUint32, i, *nccl_comm, stream));
        CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(round1_recv_histo.data()) + i, 1, ncclUint32, i, *nccl_comm, stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    size_t total_indices_to_receive = thrust::reduce(round1_recv_histo.begin(), round1_recv_histo.end());
    printf("total_indices_to_receive: %zu\n", total_indices_to_receive);
    
    thrust::device_vector<IndexType> round2_recv_indices(total_indices_to_receive);
    thrust::exclusive_scan(round1_recv_histo.begin(), round1_recv_histo.end(), round1_recv_offsets.begin());

    printf("round1_recv_offsets: ");
    for(size_t i = 0; i < num_ranks; i++){
      unsigned int tmp = round1_recv_offsets[i];
      printf("%u ", tmp);
    }
    printf("\n");

    // ===== Round 2: All2All exchange request indices =====
    CHECK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < num_ranks; ++i) {
        unsigned int indices_to_send_to_rank_i = round1_send_histo[i];
        unsigned int send_offset_for_rank_i = round1_send_offsets[i];
        unsigned int indices_to_recv_from_rank_i = round1_recv_histo[i];
        unsigned int recv_offset_for_rank_i = round1_recv_offsets[i];
        if (indices_to_send_to_rank_i > 0) {
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(round2_send_indices.data()) + send_offset_for_rank_i,
              indices_to_send_to_rank_i * sizeof(IndexType), ncclInt8, i, *nccl_comm, stream));
        }
        
        if (indices_to_recv_from_rank_i > 0) {
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(round2_recv_indices.data()) + recv_offset_for_rank_i,
                    indices_to_recv_from_rank_i * sizeof(IndexType), ncclInt8, i, *nccl_comm, stream));
        }
    }
    CHECK_NCCL(ncclGroupEnd());
    cudaStreamSynchronize(stream);
    printf("round2_recv_indices: ");
    for(size_t i = 0; i < total_indices_to_receive; i++){
      unsigned int tmp = round2_recv_indices[i];
      printf("%u ", tmp);
    }
    printf("\n");

    thrust::device_vector<DataType> round3_send_data(total_indices_to_receive);
    pack_send_data_kernel<<<grid_size, block_size>>>(input_ptr, thrust::raw_pointer_cast(round2_recv_indices.data()),
    total_indices_to_receive, thrust::raw_pointer_cast(round3_send_data.data()), rank_id, local_input_count);

    // ===== Round 3: All2All exchange actual data =====
     CHECK_NCCL(ncclGroupStart());
     for (size_t i = 0; i < num_ranks; ++i) {
         unsigned int data_to_send_to_rank_i = round1_send_histo[i];
         unsigned int send_offset_for_rank_i = round1_send_offsets[i];
         unsigned int data_to_recv_from_rank_i = round1_recv_histo[i];
         unsigned int recv_offset_for_rank_i = round1_recv_offsets[i];
         if (data_to_recv_from_rank_i > 0) {
          CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(round3_send_data.data()) + recv_offset_for_rank_i,
          data_to_recv_from_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
       }
         if (data_to_send_to_rank_i > 0) {
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(round3_recv_data.data()) + send_offset_for_rank_i,
            data_to_send_to_rank_i * sizeof(DataType), ncclInt8, i, *nccl_comm, stream));
         }   
     }
     CHECK_NCCL(ncclGroupEnd());
    // ===== Final step: Unpack received data to output =====
    unpack_recv_data_kernel<<<grid_size, block_size>>>(output_ptr, thrust::raw_pointer_cast(round2_request_positions.data()),
     num_requests, thrust::raw_pointer_cast(round3_recv_data.data()));
     printf("rank %d finished\n", rank_id);


}
 
 template <Type::Code CODE, int32_t DIM>
 struct All2AllImplBody<VariantKind::CPU, CODE, DIM> {
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
 template <Type::Code CODE, int32_t DIM>
 struct All2AllImplBody<VariantKind::OMP, CODE, DIM> {
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
 
 template <Type::Code CODE, int32_t DIM>
struct All2AllImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = type_of<CODE>;
  
  template <Type::Code INDEX_CODE>
  void execute_with_index_type(TaskContext& context,
    const legate::PhysicalStore& input_array,
    const legate::PhysicalStore& index_array,
    const legate::PhysicalStore& output_array,
    const bool is_index_space,
    const size_t rank,
    const size_t num_ranks,
    const std::vector<comm::Communicator>& comms)
  {
    using INDEX_VAL = type_of<INDEX_CODE>;
    
    auto input_rect = input_array.shape<DIM>();
    auto index_rect = index_array.shape<DIM>();
    auto output_rect = output_array.shape<DIM>();
    
    auto input = input_array.read_accessor<VAL, DIM>(input_rect);
    auto index = index_array.read_accessor<INDEX_VAL, DIM>(index_rect);
    auto output = output_array.read_write_accessor<VAL, DIM>(output_rect);
 
     // we allow empty domains for distributed sorting
    //  assert(input_rect.empty() || input.accessor.is_dense_row_major(input_rect));
    //  assert(index_rect.empty() || index.accessor.is_dense_row_major(index_rect));
    //  assert(output_rect.empty() || output.accessor.is_dense_row_major(output_rect));
     
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
       

       global_all2all<VAL, INDEX_VAL, DIM>(
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
      
     }
 
     CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
   }
   
   void operator()(TaskContext& context,
     const legate::PhysicalStore& input_array,
     const legate::PhysicalStore& index_array,
     const legate::PhysicalStore& output_array,
     const bool is_index_space,
     const size_t rank,
     const size_t num_ranks,
     const std::vector<comm::Communicator>& comms)
   {
     // Dispatch based on index array type
     auto index_code = index_array.code();
     switch (index_code) {
       case legate::Type::Code::INT32:
         execute_with_index_type<legate::Type::Code::INT32>(context, input_array, index_array, output_array, is_index_space, rank, num_ranks, comms);
         break;
       case legate::Type::Code::INT64:
         execute_with_index_type<legate::Type::Code::INT64>(context, input_array, index_array, output_array, is_index_space, rank, num_ranks, comms);
         break;
       case legate::Type::Code::UINT32:
         execute_with_index_type<legate::Type::Code::UINT32>(context, input_array, index_array, output_array, is_index_space, rank, num_ranks, comms);
         break;
       case legate::Type::Code::UINT64:
         execute_with_index_type<legate::Type::Code::UINT64>(context, input_array, index_array, output_array, is_index_space, rank, num_ranks, comms);
         break;
       case legate::Type::Code::FIXED_ARRAY:
         // Point<1> (struct int64[1]) can be treated as INT64
         // Note: STRUCT has value 18, not 17 as might be expected
         execute_with_index_type<legate::Type::Code::INT64>(context, input_array, index_array, output_array, is_index_space, rank, num_ranks, comms);
         break;
       default:
         printf("Unsupported index type code: %d\n", (int)index_code);
         assert(false && "Unsupported index type");
         break;
     }
   }
 };

 template <VariantKind KIND>
 struct All2AllImpl {
   template <Type::Code CODE, int DIM>
   void operator()(All2AllArgs& args, TaskContext& context, 
     std::vector<comm::Communicator> comms) const
   {
     using VAL = type_of<CODE>;
     auto rect_input = args.input.shape<DIM>();
     auto rect_index_array = args.index_array.shape<DIM>();
     auto rect_output = args.output.shape<DIM>();
    
     Pitches<DIM - 1> pitches;
     size_t input_volume = pitches.flatten(rect_input);
     size_t index_volume = pitches.flatten(rect_index_array);
     size_t output_volume = pitches.flatten(rect_output);
     
     if (input_volume == 0 && index_volume == 0 && output_volume == 0) {
       return;
     }
 
     All2AllImplBody<KIND, CODE, DIM>()(
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
   auto dim = std::max(1, std::max(args.input.dim(), args.index_array.dim()));
   dim = std::max(dim, args.output.dim());
   printf("args.input.code(): %d\n", args.input.code());
   printf("args.index_array.code(): %d\n", args.index_array.code());
   printf("args.output.code(): %d\n", args.output.code());  
   double_dispatch(
     dim, args.input.code(), All2AllImpl<KIND>{}, args, context, context.communicators());
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