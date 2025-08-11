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
 template<typename DataType, int DIM>
void global_all2all(
  const DataType* input_ptr, 
  const DataType* index_ptr, 
  DataType* output_ptr, 
 //  auto input_rect,
 //  auto index_rect,
 //  auto output_rect,
  int rank_id, 
  int num_ranks,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
) {
  // TODO: Implement global all2all communication using NCCL
  // For now, this is a placeholder that avoids the segmentation fault
  // caused by improper NCCL communicator handling
  
  // The communicator should be valid at this point since we included nccl.h
  // Basic validation to ensure nccl_comm is not null
  if (nccl_comm == nullptr) {
    return; // Early return if communicator is invalid
  }
  
  // Placeholder implementation - actual all2all logic would go here
  // This would typically involve:
  // 1. Calculating send/receive counts per rank
  // 2. Using ncclGroupStart()/ncclGroupEnd() with ncclSend/ncclRecv
  // 3. Proper data movement based on index_ptr values
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
 
   void operator()(TaskContext& context,
     const legate::PhysicalStore& input_array,
     const legate::PhysicalStore& index_array,
     const legate::PhysicalStore& output_array,
     const bool is_index_space,
     const size_t rank,
     const size_t num_ranks,
     const std::vector<comm::Communicator>& comms)
   {
     auto input_rect = input_array.shape<DIM>();
     auto index_rect = index_array.shape<DIM>();
     auto output_rect = output_array.shape<DIM>();
     
     auto input = input_array.read_accessor<VAL, DIM>(input_rect);
     auto index = index_array.read_accessor<VAL, DIM>(index_rect);
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
       const VAL* index_ptr = index.ptr(index_rect.lo);
       VAL* output_ptr = output.ptr(output_rect.lo);
       

       global_all2all<VAL, DIM>(
           input_ptr,
           index_ptr,
           output_ptr,
           rank,
           num_ranks,
           comms[0].get<ncclComm_t*>(),
           stream
       );
      
     }
 
     CUPYNUMERIC_CHECK_CUDA_STREAM(stream);
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