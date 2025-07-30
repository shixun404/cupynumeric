// /* Copyright 2024 NVIDIA Corporation
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  *
//  */

// #pragma once

// #include "cupynumeric/cupynumeric_task.h"
// #include "cupynumeric/cuda_help.h"

// #include <thrust/random.h>
// #include <curand_kernel.h>

// namespace cupynumeric {

// using namespace legate;

// enum class ShuffleMethod : int32_t {
//   KEY_SORT = 0,
//   FISHER_YATES = 1,
//   FEISTEL = 2,
//   FEISTEL_BIDIRECTIONAL = 3
// };

// enum class DistributedShuffleMethod : int32_t {
//   FEISTEL_ALL2ALL = 0,      // Feistel forward + all2all exchange
//   FEISTEL_BIDIRECTIONAL = 1, // Feistel forward + backward (no communication)
//   FISHER_YATES_GLOBAL = 2    // Fisher-Yates global permutation
// };

// struct ShuffleArgs {
//   legate::PhysicalStore input_output;  // in-place shuffle
//   ShuffleMethod method;
//   size_t volume;
//   size_t first_axis_size;
//   bool is_index_space;  // !single_task
//   size_t local_rank;
//   size_t num_ranks;
//   size_t num_shuffle_ranks;
// };

// template <typename T>
// struct ShufflePiece {
//   legate::Buffer<T> data;
//   legate::Buffer<int64_t> indices;
//   size_t size;
// };

// // Forward declarations for bijection classes (implemented in thrust::detail namespace in .cu file)

//   class feistel_bijection;
//   template <class IndexType> class random_bijection;


// // Forward declaration for shuffle implementation
// template <VariantKind KIND, Type::Code CODE, int32_t DIM>
// struct ShuffleImplBody;

// class ShuffleTask : public CuPyNumericTask<ShuffleTask> {
//  public:
//   static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_SHUFFLE}};

//   static constexpr auto CPU_VARIANT_OPTIONS =
//     legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);
//   static constexpr auto GPU_VARIANT_OPTIONS =
//     legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);
//   static constexpr auto OMP_VARIANT_OPTIONS =
//     legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);

//  public:
//   static void cpu_variant(legate::TaskContext context);
// #if LEGATE_DEFINED(LEGATE_USE_OPENMP)
//   static void omp_variant(legate::TaskContext context);
// #endif
// #if LEGATE_DEFINED(LEGATE_USE_CUDA)
//   static void gpu_variant(legate::TaskContext context);
// #endif
// };

// }  // namespace cupynumeric 