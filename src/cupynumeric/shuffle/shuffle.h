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

#pragma once

#include "cupynumeric/cupynumeric_task.h"

#include <thrust/random.h>
#include <curand_kernel.h>

namespace cupynumeric {

enum class ShuffleMethod : int32_t {
  KEY_SORT = 0,
  FISHER_YATES = 1,
  FEISTEL = 2,
  FEISTEL_BIDIRECTIONAL = 3
};

enum class DistributedShuffleMethod : int32_t {
  FEISTEL_ALL2ALL = 0,      // Feistel forward + all2all exchange
  FEISTEL_BIDIRECTIONAL = 1, // Feistel forward + backward (no communication)
  FISHER_YATES_GLOBAL = 2    // Fisher-Yates global permutation
};

struct ShuffleArgs {
  legate::PhysicalStore input_output;  // in-place shuffle
  ShuffleMethod method;
  size_t volume;
  size_t first_axis_size;
  bool is_index_space;  // !single_task
  size_t local_rank;
  size_t num_ranks;
  size_t num_shuffle_ranks;
};

template <typename T>
struct ShufflePiece {
  legate::Buffer<T> data;
  legate::Buffer<int64_t> indices;
  size_t size;
};

// Feistel bijection implementation for CUDA
class feistel_bijection {
public:
  using index_type = std::uint64_t;
  
  template <class URBG>
  __host__ __device__ feistel_bijection(std::uint64_t m, URBG&& g);
  
  __host__ __device__ std::uint64_t nearest_power_of_two() const;
  __host__ __device__ std::uint64_t operator()(const std::uint64_t val) const;
  __host__ __device__ std::uint64_t inverse(const std::uint64_t val) const;

private:
  static __host__ __device__ void mulhilo(std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo);
  static __host__ __device__ void mulhi(std::uint64_t a, std::uint64_t b, std::uint32_t& hi);
  static __host__ __device__ void mullo(std::uint64_t a, std::uint64_t b, std::uint32_t& lo);
  static __host__ __device__ std::uint64_t get_cipher_bits(std::uint64_t m);
  
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
  __host__ __device__ random_bijection(IndexType n, URBG&& g);
  
  __host__ __device__ IndexType operator()(IndexType i) const;
  __host__ __device__ IndexType inverse(IndexType i) const;
  __host__ __device__ IndexType size() const;
};

class ShuffleTask : public CuPyNumericTask<ShuffleTask> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{CUPYNUMERIC_SHUFFLE}};

  static constexpr auto CPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);
  static constexpr auto GPU_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);
  static constexpr auto OMP_VARIANT_OPTIONS =
    legate::VariantOptions{}.with_concurrent(true).with_has_allocations(true);

 public:
  static void cpu_variant(legate::TaskContext context);
#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  static void omp_variant(legate::TaskContext context);
#endif
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace cupynumeric 