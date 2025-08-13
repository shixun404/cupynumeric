

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
    
    }} // namespace thrust::detail
    
    // Apply bijection kernel
    template<typename IndexType>
    __global__ void apply_bijection_kernel(IndexType* indices, size_t count, 
                                           thrust::detail::random_bijection<IndexType> bijection) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            indices[idx] = bijection(indices[idx]);
        }
    }
    
    // Apply inverse bijection kernel
    template<typename IndexType>
    __global__ void apply_inverse_bijection_kernel(IndexType* indices, size_t count, 
                                           thrust::detail::random_bijection<IndexType> bijection) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            indices[idx] = bijection.inverse(indices[idx]);
        }
    }
    
    // Compute send histogram kernel for 2D (vectors)
    template<typename IndexType>
    __global__ void compute_send_histogram_2d_kernel(const IndexType* indices, size_t vector_count,
                                                     unsigned int* send_histo, size_t local_vector_count, size_t total_gpus) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < vector_count) {
            size_t target_gpu = indices[idx] / local_vector_count;
            if (target_gpu < total_gpus) {
                atomicAdd(&send_histo[target_gpu], 1);
            }
        }
    }
    
    // Pack send data kernel for 2D (vectors)
    template<typename DataType, typename IndexType>
    __global__ void pack_send_data_2d_kernel(const DataType* data, const IndexType* indices, size_t vector_count,
                                              DataType* send_data, IndexType* send_indices,
                                              unsigned int* send_offsets, unsigned int* counters,
                                              size_t local_vector_count, size_t total_gpus, size_t vector_length) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < vector_count) {
            size_t target_gpu = indices[idx] / local_vector_count;
            if (target_gpu < total_gpus) {
                size_t pos = atomicAdd(&counters[target_gpu], 1);
                size_t offset = send_offsets[target_gpu] + pos;
                
                // Copy entire vector
                for (size_t v = 0; v < vector_length; v++) {
                    send_data[offset * vector_length + v] = data[idx * vector_length + v];
                }
                send_indices[offset] = indices[idx];
            }
        }
    }
    
    // Unpack received data kernel for 2D (vectors)
    template<typename DataType, typename IndexType>
    __global__ void unpack_recv_data_2d_kernel(const DataType* recv_data, const IndexType* recv_indices,
                                               size_t vector_count, DataType* output, size_t local_vector_count, 
                                               size_t gpu_vector_offset, size_t vector_length) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < vector_count) {
            IndexType global_vector_idx = recv_indices[idx];
            size_t local_vector_idx = global_vector_idx - gpu_vector_offset;
            if (local_vector_idx < local_vector_count) {
                // Copy entire vector
                for (size_t v = 0; v < vector_length; v++) {
                    output[local_vector_idx * vector_length + v] = recv_data[idx * vector_length + v];
                }
            }
        }
    }
    
    // Kernel to scatter data based on permuted indices for single rank
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

    
// Main global shuffle function for 2D data
template<typename DataType>
void global_shuffle_bidirectional(DataType* local_data, size_t local_count, size_t global_n,
                   size_t p1_nodes, size_t p2_gpus_per_node, int node_rank, int gpu_rank,
                   ncclComm_t nccl_comm, cudaStream_t stream, size_t vector_length) {
    
    nvtxRangePush("global_shuffle_bidirectional");
    
    size_t total_gpus = p1_nodes * p2_gpus_per_node;
    size_t global_gpu_id = node_rank * p2_gpus_per_node + gpu_rank;
    
    // Calculate vector counts
    size_t local_vector_count = local_count / vector_length;
    size_t global_vector_count = global_n / vector_length;
    size_t gpu_vector_offset = global_gpu_id * local_vector_count;
    
    const size_t block_size = 256;
    const size_t grid_size = (local_vector_count + block_size - 1) / block_size;
    
    // Step 3: Initialize index array with global vector IDs
    nvtxRangePush("Initialize indices");
    thrust::device_vector<uint64_t> indices(local_vector_count);
    thrust::sequence(indices.begin(), indices.end(), gpu_vector_offset);
    nvtxRangePop();
    
    #ifdef DEBUG
    // Print the first 5 vector indices before Feistel bijection
    {
        thrust::host_vector<uint64_t> h_indices_before(std::min<size_t>(5, local_vector_count));
        thrust::copy_n(indices.begin(), std::min<size_t>(5, local_vector_count), h_indices_before.begin());
        printf("rank %d, First 5 vector indices before Feistel bijection: ", gpu_rank);
        for (int i = 0; i < std::min<size_t>(5, local_vector_count); ++i) {
            printf("%zu ", h_indices_before[i]);
        }
        printf("\n");
    }
    #endif

    // Step 4: Apply Feistel bijection to vector indices
    thrust::default_random_engine rng(42);
    thrust::detail::random_bijection<uint64_t> bijection(global_vector_count, rng);

    apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);

    #ifdef DEBUG
    // Print the first 5 vector indices after Feistel bijection
    {
        cudaStreamSynchronize(stream); // Ensure kernel is done
        thrust::host_vector<uint64_t> h_indices_after(std::min<size_t>(5, local_vector_count));
        thrust::copy_n(indices.begin(), std::min<size_t>(5, local_vector_count), h_indices_after.begin());
        printf("rank %d, First 5 vector indices after Feistel bijection: ", gpu_rank);
        for (int i = 0; i < std::min<size_t>(5, local_vector_count); ++i) {
            printf("%zu ", h_indices_after[i]);
        }
        printf("\n");
    }
    #endif


    // Step 5: Compute send histogram for vectors
    thrust::device_vector<unsigned int> send_histo(total_gpus, 0);
    thrust::device_vector<unsigned int> counters(total_gpus, 0);
    
    compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()), local_vector_count,
        thrust::raw_pointer_cast(send_histo.data()), local_vector_count, total_gpus);
    
    // Compute send offsets
    thrust::device_vector<unsigned int> send_offsets(total_gpus);
    thrust::exclusive_scan(send_histo.begin(), send_histo.end(), send_offsets.begin());
    
    thrust::host_vector<unsigned int> h_send_offsets(total_gpus);
    thrust::copy_n(send_offsets.begin(), total_gpus, h_send_offsets.begin());

    #ifdef DEBUG
    {
        cudaStreamSynchronize(stream); // Ensure kernel is done
    // Print the send_offsets (exclusive_scan result) to check correctness
    printf("rank %d, send_offsets (exclusive_scan result): ", gpu_rank);
    for (size_t i = 0; i < h_send_offsets.size(); ++i) {
        printf("%u ", h_send_offsets[i]);
    }
    printf("\n");
    }
    #endif

    // Pack send data (vectors)
    size_t total_send_vectors = thrust::reduce(send_histo.begin(), send_histo.end());
    thrust::device_vector<DataType> send_data(total_send_vectors * vector_length);
    thrust::device_vector<uint64_t> send_indices(total_send_vectors);
    
    thrust::fill(counters.begin(), counters.end(), 0);
    pack_send_data_2d_kernel<<<grid_size, block_size, 0, stream>>>(
        local_data, thrust::raw_pointer_cast(indices.data()), local_vector_count,
        thrust::raw_pointer_cast(send_data.data()), thrust::raw_pointer_cast(send_indices.data()),
        thrust::raw_pointer_cast(send_offsets.data()), thrust::raw_pointer_cast(counters.data()),
        local_vector_count, total_gpus, vector_length);
    
    
    // Step 6: All2all exchange histograms
    
    thrust::host_vector<unsigned int> h_recv_histo(total_gpus);
    thrust::device_vector<unsigned int> recv_histo(total_gpus);
    thrust::host_vector<unsigned int> h_send_histo = send_histo;


    thrust::copy_n(send_histo.begin(), total_gpus, h_send_histo.begin());



    
    // ///////////////////////////////////////////////////
    // Method 2: Use Feistel backward to calculate receive histogram (2D)
    nvtxRangePush("Compute receive histogram (bidirectional)");
    thrust::device_vector<uint64_t> recv_indices(local_vector_count);
    thrust::sequence(recv_indices.begin(), recv_indices.end(), gpu_vector_offset);
    
    thrust::default_random_engine rng_1(42);
    thrust::detail::random_bijection<uint64_t> bijection_1(global_vector_count, rng_1);
    
    apply_inverse_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(recv_indices.data()), local_vector_count, bijection_1);
    
    // Compute histogram from inverse indices
    thrust::fill(recv_histo.begin(), recv_histo.end(), 0);
    compute_send_histogram_2d_kernel<<<grid_size, block_size, 0, stream>>>(
    thrust::raw_pointer_cast(recv_indices.data()), local_vector_count,
    thrust::raw_pointer_cast(recv_histo.data()), local_vector_count, total_gpus);
    thrust::copy_n(recv_histo.begin(), total_gpus, h_recv_histo.begin());
    nvtxRangePop();

    
    // Step 7: Create recv buffers for vectors
    size_t total_recv_vectors = thrust::reduce(recv_histo.begin(), recv_histo.end());
    thrust::device_vector<DataType> recv_data(total_recv_vectors * vector_length);
    
    thrust::device_vector<size_t> recv_offsets(total_gpus);
    thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
    

    thrust::host_vector<size_t> h_recv_offsets(total_gpus);
    thrust::copy_n(recv_offsets.begin(), total_gpus, h_recv_offsets.begin());



    // Step 8: All2all distribute data
    CHECK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < total_gpus; ++i) {
        if (h_send_histo[i] > 0) {
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i] * vector_length,
                    h_send_histo[i] * vector_length, ncclInt, i, nccl_comm, stream));
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices.data()) + h_send_offsets[i],
                    h_send_histo[i], ncclUint64, i, nccl_comm, stream));
        }
        
        if (h_recv_histo[i] > 0) {
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i] * vector_length,
                    h_recv_histo[i] * vector_length, ncclInt, i, nccl_comm, stream));
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
                    h_recv_histo[i], ncclUint64, i, nccl_comm, stream));
        }
    }
    CHECK_NCCL(ncclGroupEnd());

    thrust::host_vector<DataType> h_recv_data(total_recv_vectors * vector_length);
    thrust::host_vector<uint64_t> h_recv_indices(total_recv_vectors);
    thrust::copy_n(recv_data.begin(), total_recv_vectors * vector_length, h_recv_data.begin());
    thrust::copy_n(recv_indices.begin(), total_recv_vectors, h_recv_indices.begin());
    
    // Step 9: Unpack to final positions (vectors)
    const size_t unpack_grid_size = (total_recv_vectors + block_size - 1) / block_size;
    unpack_recv_data_2d_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
        total_recv_vectors, local_data, local_vector_count, gpu_vector_offset, vector_length);
    
    cudaStreamSynchronize(stream);
    nvtxRangePop(); // End of global_shuffle_bidirectional
}