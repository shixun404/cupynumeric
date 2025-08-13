template<typename DataType>
void global_shuffle_bidirectional(
  DataType* local_data, 
  size_t vector_length,
  size_t local_count, 
  size_t global_n,
  int rank_id, 
  int num_ranks,
  ncclComm_t* nccl_comm, 
  cudaStream_t stream
) {

    nvtxRangePush("global_shuffle_bidirectional");
    
    size_t total_gpus = num_ranks;
    size_t global_gpu_id = rank_id;
    
    // Calculate vector counts
    size_t local_vector_count = local_count / vector_length;
    size_t global_vector_count = global_n / vector_length;
    size_t gpu_vector_offset = rank_id * local_vector_count;
    printf("-----Shuffle Info------\n");
    printf("rank_id:             %d\n", rank_id);
    printf("num_ranks:           %zu\n", num_ranks);
    printf("total_gpus:          %zu\n", total_gpus);
    printf("global_gpu_id:       %zu\n", global_gpu_id);
    printf("vector_length:       %zu\n", vector_length);
    printf("local_count:         %zu\n", local_count);
    printf("global_n:            %zu\n", global_n);
    printf("local_vector_count:  %zu\n", local_vector_count);
    printf("global_vector_count: %zu\n", global_vector_count);
    printf("gpu_vector_offset:   %zu\n", gpu_vector_offset);
    printf("--------------------------------\n");
    const size_t block_size = 256;
    const size_t grid_size = (local_vector_count + block_size - 1) / block_size;
    
    // Step 3: Initialize index array with global vector IDs
    printf("Rank %d, Step 3, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    nvtxRangePush("Initialize indices");
    thrust::device_vector<uint64_t> indices(local_vector_count);
    thrust::sequence(indices.begin(), indices.end(), gpu_vector_offset);
    nvtxRangePop();


    // Step 4: Apply Feistel bijection to vector indices
    printf("Rank %d, Step 4, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    thrust::default_random_engine rng(42);
    random_bijection<uint64_t> bijection(global_vector_count, rng);

    apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()), local_vector_count, bijection);

    // Step 5: Compute send histogram for vectors
    printf("Rank %d, Step 5, File: %s:%d\n", rank_id, __FILE__, __LINE__);
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
    printf("Rank %d, Step 6, File: %s:%d\n", rank_id, __FILE__, __LINE__);
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
    random_bijection<uint64_t> bijection_1(global_vector_count, rng_1);
    
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
    printf("Rank %d, Step 7, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    size_t total_recv_vectors = thrust::reduce(recv_histo.begin(), recv_histo.end());
    thrust::device_vector<DataType> recv_data(total_recv_vectors * vector_length);
    
    thrust::device_vector<size_t> recv_offsets(total_gpus);
    thrust::host_vector<size_t> h_recv_offsets(total_gpus);

    thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
    thrust::copy_n(recv_offsets.begin(), total_gpus, h_recv_offsets.begin());

    // Step 8: All2all distribute data
    printf("Rank %d, Step 8, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    CHECK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < total_gpus; ++i) {
        if (h_send_histo[i] > 0) {
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i] * vector_length,
                    h_send_histo[i] * vector_length, ncclInt, i, *nccl_comm, stream));
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices.data()) + h_send_offsets[i],
                    h_send_histo[i], ncclUint64, i, *nccl_comm, stream));
        }
        
        if (h_recv_histo[i] > 0) {
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i] * vector_length,
                    h_recv_histo[i] * vector_length, ncclInt, i, *nccl_comm, stream));
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
                    h_recv_histo[i], ncclUint64, i, *nccl_comm, stream));
        }
    }
    CHECK_NCCL(ncclGroupEnd());
    
    // Step 9: Unpack to final positions (vectors)
    printf("Rank %d, Step 9, File: %s:%d\n", rank_id, __FILE__, __LINE__);
    printf("total_recv_vectors: %zu\n", total_recv_vectors);
    const size_t unpack_grid_size = (total_recv_vectors + block_size - 1) / block_size;
    unpack_recv_data_2d_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
        total_recv_vectors, local_data, local_vector_count, gpu_vector_offset, vector_length);
    
    cudaStreamSynchronize(stream);
    nvtxRangePop(); // End of global_shuffle_bidirectional
    printf("Rank %d, End of global_shuffle_bidirectional, File: %s:%d\n", rank_id, __FILE__, __LINE__);
}