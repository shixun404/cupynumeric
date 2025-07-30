// Global Shuffle Implementation for Multi-Node Multi-GPU
// Single file implementation under 500 lines

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <iostream>
#include <vector>

#include <curand_kernel.h>

// Include the random bijection classes
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

// Error checking utilities
namespace cupynumeric {
    inline void check_nccl(ncclResult_t result, const char* file, int line) {
        if (result != ncclSuccess) {
            std::cerr << "NCCL error at " << file << ":" << line 
                      << " - " << ncclGetErrorString(result) << std::endl;
            exit(1);
        }
    }
}

#define CHECK_NCCL(...)                                      \
  do {                                                       \
    ncclResult_t __result__ = (__VA_ARGS__);                 \
    cupynumeric::check_nccl(__result__, __FILE__, __LINE__); \
  } while (false)

// 1. Kernel to initialize a cuRAND state for each thread
__global__
void init_curand_states(curandState *states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// 2. Device functor that uses cuRAND
struct CurandFunctor {
    curandState* states;
    int n;
    int gpu_rank;
    __device__ int operator()() {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // return curand(&states[idx]) % 1000;
            return idx + gpu_rank * n + (gpu_rank + 1) * 10000;
        }
        return 0;
    }
};



namespace thrust { namespace detail {

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

private:
    static __host__ __device__ void mulhilo(std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo) {
        std::uint64_t product = a * b;
        hi = static_cast<std::uint32_t>(product >> 32);
        lo = static_cast<std::uint32_t>(product);
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

// Compute send histogram kernel
template<typename IndexType>
__global__ void compute_send_histogram_kernel(const IndexType* indices, size_t count,
                                              unsigned int* send_histo, size_t local_size, size_t total_gpus) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        size_t target_gpu = indices[idx] / local_size;
        if (target_gpu < total_gpus) {
            atomicAdd(&send_histo[target_gpu], 1);
        }
    }
}

// Pack send data kernel
template<typename DataType, typename IndexType>
__global__ void pack_send_data_kernel(const DataType* data, const IndexType* indices, size_t count,
                                       DataType* send_data, IndexType* send_indices,
                                       unsigned int* send_offsets, unsigned int* counters,
                                       size_t local_size, size_t total_gpus) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        size_t target_gpu = indices[idx] / local_size;
        if (target_gpu < total_gpus) {
            size_t pos = atomicAdd(&counters[target_gpu], 1);
            size_t offset = send_offsets[target_gpu] + pos;
            send_data[offset] = data[idx];
            send_indices[offset] = indices[idx];
        }
    }
}

// Unpack received data kernel
template<typename DataType, typename IndexType>
__global__ void unpack_recv_data_kernel(const DataType* recv_data, const IndexType* recv_indices,
                                         size_t count, DataType* output, size_t local_size, size_t gpu_offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        IndexType global_idx = recv_indices[idx];
        size_t local_idx = global_idx - gpu_offset;
        if (local_idx < local_size) {
            output[local_idx] = recv_data[idx];
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

// Main global shuffle function
template<typename DataType>
void global_shuffle(DataType* local_data, size_t local_count, size_t global_n,
                   size_t p1_nodes, size_t p2_gpus_per_node, int node_rank, int gpu_rank,
                   ncclComm_t nccl_comm, cudaStream_t stream) {
    
    size_t total_gpus = p1_nodes * p2_gpus_per_node;
    size_t global_gpu_id = node_rank * p2_gpus_per_node + gpu_rank;
    size_t gpu_offset = global_gpu_id * local_count;
    
    const size_t block_size = 256;
    const size_t grid_size = (local_count + block_size - 1) / block_size;
    
    // Step 3: Initialize index array with global IDs
    thrust::device_vector<uint64_t> indices(local_count);
    thrust::sequence(indices.begin(), indices.end(), gpu_offset);
    
    // // Step 4: Apply Feistel bijection
    // thrust::default_random_engine rng(42);
    // thrust::detail::random_bijection<uint64_t> bijection(global_n, rng);
    
    // apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
    //     thrust::raw_pointer_cast(indices.data()), local_count, bijection);
    // Print the first 5 indices before Feistel bijection
    {
        thrust::host_vector<uint64_t> h_indices_before(std::min<size_t>(5, local_count));
        thrust::copy_n(indices.begin(), std::min<size_t>(5, local_count), h_indices_before.begin());
        // printf("rank %d, First 5 indices before Feistel bijection: ", gpu_rank);
        for (int i = 0; i < std::min<size_t>(5, local_count); ++i) {
            // printf("%zu ", h_indices_before[i]);
        }
        // printf("\n");
    }

    // Step 4: Apply Feistel bijection
    thrust::default_random_engine rng(42);
    thrust::detail::random_bijection<uint64_t> bijection(global_n, rng);

    apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()), local_count, bijection);

    // Print the first 5 indices after Feistel bijection
    {
        cudaStreamSynchronize(stream); // Ensure kernel is done
        thrust::host_vector<uint64_t> h_indices_after(std::min<size_t>(5, local_count));
        thrust::copy_n(indices.begin(), std::min<size_t>(5, local_count), h_indices_after.begin());
        // printf("rank %d, First 5 indices after Feistel bijection: ", gpu_rank);
        for (int i = 0; i < std::min<size_t>(5, local_count); ++i) {
            // printf("%zu ", h_indices_after[i]);
        }
        // printf("\n");
    }
    // Step 5: Compute send histogram
    thrust::device_vector<unsigned int> send_histo(total_gpus, 0);
    thrust::device_vector<unsigned int> counters(total_gpus, 0);
    
    compute_send_histogram_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()), local_count,
        thrust::raw_pointer_cast(send_histo.data()), local_count, total_gpus);
    
    // Compute send offsets
    thrust::device_vector<unsigned int> send_offsets(total_gpus);
    thrust::exclusive_scan(send_histo.begin(), send_histo.end(), send_offsets.begin());
    

    thrust::host_vector<unsigned int> h_send_offsets(total_gpus);
    thrust::copy_n(send_offsets.begin(), total_gpus, h_send_offsets.begin());
    {
        cudaStreamSynchronize(stream); // Ensure kernel is done
    // Print the send_offsets (exclusive_scan result) to check correctness
    // printf("rank %d, send_offsets (exclusive_scan result): ", gpu_rank);
    for (size_t i = 0; i < h_send_offsets.size(); ++i) {
        // printf("%u ", h_send_offsets[i]);
    }

    // printf("\n");
    }

    // Pack send data
    size_t total_send = thrust::reduce(send_histo.begin(), send_histo.end());
    thrust::device_vector<DataType> send_data(total_send);
    thrust::device_vector<uint64_t> send_indices(total_send);
    
    thrust::fill(counters.begin(), counters.end(), 0);
    pack_send_data_kernel<<<grid_size, block_size, 0, stream>>>(
        local_data, thrust::raw_pointer_cast(indices.data()), local_count,
        thrust::raw_pointer_cast(send_data.data()), thrust::raw_pointer_cast(send_indices.data()),
        thrust::raw_pointer_cast(send_offsets.data()), thrust::raw_pointer_cast(counters.data()),
        local_count, total_gpus);
    
    // printf("total_send: %zu\n", total_send);
    // Step 6: All2all exchange histograms
    
    thrust::host_vector<unsigned int> h_recv_histo(total_gpus);
    thrust::device_vector<unsigned int> recv_histo(total_gpus);
    thrust::host_vector<unsigned int> h_send_histo = send_histo;


    thrust::copy_n(send_histo.begin(), total_gpus, h_send_histo.begin());
    for (int i = 0; i < total_gpus; ++i) {
        // printf("before all2all: send_histo[%d]: %u, recv_histo[%d]: %u\n", i, h_send_histo[i], i, h_recv_histo[i]);
    }
    
    CHECK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < total_gpus; ++i) {
        CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_histo.data()) + i, 1, ncclUint32, i, nccl_comm, stream));
        CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_histo.data()) + i, 1, ncclUint32, i, nccl_comm, stream));
    }
    CHECK_NCCL(ncclGroupEnd());
    

    thrust::copy_n(recv_histo.begin(), total_gpus, h_recv_histo.begin());
    for (int i = 0; i < total_gpus; ++i) {
        // printf("gpu_rank: %d, after all2all: send_histo[%d]: %u, recv_histo[%d]: %u\n", gpu_rank, i, h_send_histo[i], i, h_recv_histo[i]);
    }
    
    // Step 7: Create recv buffers
    size_t total_recv = thrust::reduce(recv_histo.begin(), recv_histo.end());
    // printf("gpu_rank: %d, total_recv: %zu\n", gpu_rank, total_recv);
    thrust::device_vector<DataType> recv_data(total_recv);
    thrust::device_vector<uint64_t> recv_indices(total_recv);
    
    thrust::device_vector<size_t> recv_offsets(total_gpus);
    thrust::exclusive_scan(recv_histo.begin(), recv_histo.end(), recv_offsets.begin());
    

    thrust::host_vector<size_t> h_recv_offsets(total_gpus);
    thrust::copy_n(recv_offsets.begin(), total_gpus, h_recv_offsets.begin());



    thrust::host_vector<DataType> h_send_data(total_send);
    thrust::host_vector<uint64_t> h_send_indices(total_send);
    thrust::copy_n(send_data.begin(), total_send, h_send_data.begin());
    thrust::copy_n(send_indices.begin(), total_send, h_send_indices.begin());
    for (int i = 0; i < total_recv; ++i) {
        // printf("gpu_rank: %d, send_data[%d]: %d, send_indices[%d]: %zu\n", gpu_rank, i, h_send_data[i], i, h_send_indices[i]);
    }

    for (int i = 0; i < total_gpus; ++i) {
        // printf("gpu_rank: %d, send_offsets[%d]: %zu, recv_offsets[%d]: %zu, h_send_histo[%d]: %u, h_recv_histo[%d]: %u\n", gpu_rank, i, h_send_offsets[i], i, h_recv_offsets[i], i, h_send_histo[i], i, h_recv_histo[i]);
    }

    // Step 8: All2all distribute data
    CHECK_NCCL(ncclGroupStart());
    for (size_t i = 0; i < total_gpus; ++i) {
        if (h_send_histo[i] > 0) {
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_data.data()) + h_send_offsets[i],
                    h_send_histo[i], ncclInt, i, nccl_comm, stream));
            CHECK_NCCL(ncclSend(thrust::raw_pointer_cast(send_indices.data()) + h_send_offsets[i],
                    h_send_histo[i], ncclUint64, i, nccl_comm, stream));
        }
        
        if (h_recv_histo[i] > 0) {
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_data.data()) + h_recv_offsets[i],
                    h_recv_histo[i], ncclInt, i, nccl_comm, stream));
            CHECK_NCCL(ncclRecv(thrust::raw_pointer_cast(recv_indices.data()) + h_recv_offsets[i],
                    h_recv_histo[i], ncclUint64, i, nccl_comm, stream));
        }
    }
    CHECK_NCCL(ncclGroupEnd());

    thrust::host_vector<DataType> h_recv_data(total_recv);
    thrust::host_vector<uint64_t> h_recv_indices(total_recv);
    thrust::copy_n(recv_data.begin(), total_recv, h_recv_data.begin());
    thrust::copy_n(recv_indices.begin(), total_recv, h_recv_indices.begin());
    for (int i = 0; i < total_recv; ++i) {
        // printf("gpu_rank: %d, recv_data[%d]: %d, recv_indices[%d]: %zu\n", gpu_rank, i, h_recv_data[i], i, h_recv_indices[i]);
    }
    // Step 9: Unpack to final positions
    const size_t unpack_grid_size = (total_recv + block_size - 1) / block_size;
    unpack_recv_data_kernel<<<unpack_grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(recv_data.data()), thrust::raw_pointer_cast(recv_indices.data()),
        total_recv, local_data, local_count, gpu_offset);
    
    cudaStreamSynchronize(stream);
}

// Single rank shuffle function - simplified version for single GPU
// This function performs a local shuffle when there's only one GPU/rank involved.
// It applies the same Feistel bijection as the global shuffle but avoids all the
// inter-GPU communication overhead since all data is local.
template<typename DataType>
void single_rank_shuffle(DataType* data, size_t count, size_t global_n, cudaStream_t stream) {
    const size_t block_size = 256;
    const size_t grid_size = (count + block_size - 1) / block_size;
    
    // Step 1: Initialize index array with local IDs
    thrust::device_vector<uint64_t> indices(count);
    thrust::sequence(indices.begin(), indices.end(), 0);
    
    // Print the first 5 indices before Feistel bijection
    {
        thrust::host_vector<uint64_t> h_indices_before(std::min<size_t>(5, count));
        thrust::copy_n(indices.begin(), std::min<size_t>(5, count), h_indices_before.begin());
        // printf("Single rank - First 5 indices before Feistel bijection: ");
        for (int i = 0; i < std::min<size_t>(5, count); ++i) {
            // printf("%zu ", h_indices_before[i]);
        }
        // printf("\n");
    }
    
    // Step 2: Apply Feistel bijection
    thrust::default_random_engine rng(42);
    thrust::detail::random_bijection<uint64_t> bijection(global_n, rng);
    
    apply_bijection_kernel<<<grid_size, block_size, 0, stream>>>(
        thrust::raw_pointer_cast(indices.data()), count, bijection);
    
    // Print the first 5 indices after Feistel bijection
    {
        cudaStreamSynchronize(stream); // Ensure kernel is done
        thrust::host_vector<uint64_t> h_indices_after(std::min<size_t>(5, count));
        thrust::copy_n(indices.begin(), std::min<size_t>(5, count), h_indices_after.begin());
        // printf("Single rank - First 5 indices after Feistel bijection: ");
        for (int i = 0; i < std::min<size_t>(5, count); ++i) {
            // printf("%zu ", h_indices_after[i]);
        }
        // printf("\n");
    }
    
    // Step 3: Create temporary buffer for shuffled data
    thrust::device_vector<DataType> temp_data(count);
    
    // Step 4: Rearrange data based on shuffled indices
    // We need a kernel to scatter data based on the permuted indices
    single_rank_scatter_kernel<<<grid_size, block_size, 0, stream>>>(
        data, thrust::raw_pointer_cast(indices.data()), count, 
        thrust::raw_pointer_cast(temp_data.data()));
    
    // Step 5: Copy shuffled data back to original array
    thrust::copy(thrust::cuda::par.on(stream), temp_data.begin(), temp_data.end(), data);
    
    cudaStreamSynchronize(stream);
}

//------------------------------------------------------------------------------
// Main function modified to assign exactly one GPU per MPI rank
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    printf("main\n");
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    printf("MPI_Init\n");
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    printf("MPI_Comm_rank: %d, MPI_Comm_size: %d\n", world_rank, world_size);
    if (argc != 3) {
        if (world_rank == 0) {
            std::cout << "Usage: " << argv[0] << " <global_n> <gpus_per_node>\n";
            std::cout << "Run with, for example:\n"
                      << "  mpirun -n 2 " << argv[0] << " 1024 2\n";
        }
        MPI_Finalize();
        return 1;
    }
    printf("argc: %d, argv[1]: %s, argv[2]: %s\n", argc, argv[1], argv[2]);
    size_t global_n = std::stoull(argv[1]);
    size_t p2_gpus_per_node = std::stoull(argv[2]);
    printf("global_n: %zu, p2_gpus_per_node: %zu\n", global_n, p2_gpus_per_node);
    // The total number of GPUs is equal to the total number of ranks
    size_t total_gpus = static_cast<size_t>(world_size);
    size_t local_count = global_n / total_gpus;
    printf("total_gpus: %zu, local_count: %zu\n", total_gpus, local_count);
    // Obtain a unique NCCL ID on rank 0, then broadcast
    ncclUniqueId nccl_id;
    if (world_rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Determine the local rank on this node (e.g. for Open MPI)
    // so that each MPI rank can select a different CUDA device.
    int local_rank = 0;
    if (const char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
        local_rank = std::atoi(env_str);
    }
    // Device ID is chosen based on local rank
    int device_id = local_rank % static_cast<int>(p2_gpus_per_node);
    cudaSetDevice(device_id);
    printf("device_id: %d\n", device_id);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Initialize an NCCL communicator for this rank
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, total_gpus, nccl_id, world_rank));
    printf("ncclCommInitRank\n");
    // Compute or guess node_rank: for multi-node, (rank / p2_gpus_per_node)
    int node_rank = world_rank / static_cast<int>(p2_gpus_per_node);
    printf("node_rank: %d\n", node_rank);
    // Prepare random data using cuRAND states
    curandState* d_states = nullptr;
    cudaMalloc(&d_states, local_count * sizeof(curandState));
    // printf("world_rank: %d, node_rank: %d, local_rank: %d, device_id: %d, local_count: %zu\n", world_rank, node_rank, local_rank, device_id, local_count);
    // Launch init kernel
    dim3 block(256);
    dim3 grid((local_count + block.x - 1) / block.x);
    init_curand_states<<<grid, block>>>(d_states, 1234 + world_rank , local_count);
    printf("init_curand_states\n");
    thrust::device_vector<int> local_data(local_count);

    // Fill local_data with random values from the cuRAND states
    // A quick way is to do thrust::generate with a device lambda
    // that uses d_states:
    {
        // Example approach: we can do a manual kernel or just re-use CurandFunctor
        CurandFunctor functor;
        functor.states = d_states;
        functor.n = static_cast<int>(local_count);
        functor.gpu_rank = static_cast<int>(world_rank);
        thrust::generate(local_data.begin(), local_data.end(), functor);
    }
    printf("local_data\n");             
    // Print first few values before shuffle
    {
        thrust::host_vector<int> h_before = local_data;
        std::cout << "[Rank " << world_rank << " GPU " << device_id
                  << "] Before shuffle: ";
        for (size_t i = 0; i < std::min<size_t>(5, local_count); ++i) {
            std::cout << h_before[i] << " ";
        }
        std::cout << std::endl;
    }

    // Perform shuffle (single rank or global)
    if (world_size == 1) {
        // Single rank case - use simplified shuffle
        std::cout << "[Rank " << world_rank << "] Using single rank shuffle" << std::endl;
        single_rank_shuffle<int>(
            thrust::raw_pointer_cast(local_data.data()),
            local_count,
            global_n,
            stream
        );
    } else {
        // Multi-rank case - use global shuffle
        std::cout << "[Rank " << world_rank << "] Using global shuffle" << std::endl;
        global_shuffle<int>(
            thrust::raw_pointer_cast(local_data.data()),
            local_count,
            global_n,
            /* p1_nodes = world_size / p2_gpus_per_node, if multi-node */
            world_size / p2_gpus_per_node,
            p2_gpus_per_node,
            node_rank,
            device_id,  // here we use device_id for the gpu_rank
            nccl_comm,
            stream
        );
    }

    // Print first few values after shuffle
    {
        thrust::host_vector<int> h_after = local_data;
        std::cout << "[Rank " << world_rank << " GPU " << device_id
                  << "] After shuffle:  ";
        for (size_t i = 0; i < std::min<size_t>(5, local_count); ++i) {
            std::cout << h_after[i] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    CHECK_NCCL(ncclCommDestroy(nccl_comm));
    cudaStreamDestroy(stream);
    cudaFree(d_states);

    MPI_Finalize();
    return 0;
}