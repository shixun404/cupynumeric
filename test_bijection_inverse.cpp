#include <thrust/random.h>
#include <curand_kernel.h>


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



template <class URBG>
__host__ __device__ feistel_bijection::feistel_bijection(std::uint64_t m, URBG&& g) {
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

__host__ __device__ std::uint64_t feistel_bijection::nearest_power_of_two() const {
  return 1ull << (left_side_bits + right_side_bits);
}

__host__ __device__ std::uint64_t feistel_bijection::operator()(const std::uint64_t val) const {
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

__host__ __device__ std::uint64_t feistel_bijection::inverse(const std::uint64_t val) const {
  std::uint32_t state[2] = {static_cast<std::uint32_t>(val >> right_side_bits),
                            static_cast<std::uint32_t>(val & right_side_mask)};
  for (std::uint32_t i = num_rounds - 1; i >= 0; i--) {
    std::uint32_t hi, lo;
    constexpr std::uint64_t M_inv = UINT64_C(0x6C6D4A4B2DB9919B);
    constexpr std::uint64_t M0 = UINT64_C(0xD2B74407B1CE6E93);
    mullo(M_inv, state[1], lo);
    mulhi(M0, lo, hi);
    
    state[1] = ((hi ^ key[i]) ^ state[0]) & left_side_mask;
    state[0] = lo;
    
  }
  return (static_cast<std::uint64_t>(state[0]) << right_side_bits) | static_cast<std::uint64_t>(state[1]);
}


__host__ __device__ void feistel_bijection::mulhilo(std::uint64_t a, std::uint64_t b, std::uint32_t& hi, std::uint32_t& lo) {
  std::uint64_t product = a * b;
  hi = static_cast<std::uint32_t>(product >> 32);
  lo = static_cast<std::uint32_t>(product);
}

__host__ __device__ void feistel_bijection::mullo(std::uint64_t a, std::uint64_t b, std::uint32_t& lo) {
  std::uint64_t product = a * b;
  lo = static_cast<std::uint32_t>(product);
}

__host__ __device__ void feistel_bijection::mulhi(std::uint64_t a, std::uint64_t b, std::uint32_t& hi) {
  std::uint64_t product = a * b;
  hi = static_cast<std::uint32_t>(product >> 32);
}

__host__ __device__ std::uint64_t feistel_bijection::get_cipher_bits(std::uint64_t m) {
  if (m <= 16) return 4;
  std::uint64_t i = 0;
  m--;
  while (m != 0) { i++; m >>= 1; }
  return i;
}

template <class IndexType>
template <class URBG>
__host__ __device__ random_bijection<IndexType>::random_bijection(IndexType n, URBG&& g) 
  : bijection(n, g), n(n) {}

template <class IndexType>
__host__ __device__ IndexType random_bijection<IndexType>::operator()(IndexType i) const {
  auto upcast_i = static_cast<std::uint64_t>(i);
  auto upcast_n = static_cast<std::uint64_t>(n);
  
  if (upcast_i >= upcast_n) return upcast_i;
  
  do {
    upcast_i = bijection(upcast_i);
  } while (upcast_i >= upcast_n);
  return static_cast<IndexType>(upcast_i);
}

template <class IndexType>
__host__ __device__ IndexType random_bijection<IndexType>::inverse(IndexType i) const {
  auto upcast_i = static_cast<std::uint64_t>(i);
  auto upcast_n = static_cast<std::uint64_t>(n);
  
  if (upcast_i >= upcast_n) return upcast_i;
  
  do {
    upcast_i = bijection.inverse(upcast_i);
  } while (upcast_i >= upcast_n);
  return static_cast<IndexType>(upcast_i);
}

template <class IndexType>
__host__ __device__ IndexType random_bijection<IndexType>::size() const { 
  return n; 
}


int main() {
    thrust::default_random_engine rng(42);
    std::uint64_t n = 100;
    random_bijection<std::uint64_t> bijection(n, rng);
    for(std::uint64_t i = 0; i < n; i++) {
        std::uint64_t forward = bijection(i);
        std::uint64_t inverse = bijection.inverse(forward);
        if(i != inverse) {
            std::cout << "i: " << i << " forward: " << forward << " inverse: " << inverse << std::endl;
            break;
        }
    }
    return 0;
}
