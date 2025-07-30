# CuPyNumeric Shuffle Implementation

This directory contains the CUDA implementation of shuffle operations for cuPyNumeric, based on the verified `global_shuffle.cu` code and integrated with the Legate/cuPyNumeric framework.

## Implementation Overview

### **Shuffle Methods Supported:**

1. **`"key_sort"`** - Random key generation + Thrust sort
   - Best for: Small to medium arrays
   - Uses cuPyNumeric's Thrust integration

2. **`"fisher_yates"`** - Fisher-Yates algorithm with cuRAND
   - Best for: Classic distributed shuffle
   - Uses CUDA kernels with cuRAND states

3. **`"feistel"`** - Feistel bijection for cryptographic uniformity
   - Best for: Guaranteed uniform distribution
   - Uses cryptographic Feistel networks

4. **`"feistel_bidirectional"`** - Advanced dual Feistel optimization
   - Best for: Maximum performance on large arrays
   - Combines forward and backward Feistel bijections

### **Distributed Features:**

**3 Different Distributed Permutation Methods:**

1. **Feistel Forward + All2All** (`"feistel"`)
   - Uses Feistel bijection to calculate send buffer
   - Uses NCCL all2all exchange to get receive buffer
   - Still requires all2all communication for data exchange
   - Best for: Standard distributed shuffling

2. **Feistel Bidirectional** (`"feistel_bidirectional"`)
   - Uses Feistel forward to calculate send buffer
   - Uses Feistel backward (inverse) to calculate receive buffer directly
   - **No communication needed for send/recv buffer calculation!**
   - Still requires all2all communication for actual data exchange
   - Best for: Scenarios where computing receive buffer is faster than communicating it

3. **Fisher-Yates Global Permutation** (`"fisher_yates"`)
   - Each rank uses Fisher-Yates **on CPU** with same seed
   - Generates identical global permutation on all ranks
   - Derives both send and receive buffers from global permutation
   - Still requires all2all communication for data exchange
   - Best for: Ensuring identical shuffle across all ranks

## Files Added/Modified

### **New Files:**
- `src/cupynumeric/shuffle/shuffle.h` - Header with task definitions and classes
- `src/cupynumeric/shuffle/shuffle.cu` - CUDA implementation
- `test_shuffle_integration.py` - Integration test script

### **Modified Files:**
- `src/cupynumeric/cupynumeric_c.h` - Added `CUPYNUMERIC_SHUFFLE` task ID
- `cupynumeric/config.py` - Added `SHUFFLE` opcode 
- `cupynumeric/_thunk/_shuffle.py` - Modified to call CUDA tasks
- `cupynumeric/_thunk/deferred.py` - Updated shuffle method signatures
- `cupynumeric/_array/array.py` - Updated public API
- `cupynumeric/_module/ssc_sorting.py` - Updated module API

## Build Instructions

### **1. Add to CMake Build System**
Add the following to your CMakeLists.txt:

```cmake
# Add shuffle source files
target_sources(cupynumeric PRIVATE
    src/cupynumeric/shuffle/shuffle.cu
)

# Make sure NCCL is linked for distributed shuffle
find_package(NCCL REQUIRED)
target_link_libraries(cupynumeric PRIVATE NCCL::NCCL)
```

### **2. Build cuPyNumeric**
```bash
# Configure with CUDA support
cmake -DCMAKE_BUILD_TYPE=Release -DCUPYNUMERIC_USE_CUDA=ON ..

# Build
make -j$(nproc)

# Install
make install
```

## Testing

### **Basic Test:**
```bash
python test_shuffle_integration.py
```

### **Manual Testing:**
```python
import numpy as np
import cupynumeric as cp

# Test all methods
data = cp.arange(1000, dtype=np.int64)
for method in ["key_sort", "fisher_yates", "feistel", "feistel_bidirectional"]:
    data_copy = data.copy()
    data_copy.shuffle(method=method)
    print(f"{method}: {data_copy[:5]}...")  # Show first 5 elements
    assert sorted(data_copy.tolist()) == sorted(data.tolist())
```

### **Distributed Testing:**
```bash
# Multi-GPU test (requires 2+ GPUs)
mpirun -np 2 python -c "
import cupynumeric as cp
data = cp.arange(1000, dtype=cp.int64)
data.shuffle(method='feistel')  # Uses distributed shuffle
print('Distributed shuffle successful!')
"
```

## Key Features from global_shuffle.cu

### **1. Feistel Bijection**
- 24-round Feistel network for cryptographic-quality shuffling
- Guaranteed bijective mapping (no duplicate/missing elements)
- Host/device compatible implementation

### **2. Distributed Communication**
- NCCL all2all for efficient multi-GPU data exchange
- Histogram-based load balancing
- Optimized send/receive buffer management

### **3. Performance Optimizations**
- Thrust library integration for GPU algorithms
- Memory-efficient buffer management
- Stream-based asynchronous execution
- Coalesced memory access patterns

## Data Types Supported

- **Current:** `int64` (as requested)
- **Future:** Easy to extend to other types by adding template instantiations

## Integration Points

### **Python Frontend:**
```python
# All methods available through consistent API
arr.shuffle(method="feistel")  # In-place shuffle
cp.shuffle(arr, method="fisher_yates")  # Module function
```

### **CUDA Backend:**
- Integrated with Legate task system
- Proper memory management through Legate buffers
- NCCL communicator integration for distributed operations

## Performance Characteristics

### **Local Methods:**
- **key_sort**: O(n log n), moderate memory usage
- **fisher_yates**: O(n), high memory usage (cuRAND states)  
- **feistel**: O(n), low memory usage, best uniformity
- **feistel_bidirectional**: O(n), moderate memory usage

### **Distributed Methods:**

| Method | Send/Recv Buffer Calc | Data Exchange | Computation | Memory | Best Use Case |
|--------|----------------------|---------------|-------------|---------|---------------|
| **Feistel + All2All** | All2all exchange | All2all | Low (single bijection) | Low | General distributed shuffle |
| **Feistel Bidirectional** | **Local computation** | All2all | Medium (forward + inverse) | Low | Fast send/recv calc |
| **Fisher-Yates Global** | **Local computation** | All2all | High (global permutation on CPU) | Medium | Reproducible results |

### **Trade-offs:**
- **All methods require all2all communication** for actual data exchange
- **Feistel Bidirectional** saves one all2all round trip for buffer calculation
- **Fisher-Yates Global** uses CPU computation and provides deterministic results
- **Feistel + All2All** is the most general-purpose approach

## Next Steps

1. **Add more data types** (float32, float64, etc.)
2. **CPU/OMP variants** for non-GPU systems
3. **Multi-dimensional shuffle** along different axes
4. **Performance benchmarking** against other implementations 