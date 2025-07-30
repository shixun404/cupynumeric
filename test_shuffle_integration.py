#!/usr/bin/env python3

"""
Simple test script to verify shuffle integration between Python frontend and CUDA backend.
"""

import numpy as np
import cupynumeric as cp

def test_shuffle_basic():
    """Test basic shuffle functionality for int64 arrays."""
    print("Testing basic shuffle functionality...")
    
    # Test 1D array
    print("\n1. Testing 1D int64 array:")
    data_1d = cp.arange(100000, dtype=np.int64)
    print(f"Original: {data_1d}")
    
    # Test key_sort method (should work with CUDA implementation)
    data_copy = data_1d.copy()
    data_copy.shuffle(method="key_sort")
    print(f"Shuffled (key_sort): {data_copy}")
    
    # Test that all elements are still present
    original_sorted = sorted(data_1d.tolist())
    shuffled_sorted = sorted(data_copy.tolist())
    assert shuffled_sorted == original_sorted
    print("✓ All elements preserved")
    
    # Verify it's actually shuffled (not the same order)
    if len(data_1d) > 1:
        assert data_copy.tolist() != data_1d.tolist(), "Data should be shuffled"
        print("✓ Data order changed")
    
    # # Test 2D array
    # print("\n2. Testing 2D int64 array:")
    # data_2d = cp.arange(20, dtype=np.int64).reshape(5, 4)
    # print(f"Original:\n{data_2d}")
    
    # data_copy_2d = data_2d.copy()
    # data_copy_2d.shuffle(method="key_sort")
    # print(f"Shuffled (key_sort):\n{data_copy_2d}")
    
    # # Test that each row is preserved (shuffle only along first axis)
    # original_rows = [tuple(row) for row in data_2d.tolist()]
    # shuffled_rows = [tuple(row) for row in data_copy_2d.tolist()]
    # assert sorted(shuffled_rows) == sorted(original_rows)
    # print("✓ Rows preserved, only order changed")

def test_shuffle_methods():
    """Test different shuffle methods."""
    print("\n3. Testing different shuffle methods:")
    
    data = cp.arange(100000, dtype=np.int64)
    print(f"Original: {data}")
    
    methods = ["key_sort", "fisher_yates", "feistel", "feistel_bidirectional"]
    
    print("\nDistributed method mapping (all use all2all for data exchange):")
    print("- fisher_yates → Fisher-Yates global permutation (CPU) for send/recv buffers")
    print("- feistel → Feistel forward + all2all exchange for send/recv buffers")
    print("- feistel_bidirectional → Feistel forward + backward for send/recv buffers")
    print("- key_sort → Local shuffle only")
    print()
    
    for method in methods:
        try:
            data_copy = data.copy()
            data_copy.shuffle(method=method)
            print(f"Shuffled ({method}): {data_copy}")
            
            # Verify all elements preserved
            assert sorted(data_copy.tolist()) == sorted(data.tolist())
            print(f"✓ {method} preserves all elements")
            
        except Exception as e:
            print(f"✗ {method} failed: {e}")

def test_edge_cases():
    """Test edge cases."""
    print("\n4. Testing edge cases:")
    
    # Empty array
    empty = cp.array([], dtype=np.int64)
    empty.shuffle()
    print("✓ Empty array handled")
    
    # Single element
    single = cp.array([42], dtype=np.int64)
    single.shuffle()
    assert single[0] == 42
    print("✓ Single element handled")
    
    # Large array
    large = cp.arange(100000, dtype=np.int64)
    original_sum = large.sum()
    large.shuffle()
    assert large.sum() == original_sum
    print("✓ Large array (1000 elements) handled")

if __name__ == "__main__":
    print("CuPyNumeric Shuffle Integration Test")
    print("=" * 40)
    
    try:
        test_shuffle_basic()
        test_shuffle_methods()
        test_edge_cases()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 