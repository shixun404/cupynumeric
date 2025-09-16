# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pytest

import cupynumeric as num


def mk_test_array(lib, shape):
    """Create a test array with sequential values for easy verification."""
    total_size = np.prod(shape)
    arr = lib.arange(total_size, dtype=np.int64)
    return arr.reshape(shape)


def test_shuffle_1d():
    """Test shuffle on 1D arrays."""
    for size in [4, 8, 16, 100]:
        # Create test arrays
        np_arr = mk_test_array(np, (size,))
        num_arr = mk_test_array(num, (size,))
        
        # Generate shuffle indices
        index = num.arange(size, dtype=np.int64)
        num.random.seed(42)
        index.shuffle()
        index_np = np.array(index)
        
        # Test shuffle with same seed
        num.random.seed(42) 
        num_arr_copy = num.array(num_arr)
        num_arr_copy.shuffle()
        
        # Verify shuffle matches expected permutation
        expected = np_arr[index_np]
        assert np.array_equal(num_arr_copy, expected), f"1D shuffle failed for size {size}"


def test_shuffle_2d():
    """Test shuffle on 2D arrays."""
    shapes = [(4, 2), (3, 5), (8, 4), (10, 6)]
    
    for shape in shapes:
        N1, N2 = shape
        
        # Create test arrays
        np_arr = mk_test_array(np, shape)
        num_arr = mk_test_array(num, shape)
        
        # Generate shuffle indices for first dimension
        index = num.arange(N1, dtype=np.int64)
        num.random.seed(123)
        index.shuffle()
        index_np = np.array(index)
        
        # Test shuffle with same seed
        num.random.seed(123)
        num_arr_copy = num.array(num_arr)
        num_arr_copy.shuffle()
        
        # Verify shuffle affects only first dimension
        expected = np_arr[index_np]
        assert np.array_equal(num_arr_copy, expected), f"2D shuffle failed for shape {shape}"


def test_shuffle_3d():
    """Test shuffle on 3D arrays."""
    shapes = [(3, 2, 4), (4, 3, 2), (5, 4, 3)]
    
    
    for shape in shapes:
        N1, N2, N3 = shape
        
        # Create test arrays
        np_arr = mk_test_array(np, shape)
        num_arr = mk_test_array(num, shape)
        
        # Generate shuffle indices for first dimension
        index = num.arange(N1, dtype=np.int64)
        num.random.seed(456)
        index.shuffle()
        index_np = np.array(index)
        
        # Test shuffle with same seed
        num.random.seed(456)
        num_arr_copy = num.array(num_arr)
        num_arr_copy.shuffle()
        
        # Verify shuffle affects only first dimension
        expected = np_arr[index_np]
        assert np.array_equal(num_arr_copy, expected), f"3D shuffle failed for shape {shape}"


def test_shuffle_4d():
    """Test shuffle on 4D arrays."""
    # Use smaller sizes for 4D to manage memory usage
    shapes = [(3, 2, 2, 2), (4, 2, 3, 2), (2, 3, 4, 2)]
    
    for shape in shapes:
        N1, N2, N3, N4 = shape
        
        # Create test arrays
        np_arr = mk_test_array(np, shape)
        num_arr = mk_test_array(num, shape)
        
        # Generate shuffle indices for first dimension
        index = num.arange(N1, dtype=np.int64)
        num.random.seed(789)
        index.shuffle()
        index_np = np.array(index)
        
        # Test shuffle with same seed
        num.random.seed(789)
        num_arr_copy = num.array(num_arr)
        num_arr_copy.shuffle()
        
        # Verify shuffle affects only first dimension
        expected = np_arr[index_np]
        assert np.array_equal(num_arr_copy, expected), f"4D shuffle failed for shape {shape}"


@pytest.mark.parametrize("dtype", [np.int64])
def test_shuffle_different_dtypes(dtype):
    """Test shuffle with different data types."""
    shape = (6, 4)
    
    # Create test arrays
    np_arr = mk_test_array(np, shape).astype(dtype)
    num_arr = mk_test_array(num, shape).astype(dtype)
    
    # Generate shuffle indices
    index = num.arange(shape[0], dtype=np.int64)
    num.random.seed(999)
    index.shuffle()
    index_np = np.array(index)
    
    # Test shuffle with same seed
    num.random.seed(999)
    num_arr_copy = num.array(num_arr)
    num_arr_copy.shuffle()
    
    # Verify results
    expected = np_arr[index_np]
    assert np.array_equal(num_arr_copy, expected), f"Shuffle failed for dtype {dtype}"


def test_shuffle_reproducibility():
    """Test that shuffle produces the same result with the same seed."""
    shape = (8, 3)
    num_arr = mk_test_array(num, shape)
    
    # Test reproducibility with same seed
    num.random.seed(42)
    arr1 = num.array(num_arr)
    arr1.shuffle()
    
    num.random.seed(42)
    arr2 = num.array(num_arr)
    arr2.shuffle()
    
    assert np.array_equal(arr1, arr2), "Shuffle is not reproducible with same seed"
    
    # Test different results with different seed
    num.random.seed(43)
    arr3 = num.array(num_arr)
    arr3.shuffle()
    
    assert not np.array_equal(arr1, arr3), "Shuffle with different seeds produced same result"


def test_shuffle_randomness():
    """Test that shuffle without resetting seed produces different results."""
    shape = (10, 2)
    num_arr = mk_test_array(num, shape)
    
    # Set initial seed once
    num.random.seed(100)
    
    # Multiple shuffles should produce different results
    results = []
    for i in range(5):
        arr = num.array(num_arr)
        arr.shuffle()
        results.append(np.array(arr))
    
    # Verify not all results are identical
    all_same = all(np.array_equal(results[0], result) for result in results[1:])
    assert not all_same, "Multiple shuffles produced identical results"



def test_shuffle_general_nd():
    """Test shuffle correctness for general N-dimensional arrays."""
    test_shapes = [
        (6,),                    # 1D
        (4, 5),                  # 2D
        (3, 4, 2),              # 3D
        (2, 3, 4, 2),           # 4D
        (5, 2, 3),              # 3D
    ]
    
    for shape in test_shapes:
        N1 = shape[0]
        
        # Create test arrays
        np_arr = mk_test_array(np, shape)
        num_arr = mk_test_array(num, shape)
        
        # Generate shuffle indices
        index = num.arange(N1, dtype=np.int64)
        seed = 42 + sum(shape)
        num.random.seed(seed)
        index.shuffle()
        index_np = np.array(index)
        
        # Test shuffle with same seed
        num.random.seed(seed)
        num_arr_copy = num.array(num_arr)
        num_arr_copy.shuffle()
        
        # Verify correctness
        expected = np_arr[index_np]
        assert np.array_equal(num_arr_copy, expected), \
            f"General ND shuffle failed for shape {shape}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))