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
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..runtime import runtime
from .._utils.array import to_core_type
from ..random._random import uniform

if TYPE_CHECKING:
    from .._thunk.deferred import DeferredArray


def _debug_print_distributed_data(array: "DeferredArray", label: str, max_elements: int = 5) -> None:
    """
    Debug function to print the first few elements from each GPU/node
    to understand data distribution.
    """
    try:
        # Convert to numpy to see actual values
        # This will gather data from all nodes for debugging
        if array.size > 0:
            # Get a small slice for debugging to avoid expensive operations
            debug_size = min(max_elements, array.size)
            if array.ndim == 1:
                debug_slice = array[:debug_size]
            else:
                # For multi-dimensional arrays, take first few elements along first axis
                debug_slice = array[:min(max_elements, array.shape[0])]
            
            # Convert to numpy array to see values
            debug_values = np.array(debug_slice)
            print(f"[SHUFFLE DEBUG] {label}: shape={array.shape}, "
                  f"size={array.size}, first_{debug_size}_elements={debug_values}")
        else:
            print(f"[SHUFFLE DEBUG] {label}: empty array")
    except Exception as e:
        print(f"[SHUFFLE DEBUG] {label}: Error accessing data - {e}")


def _debug_print_local_info() -> None:
    """Print information about the current process/GPU for debugging."""
    try:
        num_procs = runtime.num_procs
        num_gpus = runtime.num_gpus
        print(f"[SHUFFLE DEBUG] Runtime info: num_procs={num_procs}, num_gpus={num_gpus}")
    except Exception as e:
        print(f"[SHUFFLE DEBUG] Runtime info error: {e}")


def shuffle_deferred(array: "DeferredArray") -> None:
    """
    Shuffle the array along the first axis using a distributed algorithm.
    
    For multi-dimensional arrays, only the order along the first axis changes.
    The contents of sub-arrays remain unchanged.
    """
    print("Starting shuffle_deferred function")
    
    if array.size == 0:
        print("Array is empty, returning early")
        return
    
    # Get the size of the first axis for generating permutation
    first_axis_size = array.shape[0]
    print(f"First axis size: {first_axis_size}")
    
    if first_axis_size <= 1:
        print("First axis size <= 1, no shuffling needed")
        return  # Nothing to shuffle if first axis has 0 or 1 elements
    
    # Debug: Show array distribution information
    from cupynumeric.runtime import runtime
    print(f"Array is eager: {runtime.is_eager_array(array)}")
    print(f"Array is deferred: {runtime.is_deferred_array(array)}")
    print(f"Max eager volume: {runtime.max_eager_volume}")
    
    # For 1D arrays, use direct indexing approach
    if array.ndim == 1:
        print("Using 1D direct indexing approach")
        
        # Step 1: Generate random keys for sorting
        print("Generating random keys...")
        random_keys = cast(
            "DeferredArray",
            runtime.create_empty_thunk(
                (first_axis_size,), dtype=to_core_type(np.float64), inputs=[array]
            ),
        )
        
        # Fill with random values - correct approach: generate with shape and copy
        random_values = uniform(0.0, 1.0, size=(first_axis_size,), dtype=np.float64)
        random_keys.copy(random_values._thunk, deep=False)
        print("Random keys generated")
        
        # Step 2: Get permutation indices by sorting the random keys
        print("Getting permutation indices...")
        permutation_indices = cast(
            "DeferredArray",
            runtime.create_empty_thunk(
                (first_axis_size,), dtype=to_core_type(np.int64), inputs=[random_keys]
            ),
        )
        permutation_indices.sort(random_keys, argsort=True, axis=-1)
        print("Permutation indices obtained")
        
        # Step 3: Apply the permutation using advanced indexing
        print("Applying permutation using advanced indexing...")
        shuffled_result = array.get_item(permutation_indices)
        print("Permutation applied")
        
        # Step 4: Copy result back to original array  
        print("Copying result back to original array...")
        array.copy(shuffled_result, deep=False)
        print("Copy completed")
        
    else:
        print(f"Using multi-dimensional approach for {array.ndim}D array")
        
        # For multi-dimensional arrays, we need to shuffle indices along the first axis
        # while keeping the shape of sub-arrays intact
        
        # Generate random keys for the first axis
        print("Generating random keys for first axis...")
        random_keys = cast(
            "DeferredArray", 
            runtime.create_empty_thunk(
                (first_axis_size,), dtype=to_core_type(np.float64), inputs=[array]
            ),
        )
        
        # Fill with random values - correct approach: generate with shape and copy
        random_values = uniform(0.0, 1.0, size=(first_axis_size,), dtype=np.float64)
        random_keys.copy(random_values._thunk, deep=False)
        print("Random keys generated")
        
        # Get permutation indices
        print("Getting permutation indices...")
        permutation_indices = cast(
            "DeferredArray",
            runtime.create_empty_thunk(
                (first_axis_size,), dtype=to_core_type(np.int64), inputs=[random_keys]
            ),
        )
        permutation_indices.sort(random_keys, argsort=True, axis=-1)
        print("Permutation indices obtained")
        
        # Apply permutation along first axis using advanced indexing
        print("Applying permutation along first axis...")
        shuffled_result = array.get_item(permutation_indices)
        print("Permutation applied")
        
        # Copy result back
        print("Copying result back to original array...")
        array.copy(shuffled_result, deep=False)
        print("Copy completed")
    
    print("shuffle_deferred function completed") 