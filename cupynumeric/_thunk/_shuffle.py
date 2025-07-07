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

if TYPE_CHECKING:
    from .._thunk.deferred import DeferredArray


def shuffle_deferred(
    array: DeferredArray,
) -> None:
    """
    Modify a sequence in-place by shuffling its contents.
    
    This function only shuffles the array along the first axis of a 
    multi-dimensional array. The order of sub-arrays is changed but 
    their contents remains the same.
    
    Parameters
    ----------
    array : DeferredArray
        The array to shuffle in-place
    """
    if array.size == 0:
        # Empty array, nothing to shuffle
        return
        
    if array.ndim == 0:
        # Scalar, nothing to shuffle
        return
    
    # Get the size of the first axis (the one we're shuffling)
    first_axis_size = array.shape[0]
    
    if first_axis_size <= 1:
        # Only one element along first axis, nothing to shuffle
        return
    
    # Generate random permutation indices for the first axis
    # Use Fisher-Yates shuffle approach: generate random keys and sort
    random_keys = cast(
        "DeferredArray",
        runtime.create_empty_thunk(
            (first_axis_size,), dtype=np.float64, inputs=[array]
        ),
    )
    
    # Fill with random values
    random_keys.random_uniform()
    
    # Use argsort to get the permutation indices
    permutation_indices = cast(
        "DeferredArray",
        runtime.create_empty_thunk(
            (first_axis_size,), dtype=np.int64, inputs=[random_keys]
        ),
    )
    
    # Import sort_deferred to use for generating permutation
    from ._sort import sort_deferred
    sort_deferred(permutation_indices, random_keys, argsort=True, axis=0, stable=False)
    
    # Apply the permutation to the array along axis 0 using advanced indexing
    # Create indexing tuple: permutation_indices for axis 0, slice(None) for others
    if array.ndim == 1:
        # 1D array: direct indexing
        shuffled_result = array.get_item(permutation_indices)
    else:
        # Multi-dimensional array: index only the first axis
        # This is equivalent to array[permutation_indices, :, :, ...]
        shuffled_result = array.get_item(permutation_indices)
    
    # Copy the shuffled result back to the original array (in-place modification)
    array.copy(shuffled_result, deep=True) 