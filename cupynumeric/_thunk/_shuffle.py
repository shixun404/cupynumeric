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

from typing import TYPE_CHECKING, cast, Literal

import numpy as np

from legate.core import get_legate_runtime, types as ty

from ..runtime import runtime
from .._utils.array import to_core_type
from ..random._random import uniform
from ..config import CuPyNumericOpCode

if TYPE_CHECKING:
    from .._thunk.deferred import DeferredArray

# Type alias for shuffle methods
ShuffleMethod = Literal["key_sort", "fisher_yates", "feistel", "feistel_bidirectional"]


def shuffle_deferred(
    array: "DeferredArray", 
    method: ShuffleMethod = "key_sort"
) -> None:
    """
    Shuffle the array along the first axis using various distributed algorithms.
    
    For multi-dimensional arrays, only the order along the first axis changes.
    The contents of sub-arrays remain unchanged.
    
    Parameters
    ----------
    array : DeferredArray
        The array to shuffle along the first axis
    method : {"key_sort", "fisher_yates", "feistel", "feistel_bidirectional"}
        The shuffle algorithm to use:
        
        - "key_sort": Random key generation + sort (current implementation)
          Best for: Small to medium arrays, uses cuPyNumeric operations
          
        - "fisher_yates": Fisher-Yates algorithm + NCCL all2all
          Best for: Classic distributed shuffle, stable performance
          
        - "feistel": Feistel bijection + all2all communication  
          Best for: Cryptographically uniform distribution
          
        - "feistel_bidirectional": Advanced Feistel forward/backward
          Best for: Maximum performance on large distributed arrays
    """
    
    if array.size == 0:
        return
    
    # Get the size of the first axis for generating permutation
    first_axis_size = array.shape[0]
    
    if first_axis_size <= 1:
        return  # Nothing to shuffle if first axis has 0 or 1 elements

    # Call our new CUDA shuffle task
    _shuffle_task(array, method)


def _shuffle_task(array: "DeferredArray", method: str) -> None:
    """
    Call the CUDA shuffle task implementation.
    """
    legate_runtime = get_legate_runtime()
    task = legate_runtime.create_auto_task(
        array.library, CuPyNumericOpCode.SHUFFLE
    )

    # Add input/output (in-place shuffle)
    task.add_output(array.base)
    task.add_input(array.base)
    task.add_alignment(array.base, array.base)

    # Add communicators if needed for distributed shuffle
    if runtime.num_gpus > 1:
        task.add_nccl_communicator()
    elif runtime.num_gpus == 0 and runtime.num_procs > 1:
        task.add_cpu_communicator()

    # Convert method string to enum value
    method_map = {
        "key_sort": 0,
        "fisher_yates": 1, 
        "feistel": 2,
        "feistel_bidirectional": 3
    }
    method_value = method_map.get(method, 0)

    # Add scalar arguments
    task.add_scalar_arg(method_value, ty.int32)  # shuffle method
    task.add_scalar_arg(array.size, ty.int64)   # total volume
    task.add_scalar_arg(array.shape[0], ty.int64)  # first axis size
    task.add_scalar_arg(runtime.num_procs > 1, ty.bool_)  # is_index_space
    
    task.execute()


# All shuffle method implementations are now handled by the CUDA task above


# All methods now implemented in the CUDA task