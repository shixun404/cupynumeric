#!/usr/bin/env python3

"""
Example demonstrating the NumPy-compatible shuffle functionality in cuPyNumeric.
This matches the interface of numpy.random.shuffle().
"""

# import numpy as np
import cupynumeric as np
from mpi4py import MPI
import numpy
import subprocess
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rank_1 = 0
hostname = ""
def demonstrate_2d_behavior(N=1000000,M = 10):
    """Demonstrate the specific 2D behavior clearly"""
    print("Demonstrating 2D shuffle behavior:")
    print("(Only rows are reordered, row contents stay the same)")
    
    # Create a clearly structured 2D array
    # arr = np.random.randint(low=0, high=5, size=10)
    arr = np.arange(N)
    
    print(f"[Rank {rank}/{size}]" + "Original array:")
    output = f"[Rank {rank}/{size}]"
    
    # if rank == 0:
    
    print(output, arr[rank*(N // size):(rank*(N // size) + M)])
    # perm = numpy.random.permutation(N)
    # res = arr[perm]
    # print(output, res[rank*(N // size):(rank*(N // size) + M)])
    # print(output, arr[perm])
    # if hostname in "lego-cg1-qs-139":
    #     print(output, arr[0])
    
    
    
    arr.shuffle()
    print(f"[Rank {rank}/{size}]" + "After shuffle:")
    output = f"[Rank {rank}/{size}]"
    # if rank == 0:
    print(output, arr[rank*(N // size):(rank*(N // size) + M)])
    

if __name__ == "__main__":
    print("cuPyNumeric Shuffle Implementation (NumPy-compatible)")
    print("=" * 55)
    result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
    hostname = result.stdout.strip()
    print(f"[Rank {rank}/{size}] {result.stdout.strip()}")
    try:
        demonstrate_2d_behavior()
        
        # print("All tests completed successfully!")
        # print("\nUsage:")
        # print("  arr = cnp.arange(10)")
        # print("  arr.shuffle()  # Modifies arr in-place")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This implementation uses existing cuPyNumeric operations")
        print("and should work with the current codebase.") 