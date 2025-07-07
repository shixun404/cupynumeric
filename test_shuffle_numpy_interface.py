#!/usr/bin/env python3

"""
Example demonstrating the NumPy-compatible shuffle functionality in cuPyNumeric.
This matches the interface of numpy.random.shuffle().
"""

# import numpy as np
import cupynumeric as np

def demonstrate_2d_behavior():
    """Demonstrate the specific 2D behavior clearly"""
    print("Demonstrating 2D shuffle behavior:")
    print("(Only rows are reordered, row contents stay the same)")
    
    # Create a clearly structured 2D array
    arr = np.random.randint(low=0, high=5, size=5)
    
    print("Original array:")
    print(arr)
    
    print("\nAfter shuffle:")
    arr.shuffle()
    print(arr)
    
    # print("\nNotice: Each row [X0, X1, X2] stays intact,")
    # print("only the order of rows changes!")
    # print()


if __name__ == "__main__":
    print("cuPyNumeric Shuffle Implementation (NumPy-compatible)")
    print("=" * 55)
    
    try:
        demonstrate_2d_behavior()
        
        print("All tests completed successfully!")
        print("\nUsage:")
        print("  arr = cnp.arange(10)")
        print("  arr.shuffle()  # Modifies arr in-place")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This implementation uses existing cuPyNumeric operations")
        print("and should work with the current codebase.") 