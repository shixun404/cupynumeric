#!/usr/bin/env python3

"""
Test script to demonstrate the 3 different distributed shuffle methods.
Run with: mpirun -np 2 python test_distributed_shuffle_methods.py
"""

import numpy as np
import cupynumeric as cp

def test_distributed_methods():
    """Test all 3 distributed shuffle methods."""
    print("Testing Distributed Shuffle Methods")
    print("=" * 40)
    
    # Create test data
    data_size = 20
    data = cp.arange(data_size, dtype=np.int64)
    print(f"Original data: {data}")
    print()
    
    methods = [
        ("feistel", "Feistel Forward + All2All Exchange"),
        ("feistel_bidirectional", "Feistel Bidirectional (No Buffer Communication)"), 
        ("fisher_yates", "Fisher-Yates Global Permutation (CPU)")
    ]
    
    for method, description in methods:
        print(f"Testing: {description}")
        print(f"Method: '{method}'")
        
        try:
            data_copy = data.copy()
            data_copy.shuffle(method=method)
            print(f"Result: {data_copy}")
            
            # Verify all elements preserved
            original_sorted = sorted(data.tolist())
            shuffled_sorted = sorted(data_copy.tolist())
            assert shuffled_sorted == original_sorted
            print("✓ All elements preserved")
            
            # Check if actually shuffled
            if data_copy.tolist() != data.tolist():
                print("✓ Data order changed")
            else:
                print("⚠ Data order unchanged (may happen with small arrays)")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 40)

def test_performance_comparison():
    """Compare performance characteristics of different methods."""
    print("\nPerformance Characteristics:")
    print("=" * 40)
    
    print("📊 Communication Requirements:")
    print("  ALL METHODS          → NCCL all2all for data exchange")
    print("  feistel              → + all2all for recv buffer calculation")
    print("  feistel_bidirectional → + local computation for recv buffer")
    print("  fisher_yates         → + local computation for recv buffer")
    print()
    
    print("🧮 Computational Complexity:")
    print("  feistel              → Low (single bijection)")
    print("  feistel_bidirectional → Medium (forward + inverse bijection)")
    print("  fisher_yates         → High (global permutation on CPU)")
    print()
    
    print("💾 Memory Usage:")
    print("  feistel              → Low")
    print("  feistel_bidirectional → Low")
    print("  fisher_yates         → Medium (global permutation on host)")
    print()
    
    print("🎯 Best Use Cases:")
    print("  feistel              → General-purpose distributed shuffle")
    print("  feistel_bidirectional → Save one communication round trip")
    print("  fisher_yates         → Reproducible results across ranks")

if __name__ == "__main__":
    try:
        test_distributed_methods()
        test_performance_comparison()
        
        print("\n" + "=" * 40)
        print("🎉 All distributed shuffle methods tested!")
        print("Run with more processes to see distributed behavior:")
        print("  mpirun -np 4 python test_distributed_shuffle_methods.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 