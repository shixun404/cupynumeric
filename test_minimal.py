#!/usr/bin/env python3
"""
Minimal test script to verify Legate setup
"""
import sys
import time

print("=== Legate Minimal Test ===")
print(f"Python version: {sys.version}")

try:
    import legate.numpy as np
    print("✓ Legate NumPy imported successfully")
    
    # Simple array operations
    print("Testing basic array operations...")
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    c = a + b
    print(f"✓ Array addition: {a} + {b} = {c}")
    
    # Test GPU operations if available
    print("Testing matrix operations...")
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    z = np.dot(x, y)
    print(f"✓ Matrix multiplication: {x.shape} @ {y.shape} = {z.shape}")
    
    print("=== Test completed successfully! ===")
    
except ImportError as e:
    print(f"✗ Failed to import Legate NumPy: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
