#!/usr/bin/env python3

"""
Script to check cupynumeric max_eager_volume values and thunk selection criteria.
This helps understand when arrays use eager vs deferred execution.
"""

import cupynumeric as cnp
import numpy as np
from mpi4py import MPI
from cupynumeric.runtime import runtime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def print_system_info():
    """Display system and runtime configuration"""
    print(f"{'='*60}")
    print("CuPyNumeric System Information")
    print(f"{'='*60}")
    
    print(f"[Rank {rank}/{size}] Number of processors: {runtime.num_procs}")
    print(f"[Rank {rank}/{size}] Number of GPUs: {runtime.num_gpus}")
    print(f"[Rank {rank}/{size}] Max eager volume: {runtime.max_eager_volume}")
    
    # 显示具体的阈值值
    if rank == 0:
        print(f"\n🔧 **Eager Volume Thresholds (from C++ constants):**")
        print(f"   - GPU systems: 65,536 elements (MIN_GPU_CHUNK_DEFAULT)")
        print(f"   - OpenMP systems: 8,192 elements (MIN_OMP_CHUNK_DEFAULT)")  
        print(f"   - CPU systems: 1,024 elements (MIN_CPU_CHUNK_DEFAULT)")
        print(f"   - Test mode: 2 elements (for all systems)")
        
        print(f"\n💡 **Current system is using: {runtime.max_eager_volume} elements as threshold**")
        
        # 判断当前系统类型
        if runtime.num_gpus > 0:
            system_type = "GPU"
            expected_value = 65536
        elif runtime.num_procs > 1:
            system_type = "OpenMP"  
            expected_value = 8192
        else:
            system_type = "CPU"
            expected_value = 1024
            
        print(f"   - Detected system type: {system_type}")
        print(f"   - Expected threshold: {expected_value}")
        
        if runtime.max_eager_volume != expected_value:
            if runtime.max_eager_volume == 2:
                print(f"   - ⚠️  Running in TEST MODE (LEGATE_TEST=1)")
            else:
                print(f"   - ⚠️  Custom threshold set via environment variable")

def test_array_thunk_selection():
    """Test which arrays become eager vs deferred"""
    print(f"\n{'='*60}")
    print("Array Thunk Selection Testing")
    print(f"{'='*60}")
    
    # 测试不同大小的数组
    test_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]
    
    for size_val in test_sizes:
        try:
            arr = cnp.arange(size_val)
            is_eager = runtime.is_eager_array(arr._thunk)
            is_deferred = runtime.is_deferred_array(arr._thunk)
            
            # 判断应该是什么类型
            should_be_eager = size_val <= runtime.max_eager_volume
            
            status_icon = "✅" if (is_eager == should_be_eager) else "❌"
            thunk_type = "Eager" if is_eager else "Deferred"
            
            print(f"[Rank {rank}] {status_icon} Size {size_val:>7}: {thunk_type:>8} "
                  f"(volume <= {runtime.max_eager_volume}: {should_be_eager})")
                  
        except Exception as e:
            print(f"[Rank {rank}] ❌ Size {size_val:>7}: Error - {e}")

def demonstrate_environment_control():
    """Show how to control thunk selection via environment variables"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("Environment Variable Controls")
        print(f"{'='*60}")
        
        print("🔧 **Force Eager Mode:**")
        print("   export CUPYNUMERIC_FORCE_THUNK=eager")
        print("   legate --gpus 2 your_script.py")
        print("   (All arrays become eager, like NumPy)")
        
        print("\n🚀 **Force Deferred Mode:**")
        print("   export CUPYNUMERIC_FORCE_THUNK=deferred")
        print("   legate --gpus 2 your_script.py")
        print("   (All arrays become deferred, distributed)")
        
        print("\n📊 **Custom Thresholds:**")
        print("   export CUPYNUMERIC_MIN_GPU_CHUNK=1000000")
        print("   export CUPYNUMERIC_MIN_CPU_CHUNK=10000")
        print("   export CUPYNUMERIC_MIN_OMP_CHUNK=50000")
        print("   (Custom eager volume thresholds)")
        
        print("\n🧪 **Test Mode:**")
        print("   export LEGATE_TEST=1")
        print("   (Sets eager threshold to 2 elements)")

def get_eager_volume_programmatically():
    """Show how to get the eager volume value in your code"""
    if rank == 0:
        print(f"\n{'='*60}")
        print("Programmatic Access")
        print(f"{'='*60}")
        
        print("🐍 **How to get max_eager_volume in your Python code:**")
        print(f"   from cupynumeric.runtime import runtime")
        print(f"   max_vol = runtime.max_eager_volume")
        print(f"   print(f'Current threshold: {{max_vol}} elements')")
        print(f"   # Output: Current threshold: {runtime.max_eager_volume} elements")
        
        print(f"\n🔍 **Check if array will be eager:**")
        print(f"   size = 50000")
        print(f"   will_be_eager = size <= runtime.max_eager_volume")
        print(f"   print(f'Array of size {{size}} will be eager: {{will_be_eager}}')")
        
        # 实际演示
        size = 50000
        will_be_eager = size <= runtime.max_eager_volume
        print(f"   # Example: Array of size {size} will be eager: {will_be_eager}")

if __name__ == "__main__":
    comm.Barrier()  # 同步所有rank
    
    # 只让rank 0打印系统信息概述
    if rank == 0:
        print_system_info()
    
    comm.Barrier()
    
    # 所有rank测试数组选择
    test_array_thunk_selection()
    
    comm.Barrier()
    
    # 只让rank 0打印控制信息
    demonstrate_environment_control()
    get_eager_volume_programmatically()
    
    comm.Barrier()
    print(f"[Rank {rank}] Testing completed!") 