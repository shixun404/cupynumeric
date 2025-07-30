#!/usr/bin/env python3

"""
解决 cupynumeric reshape 导致 UCX 超时的问题

问题：arr = np.reshape(arr_1, (N, 2)) 导致 UCX 通信超时
原因：大数组的 reshape 操作在分布式环境中触发数据重分布
"""

import cupynumeric as np
from mpi4py import MPI
import os
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def solution_1_reduce_array_size():
    """
    解决方案1: 减少数组大小，逐步调试
    """
    print(f"[Rank {rank}] === 解决方案1: 减少数组大小 ===")
    
    # 从小数组开始测试
    test_sizes = [1000, 10000, 100000, 1000000]
    
    for N in test_sizes:
        try:
            print(f"[Rank {rank}] 测试 N={N}")
            start_time = time.time()
            
            arr_1 = np.arange(N * 2)
            print(f"[Rank {rank}] 创建 arr_1 完成: {time.time() - start_time:.2f}s")
            
            # 添加同步点
            comm.Barrier()
            
            reshape_start = time.time()
            arr = arr_1.reshape(N, 2)
            reshape_time = time.time() - reshape_start
            
            print(f"[Rank {rank}] N={N} reshape成功: {reshape_time:.2f}s, shape={arr.shape}")
            
            # 验证数据正确性
            if N <= 10:
                print(f"[Rank {rank}] 前几个元素: {arr[:min(3, N)]}")
            
            comm.Barrier()
            
        except Exception as e:
            print(f"[Rank {rank}] N={N} 失败: {e}")
            break

def solution_2_force_eager_mode():
    """
    解决方案2: 强制使用eager模式，避免分布式操作
    """
    print(f"[Rank {rank}] === 解决方案2: 强制Eager模式 ===")
    
    try:
        from cupynumeric.runtime import runtime
        print(f"[Rank {rank}] 当前eager阈值: {runtime.max_eager_volume}")
        
        N = 100000  # 使用较小的N进行测试
        
        # 强制创建eager数组
        arr_1 = np.ndarray(shape=(N * 2,), dtype=np.int64, force_thunk="eager")
        arr_1[:] = np.arange(N * 2)
        
        print(f"[Rank {rank}] 创建eager数组完成")
        
        # 检查是否是eager
        is_eager = runtime.is_eager_array(arr_1._thunk)
        print(f"[Rank {rank}] arr_1 是eager: {is_eager}")
        
        if is_eager:
            # eager数组的reshape应该更快
            arr = arr_1.reshape(N, 2)
            print(f"[Rank {rank}] Eager reshape成功: {arr.shape}")
        else:
            print(f"[Rank {rank}] 警告: 数组不是eager模式")
            
    except Exception as e:
        print(f"[Rank {rank}] Eager模式失败: {e}")

def solution_3_manual_reshape():
    """
    解决方案3: 手动重塑，避免自动数据重分布
    """
    print(f"[Rank {rank}] === 解决方案3: 手动重塑 ===")
    
    try:
        N = 100000
        
        # 直接创建目标形状的数组
        arr = np.empty((N, 2), dtype=np.int64)
        
        # 手动填充数据，避免reshape操作
        flat_data = np.arange(N * 2)
        
        # 分块赋值，减少内存压力
        chunk_size = min(10000, N)
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            start_idx = i * 2
            end_idx = end_i * 2
            
            # 将一维数据重新组织为二维
            chunk_data = flat_data[start_idx:end_idx].reshape(-1, 2)
            arr[i:end_i] = chunk_data
        
        print(f"[Rank {rank}] 手动重塑成功: {arr.shape}")
        print(f"[Rank {rank}] 前几行: {arr[:3]}")
        
    except Exception as e:
        print(f"[Rank {rank}] 手动重塑失败: {e}")

def solution_4_alternative_creation():
    """
    解决方案4: 直接创建目标形状，避免reshape
    """
    print(f"[Rank {rank}] === 解决方案4: 直接创建目标形状 ===")
    
    try:
        N = 100000
        
        # 方法1: 直接创建2D数组
        arr_2d = np.arange(N * 2).reshape(N, 2)  # 这个可能仍有问题
        
        # 方法2: 使用stack或concatenate
        col1 = np.arange(0, N * 2, 2)  # 偶数索引
        col2 = np.arange(1, N * 2, 2)  # 奇数索引
        
        # 使用column_stack避免reshape
        arr_stack = np.column_stack((col1, col2))
        
        print(f"[Rank {rank}] 替代方法成功: {arr_stack.shape}")
        print(f"[Rank {rank}] 前几行: {arr_stack[:3]}")
        
    except Exception as e:
        print(f"[Rank {rank}] 替代方法失败: {e}")

def solution_5_environment_tuning():
    """
    解决方案5: 环境变量调优
    """
    print(f"[Rank {rank}] === 解决方案5: 环境变量调优 ===")
    
    # 显示当前UCX相关的环境变量
    ucx_vars = [
        'UCX_TLS', 'UCX_NET_DEVICES', 'UCX_IB_TIMEOUT', 
        'UCX_TCP_KEEPALIVE', 'UCX_RNDV_THRESH', 'UCX_MEMTYPE_CACHE'
    ]
    
    print(f"[Rank {rank}] 当前UCX环境变量:")
    for var in ucx_vars:
        value = os.environ.get(var, "未设置")
        print(f"[Rank {rank}]   {var}={value}")
    
    print(f"[Rank {rank}] 建议的UCX优化设置:")
    print(f"[Rank {rank}]   export UCX_TLS=tcp,self")
    print(f"[Rank {rank}]   export UCX_TCP_KEEPALIVE=1") 
    print(f"[Rank {rank}]   export UCX_IB_TIMEOUT=20s")
    print(f"[Rank {rank}]   export UCX_RNDV_THRESH=1048576")

def solution_6_chunked_processing():
    """
    解决方案6: 分块处理大数组
    """
    print(f"[Rank {rank}] === 解决方案6: 分块处理 ===")
    
    try:
        N = 1000000  # 原始大小
        chunk_size = N // (size * 4)  # 每个rank处理更小的块
        
        print(f"[Rank {rank}] 总大小: {N}, 块大小: {chunk_size}")
        
        # 每个rank处理自己的数据块
        start_idx = rank * chunk_size
        end_idx = min((rank + 1) * chunk_size, N)
        local_size = end_idx - start_idx
        
        if local_size > 0:
            # 创建本地数据
            local_arr_1 = np.arange(start_idx * 2, end_idx * 2)
            local_arr = local_arr_1.reshape(local_size, 2)
            
            print(f"[Rank {rank}] 本地处理成功: 范围[{start_idx}:{end_idx}], shape={local_arr.shape}")
            print(f"[Rank {rank}] 本地数据样本: {local_arr[:2]}")
        else:
            print(f"[Rank {rank}] 无数据分配")
        
        comm.Barrier()
        
    except Exception as e:
        print(f"[Rank {rank}] 分块处理失败: {e}")

def main():
    """主函数，按顺序尝试各种解决方案"""
    
    if rank == 0:
        print("CuPyNumeric UCX 超时问题解决方案")
        print("=" * 50)
        print("问题: arr = np.reshape(arr_1, (N, 2)) 导致 UCX 超时")
        print("=" * 50)
    
    comm.Barrier()
    
    # 逐个尝试解决方案
    try:
        solution_1_reduce_array_size()
        comm.Barrier()
        
        solution_2_force_eager_mode() 
        comm.Barrier()
        
        # solution_3_manual_reshape()
        # comm.Barrier()
        
        solution_4_alternative_creation()
        comm.Barrier()
        
        solution_5_environment_tuning()
        comm.Barrier()
        
        solution_6_chunked_processing()
        comm.Barrier()
        
    except KeyboardInterrupt:
        print(f"[Rank {rank}] 用户中断")
    except Exception as e:
        print(f"[Rank {rank}] 程序错误: {e}")
    
    # if rank == 0:
    print("\n" + "=" * 50)
    print("✅ 建议的最佳实践:")
    print("1. 使用较小的数组大小进行调试")
    print("2. 设置 CUPYNUMERIC_FORCE_THUNK=eager")
    print("3. 使用分块处理替代大数组reshape")
    print("4. 优化UCX环境变量")
    print("5. 直接创建目标形状而非reshape")

if __name__ == "__main__":
    main() 