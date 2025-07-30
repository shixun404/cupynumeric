#!/usr/bin/env python3

"""
快速修复 UCX 超时问题的代码替换

直接替换原始代码中有问题的 reshape 操作
"""

import cupynumeric as np
from mpi4py import MPI
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def demonstrate_2d_behavior_fixed(N=100000, M=10):  # 减小默认N值
    """修复版本的2D行为演示"""
    print(f"[Rank {rank}] 开始演示 (修复版本)")
    print(f"[Rank {rank}] (只重新排列行，行内容保持不变)")
    
    # ===== 🔧 修复方案1: 强制eager模式 =====
    try:
        print(f"[Rank {rank}] 方案1: 强制eager模式")
        
        # 创建eager数组，避免分布式开销
        arr_1 = np.ndarray(shape=(N * 2,), dtype=np.int64, force_thunk="eager")
        arr_1[:] = range(N * 2)
        
        # 检查是否真的是eager
        from cupynumeric.runtime import runtime
        is_eager = runtime.is_eager_array(arr_1._thunk)
        print(f"[Rank {rank}] arr_1 是 eager模式: {is_eager}")
        
        if is_eager:
            # Eager模式的reshape应该很快
            arr = arr_1.reshape(N, 2)
            print(f"[Rank {rank}] ✅ Eager reshape 成功: {arr.shape}")
            return arr
        else:
            print(f"[Rank {rank}] ⚠️  警告: 不是eager模式，尝试其他方案")
    except Exception as e:
        print(f"[Rank {rank}] 方案1失败: {e}")
    
    # ===== 🔧 修复方案2: 直接创建2D数组 =====
    try:
        print(f"[Rank {rank}] 方案2: 直接创建2D数组")
        
        # 避免中间的1D数组，直接创建目标形状
        arr = np.empty((N, 2), dtype=np.int64)
        
        # 手动填充，分块进行
        chunk_size = min(1000, N)
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            arr[i:end_i, 0] = range(i * 2, end_i * 2, 2)      # 偶数
            arr[i:end_i, 1] = range(i * 2 + 1, end_i * 2, 2) # 奇数
        
        print(f"[Rank {rank}] ✅ 直接创建成功: {arr.shape}")
        return arr
    except Exception as e:
        print(f"[Rank {rank}] 方案2失败: {e}")
    
    # ===== 🔧 修复方案3: 使用stack/concatenate =====
    try:
        print(f"[Rank {rank}] 方案3: 使用stack方法")
        
        # 分别创建两列
        col1 = np.arange(0, N * 2, 2, dtype=np.int64)  # 偶数索引 [0, 2, 4, ...]
        col2 = np.arange(1, N * 2, 2, dtype=np.int64)  # 奇数索引 [1, 3, 5, ...]
        
        # 使用column_stack组合
        arr = np.column_stack((col1, col2))
        
        print(f"[Rank {rank}] ✅ Stack方法成功: {arr.shape}")
        return arr
    except Exception as e:
        print(f"[Rank {rank}] 方案3失败: {e}")
    
    # ===== 🔧 修复方案4: 分布式分块处理 =====
    try:
        print(f"[Rank {rank}] 方案4: 分布式分块处理")
        
        # 每个rank只处理一部分数据
        local_size = N // size
        start_idx = rank * local_size
        end_idx = min((rank + 1) * local_size, N)
        actual_local_size = end_idx - start_idx
        
        if actual_local_size > 0:
            # 创建本地的小数组
            local_arr_1 = np.arange(start_idx * 2, end_idx * 2, dtype=np.int64)
            local_arr = local_arr_1.reshape(actual_local_size, 2)
            
            print(f"[Rank {rank}] ✅ 本地分块成功: {local_arr.shape}, 范围[{start_idx}:{end_idx}]")
            return local_arr
        else:
            print(f"[Rank {rank}] 当前rank无数据分配")
            return np.empty((0, 2), dtype=np.int64)
    except Exception as e:
        print(f"[Rank {rank}] 方案4失败: {e}")
    
    # 所有方案都失败
    print(f"[Rank {rank}] ❌ 所有修复方案都失败，请检查环境配置")
    return None

def main():
    """主函数"""
    if rank == 0:
        print("CuPyNumeric Reshape UCX 超时问题快速修复")
        print("=" * 55)
    
    # 获取主机名
    try:
        result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
        hostname = result.stdout.strip()
        print(f"[Rank {rank}/{size}] 运行在 {hostname}")
    except:
        print(f"[Rank {rank}/{size}] 主机名获取失败")
    
    comm.Barrier()
    
    # 运行修复版本的演示
    try:
        # 从小数组开始测试
        for test_N in [1000, 10000, 100000]:
            if rank == 0:
                print(f"\n🧪 测试 N={test_N}")
            
            comm.Barrier()
            arr = demonstrate_2d_behavior_fixed(N=test_N)
            
            if arr is not None:
                print(f"[Rank {rank}] N={test_N} 成功! 形状: {arr.shape}")
                if test_N <= 10:
                    print(f"[Rank {rank}] 前几行: {arr[:3]}")
            else:
                print(f"[Rank {rank}] N={test_N} 失败!")
                break
            
            comm.Barrier()
        
        if rank == 0:
            print(f"\n✅ 修复完成!")
            print(f"\n💡 最佳实践:")
            print(f"1. 设置环境变量: export CUPYNUMERIC_FORCE_THUNK=eager")
            print(f"2. 使用较小的数组大小进行调试")
            print(f"3. 优化UCX设置: export UCX_TLS=tcp,self")
            print(f"4. 直接创建目标形状，避免reshape")
            
    except Exception as e:
        print(f"[Rank {rank}] 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 