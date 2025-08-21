# # import cupy as np

# import cupynumeric as np
# from legate.timing import time


# N_ROWS = 1_000_000
# N_COLS = 32


# def take_with_pairs(data, pairs):
#     return np.take(data, pairs, axis=1)


# data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32)
# # .reshape((N_ROWS, N_COLS))
# print("Data generated")

# print("Taking pairs")
# pair_list = [(i * N_COLS + j + k) % N_ROWS for i in range(N_ROWS) for j in range(N_COLS) for  k in range(16)]
# pairs_array = np.array(pair_list)
# print(len(pair_list))

# start_time = time()
# # result_array = take_with_pairs(data_array, pairs_array)
# result_array = data_array[pairs_array]
# duration = time() - start_time

# print(result_array.shape)

# print(duration / 1e6)


# import cupy as np

import cupynumeric as np
from legate.timing import time


N_ROWS = 4
N_COLS = 4


def take_with_pairs(data, pairs):
    return np.take(data, pairs, axis=1)


data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32).reshape((N_ROWS, N_COLS))
# data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32)
# print("Data generated")
print(data_array)

# print("Taking pairs")
pair_list = [[i, i] for i in range(2)]
pairs_array = np.array(pair_list)
print(pairs_array.shape)

start_time = time()
# result_array = take_with_pairs(data_array, pairs_array)
result_array = data_array[[[0, 1], [2, 3]]]
duration = time() - start_time

print(result_array.shape)
print(result_array)
# print(duration / 1e6)