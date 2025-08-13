# import cupy as np

import cupynumeric as np
from legate.timing import time


N_ROWS = 1_000_000
N_COLS = 32


def take_with_pairs(data, pairs):
    return np.take(data, pairs, axis=1)


data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32).reshape((N_ROWS, N_COLS))
print("Data generated")

print("Taking pairs")
pair_list = [[i, j] for i in range(N_COLS) for j in range(i + 1, N_COLS)]
pairs_array = np.array(pair_list)
print(len(pair_list))

start_time = time()
result_array = take_with_pairs(data_array, pairs_array)
duration = time() - start_time

print(result_array.shape)

print(duration / 1e6)