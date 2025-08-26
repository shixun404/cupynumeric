# import cupynumeric as np
# from legate.timing import time


# N_ROWS = 1_000_000
# N_COLS = 64
# # N_ROWS = 1_000_000
# # N_COLS = 32


# def take_with_pairs(data, pairs):
#     return np.take(data, pairs, axis=1)


# data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32)
# # .reshape((N_ROWS, N_COLS))
# print("Data generated")

# print("Taking pairs")
# # pair_list = [(i * N_COLS + j + k) % N_ROWS for i in range(N_ROWS) for j in range(N_COLS) for  k in range(16)]
# pair_list = np.arange(N_ROWS * N_COLS, dtype=np.int64)
# pairs_array = np.array(pair_list)
# # print(len(pair_list))
# print(data_array.shape)

# start_time = time()
# result_array = data_array[pairs_array]




# # start_time = time()
# result_array_1 = data_array[pairs_array]
# result_array_2 = data_array[pairs_array]
# duration = time() - start_time
# print(duration / 1e6)

# start_time = time()
# result_array = data_array[pairs_array]
# duration = time() - start_time
# print(duration / 1e6)



###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

# import cupynumeric as np
# from legate.timing import time


# N_ROWS = 4
# N_COLS = 4


# def take_with_pairs(data, pairs):
#     return np.take(data, pairs, axis=1)


# # data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32).reshape((N_ROWS, N_COLS))
# data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32)
# # data_array = np.array([1, 2, 3, 4], dtype=np.float32)
# # data_array = np.arange(N_ROWS * N_COLS, dtype=np.float32)
# # print("Data generated")
# print(data_array)

# # print("Taking pairs")
# # pair_list = [[i, i] for i in range(2)]
# # pair_list = [[i, i] for i in range(4)]
# pair_list = []
# num_ranks = 8
# for i in range(num_ranks):
#     pair_list.append( ((i + 1) % num_ranks) * 2)
#     pair_list.append( ((i + 1) % num_ranks) * 2 + 1)
# # pair_list = np.arange(N_ROWS * N_COLS, dtype=np.int64)
# pairs_array = np.array(pair_list)
# print(pairs_array.shape)

# start_time = time()
# # result_array = take_with_pairs(data_array, pairs_array)
# result_array = data_array[pair_list]
# duration = time() - start_time

# print(result_array.shape)
# print(result_array)
# # print(duration / 1e6)


########################################################
########################################################
########################################################
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
for i in range(10):
    result_array = take_with_pairs(data_array, pairs_array)
duration = time() - start_time

print(result_array.shape)

print(duration / 1e6)
# print(result_array)