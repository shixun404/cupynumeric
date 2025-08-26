import cupynumeric as np
import numpy

# input = numpy.array([0,1,2,3,4,5,6,7,8,9])
input = numpy.arange(32)
# input = input.reshape(10, 10)

a = np.array(input, dtype=np.float32)
print(a)
# b = a[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
b = a[[0,1,2,3]]
# b = a[[1,2,3,4]]
print(b)




# input = numpy.array([[1,2,3,4]])
# # input = input.reshape(2, 5)

# a = np.array(input)
# print(a)
# b = a[:, [3, 2, 1, 0]]
# print(b)



# input.sort()
# print(input[:5])

