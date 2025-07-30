import cupynumeric as np
import numpy

input = numpy.random.randint(0, 100, 10000000)
input = input.reshape(100, 100000)
a = np.array(input)
a.sort()
print(a[:5])


# input.sort()
# print(input[:5])

