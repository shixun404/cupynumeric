import cupynumeric as np
import numpy
d1 = 100000
d2 = 100
input = numpy.random.randint(0, 100, d1 * d2)
input = input.reshape(d1, d2)
a = np.array(input)
print(a[:5, :5])
a.shuffle()
print(a[:5, :5])


# input.sort()
# print(input[:5])

