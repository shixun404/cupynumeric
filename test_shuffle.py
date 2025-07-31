import cupynumeric as np
import numpy
d1 = 10
d2 = 2
input = numpy.arange(d1 * d2)
input = input.reshape(d1, d2)
a = np.array(input)
print(type(a[0, 0]))
print(type(a))
print(a)
a.shuffle()
print(a)


# input.sort()
# print(input[:5])

