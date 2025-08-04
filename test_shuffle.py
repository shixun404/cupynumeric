import cupynumeric as np
import numpy
d1 = 100000000
d2 = 2
input = numpy.arange(d1 * d2)
input = input.reshape(d1, d2)
a = np.array(input)

print(a)

a.shuffle()

print(a)