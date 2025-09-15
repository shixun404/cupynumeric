import cupynumeric as np
import numpy
d1 = 3
d2 = 4
d3 = 5
input = numpy.arange(d1 * d2 * d3)
input = input.reshape(d1, d2, d3)
a = np.array(input)

print(a)

a.shuffle()

print(a)