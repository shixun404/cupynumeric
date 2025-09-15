import cupynumeric as np
import numpy
d1 = 4
d2 = 2
d3 = 1
input = numpy.arange(d1 * d2 * d3)
input = input.reshape(d1, d2)
# input = input.reshape(d1, d2, d3)
for i in range(5):
    a = np.array(input)
    a.shuffle()
    print(a)