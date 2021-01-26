import numpy as np
import operator
n, p = 1, 0.1
s = np.random.binomial(n, p, 1)
print(s[0])
values = [20, 50, 10, 25]
max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
print(max_index, max_value)