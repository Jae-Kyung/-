import numpy as np
import matplotlib.pyplot as plt
from cda_class import cda

# random seed 설정
np.random.seed(1)

# data genreation
n = 1000
p = 1
x = np.zeros((n, p))
for j in range(p):
    x[:, j] = np.linspace(0.0, 1.0, n)
f = 2 + 3 * x[:, 0]
y = f + np.random.normal(0, 1, size = n)

# setting model with the form f = a + bx
basis = np.c_[np.ones(n), x]
fit = cda(basis, y)
fit.training()

plt.scatter(x, y, s = 3, color = 'silver')
plt.plot(x, f, 'black', linestyle = '--')
plt.plot(x, fit.fitted_values, 'r')
plt.show()
