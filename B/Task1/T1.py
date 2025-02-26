# This is the .py file for Task 1 of Section 7B

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol

sobol_gen = Sobol(d=2, scramble=False)
sobol_points = sobol_gen.random_base2(m=6)

plt.figure(figsize=(8, 8))
plt.scatter(sobol_points[:50, 0], sobol_points[:50, 1], c='blue', marker='o')
plt.title('First 50 Elements of a 2D Sobol Sequence')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()
plt.savefig('Task1.png')
