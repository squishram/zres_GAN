"""
current use of test.misc:
testing interpolation functions for the z-spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate

x = np.linspace(0, 20, 10)
y = np.sin(x)

xx = np.linspace(0, 20, 100)
yy = pchip_interpolate(x, y, xx)

xx = xx[::11]
yy = yy[::11]

plt.plot(x, y)
plt.plot(xx, yy)
plt.show()
