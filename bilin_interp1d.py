import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

start = 0
stop = 20

# undersampled
x1 = np.linspace(start, stop, 10)
y1 = np.sin(x1)

# well-sampled
x2 = np.linspace(start, stop, 20)
y2 = np.sin(x2)

f = interpolate.interp1d(x1, y1, "cubic")
yi = f(x2)

plt.plot(x1, y1, "o--")
plt.plot(x2, y2, "-")
plt.plot(x2, yi, "-")
plt.show()
