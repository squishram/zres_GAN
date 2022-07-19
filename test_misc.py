"""
current use of test.misc:
testing interpolation functions for the z-spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate


n1_samples = 10
n2_samples = 20
n3_samples = 100

downsample_2d1 = int((n2_samples / n1_samples))
downsample_3d2 = int((n3_samples / n2_samples))
downsample_3d1 = int((n3_samples / n1_samples))

x1 = np.linspace(0, 20, n1_samples)
y1 = np.sin(x1)

x2 = np.linspace(0, 20, n2_samples)
y2 = np.sin(x2)

y1i2 = pchip_interpolate(x1, y1, x2)

x3 = np.linspace(0, 20, n3_samples)
y1i3 = pchip_interpolate(x1, y1, x3)

y3d2 = y1i3[::downsample_3d2]

# plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y1i3)
plt.plot(x2, y3d2)
plt.legend(
    [
        # f"sin, {n1_samples} samples",
        f"sin, {n2_samples} samples",
        f"sin, cubic monotonic interpolation from {n1_samples} to {n3_samples} samples",
        f"sin, downsampled from {n3_samples} to {n2_samples} samples",
    ]
)

# plt.savefig("sin_waves.pdf", format="pdf")

plt.show()
