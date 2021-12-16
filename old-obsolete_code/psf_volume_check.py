import numpy as np
import scipy.integrate as integrate

"""
This program:
- builds a 3D PSF with sig_z = 3*sig_xy
- calculates the percentage of signal within the volume
  enclosed by a distance of 1 X sigma from the mean
(which should be very high)
Thus this program is basically a volume integral that consitutes a sanity check
"""


# This is a 1D gaussian with variable x
def gauss(sig):
    f = lambda x: np.exp(-x ** 2 / (2 * sig ** 2))

    return f


# The sigmas and the intensity
sig_x = 1
sig_y = 1
sig_z = 3
intensity = 1

# intensity scaler
intensity *= (((2 * np.pi) ** 1.5) * sig_x * sig_y * sig_z) ** -1

# integral (area) calculation along each dimension for range of 1 sigma
integral_x = integrate.quad(gauss(sig_x), -sig_x, sig_x)
integral_y = integrate.quad(gauss(sig_y), -sig_y, sig_y)
integral_z = integrate.quad(gauss(sig_z), -sig_z, sig_z)

# integral along each dimension for boundless area
integral_x_inf = integrate.quad(gauss(sig_x), -np.inf, np.inf)
integral_y_inf = integrate.quad(gauss(sig_y), -np.inf, np.inf)
integral_z_inf = integrate.quad(gauss(sig_z), -np.inf, np.inf)

# bringing it together to find the volume
gaussian_siglim = intensity * integral_x[0] * integral_y[0] * integral_z[0]
gaussian_inflim = intensity * integral_x_inf[0] * integral_y_inf[0] * integral_z_inf[0]

print("The volume encloses ", gaussian_siglim, " within 1 sigma.")
print("The volume encloses ", gaussian_inflim, " when unbounded.")
print("Thus the ratio of the total volume enclosed by 1 sigma is ", gaussian_siglim / gaussian_inflim)
