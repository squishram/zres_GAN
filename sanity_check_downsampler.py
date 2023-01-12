from scipy.ndimage import gaussian_filter, zoom
import tifffile
import numpy as np


# image paths
hires_img = "/mnt/linux_data/code/python/zres_GAN/images/sims/microtubules/hires_test/sim_mtubs_1_hires"
lores_img = "/mnt/linux_data/code/python/zres_GAN/images/sims/microtubules/lores_test/sim_mtubs_1_lores"

# import images
lores_img = tifffile.imread(lores_img)
hires_img = tifffile.imread(hires_img)

# z-resolution (isotropic)
zres_hi = 24
# z-resolution (anisotropic)
zres_lo = 5 * zres_hi
# pixel size
size_pix_nm = 10
# downsample factor
downsample_factor = 3

# sigma for real (lores) and isomorphic (hires) data
sig_lores = zres_lo / size_pix_nm
sig_hires = zres_hi / size_pix_nm
# sig_extra is derived from the formula of convolving 2 gaussians, where it is defined as:
# gaussian(sig_hires) (*) gaussian(sig_extra) = gaussian(sig_lores)
sig_extra = [np.sqrt(sig_lores**2 - sig_hires**2), 0, 0]

# Create the 1D Gaussian kernel
# the size of the kernel - choose 6 * sigma to ensure that the "whole gaussian" is captured
# (right until it drops roughly to 0)
size = int(6 * sig_extra[0] + 1)
# plot the x range
x = np.linspace(-size // 2 + 1, size // 2 + 1, size)
# gaussian kernel
kernel = np.exp(-(x**2) / (2 * sig_extra[0]**2))
kernel = kernel / kernel.sum()

# Perform the convolution along the z-axis
image_blurred = gaussian_filter(hires_img, sig_extra)
# Downsample the image along the z-axis
output = zoom(image_blurred, [1 / downsample_factor, 1, 1])
# take the difference of the images
diffimg = np.sqrt((output - lores_img)**2)

print(f"sig_lores = {sig_lores}")
print(f"sig_hires = {sig_hires}")
print(f"sig_extra = {sig_extra}")

print(f"sig_lores = {sig_lores}")
print(f"sig_hires = {sig_hires}")
print(f"sig_extra = {sig_extra}")

tifffile.imwrite("input.tif", hires_img)
tifffile.imwrite("test.tif", lores_img)
tifffile.imwrite("output.tif", output)
tifffile.imwrite("difference.tif", diffimg)
