from pathlib import Path
import math
from typing import List, Union
import matplotlib.pyplot as plt
import torch
import numpy as np
from tifffile import imsave
from astropy.nddata import CCDData


Number = Union[int, float]


def fwhm_to_sigma(fwhm: float):
    """
    Convert FWHM to standard deviation
    """

    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def load_cube(zres: Number, prefix: str = "") -> torch.Tensor:
    """
    Function for loading some data, assumes res is in the filename
    """

    data = []
    print('{}zres-{}-slice-*.fits'.format(prefix, int(zres)))

    for i in sorted(Path('tmp/').glob('{}zres-{}-slice-*.fits'.format(prefix, int(zres)))):
        print(i)
        # Not sure what adu unit it. Arbitrary Data Unit? Seems to
        # correspond to the numbers in the file.
        data.append(np.asarray(CCDData.read(i, unit='adu')))

    # data index is [y, x], i.e. row, col, so add a z dimension and stack
    # Expand and contatenate to make [z, y, x]
    return torch.tensor(np.concatenate([np.expand_dims(n, 0) for n in data], 0), dtype=torch.float)


def cosine_window(N: int, coefficients: List[float]):
    """
    General cosine window: a sum of increasing cosines with alternating signs.
    N+1 is the number of samples in the range [0 to 2pi],
    so N represents a discrete approximation of [0, 2pi)
    """

    x = torch.tensor(range(0, N)) * 2 * math.pi / N

    result = torch.zeros(N)

    for i, c in enumerate(coefficients):
        result += c * torch.cos(x * i) * ((-1)**i)

    return result


def blackman_harris_window(N: int):
    """
    Create a Blackman-Harris cosine window,
    https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window
    """

    return cosine_window(N, [0.35875, 0.48829, 0.14128, 0.01168])


def complex_abs_sq(data: torch.Tensor):
    """
    Compute squared magnitude, assuming the last dimension is
    [re, im], and therefore of size 2
    """

    # assert data.size(-1) == 2
    # "Last dimension size must be 2, representing [re, im]"
    # return torch.sqrt(torch.sum(data**2, data.ndim-1))
    return torch.sqrt(torch.sum(data**2))


def gaussian_kernel(sigma: float, sigmas: float = 3.0) -> torch.Tensor:
    """
    Make a normalized 1D Gaussian Kernel
    """

    radius = math.ceil(sigma * sigmas)
    xs = np.array(range(-radius, radius+1))
    kernel = np.exp(-xs**2 / (2 * sigma**2))
    return torch.tensor(kernel / sum(kernel), dtype=torch.float)


def conv_1D_z_axis(data: torch.Tensor, kernel: torch.Tensor, pad: bool = False) -> torch.Tensor:
    """
    Assuming data is a cube [z,y,x],
    then perform a 1D convolution along the z axis
    """

    assert len(kernel.size()) == 1, "Kernel is not 1D"
    assert len(data.size()) == 3, "Data is not a cuboid"
    assert len(kernel) % 2 == 1, "Kernel must be odd sized (symmetric and centred on the middle element"

    # Data is [z y x]. We need [batch, channels, z, y, x] = [1,1,z,y,z]
    d = data.unsqueeze(0).unsqueeze(0)

    # Kernel is [z]. We need [channels out, channels in, z, y, x] = [1,1,z,1,1]
    kz = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4)
    radius = int(len(kernel-1)/2)

    if pad == "zero":
        padding = (radius, 0, 0)
    elif pad == "edge":
        padding = (0, 0, 0)
        bottom = data[0, :, :].expand(radius, data.size(1), data.size(2))
        top = data[-1, :, :].expand(radius, data.size(1), data.size(2))
        d = torch.cat([bottom, data, top], dim=0).unsqueeze(0).unsqueeze(0)

    else:
        assert False, "bad pad " + pad

    result = torch.nn.functional.conv3d(d, kz, bias=None, padding=padding)

    return result.squeeze(0).squeeze(0)


def xyz_projections(data: torch.Tensor) -> torch.Tensor:
    """Project the cube on to the 3 1D axes. 0th index goes as x, y, z"""
    assert data.size(0) == data.size(1) and data.size(0) == data.size(2), "Data is not a cube"
    x_projection = data.sum(1).sum(0)
    y_projection = data.sum(2).sum(0)
    z_projection = data.sum(2).sum(1)
    return torch.stack([x_projection, y_projection, z_projection], dim=0)


def windowed_projected_PSD(data: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Return the windowed power spectral density (onesided) of the provided data cube.
    0th index goes as x, y, z"""
    assert data.size(0) == data.size(1) and data.size(0) == data.size(2), "Data is not a cube"
    assert len(window.size()) == 1, "Window must be 1D"
    assert window.size(0) == data.size(0), "Window must match data size"

    windowed_projections = xyz_projections(data) * window.expand((3, window.size(0)))
    return complex_abs_sq(torch.fft.rfft(windowed_projections))


def fourier_anisotropy_loss_pnorm(data: torch.Tensor, window: torch.Tensor, filter_ft: torch.Tensor, p: float = 2)->torch.Tensor:
    """Compute the XZ and YZ fourier losses (error between PSD on X compared to Z axis), with
    the provided windowing function and fourier domain filter."""
    assert data.size(0) == data.size(1) and data.size(0) == data.size(2), "Data is not a cube"
    assert len(window.size()) == 1, "Window must be 1D"
    assert window.size(0) == data.size(0), "Window must match data size"
    assert len(filter_ft.size()) == 1, "Filter must be real 1D"
    assert len(filter_ft) == math.floor(data.size(0)/2)+1, "Filter must match size of 1 sided real FT of the data"

    filtered_psd = windowed_projected_PSD(data, window) * filter_ft.expand((3, filter_ft.size(0)))

    xy = filtered_psd[0:2, :]
    zz = filtered_psd[2, :].expand(0, filtered_psd.size(1))
    zz = filtered_psd[2, :].expand((2, filtered_psd.size(1)))

    return torch.sum(torch.pow(torch.abs(xy - zz), p), dim=1)


def highpass_gaussian_kernel_fourier(sigma: float, N: int, onesided: bool = True) -> torch.Tensor:
    """
    Make an unshifted Gaussian highpass filter in Fourier space. All real
    The thinking here is that a highpass filter will only pick out high
    frequency structures
    Wheras low frequency structures might not actually be isotropic
    """

    centre = math.floor(N/2)
    xs = torch.tensor(range(0, N), dtype=torch.float) - centre

    space_domain = torch.exp(-xs**2/ (2*sigma**2))
    space_domain /= sum(space_domain)

    #fourier = torch.fft.rfft(space_domain, signal_ndim=1, onesided=onesided)
    fourier = torch.fft.rfft(space_domain)

    return 1 - torch.sqrt(complex_abs_sq(fourier))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nm_per_pix = 100.0
z_res_full = 240.0
z_res_real = 600.0

data_normal_cpu = load_cube(z_res_real, prefix="noisy-")
data_normal = data_normal_cpu.to(device)
data_hires = load_cube(z_res_full)

sigma_extra = math.sqrt(fwhm_to_sigma(z_res_real/nm_per_pix)**2 - fwhm_to_sigma(z_res_full/nm_per_pix)**2)
print("extra sigma = {}".format(sigma_extra))

# dl = conv_1D_z_axis(data_hires, gaussian_kernel(sigma_extra, 6.0))
# for z in range(1, data_normal.size(0)):
#    hires_orig = data_hires[z, :, :]
#    lores_orig = data_normal[z, :, :]
#    lores_new = dl[z, :, :]
#    res = torch.cat((hires_orig, lores_orig, lores_new, (lores_orig-lores_new) * 1e7), 1)
#    imsave(Path("splat")/"catted-{0:03d}.tiff".format(z), res.numpy())


hpf_ft = highpass_gaussian_kernel_fourier(fwhm_to_sigma(z_res_real/nm_per_pix),
                                          N=data_normal.size(2)).to(device)
sampling_window = blackman_harris_window(128).to(device)
z_kernel = gaussian_kernel(sigma_extra, 3.0).to(device)

# Start the reconstruction at the data
reconstruction = torch.sqrt(data_normal.clone()).to(device)
reconstruction.requires_grad = True


optimizer = torch.optim.LBFGS([reconstruction], lr=1)


for _ in range(1000):
    # 50 is the batch size
    for _ in range(50):
        def closure():

            # RENDERING LOSS
            # -----------------------------------------------------------------
            optimizer.zero_grad()
            reco = reconstruction ** 2
            rendering = conv_1D_z_axis(reco, z_kernel, pad="edge")
            rendering_loss = torch.sum((rendering - data_normal) ** 2)
            # fourier_loss = fourier_anisotropy_loss_pnorm(reco, sampling_window, hpf_ft)

            # FOURIER LOSS
            # -----------------------------------------------------------------
            # do a projection down all 3 axes
            # 0-z, 1=y, 2=x
            data_x_projection = data_normal.sum(1).sum(0)
            data_y_projection = data_normal.sum(2).sum(0)
            reco_z_projection = reco.sum(2).sum(1)
            # build them into a single data structure
            projections = torch.stack([data_x_projection,
                                       data_y_projection,
                                       reco_z_projection],
                                      dim=0)
            # stop the edge effects
            windowed_projections = projections * sampling_window.expand((3, sampling_window.size(0)))
            # power spectra is the square of the fourier transform
            power_spectra = complex_abs_sq(torch.fft.rfft(windowed_projections))
            # filtered power spectral density
            # filtered in that it has:
            # a high pass filter
            # no edge effects problems thanks to window
            filtered_psd = power_spectra * hpf_ft.expand((3, hpf_ft.size(0)))

            xy = filtered_psd[0:2, :]
            zz = filtered_psd[2, :].expand(0, filtered_psd.size(1))
            zz = filtered_psd[2, :].expand((2, filtered_psd.size(1)))

            # calculate the fourier loss
            fourier_loss = torch.sum(torch.pow(torch.abs(torch.log(xy) -
                                     torch.log(zz)), 2), dim=1) * 400000

            print(rendering_loss.item(), fourier_loss.cpu().detach().numpy())

            loss = rendering_loss + sum(fourier_loss)
            loss.backward()
            return loss

        optimizer.step(closure)

    # loss = fourier_loss + rendering_loss

    reco = reconstruction**2
    loss = closure()
    recodata = reco.clone().detach()

    plt.clf()
    plt.subplot(2, 4, 1)
    proj_data = xyz_projections(data_normal).to("cpu")
    proj_recon = xyz_projections(recodata).to("cpu")
    plt.plot(proj_data[0, :], label='Data x')
    plt.plot(proj_data[1, :], label='Data y')
    plt.plot(proj_data[2, :], label='Data z')
    plt.plot(proj_recon[0, :], label='Recon x')
    plt.plot(proj_recon[1, :], label='Recon y')
    plt.plot(proj_recon[2, :], label='Recon z')
    plt.title("XYZ projections")

    plt.subplot(2, 4, 2)
    psd_data = windowed_projected_PSD(data_normal, sampling_window).to("cpu")
    psd_recon = windowed_projected_PSD(recodata.clone().detach(),
                                       sampling_window).to("cpu")

    plt.semilogy(psd_data[0, :], label='Data x')
    plt.semilogy(psd_data[1, :], label='Data y')
    plt.semilogy(psd_data[2, :], label='Data z')
    plt.semilogy(psd_recon[0, :], label='Recon x')
    plt.semilogy(psd_recon[1, :], label='Recon y')
    plt.semilogy(psd_recon[2, :], label='Recon z')
    plt.title("Fourier spectra")

    plt.subplot(2, 4, 3)
    plt.plot(0, label='Data x')
    plt.plot(0, label='Data y')
    plt.plot(0, label='Data z')
    plt.plot(0, label='Recon x')
    plt.plot(0, label='Recon y')
    plt.plot(0, label='Recon z')
    plt.legend()
    plt.axis(False)

    plt.subplot(2, 4, 5)
    plt.imshow(data_normal_cpu[100, :, :])
    plt.subplot(2, 4, 6)
    plt.imshow(recodata[100, :, :].to("cpu"))

    plt.subplot(2, 4, 7)
    plt.imshow(data_hires[100, :, :])

    plt.subplot(2, 4, 8)
    plt.imshow(recodata[100, :, :].to("cpu") - data_hires[100, :, :])

    plt.show(block=False)
    print(loss)

    for z in range(1, data_normal.size(0)):
        hires_orig = data_hires[z, :, :]
        lores_orig = data_normal_cpu[z, :, :]
        hires_new = recodata[z, :, :].cpu()
        res = torch.cat((hires_orig, lores_orig, hires_new, (hires_orig-hires_new)), 1)
        imsave(Path("out")/"catted-{0:03d}.tiff".format(z), res.numpy())

    plt.waitforbuttonpress()


# plt.clf()
# x_only = data_normal.sum(1).sum(0) * blackman_harris_window(128)
# y_only = data_normal.sum(2).sum(0) * blackman_harris_window(128)
# z_only = data_normal.sum(2).sum(1) * blackman_harris_window(128)
# plt.subplot(1, 2, 1)
# plt.plot(x_only, label='x')
# plt.plot(y_only, label='y')
# plt.plot(z_only, label='z')
# plt.legend()


# What's a good fourier measure?

# Only care about high frequency stuff
# fta = complex_abs(torch.fft.rfft(x_only, signal_ndim=1, normalized=False, onesided=True))
# plt.subplot(1, 2, 2)
# plt.semilogy((abs(np.fft.fft(x_only))), label='x')
# plt.semilogy(psd[0, :], label='xx')
# plt.semilogy((abs(np.fft.fft(y_only))), label='y')
# plt.semilogy(psd[1, :], label='yy')
# plt.semilogy((abs(np.fft.fft(z_only))), label='z')
# plt.semilogy(psd[2, :], label='zz')
# plt.semilogy(filter_ft, label='kern')
# plt.semilogy((fta), label='zz', LineWidth=1)
# plt.legend()
# plt.show(block=False)

# plt.clf();
# x = blackman_harris_window(128)
# plt.plot(x)
# plt.show(block=False)
