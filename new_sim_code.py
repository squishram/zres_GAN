"""
This code generates 3D images of simulated microtubules.
1. Uses a 3D random walk with constant step sizes
and limited 'turn sharpness' to generate a list of coordinates
(the random_walk() function)
2. assigns a semi-randomised sigma_xy, sigma_z, and intensity to each coordinate
3. creates a matrix of indices
4. adds each coordinate's signal contribution to every pixel in the image in a for loop

THE CHALLENGE:
Although we are provided with a list of coordiantes, it is still difficult to make a GT and non-GT version of the same image.
Converting between resolutions is easy (we simply apply a different sigma_z to the same coordinate set)
Converting between samplings is difficult, because we actually need to change the z-coordinates of the array and I'm not sure how
(if we divide all the z-values by (sigma_z/sigma_xy), it sort of compresses it and it doesn't look right, but maybe I am doing it wrong - should this work?)
"""

from datetime import date
import math
import os
import random as r
import numpy as np
from time import perf_counter
import torch
from tifffile import imwrite


def rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians.
    // axis = the axis of rotation, described as a 3D vector
    // angle = the rotation in radians
    """

    # normalise axis length
    axis = axis / (torch.linalg.norm(axis))

    # these are the components required to define the rotation matrix
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = torch.cos(angle)
    s = torch.sin(angle)

    # and here it is!
    rotmat = torch.tensor(
        [
            [
                x**2 + (y**2 + z**2) * c,
                x * y * (1 - c) - z * s,
                x * z * (1 - c) + y * s,
            ],
            [
                x * y * (1 - c) + z * s,
                y**2 + (x**2 + z**2) * c,
                y * z * (1 - c) - x * s,
            ],
            [
                x * z * (1 - c) - y * s,
                y * z * (1 - c) + x * s,
                z**2 + (x**2 + y**2) * c,
            ],
        ]
    )

    return rotmat


def random_walk(
    t: int, size_img: torch.Tensor, max_step: float = 0.25, sharpest: float = torch.pi
):
    """
    Sets up a random walk in three dimensions:
    // t = number of steps taken on each walk, dtype = uint
    // size_img = dimensions of space in which walk takes place,
    [xsize, ysize, zsize].
    Faster if fed in as a numpy array.
    // max_step = the size of each step in the walk, dtype = uint
    // sharpest = the sharpest turn between each step, dtype = float
    // reinitialise = whether or not the walk is reinitialised
    at a random location when it leaves the space (saves memory), dtype = bool
    """

    # x, y, z will contain a list of all of the positions
    x = torch.zeros(t)
    y = torch.zeros(t)
    z = torch.zeros(t)

    # unit vectors (j not needed):
    i = torch.tensor([1, 0, 0], dtype=torch.double)
    k = torch.tensor([0, 0, 1], dtype=torch.double)

    # this is the step along each axis
    step_size = max_step / (np.sqrt(3))

    # random starting point:
    x[0] = r.uniform(0, size_img[0].item())
    y[0] = r.uniform(0, size_img[1].item())
    z[0] = r.uniform(0, size_img[2].item())

    # random first step:
    v = torch.from_numpy(np.random.uniform(-step_size, step_size, 3))
    # ensure it's the right length
    v = (v * max_step) / torch.linalg.norm(v)

    for q in range(1, t):
        # add the last step to the last position to get the new position:
        x[q] = x[q - 1] + v[0]
        y[q] = y[q - 1] + v[1]
        z[q] = z[q - 1] + v[2]

        # if the microtubule leaves the imaging area
        # just re-initialize it somewhere else:
        if (
            ((x[q] > (size_img[0] + 1)) or (x[q] < -1))
            or ((y[q] > (size_img[1] + 1)) or (y[q] < -1))
            or ((z[q] > (size_img[2] + 1)) or (z[q] < -1))
        ):
            # new random starting point:
            x[q] = r.uniform(0, size_img[0].item())
            y[q] = r.uniform(0, size_img[1].item())
            z[q] = r.uniform(0, size_img[2].item())
            # new random first step:
            v = torch.from_numpy(np.random.uniform(-step_size, step_size, 3))

            # if the microtubule is still within the box
            # its next step must be constrained so it is not too sharp
        else:
            # initialise random polar angle
            theta = r.uniform(0, sharpest)
            # initialise random azimuthal angle
            phi = r.uniform(0, 2 * np.pi)
            # make the vector unit length
            v = v / torch.linalg.norm(v)

            # rotate v about the normal to the plane created by v and k
            # unless v is parallel to k, in which case rotate v about i
            if torch.dot(v, k) == 1:
                axis = i
            else:
                axis = torch.cross(v, k)

            # find the polar rotation matrix about axis
            r_pol = rotation_matrix(axis, torch.tensor(theta))
            # find the azimuth rotation matrix about v1
            r_azi = rotation_matrix(v, torch.tensor(phi))

            # apply rotations to create a random vector within an angle of phi
            v = r_azi @ r_pol @ v

        # ensure step is consistent length:
        v = (v * max_step) / torch.linalg.norm(v)

    data = torch.stack((x, y, z), 1).to(device)

    return data


def semirandomised_values(mean: float, uncertainty: float, size: int) -> torch.Tensor:
    """
    creates a 1D tensor of values distributed with "uncertainty" about a "mean" of length "size"
    """

    output = torch.tensor(
        [mean * (1 + r.uniform(-uncertainty, uncertainty)) for _ in range(size)]
    ).to(device)

    return output


def simulated_image(coordinates, img_size, intensity, sigma_xy, sigma_z):

    #####################
    # TENSOR OF INDICES #
    #####################

    # generate tensor of indices
    img_indices = torch.from_numpy(np.indices((img_size))).to(device)
    # resize to (n_molecules, img_size[0], img_size[1], img_size[2], 2)
    img_indices = (
        img_indices.unsqueeze(0)
        .expand(
            coordinates.shape[0],
            img_size.shape[0],
            img_size[0],
            img_size[1],
            img_size[2],
        )
        .to(device)
    )

    # pull out x y and z indices
    x_indices = img_indices[:, 0, :, :, :].to(device)
    y_indices = img_indices[:, 1, :, :, :].to(device)
    z_indices = img_indices[:, 2, :, :, :].to(device)

    #######################
    # TENSOR OF MOLECULES #
    #######################

    for _ in range(img_size.shape[0]):
        coordinates = coordinates.unsqueeze(-1)
    coordinates = coordinates.expand(img_indices.shape).to(device)

    # pull out x y and z indices
    x_coordinates = coordinates[:, 0, :, :, :].to(device)
    y_coordinates = coordinates[:, 1, :, :, :].to(device)
    z_coordinates = coordinates[:, 2, :, :, :].to(device)

    # broadcast the intensity and sigma values too
    for _ in range(img_size.shape[0]):
        sigma_xy = sigma_xy.unsqueeze(-1)
        sigma_z = sigma_z.unsqueeze(-1)
        intensity = intensity.unsqueeze(-1)
    sigma_xy = sigma_xy.expand(x_coordinates.shape).to(device)
    sigma_z = sigma_z.expand(x_coordinates.shape).to(device)
    intensity = intensity.expand(x_coordinates.shape).to(device)

    # add the gaussian contribution to the spot
    gaussians = torch.zeros((img_size[0], img_size[1], img_size[2])).to(device)
    for i in range(coordinates.shape[0]):
        gaussians += (
            # normalisation constant for 3D gaussian
            (intensity[i] / ((sigma_xy[i] ** 3) * (2 * np.pi) ** 1.5))
            # gaussian equation
            * torch.exp(
                -(
                    ((x_indices[i] - x_coordinates[i]) ** 2) / (2 * sigma_xy[i] ** 2)
                    + ((y_indices[i] - y_coordinates[i]) ** 2) / (2 * sigma_xy[i] ** 2)
                    + ((z_indices[i] - z_coordinates[i]) ** 2) / (2 * sigma_z[i] ** 2)
                )
            )
        ).to(device)

    return gaussians


# Initialize timer
time1 = perf_counter()

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########
# STORAGE #
###########

# get the date
today = str(date.today())
today = today.replace("-", "")
# path to data
path_lores = os.path.join(os.getcwd(), "images/sims/microtubules/lores_test/")
path_hires = os.path.join(os.getcwd(), "images/sims/microtubules/hires_test/")
# make directories if they don't already exist so images have somewhere to go
os.makedirs(path_lores, exist_ok=True)
os.makedirs(path_hires, exist_ok=True)

###################
# finishing alarm #
###################

# how long the sound goes on for, in seconds
duration = 1
# the frequency of the sine wave (i.e. the pitch)
freq = 440

##############
# file specs #
##############

# number of images to produce (for each resolution if making GT as well):
n_imgs = 1
# file name root (include underscore at the end):
filename = "sim_mtubs_"
# bittage of final image - 8 | 16 | 32 | 64
# 16-bit is as high as cameras usually go anyway
img_bit = 16


#####################
# microtubule specs #
#####################

# total length of all fibres:
t = 2500
# size of final image in pixels:
size_img_lores = torch.tensor([96, 96, 32]).to(device)
size_img_hires = torch.tensor([96, 96, 96]).to(device)
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 0.5
# how sharply can the path bend each step?
sharpest = (np.pi * max_step) / 10
# do we want the hires data to have the same coordinates as the lores data?
same = True

#############
# PSF specs #
#############

# What is the mean intensity (in AU) and its uncertainty
# (as a fraction of the mean value)?
mean_int = 1000
int_unc = 0.2
# pixel size in nm
size_pix_nm = 10.0
# x/y-resolution
xres = 24.0
# z-resolution
zres = xres * 5
# What is the mean sigma (in voxels) and the sigma uncertainty
# (as a fraction of the mean value)?
sig_unc = 0.2

#######################################
# calculating & optimising parameters #
#######################################

# convert to sigma
mean_sigxy = (xres / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
mean_sigz = (zres / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))

#################
# training loop #
#################

for i in range(n_imgs):
    # generate data as list of 3d coordinates
    lores_data = random_walk(t, size_img_lores, max_step, sharpest)

    # the if statement checks the 'same' variable
    # which is whether we want the hires and lores data to have the same coordinates
    if not same:
        hires_data = random_walk(t, size_img_hires, max_step, sharpest)
    else:
        ratio = (size_img_hires / size_img_lores).expand(len(lores_data), -1)
        hires_data = lores_data * ratio

    # broadcast intensity & sigma values into arrays with slight randomness to their values
    intensity = semirandomised_values(mean_int, int_unc, len(lores_data))
    sigma_xy = semirandomised_values(mean_sigxy, sig_unc, len(lores_data))
    sigma_z = semirandomised_values(mean_sigz, sig_unc, len(lores_data))

    # function to make an image of gaussians from molecule coordinate data
    sim_img_lores = simulated_image(
        lores_data, size_img_lores, intensity, sigma_xy, sigma_z
    ).to(device)
    sim_img_hires = simulated_image(
        hires_data, size_img_hires, intensity, sigma_xy, sigma_xy
    ).to(device)

    # normalise all the brightness values
    # then scale them up so that the brightest value is 255
    # scale according to the z-projection in order that the  high-and-low-res-images aren't affected differently
    # which should be the same for both
    # z_projection = data.sum(2).sum(1)
    # TODO: finish this idea! NOTE - will not work for different sampling (as in this code)

    # tiff writing in python gets the axes wrong so rotate the image before writing
    sim_img_lores = torch.rot90(sim_img_lores, 1, (0, 2)).short()
    sim_img_hires = torch.rot90(sim_img_hires, 1, (0, 2)).short()
    # convert to numpy array
    sim_img_lores = sim_img_lores.cpu().detach().numpy().astype(np.uint16)
    sim_img_hires = sim_img_hires.cpu().detach().numpy().astype(np.uint16)

    # add an offset
    # sim_img += 100

    # write to file
    # isotropic version:
    filename_ind = filename + str(i + 1)
    lores_file_path = os.path.join(path_lores, filename_ind + "_lores")
    hires_file_path = os.path.join(path_hires, filename_ind + "_hires")
    print(f"Writing to tiff: {i + 1}")
    imwrite(lores_file_path, sim_img_lores)
    imwrite(hires_file_path, sim_img_hires)


time2 = perf_counter()

print(f"The low-res image dimensions are {size_img_lores[:]}")
print(f"The high-res image dimensions are {size_img_hires[:]}")
print(f"The number of total steps is {t}")
print(f"the mean xy-sigma is {mean_sigxy}")
print(f"the mean z-sigma is {mean_sigz}")
print("Done!")
print(f"To make {n_imgs} {img_bit}-bit images took {time2 - time1} seconds")

# play an alarm to signal the code has finished running!
os.system("play -nq -t alsa synth {} sine {}".format(duration, freq))

############
# METADATA #
############

# make a metadata file
metadata = today + "simulated_microtubules_metadata.txt"
metadata = os.path.join(path_lores, metadata)
# make sure to remove any other metadata files in the subdirectory
if os.path.exists(metadata):
    os.remove(metadata)
with open(metadata, "a") as file:
    file.writelines(
        [
            os.path.basename(__file__),
            f"\nlow-res image dimensions, voxels: {size_img_lores}",
            f"\nhigh-res image dimensions, voxels: {size_img_hires}",
            f"\nmean emitter intensity, AU1: {mean_int}",
            f"\nemitter intensity variance, AU1: {int_unc * mean_int}",
            f"\nvoxel dimensions, nm: {(size_pix_nm, size_pix_nm, size_pix_nm)}",
            f"\nxy-resolution, nm: {xres}",
            f"\nz-resolution, nm: {zres}",
            f"\nxy-resolution variance, nm: {int_unc * xres}",
            f"\nz-resolution variance, nm: {int_unc * zres}",
            f"\ntotal fibre length: {t}",
            f"\nmean xy-sigma: {mean_sigxy}",
            f"\nmean  z-sigma: {mean_sigz}",
            f"\nnumber of images: {n_imgs}",
            f"\nimage bit-depth: {img_bit}",
            f"\ntotal time taken, seconds: {time2 - time1}",
        ]
    )
