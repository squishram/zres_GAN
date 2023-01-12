"""
This code generates 3D images of simulated microtubules. Workflow:
1. Uses a 3D random walk with constant step sizes
and limited 'turn sharpness' to generate coordinates
2. Creates an empty 3 dimensional array
3. Sums up all gaussian contributions to each pixel in 'patches'
   i.e volume subsections of the final image
4. Scales up the signal to match the desired image bittage
and saves the final array as a tiff

TODO
1. make the microtubules have length and start position/ stop position constraints
2. get rid of the hires/ lores version of the data
3. make it 2D?
4. get rid of the to(device) stuff if necessary
"""

from datetime import date
import math
import os
import random as r
import numpy as np
from time import perf_counter
import torch
from tifffile import imwrite
from torch._C import uint8


def rotation_matrix(axis: torch.Tensor, angle):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians.
    // axis = the axis of rotation, described as a 3D vector
    // angle = the rotation in radians
    """

    # ensure axis is a torch tensor
    if axis.type() != torch.Tensor:
        axis = torch.tensor(axis)
    # normalise axis length
    axis = axis / (torch.linalg.norm(axis))

    # these are the components required to define the rotation matrix
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = torch.cos(angle)
    s = torch.sin(angle)

    # and here it is!
    rotmat2d = torch.tensor(
        [
            [c s],
            [-s c]
        ]
    )
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


def random_walk(t: int, size_img, max_step=0.25, sharpest=torch.pi):
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
    x[0] = r.uniform(0, size_img[0])
    y[0] = r.uniform(0, size_img[1])
    z[0] = r.uniform(0, size_img[2])

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
            x[q] = r.uniform(0, size_img[0])
            y[q] = r.uniform(0, size_img[1])
            z[q] = r.uniform(0, size_img[2])
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
            r_pol = rotation_matrix(axis, theta)
            # find the azimuth rotation matrix about v1
            r_azi = rotation_matrix(v, phi)

            # apply rotations to create a random vector within an angle of phi
            v = r_azi @ r_pol @ v

        # ensure step is consistent length:
        v = (torch.tensor(v) * max_step) / torch.linalg.norm(v)

    data = torch.stack((x, y, z), 1)

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
    # resize to (n_molecules, img_size[0], img_size[1], 2)
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

    # add the gaussian contribution to the spot
    gaussians = (
        (
            # normalisation constant for 3D gaussian
            (intensity / ((sigma_xy**3) * (2 * np.pi) ** 1.5))
            # gaussian equation
            * torch.exp(
                -(
                    ((x_indices - x_coordinates) ** 2) / (2 * sigma_xy**2)
                    + ((y_indices - y_coordinates) ** 2) / (2 * sigma_xy**2)
                    + ((z_indices - z_coordinates) ** 2) / (2 * sigma_z**2)
                )
            )
        )
        .sum(0)
        .to(device)
    )

    return gaussians


# Initialize timer
time1 = perf_counter()

# use gpu if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################
# USER INPUT STARTS #
#####################

###########
# STORAGE #
###########

# get the date
today = str(date.today())
today = today.replace("-", "")
# path to data
path_data = os.path.join(os.getcwd(), "images/sims/microtubules/noGT_LD/")
# make directories if they don't already exist so images have somewhere to go
os.makedirs(path_data, exist_ok=True)

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
# file name root:
filename = "sim_img_sim_noGT_"
# bittage of final image - 8 | 16 | 32 | 64
# 16-bit is as high as cameras usually go anyway
img_bit = 16


#####################
# microtubule specs #
#####################

# total length of all fibres:
t = 2500
# size of final image in pixels:
size_img = torch.tensor([96, 96, 32]).to(device)
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 0.5
# how sharply can the path bend each step?
sharpest = (np.pi * max_step) / 10

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

###################
# USER INPUT ENDS #
###################

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
    lores_data = random_walk(t, size_img, max_step, sharpest)
    hires_data = (size_img[0] / size_img[2]) * lores_data
    # broadcast intensity & sigma values into arrays with slight randomness to their values
    intensity = semirandomised_values(mean_int, int_unc, len(lores_data[0]))
    sigma_xy = semirandomised_values(mean_sigxy, sig_unc, len(lores_data[0]))
    sigma_z = semirandomised_values(mean_sigz, sig_unc, len(lores_data[0]))

    # function to make an image of gaussians from molecule coordinate data
    sim_img_lores = simulated_image(lores_data, size_img, intensity, sigma_xy, sigma_z).to(device)
    sim_img_hires = simulated_image(hires_data, size_img, intensity, sigma_xy, sigma_xy).to(device)

    # normalise all the brightness values
    # then scale them up so that the brightest value is 255
    # scale according to the z-projection in order that the  high-and-low-res-images aren't affected differently
    # which should be the same for both
    # z_projection = data.sum(2).sum(1)
    # TODO: finish this idea! NOTE - will not work for different sampling (as in this code)

    # tiff writing in python gets the axes wrong so rotate the image before writing
    sim_img_lores = torch.rot90(sim_img_lores, 1, (0, 2))
    # convert to torch tensor
    sim_img_lores = sim_img_lores.cpu().detach().numpy()

    # add an offset
    # sim_img += 100

    # write to file
    # isotropic version:
    filename_ind = filename + str(i + 1)
    file_path = os.path.join(path_data, filename_ind)
    print(f"Writing to tiff: {i + 1}")
    imwrite(file_path, sim_img_lores.astype(torch.ShortTensor))
    imwrite(file_path, sim_img_lores.astype(torch.ShortTensor))


time2 = perf_counter()

print(f"The image dimensions are {size_img}")
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
metadata = os.path.join(path_data, metadata)
# make sure to remove any other metadata files in the subdirectory
if os.path.exists(metadata):
    os.remove(metadata)
with open(metadata, "a") as file:
    file.writelines(
        [
            os.path.basename(__file__),
            f"\nimage dimensions, voxels: {size_img}",
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
