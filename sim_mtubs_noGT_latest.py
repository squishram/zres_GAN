"""
This code generates 3D images of simulated microtubules. Workflow:
1. Uses a 3D random walk with constant step sizes
and limited 'turn sharpness' to generate coordinates
2. Creates an empty 3 dimensional array
3. Sums up all gaussian contributions to each pixel in 'patches'
   i.e volume subsections of the final image
4. Scales up the signal to match the desired image bittage
and saves the final array as a tiff

NOTE: cursory testing found 5^3 chunks for 96^3 voxel image to be fastest
(faster than 4 chunks and 6 chunks for the same data)
This translates to a (ROUGHLY) optimal voxels/chunk of 19 assuming linear relationship
"""

from datetime import date
import math
from pathlib import Path
import os
import random as r
import numpy as np
from time import perf_counter
from tifffile import imwrite


def rotation_matrix(axis, angle):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians.
    // axis = the axis of rotation, described as a 3D vector
    // angle = the rotation in radians
    """

    # ensure axis is a numpy array
    axis = np.array(axis)
    # normalise axis length
    axis = axis / (np.linalg.norm(axis))

    # these are the components required to define the rotation matrix
    x = axis[0]
    y = axis[1]
    z = axis[2]
    c = np.cos(angle)
    s = np.sin(angle)

    # and here it is!
    rotmat = np.array(
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


def random_walk(t, size_img, max_step=0.25, sharpest=np.pi):
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
    x = np.zeros(t)
    y = np.zeros(t)
    z = np.zeros(t)

    # unit vectors (j not needed):
    i = np.array([1, 0, 0])
    k = np.array([0, 0, 1])

    # this is the step along each axis
    step_size = max_step / (np.sqrt(3))

    # random starting point:
    x[0] = r.uniform(0, size_img[0])
    y[0] = r.uniform(0, size_img[1])
    z[0] = r.uniform(0, size_img[2])

    # random first step:
    v = np.random.uniform(-step_size, step_size, 3)
    # ensure it's the right length
    v = (v * max_step) / np.linalg.norm(v)

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
            v = np.random.uniform(-step_size, step_size, 3)

            # if the microtubule is still within the box
            # its next step must be constrained so it is not too sharp
        else:
            # initialise random polar angle
            theta = r.uniform(0, sharpest)
            # initialise random azimuthal angle
            phi = r.uniform(0, 2 * np.pi)
            # make the vector unit length
            v = v / np.linalg.norm(v)

            # rotate v about the normal to the plane created by v and k
            # unless v is parallel to k, in which case rotate v about i
            if np.dot(v, k) == 1:
                axis = i
            else:
                axis = np.cross(v, k)

            # find the polar rotation matrix about axis
            r_pol = rotation_matrix(axis, theta)
            # find the azimuth rotation matrix about v1
            r_azi = rotation_matrix(v, phi)

            # apply rotations to create a random vector within an angle of phi
            v = r_azi @ r_pol @ v

        # ensure step is consistent length:
        v = (v * max_step) / np.linalg.norm(v)

    data = np.concatenate(([x], [y], [z]), axis=0)
    data = np.array(data)

    return data


def image_of_gaussians(data, size_img, n_chunks, overlap):
    """
    Breaks up coordinate data into 3D chunks to decrease runtime,
    Retrieves gaussian contributions to each pixel
    using coordinate, intensity, and sigma data
    outputs n_chunks arrays
    each with the data that could contribute to chunk n_chunks.
    // data is the coordinates of the points that will be made into an image
    AND their intensity, sigma_xy and sigma_z.
    It is an array with dimensions (6, n_points)
    // size_img is the dimensions of the final image,
    tuple (size_x, size_y, size_z)
    // n_chunks should be a 3 element vector
    containing x, y, z chunking values respectively
    """

    # this output will contain the final image with illuminated pixels
    img = np.zeros(tuple(size_img))

    # the size of each chunk
    # // is 'floor division' i.e. divide then round down the result
    size_chunk = np.array([size_img[i] // n_chunks[i] for i in range(3)])

    # make an object array to contain all the data inside each chunk
    # (roughly equivalent to matlab cell array in that
    # each object/unit/cell can contain anything i.e. an array of any size)
    chunked_data = np.empty([n_chunks[0], n_chunks[1], n_chunks[2]], dtype=object)

    # assign an empty array as each object in chunked_data
    for x in range(n_chunks[0]):
        for y in range(n_chunks[1]):
            for z in range(n_chunks[2]):
                chunked_data[x][y][z] = []

    # This loop loads up the empty arrays with chunked_data
    for j in range(len(data[0])):
        for x in range(n_chunks[0]):
            xstart = (size_img[0] * x) // n_chunks[0]
            for y in range(n_chunks[1]):
                ystart = (size_img[1] * y) // n_chunks[1]
                for z in range(n_chunks[2]):
                    zstart = (size_img[2] * z) // n_chunks[2]

                    # edited to include the sigma & intensity information
                    if (
                        (
                            data[0][j] < xstart - overlap[0]
                            or data[0][j] >= (xstart + size_chunk[0] + overlap[0])
                        )
                        or (
                            data[1][j] < ystart - overlap[1]
                            or data[1][j] >= (ystart + size_chunk[1] + overlap[1])
                        )
                        or (
                            data[2][j] < zstart - overlap[2]
                            or data[2][j] >= (zstart + size_chunk[2] + overlap[2])
                        )
                    ):
                        continue
                    # if the point is inside the chunk, append it to that chunk
                    chunked_data[x][y][z].append([data[i][j] for i in range(6)])

    # creates a matrix of indices for each dimension (x, y, and z) each is 1 patch large -
    chunk_ind = np.indices((size_chunk[0], size_chunk[1], size_chunk[2]))

    # This loop sums the contributions from each local gaussian to each chunk
    for x in range(n_chunks[0]):
        xstart = (size_img[0] * x) // n_chunks[0]
        for y in range(n_chunks[1]):
            ystart = (size_img[1] * y) // n_chunks[1]
            for z in range(n_chunks[2]):
                zstart = (size_img[2] * z) // n_chunks[2]

                intensityspot = np.zeros((size_chunk[0], size_chunk[1], size_chunk[2]))

                # TODO do we want to include a patch calculation here?
                for cx, cy, cz, cintensity, csig_xy, csig_z in chunked_data[x][y][z]:
                    # define the normalisation constant for the gaussian
                    const_norm = cintensity / ((csig_xy**3) * (2 * np.pi) ** 1.5)
                    # add the gaussian contribution to the spot
                    intensityspot += const_norm * np.exp(
                        -(
                            ((chunk_ind[0] + xstart - cx) ** 2) / (2 * csig_xy**2)
                            + ((chunk_ind[1] + ystart - cy) ** 2) / (2 * csig_xy**2)
                            + ((chunk_ind[2] + zstart - cz) ** 2) / (2 * csig_z**2)
                        )
                    )
                xend = xstart + size_chunk[0]
                yend = ystart + size_chunk[1]
                zend = zstart + size_chunk[2]
                img[xstart:xend, ystart:yend, zstart:zend] = intensityspot

    return np.array(img)


# Initialize timer
time1 = perf_counter()


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

# number of images to produce for each resolution:
n_imgs = 300
# file name root:
filename = "mtubs_sim_noGT_"
# bittage of final image - 8 | 16 | 32 | 64
# 16-bit is as high as cameras usually go anyway
img_bit = 16


#####################
# microtubule specs #
#####################

# total length of all fibres:
t = 2500
# size of final image in pixels:
size_img = np.array([96, 96, 32])
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 0.5
# how sharply can the path bend each step?
sharpest = (np.pi * max_step) / 10
# chunk optimisation factor
# numbers that give good results: 18
chunk_opt = 18
# chunk overlap factor
# numbers that give good results: 7
chunk_overlap_factor = 7

#############
# PSF specs #
#############

# What is the mean intensity (in AU) and its uncertainty
# (as a fraction of the mean value)?
intensity_mean = 1000
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
sigma_xy_mean = (xres / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
sigma_z_mean = (zres / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))

# here we check if size_img % chunk_size == 0, so the number of chunks fits cleanly into the image size
# if it doesn't, we check chunk_size == 20 and then chunk_size == 18, then chunk_size == 21...
# how many chunks are we splitting the data into along each dimension?
# (optimal found to be 5 for 96x96x96 voxels, assume linear relation)
n_chunks_0 = [(size_img[i] // chunk_opt) for i in range(3)]
n_chunks = [(size_img[i] // chunk_opt) for i in range(3)]

for i in range(len(size_img)):
    counter = 0
    go_up = True
    while size_img[i] % n_chunks[i] != 0:
        n_chunks[i] = n_chunks_0[i]
        if go_up:
            n_chunks[i] += 1 + (counter // 2)
            go_up = False
        elif not go_up:
            n_chunks[i] -= 1 + (counter // 2)
            go_up = True
        counter += 1

# how much do the chunks overlap?
chunk_overlap = np.array(
    [
        int(chunk_overlap_factor * sigma_xy_mean),
        int(chunk_overlap_factor * sigma_xy_mean),
        int(chunk_overlap_factor * sigma_z_mean),
    ]
)


#################
# training loop #
#################

for i in range(n_imgs):
    # generate data as list of 3d coordinates
    data = random_walk(t, size_img, max_step, sharpest)
    # broadcast intensity & sigma values into arrays with slight randomness to their values
    intensity = np.array(
        [
            intensity_mean * (1 + r.uniform(-int_unc, int_unc))
            for _ in range(len(data[0]))
        ]
    )
    sigma_xy = np.array(
        [
            sigma_xy_mean * (1 + r.uniform(-sig_unc, sig_unc))
            for _ in range(len(data[0]))
        ]
    )

    sigma_z = np.array(
        [sigma_z_mean * (1 + r.uniform(-sig_unc, sig_unc)) for _ in range(len(data[0]))]
    )

    # put coordinates, intensity, sigma_xy, sigma_z data into one structure
    data_lores = np.concatenate(
        ([data[0]], [data[1]], [data[2]], [intensity], [sigma_xy], [sigma_z]),
        axis=0,
    )

    # This function breaks the data into "chunks" for efficiency,
    # then uses it to 'fill up' the empty image array:
    mtubs = image_of_gaussians(data_lores, size_img, n_chunks, chunk_overlap)

    # normalise all the brightness values
    # then scale them up so that the brightest value is 255
    # scale according to the z-projection in order that the  high-and-low-res-images aren't affected differently
    # which should be the same for both
    # z_projection = data.sum(2).sum(1)
    # TODO - finish this idea! NOTE - will not work for different sampling (as in this code)

    # tiff writing in python gets the axes wrong
    # rotate the image before writing so it doesn't!
    mtubs = np.rot90(mtubs, 1, (0, 2))

    # add an offset
    # mtubs += 100

    # write to file
    # isotropic version:
    filename_ind = filename + str(i + 1)
    file_path = os.path.join(path_data, filename_ind)
    print(f"Writing to tiff: {i + 1}")
    imwrite(file_path, mtubs.astype(np.uint16))


time2 = perf_counter()

print(f"The image dimensions are {size_img}")
print(f"The number of chunks along each dimension is: {n_chunks}")
print(f"The overlap, in voxels, is {chunk_overlap}")
print(f"The number of total steps is {t}")
print(f"the mean xy-sigma is {sigma_xy_mean}")
print(f"the mean z-sigma is {sigma_z_mean}")
print("Done!")
print(
    f"To make {n_imgs} {img_bit}-bit images took {time2 - time1} seconds"
)

# play an alarm to signal the code has finished running!
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

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
            f"\nimage dimensions, voxels: {size_img}\n",
            f"mean emitter intensity, AU1: {intensity_mean}\n",
            f"emitter intensity variance, AU1: {int_unc * intensity_mean}\n",
            f"voxel dimensions, nm: {(size_pix_nm, size_pix_nm, size_pix_nm)}\n",
            f"xy-resolution, nm: {xres}\n",
            f"z-resolution, nm: {zres}\n",
            f"xy-resolution variance, nm: {int_unc * xres}\n",
            f"z-resolution variance, nm: {int_unc * zres}\n",
            f"number of chunks: {n_chunks}\n",
            f"chunk overlap: {chunk_overlap}\n",
            f"total fibre length: {t}\n",
            f"mean xy-sigma: {sigma_xy_mean}\n",
            f"mean  z-sigma: {sigma_z_mean}\n",
            f"number of images: {n_imgs}\n",
            f"image bit-depth: {img_bit}\n",
            f"total time taken, seconds: {time2 - time1}\n",
        ]
    )
