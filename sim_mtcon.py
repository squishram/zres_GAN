"""
This code generates 3D images of simulated microtubules. Workflow:
1. Uses a 3D random walk with constant step sizes
and limited 'turn sharpness' to generate coordinates
2. Creates an empty 3 dimensional array
3. Sums up all gaussian contributions to each pixel in 'patches'
   i.e volume subsections of the final image
4. Scales up the signal to match the desired image bittage
and saves the final array as a tiff


To do:
- NEEDS TO ADJUST CHUNK SIZES IF IMAGE SIZE IS NOT PERFECT CUBE
- Add patch calculation to potentially reduce overlap effects?
- Get all the 'circle points' into a single array (3, n_points * n_circle_points)

Questions for Susan:
How do I cange this part of the code (lines ~194-199)
to accomodate non-perfectly cuboidal chunks?

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
from time import time
from tifffile import imwrite
import torch


def rotation_matrix(axis, angle):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians.
    // axis = the axis of rotation, described as a 3D vector
    // angle = the magnitude of rotation in radians
    """

    # ensure axis is a numpy array
    axis = np.asarray(axis)
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


def circle(n_points, normal, centre, radius):
    """
    generate a circle given
    // n_points, the number of points to plot on the circle perimeter, int
    // normal, the normal to the plane containing the circle, list-like
    // centre, the centre of the circle, list-like
    // radius, the radius of the circle, int
    """

    # we need to create two arbitrary vectors parallel to the circle plane to draw it
    # first vector on the plane is an arbitrary vector perpendicular to normal
    # obtain the indices of the two largest values in the normal
    normal_largest_indices = (-normal).argsort()[:2]
    # swap them around, make the largest one negative, and the remaining component 0
    line1 = np.zeros(3)
    line1[normal_largest_indices[1]] = -normal[normal_largest_indices[0]]
    line1[normal_largest_indices[0]] = normal[normal_largest_indices[1]]
    # second vector on the plane is simply normal x line1 (needs to be any vector perpendicular to both)
    line2 = np.cross(normal, line1)
    # normalise
    line1 = line1 / np.linalg.norm(line1)
    line2 = line2 / np.linalg.norm(line2)

    # n_points values between 0 and 2*pi
    points = np.linspace(0, 2 * np.pi, n_points)
    # parametric circle equation in 3D space
    circle = np.zeros((3, n_points))
    for i in range(3):
        circle[i, :] = (
            centre[i]
            + (radius * line1[i] * np.cos(points))
            + (radius * line2[i] * np.sin(points))
        )

    return circle


def random_walk(t, size_img, radius, max_step=0.25, sharpest=np.pi, n_circle_points=16):
    """
    Sets up a random walk in three dimensions:
    // t = number of steps taken on each walk, uint
    // size_img = dimensions of space in which walk takes place, list-like
    // radius = radius of the circle plotted around each step in the random walk, uint/float
    // max_step = the size of each step in the walk, uint/float
    // sharpest = the sharpest turn between each step, float
    // n_circle_points = number of points to be plotted in a circle around each step in the random walk, uint
    """

    # data will contain a list of all positions along the walk, with circles drawn around them
    circle_data = np.zeros((3, t * n_circle_points))

    # current random walk position (the centre of each circle)
    c = np.zeros(3)

    # unit vectors (j not needed):
    i = np.array([1, 0, 0])
    k = np.array([0, 0, 1])

    # this is the step along each axis
    step_size = max_step / (np.sqrt(3))

    for q in range(0, t):

        # add a randomness to the circle's radius
        new_radius = radius * r.uniform(0.5, 1.5)

        # if the microtubule leaves the imaging area (or if it's the first step)
        # just re-initialize the walk afresh
        if (
            ((c[0] > (size_img[0] + 1)) or (c[0] < -1))
            or ((c[1] > (size_img[1] + 1)) or (c[1] < -1))
            or ((c[2] > (size_img[2] + 1)) or (c[2] < -1))
            or (q == 0)
        ):
            # new random starting point:
            c[0] = r.uniform(0, size_img[0])
            c[1] = r.uniform(0, size_img[1])
            c[2] = r.uniform(0, size_img[2])
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

        # accumulate circle points
        circle_data[:, q * n_circle_points : (q + 1) * n_circle_points] = circle(
            n_circle_points, v, c, new_radius
        )

        # add the last step to the last position to get the new position:
        for idx in range(3):
            c[idx] += v[idx]

    return circle_data


def image_of_gaussians(data, size_img, overlap, size_patch=5):
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
    img = torch.zeros(tuple(size_img))

    # the size of each chunk
    # // is 'floor division' i.e. divide then round down the result
    size_patch = np.array([size_img[i] // n_chunks[i] for i in range(3)])

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
                            or data[0][j] >= (xstart + size_patch[0] + overlap[0])
                        )
                        or (
                            data[1][j] < ystart - overlap[1]
                            or data[1][j] >= (ystart + size_patch[1] + overlap[1])
                        )
                        or (
                            data[2][j] < zstart - overlap[2]
                            or data[2][j] >= (zstart + size_patch[2] + overlap[2])
                        )
                    ):
                        continue
                    # if the point is inside the chunk, append it to that chunk
                    chunked_data[x][y][z].append([data[i][j] for i in range(6)])

    # TODO this needs to be fixed to accomodate potentially varying sizes of chunk
    # creates a matrix of indices for each dimension (x, y, and z)
    N = torch.tensor(range(size_patch[0]))
    zi = N.expand(size_patch[0], size_patch[1], size_patch[2])
    xi = zi.transpose(0, 2)
    yi = zi.transpose(1, 2)

    # lists are faster to iterate through
    data = data.T
    data = data.tolist()

    # This loop sums the contributions from each local gaussian to each chunk
    for x in range(n_chunks[0]):
        xstart = (size_img[0] * x) // n_chunks[0]
        for y in range(n_chunks[1]):
            ystart = (size_img[1] * y) // n_chunks[1]
            for z in range(n_chunks[2]):
                zstart = (size_img[2] * z) // n_chunks[2]

                intensityspot = torch.zeros(
                    (size_patch[0], size_patch[1], size_patch[2])
                )

                # NOTE do we want to include a patch calculation here?
                for cx, cy, cz, cintensity, csig_xy, csig_z in chunked_data[x][y][z]:
                    # define the normalisation constant for the gaussian
                    const_norm = cintensity / ((csig_xy**3) * (2 * np.pi) ** 1.5)
                    # add the gaussian contribution to the spot
                    intensityspot += const_norm * torch.exp(
                        -(
                            ((xi + xstart - cx) ** 2) / (2 * csig_xy**2)
                            + ((yi + ystart - cy) ** 2) / (2 * csig_xy**2)
                            + ((zi + zstart - cz) ** 2) / (2 * csig_z**2)
                        )
                    )
                xend = xstart + size_patch[0]
                yend = ystart + size_patch[1]
                zend = zstart + size_patch[2]
                img[xstart:xend, ystart:yend, zstart:zend] = intensityspot

    data = np.array(data)
    data = data.T

    return np.array(img)


# Initialize timer
time1 = time()


###########
# STORAGE #
###########

# get the date
today = str(date.today())
today = today.replace("-", "")
# path to data
path_data = os.path.join(os.getcwd(), "images/sims/")
# path to training samples (low-z-resolution, high-z-resolution)
path_lores = os.path.join(path_data, Path("mitochondria/lores"))
path_hires = os.path.join(path_data, Path("mitochondria/hires"))
# make directories if they don't already exist so images have somewhere to go
os.makedirs(path_lores, exist_ok=True)
os.makedirs(path_hires, exist_ok=True)

##############
# file specs #
##############

# number of images to produce for each resolution:
n_imgs = 1
# file name root:
filename = "mtcon_sim_"

######################
# mitochondria specs #
######################

# number of steps per walk:
t = 1000
# size of final image in pixels:
size_img = np.array([96, 96, 96])
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 2
# how sharply can the path bend each step?
sharpest = (np.pi * max_step) / 20
# how many points plotted on each circle
n_circle_points = 12
# mean radius of circles, in pixels
radius = 5


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
zres = 60.0
# What is the mean sigma (in voxels) and the sigma uncertainty
# (as a fraction of the mean value)?
sig_unc = 0.2
# convert to sigma
sigma_xy_mean = (xres / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
sigma_z_mean = (zres / size_pix_nm) / (2 * math.sqrt(2 * math.log(2)))
sigma_tuple = (sigma_xy_mean, sigma_z_mean)


###############
# chunk specs #
###############

# here we check if size_img % chunk_size == 0, so the number of chunks fits cleanly into the image size
# if it doesn't, we check chunk_size == 20 and then chunk_size == 18, then chunk_size == 21...
go_up = True
counter = 0
# how many chunks are we splitting the data into along each dimension?
# (optimal found to be 5 for 96x96x96 voxels, assume linear relation)
n_chunks = 5
# chunks need to fit evenly into the data, this will iteratively try chunk sizes until this is the case
while size_img[0] % n_chunks != 0:
    n_chunks = 5
    if go_up:
        n_chunks += 1 + (counter // 2)
        go_up = False
    elif not go_up:
        n_chunks -= 1 + (counter // 2)
        go_up = True
    counter += 1

n_chunks = np.array([n_chunks for i in range(3)])
# how much do the chunks overlap?
overlap = 7 * sigma_xy_mean


#################
# training loop #
#################

for i in range(n_imgs):
    # plot all the points in the random walk, with circles around them
    circle_data = random_walk(t, size_img, radius, max_step, sharpest, n_circle_points)

    # make one isotropic, one anisotropic
    for idx, sigma_z_mean in enumerate(sigma_tuple):
        # broadcast intensity & sigma values into distributed arrays
        intensity = np.array(
            [
                intensity_mean * (1 + r.uniform(-int_unc, int_unc))
                for i in range(len(circle_data[0]))
            ]
        )
        sigma_xy = np.array(
            [
                sigma_xy_mean * (1 + r.uniform(-sig_unc, sig_unc))
                for i in range(len(circle_data[0]))
            ]
        )
        sigma_z = np.array(
            [
                sigma_z_mean * (1 + r.uniform(-sig_unc, sig_unc))
                for i in range(len(circle_data[0]))
            ]
        )

        # put coordinates, intensity, sigma_xy, sigma_z data into one structure
        circle_data = np.concatenate(
            (
                [circle_data[0]],
                [circle_data[1]],
                [circle_data[2]],
                [intensity],
                [sigma_xy],
                [sigma_z],
            ),
            axis=0,
        )

        # This function breaks the data into "chunks" for efficiency,
        # then uses it to 'fill up' the empty image array:
        mtubs = image_of_gaussians(circle_data, size_img, n_chunks, overlap)

        # normalise all the brightness values
        # then scale them up so that the brightest value is 255:
        # mtubs = mtubs - np.mean(mtubs) / np.std(mtubs)
        # mtubs = (mtubs / np.amax(mtubs)) * 255
        # print(np.amax(mtubs))
        # problem with above method: results in different scaling
        # for lowres and hires images

        # alternatively: scale according to the z-projection
        # which should be the same for both
        # z_projection = data.sum(2).sum(1)
        # TODO - finish this idea!

        # tiff writing in python gets the axes wrong
        # rotate the image before writing so it doesn't!
        mtubs = np.rot90(mtubs, 1, [0, 2])
        mtubs = np.rot90(mtubs, 1, [1, 2])

        # write to file
        # isotropic version:
        if idx == 0:
            filename_ind = filename + str(i + 1) + "_hires.tif"
            file_path = os.path.join(path_hires, filename_ind)
            print(f"Writing to tiff: hires {i + 1}")
        # anisotropic version
        elif idx == 1:
            filename_ind = filename + str(i + 1) + "_lores.tif"
            file_path = os.path.join(path_lores, filename_ind)
            print(f"Writing to tiff: lores {i + 1}")

        imwrite(file_path, mtubs.astype(np.uint16))

time2 = time()

print(f"The image size, in voxels, is {size_img}")
print(f"The overlap, in voxels, is {overlap}")
print(f"The number of total steps is {t}")
print(f"the mean xy-sigma is {sigma_xy_mean} pixels, or {sigma_xy_mean * size_pix_nm}nm")
print(f"the mean z-sigma is {sigma_z_mean} pixels, or {sigma_z_mean * size_pix_nm}nm")
print(f"made {2 * n_imgs}x16-bit tiffs with chunk dims {n_chunks}")
print(f"time taken: {time2 - time1} seconds")
