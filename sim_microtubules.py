import random as r
import numpy as np
from time import time
from tifffile import imsave

"""
This code generates 3D images of simulated microtubules. Workflow:
1. Uses a 3D random walk with constant step sizes and limited 'turn sharpness'
to generate coordinates of simulated microtubules
2. Creates an empty 3 dimensional array
3. Uses a pooling-style approach to convert the random walk coordinates into
   array 'signal' values.
   This is done by convolving each coordinate with a gaussian
   to simulate the PSF/ position uncertainty.
   Signal is then pooled from a surrounding patch of pre-determined size
   into the central voxel
4. Scales up the signal to match the desired image bittage
and saves the final array as a tiff

NOTE To do:
- Optimise for speed?
- FIND OUT ABOUT 3D NORMALISATION SO THE PSF ISN'T TOO LARGE IN gauss3d
- try taking out the += and replace with just = in plot2mat3D

"""


def rotation_matrix(axis, angle):
    """
    Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians. 
    // axis = the axis of rotation, described as a 3D vector
    // angle = the rotation in radians
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
    rotmat = np.array([[x**2 + (y**2 + z**2)*c, x*y*(1 - c) - z*s, x*z*(1 - c) + y*s],
                       [x*y*(1 - c) + z*s, y**2 + (x**2 + z**2)*c, y*z*(1 - c) - x*s],
                       [x*z*(1 - c) - y*s, y*z*(1 - c) + x*s, z**2 + (x**2 + y**2)*c]])
    
    return rotmat


def random_walk(t, size, max_step=0.25, sharpest=2*np.pi, reinitialise=True):
    """
    Sets up a random walk in three dimensions:
    // t = number of steps taken on each walk, dtype = uint
    // size = dimensions of space in which walk takes place, presented as [xsize, ysize, zsize]. 
    Faster if fed in as a numpy array.
    // max_step = the size of each step in the walk, dtype = uint
    // sharpest = the sharpest turn between each step, dtype = float
    // reinitialise = whether or not the walk is reinitialised at a random location when it leaves the space (saves memory), dtype = bool
    """

    # x, y, z will contain a list of all of the positions
    x = np.zeros(t)
    y = np.zeros(t)
    z = np.zeros(t)

    # unit vectors:
    i = np.array([1, 0, 0])
    k = np.array([0, 0, 1])

    # this is the step along each axis
    step_size = max_step/(np.sqrt(3))

    # random starting point:
    x[0] = r.uniform(0, size[0])
    y[0] = r.uniform(0, size[1])
    z[0] = r.uniform(0, size[2])

    # random first step:
    v = np.random.uniform(-step_size, step_size, 3)
    # ensure it's the right length
    v = (v*max_step)/np.linalg.norm(v)

    for q in range(1, t):

        # add the last step to the last position to get the new position:
        x[q] = x[q - 1] + v[0]
        y[q] = y[q - 1] + v[1]
        z[q] = z[q - 1] + v[2]

        # if the microtubule leaves the imaging area
        # just re-initialize it somewhere else
        if reinitialise and (((x[q] > (size[0] + 1)) or (x[q] < -1))
                             or ((y[q] > (size[1] + 1)) or (y[q] < -1))
                             or ((z[q] > (size[2] + 1)) or (z[q] < -1))):
            # new random starting point:
            x[q] = r.uniform(0, size[0])
            y[q] = r.uniform(0, size[1])
            z[q] = r.uniform(0, size[2])
            # new random first step:
            v = np.random.uniform(-step_size, step_size, 3)

        # if the microtubule is still within the box
        # its next step must be constrained so it is not too sharp
        else:
            # initialise random polar angle
            theta = r.uniform(0, sharpest)
            # initialise random azimuthal angle
            phi = r.uniform(0, 2*np.pi)
            # make the vector unit length
            v = v/np.linalg.norm(v)

            # rotate v about the normal to the plane created by v and i
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
            v = r_azi@r_pol@v

        # ensure step is consistent length:
        v = (v*max_step)/np.linalg.norm(v)

    return [x, y, z]


def patch3D(coords, boxsize=3):
    """
    Build a box of side length=boxsize around the selected array element, 
    and returns the indices of all the elements in that box in the form:
    [[x inds][y inds][z inds]]
    // boxsize = the size of the box built around coords, dtype = odd integer
    // coords = coordinate of central pixel in 3D, dtype = list/array
    """
    
    # patchsize must be an odd number
    boxend = int(boxsize - np.ceil(boxsize/2))
    # convert coordinates to numpy array for speed
    coords = np.array(coords)
    # obtain x, y, and z coordinates of the patch
    # N.B. these do NOT 'line up' as coordinates - they are merely lists of the x, y, and z coordinates that APPEAR in the box
    inds = np.array([np.arange(coords[i] - boxend, coords[i] + boxend + 1) for i in range(3)])
    
    return inds


def gauss3D(coords, intensity, mean, sigma_xy, sigma_z):
    """
    - Finds the gaussian contribution to point coords from a gaussian with parameters: intensity, mean, sigma_xy, sigma_z
    - Used to simulate PSF/ location uncertainty of molecules
    """
    
    
    gauss3D = (intensity/sigma_xy*np.sqrt(np.pi))*np.exp(-(((coords[0] - mean[0])**2)/(2*(sigma_xy**2)) + 
                                 ((coords[1] - mean[1])**2)/(2*(sigma_xy**2)) + 
                                 ((coords[2] - mean[2])**2)/(2*(sigma_z**2))))
    
    return gauss3D


def signal_contribution(coords, data, intensity, sigma_xy, sigma_z, boxsize=3):
    """
    Makes a function that obtains signal from nearby emitters
    // coords = the indices of the pixel under investigation, dtype=list or numpy array
    // data = the location of all the emitters, dtype=(list or numpy array) in format [[x] [y] [z]]
    // intensity = the brightness of each emitter, dtype=(list or numpy array)
    // sigma_xy = the spread of each emitter in xy, dtype=(list or numpy array)
    // sigma_z  = the spread of each emitter in z,  dtype=(list or numpy array)
    // boxsize = the size of the area around coords from which signal contributions are obtained, dtype = odd integer

    """
        
    # make sure all lists are defined as numpy arrays to make everything faster
    coords = np.array(coords)
    data = np.array(data)
    intensity = np.array(intensity)
    sigma_xy = np.array(sigma_xy)
    sigma_z = np.array(sigma_z)
    
    # patch3D() obtains the coordinates of the surrounding pixels within area boxsize
    patch = patch3D(coords, boxsize) 
    
    # remove all data that is not inside the box
    for j in range(3):
        # Each loop is a different dimension x, y, z. So, on the first loop, the next line works like so:
        # delete all COLUMNS (this is why 'axis=1') (each column is a single [[x] [y] [z]] coordinate)
        # for which the x value in data is smaller than min(patch[x coordinates]) or larger than max(patch[x coordinates])
        data = np.delete(data, np.where((data[j, :] < np.amin(patch[j, :])) | (data[j, :] > np.amax(patch[j, :])))[0], axis=1)
                           
    # initialize signal
    signal = 0
    # sum contributions from all voxels in the box to the central voxel (if there are any)
    if len(data[0]) > 0:
        for i in range(len(data[0])):
            mean = data[:, i]
            signal += gauss3D(coords, intensity[i], mean, sigma_xy[i], sigma_z[i])
    
    return signal


def plot2mat3D(data, size, intensity, sigma_xy, sigma_z, boxsize=3):
    """
    Builds an empty 3D image and applies signal_contribution() to each voxel
    data is the coordinates of each emitter in 3D, dtype = [[xcoords] [ycoords] [zcoords]]
    // size is the size of the 3D image in voxels, dtype=list/array
    // data = the location of all the emitters, dtype=(list or numpy array) in format [[x] [y] [z]]
    // intensity = the brightness of each emitter, dtype=(list or numpy array)
    // sigma_xy = the spread of each emitter in xy, dtype=(list or numpy array)
    // sigma_z  = the spread of each emitter in z,  dtype=(list or numpy array)
    // boxsize = the size of the area around coords from which signal contributions are obtained, dtype = odd integer

    """
    
    # the distance of the 'sides' of the box from the central voxel, in voxels
    boxend = int(boxsize - np.ceil(boxsize/2))
    # initialise empty array to contain the full image + buffer zone
    mtubs = np.empty((size[0] + 2*boxend, size[1] + 2*boxend, size[2] + 2*boxend))
    
    # iterating over every voxel
    for i in range(boxend, size[0]):
        for j in range(boxend, size[1]):
            for k in range(boxend, size[2]):
                coords = [i, j, k]
                # add all signal contributions from surrounding voxels to voxel [i, j, k]
                mtubs[i, j, k] = signal_contribution(coords, data, intensity, sigma_xy, sigma_z, boxsize)
            
    return mtubs


# Initialize timer
time1 = time()

# file specs:
# how many images do you want?
nimg = 1
# file name root:
filename = "sim_mtubs"
# bit depth of final image - 8, 16, or 32?
img_bit = 32

# PSF specs:
# What is the mean intensity (AU) and its uncertainty
# (as a proportion of mean intensity)?
intensity_mean = 5
int_unc = 0.2
# What is the mean sigma in xy (voxels?) and the sigma uncertainty
# (as a proportion of mean intensity)?
sigma_xy_mean = 0.5
sig_unc = 0.2
# by what factor is the z axis PSF less precise?
# !!!(replace this with an absolute value for sigma_z_mean)
sigfact = 3
# from what surrounding area do we retrieve psf data?
boxsize = 5
# We are splitting up the data into 'chunks' to help find ROIs more efficiently
# how many chunks per dimension?
# (this will be cubed to get the total number of chunks)
n_chunks = 2

# image + random walk specs:
# number of total random walk 'steps' in the final image
t = 2000
# size of final image in pixels:
size = np.array([100, 100, 30])
# step size in pixels each iteration
# (make it <0.5 if you want continuous microtubules):
max_step = 1
# how sharply can the path bend each step?
sharpest = (np.pi*max_step)/(10)
# reinitialise walk in a random location when they leave the AOI?
reinitialise = True

# calculate the mean sigma in z (in voxels)
sigma_z_mean = sigfact*sigma_xy_mean


for i in range(nimg):
    [x, y, z] = random_walk(t, size, max_step, sharpest, reinitialise)

    # broadcast intensity & sigma values into arrays with added random values
    # so each emitter gets its own ones:
    intensity = np.array([intensity_mean*(1 + r.uniform(-int_unc, int_unc))
                         for i in range(len(x))])
    sigma_xy = np.array([sigma_xy_mean*(1 + r.uniform(-sig_unc, sig_unc))
                        for i in range(len(x))])
    sigma_z = np.array([sigma_z_mean*(1 + r.uniform(-sig_unc, sig_unc))
                       for i in range(len(x))])

    # put them into a single array
    data = np.concatenate(([x], [y], [z]), axis=0)
    data = np.array(data)

    # convert coordinates into array
    mtubs = plot2mat3D(data, size, intensity, sigma_xy, sigma_z, boxsize)
    # normalise all the brightness values
    # then scale them up so that the brightest value is 255
    mtubs = (mtubs/np.amax(mtubs))*255

    # tiff writing in python gets the axes wrong
    mtubs = np.rot90(mtubs, 1, [0, 2])
    mtubs = np.rot90(mtubs, 1, [1, 2])

    # write to file
    filename_ind = filename + str(i) + ".tif"
    print("Writing to tiff: " + str(i + 1))
    if img_bit == 8:
        imsave(filename_ind, mtubs.astype(np.uint8))
    elif img_bit == 16:
        imsave(filename_ind, mtubs.astype(np.uint16))
    elif img_bit == 32:
        imsave(filename_ind, mtubs.astype(np.uint32))

time2 = time()

print("The image size, in voxels, is " + str(size))
print("The patch size, in voxels, is " + str(boxsize))
print("The number of total steps is " + str(t))
print("Done! To make " + str(nimg) + " " + str(img_bit) +
      "-bit images with these parameters only took " +
      str(time2 - time1) + "seconds, a mere " +
      str((time2 - time1)/60) + " minutes - wow!")
