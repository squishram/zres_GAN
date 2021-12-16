import gc
gc.collect()
import random as r
import numpy as np
from time import time
from tifffile import imsave

"""
To do:
- time the thing, optimise it
- change so that each PSF gets its own intensity/sigma/sigmaz value
"""
   

def rotation_matrix(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians. 
    axis = the axis of rotation, described as a 3D vector
    angle = the rotation in radians
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

def random_walk(n, t, size=np.array([32*5, 32*5, 32*5]), max_step=0.25, sharpest=2*np.pi, reinitialise=True):
    """
    Sets up a random walk in three dimensions:
    n = number of random walks
    t = number of steps taken on each walk
    size = dimensions of space in which walk takes place
    max_step = the size of each step in the walk
    sharpest = the sharpest turn between each step
    reinitialise = whether or not the walk is reinitialised at a random location when the walk leaves the space
    """
    
    # x, y, z will contain a list of all of the positions
    x = np.zeros((n, t))
    y = np.zeros((n, t))
    z = np.zeros((n, t))

    # unit vectors:
    i = np.array([1, 0, 0])
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    # this is the step along each axis
    step_size = max_step/(np.sqrt(3))

    for p in range(n):

        # random starting point:
        x[p, 0] = r.uniform(0, size[0])
        y[p, 0] = r.uniform(0, size[1])
        z[p, 0] = r.uniform(0, size[2])

        # random first step:
        v = np.random.uniform(-step_size, step_size, 3)
        # ensure it's consistent length
        v = (v*max_step)/np.linalg.norm(v)


        for q in range(1, t):
            
            # add the last step to the last position to get the new position:
            x[p, q] = x[p, q - 1] + v[0]
            y[p, q] = y[p, q - 1] + v[1]
            z[p, q] = z[p, q - 1] + v[2]
            
            # if the microtubule leaves the imaging area, just re-initialize it somewhere else so as not to waste computing power:
            if reinitialise==True and (((x[p, q] > (size[0] + 1)) or (x[p, q] < -1))
                or ((y[p, q] > (size[1] + 1)) or (y[p, q] < -1)) 
                or ((z[p, q] > (size[2] + 1)) or (z[p, q] < -1))):
            # new random starting point:
                x[p, q] = r.uniform(0, size[0])
                y[p, q] = r.uniform(0, size[1])
                z[p, q] = r.uniform(0, size[2])
                # new random first step:
                v = np.random.uniform(-step_size, step_size, 3)


            # if the microtubule is still within the box, its next step must be constrained so it is not too sharp
            else:
                # initialise random polar angle       
                theta = r.uniform(0, sharpest)
                # initialise random azimuthal angle
                phi = r.uniform(0, 2*np.pi)
                # make the vector unit length
                v = v/np.linalg.norm(v)

                # rotate v about the normal to the plane created by v and i, 
                # unless v is parallel to k, in which case rotate v about i
                if  np.dot(v, k) == 1:
                    axis = i
                else:
                    axis = np.cross(v, k)

                # find the polar rotation matrix about axis
                r_pol = rotation_matrix(axis, theta)
                # find the azimuth rotation matrix about v1
                r_azi = rotation_matrix(v, phi)
                
                # apply random rotations to create a random vector within an angle of phi 
                v = r_azi@r_pol@v
                
            # ensure step is consistent length:
            v = (v*max_step)/np.linalg.norm(v)
            
    return [x, y, z]        

def patch3D(coords, boxsize=3):
    """
    Build a box of side length boxsize around the selected array element, 
    and returns the indices of all the elements in that box in the form:
    [[x inds][y inds][z inds]]
    boxsize = the size of the box built around coords, dtype = odd integer
    coords = coordinate of central pixel in 3D, dtype = list/array
    """
    # patchsize must be an odd number
    boxend = int(boxsize - np.ceil(boxsize/2))

    coords = np.array(coords)

    inds = np.array([np.arange(coords[i] - boxend, coords[i] + boxend + 1) for i in range(3)])
    
    return inds

def gauss3D(coords, intensity, mean, sigma_xy, sigma_z):
    """
    - Finds the gaussian contribution to point coords from a gaussian with parameters: intensity, mean, sigma_xy, sigma_z
    - Used to simulate PSF/ location uncertainty of molecules
    """
    
    gauss3D = intensity*np.exp(-(((coords[0] - mean[0])**2)/(2*(sigma_xy**2)) + 
                                 ((coords[1] - mean[1])**2)/(2*(sigma_xy**2)) + 
                                 ((coords[2] - mean[2])**2)/(2*(sigma_z**2))))
    
    return gauss3D

def signal_contribution(coords, data, intensity, sigma_xy, sigma_z, boxsize=3):
    """
    make a function that obtains signal from nearby emitters
    coords = the indices of the pixel under investigation, dtype=list or numpy array
    boxsize = the size of the area around coords from which signal contributions are obtained, dtype = odd integer
    data = the location of all the emitters dtype=(list or numpy array) in format [[x] [y] [z]]
    intensity = the brightness of each emitter, dtype=float (to be replaced by a list or numpy array in future versions)
    sigma_xy = the spread of each emitter in xy, dtype=float (to be replaced by a list or numpy array in future versions)
    sigma_z  = the spread of each emitter in z,  dtype=float (to be replaced by a list or numpy array in future versions)
    """
    
    
    # make sure the pixel you are looking at is defined in coordinates of type==numpy array
    coords = np.array(coords)
    data = np.array(data)
    # patch3D() obtains the coordinates of the surrounding pixels within area boxsize
    patch = patch3D(coords, boxsize) 
    
    # remove all data that is not inside the box
    for j in range(3):
        data = np.delete(data, np.where((data[j, :] < np.amin(patch[j, :])) | (data[j, :] > np.amax(patch[j, :])))[0], axis=1)
        
    
    
    # remove all data that is not inside the box
    # count = len(data[0])
    # i = 0
    # while i < count:
    #     for j in range(3):
    #         # if the coordinate is not within the box, delete it!
    #         if data[j, i] < patch[:, j].min() or data[j, i] > patch[:, j].max():
    #             data = np.delete(data, i, axis=1)
    #     i += 1
    #     count = len(data[0])
    
    #if len(data[0]) > 0:
    #    print("shape is:\n", data.shape)

                    
    # initialize signal
    signal = 0
    # sum contributions from all voxels in the box to the central voxel (if there are any)
    if len(data[0]) > 0:
        for i in range(len(data[0])):
            mean = data[:, i]
            signal += gauss3D(coords, intensity, mean, sigma_xy, sigma_z)
    
    return signal

def plot2mat3D(data, size, intensity, sigma_xy, sigma_z, boxsize=3):
    """
    Builds an empty 3D image and applies signal_contribution() to each voxel
    data is the coordinates of each emitter in 3D, dtype = [[][][]]
    size is the size of the 3D image in voxels, dtype=[]
    A buffer zone of size boxend is built around the image to accomodate the pooling process
    """
    
    # the distance of the 'sides' of the box from the central voxel, in voxels
    boxend = int(boxsize - np.ceil(boxsize/2))
    # initialise empty array to contain the full image
    mtubs = np.zeros((size[0] + 2*boxend, size[1] + 2*boxend, size[2] + 2*boxend))

    for i in range(boxend, size[0] + boxend):
        for j in range(boxend, size[1] + boxend):
            for k in range(boxend, size[2] + boxend):
                coords = [i, j, k]
                # add all signal contributions to this voxel
                mtubs[i, j, k] += signal_contribution(coords, data, intensity, sigma_xy, sigma_z, boxsize)
            
    return mtubs


time1 = time()

# PSF specs: 
# What is the mean intensity (AU) and its uncertainty?
intensity = 5
int_unc = 0.2
# What is the mean sigma in xy (voxels?) and its uncertainty?
sigma_xy = 0.5
sig_unc = 0.2
# by what factor is the z axis PSF less precise?
sigfact = 3

# from what surrounding area do we retrieve psf data?
boxsize = 5

# file specs:
# how many images do you want?
nimg = 5
# file name root:
filename = "mtubs"
# bittage of final image - 8, 16, 32, or 64?
img_bit = 32

# image + random walk specs:
# number of random walks happening:
n = 3
# number of steps per walk:
t = 1000
# size of final image in pixels:
size = np.array([32*3, 32*3, 32*3])
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 1
# how sharply can the path bend each step?
sharpest = (np.pi*max_step)/(10)
# reinitialise walk in a random location when they leave the AOI?
reinitialise = True


# calculate the highest signal value for the image size (8bit, 16bit etc)
signal_max = 2**img_bit - 1
# calculate the mean sigma in z (in voxels)
sigma_z = sigfact*sigma_xy


for i in range(nimg):
    [x, y, z] = random_walk(n, t, size, max_step, sharpest, reinitialise)
        
    # columnate all of the datapoint coordinates (currently each walk has its own separate column)
    xf = x.flatten()
    yf = y.flatten()
    zf = z.flatten()

    # put them into a single array
    dataF = np.concatenate(([xf], [yf], [zf]), axis=0)
    dataF = np.array(dataF)
    

    # convert coordinates into array
    mtubs = plot2mat3D(dataF, size, boxsize, intensity, sigma_xy, sigma_z)
    # normalise all the brightness values, then scale them up to signal_max (dependent on bittage of final image)
    mtubs = mtubs*signal_max/np.amax(mtubs)

    # write to file
    filename_ind = filename + str(i) + ".tif"
    print("Writing to tiff: " + str(i))
    if img_bit == 8:
        imsave(filename_ind, mtubs.astype(np.uint8))
    elif img_bit == 16:
        imsave(filename_ind, mtubs.astype(np.uint16))
    elif img_bit == 32:
        imsave(filename_ind, mtubs.astype(np.uint32))
    elif img_bit == 64:
        imsave(filename_ind, mtubs.astype(np.uint64))

time2 = time()

print("to make" + str(nimg) + " images takes " + str(time2 - time1))

print("Done!")