import gc
gc.collect()
import random as r
import numpy as np
import imageio
import time
"""
time the thing, optimise it
convolve every point with a gaussian that is 3x larger in z
"""

def rotation_matrix(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by 'angle' radians. 
    Why the hell doesn't this already exist in numpy? It's the general case of a rotation matrix.
    Bizarre.
    axis = the vector we're rotating around
    angle = the amount of rotation in radians
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


            # if the microtubule is still within the box, it the next step must be constrained so it is not too sharp
            else:
                # initialise random polar angle       
                theta = r.uniform(0, sharpest)
                # initialise random azimuthal angle
                phi = r.uniform(0, 2*np.pi)
                # make the vector unit length
                v = v/np.linalg.norm(v)

                # rotate v in the plane created by v and i, unless v is parallel to k, then use the plane normal to i
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
def plot2mat3D(dataR, size):
    """
    Converts 3D cartesian coordinates into 3D matrix indices and sets the value of those indices to 1.
    Returns the matrix
    """
    sz = np.shape(dataR)
    ### turning a plot into an image
    mtubs = np.zeros((size[0], size[1], size[2]))
    mtubs = mtubs.astype(int)

    for m in range(sz[1]):
        ind = dataR[:, m]
        if ind[0] < size[0] and ind[1] < size[1] and ind[2] < size[2] and (ind > 0).all():
            mtubs[ind[0], ind[1], ind[2]] = 1
            
    return mtubs
def conv_PSF():
    """
    make a function that convolves with psf before converting into matrix
    """

# how many images do you want?
nimg = 2
# file name root:
filename = "mtubs"
# number of random walks happening:
n = 20
# number of steps per walk:
t = 1000
# size of final image in pixels:
size = np.array([32*5, 32*5, 32*5])
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 0.25
# how sharply can the path bend each step?
sharpest = (np.pi*max_step)/(10)
# reinitialise walk in a random location when they leave the AOI?
reinitialise = True



for i in range(nimg):
    [x, y, z] = random_walk(n, t, size, max_step, sharpest, reinitialise)
        
    # columnate all of the datapoint coordinates (currently each walk has its own separate column)
    xf = x.flatten()
    yf = y.flatten()
    zf = z.flatten()

    # put them into a single array
    dataF = np.concatenate(([xf], [yf], [zf]), axis=0)
    # round them all down (as they will become indices in the array/image)
    dataR = np.floor(dataF)
    dataR = dataR.astype(int)
    # get the size of the array
    sz = np.shape(dataR)

    # convert coordinates into array
    mtubs = plot2mat3D(dataR, size)

    filename_ind = filename + str(i) + ".tif"
    print("Writing to tiff: " + str(i))
    imageio.volwrite(filename_ind, mtubs.astype(np.int32))

print("Done!")