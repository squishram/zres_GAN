import gc
gc.collect()
import random as r
import numpy as np
from time import time
from tifffile import imsave
import torch

"""
This code generates 3D images of simulated microtubules. Workflow:
1. Uses a 3D random walk with constant step sizes and limited 'turn sharpness' to generate coordinates
2. Creates an empty 3 dimensional array
3. Uses a pooling-style approach to convert the random walk coordinates into array 'signal' values.
   This is done by convolving each coordinate with a gaussian to simulate the PSF. 
   Signal is then pooled from a surrounding patch of pre-determined size into the central voxel
4. Scales up the signal to match the desired image bittage and saves the final array as a tiff

To do:
- Convert everything from numpy to torch 
- Annotate the code in full, streamline where possible
- Find the optimal number of chunks for speed
- Set it up so it takes in varying sig/int values (from the list) - DONE

Questions for Susan:
- Is it actually a good idea to move everything from numpy to torch, or are we better off converting at the end?
As I understand it, numpy is known to be a fair bit faster for calculations on large arrays (~1.5x faster for arrays with >10^4 elements)
- As I understand it, this code pools instensity values in from every single point in each chunk. Doesn't this mean we'll get edge effects?
- I can't find the documentation on .expand() online. 
How do I cange this part of the code (lines ~224-229) to accomodate non-perfectly cuboidal chunks?

NOTE: cursory testing found 5^3 chunks for 96^3 voxel image to be fastest (faster than 4 chunks and 6 chunks for the same data)
This translates to a (ROUGHLY) optimal vox/chunk ratio of 19, so this has been incoorporated into the code
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

def random_walk(t, size_img, max_step=0.25, sharpest=np.pi):
    """
    Sets up a random walk in three dimensions:
    // t = number of steps taken on each walk, dtype = uint
    // size_img = dimensions of space in which walk takes place, presented as [xsize, ysize, zsize]. 
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
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    # this is the step along each axis
    step_size = max_step/(np.sqrt(3))

    
    # random starting point:
    x[0] = r.uniform(0, size_img[0])
    y[0] = r.uniform(0, size_img[1])
    z[0] = r.uniform(0, size_img[2])

    # random first step:
    v = np.random.uniform(-step_size, step_size, 3)
    # ensure it's the right length
    v = (v*max_step)/np.linalg.norm(v)


    for q in range(1, t):
        
        # add the last step to the last position to get the new position:
        x[q] = x[q - 1] + v[0]
        y[q] = y[q - 1] + v[1]
        z[q] = z[q - 1] + v[2]
        
        # if the microtubule leaves the imaging area, just re-initialize it somewhere else so as not to waste computing power:
        if (((x[q] > (size_img[0] + 1)) or (x[q] < -1))
            or ((y[q] > (size_img[1] + 1)) or (y[q] < -1)) 
            or ((z[q] > (size_img[2] + 1)) or (z[q] < -1))):
        # new random starting point:
            x[q] = r.uniform(0, size_img[0])
            y[q] = r.uniform(0, size_img[1])
            z[q] = r.uniform(0, size_img[2])
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
    
    data = np.concatenate(([x], [y], [z]), axis=0)
    data = np.array(data)
            
    return data

def patch3D(coords, size_patch=5):
    """
    Build a box of side length=boxsize around the selected array element, 
    and returns the indices of all the elements in that box in the form:
    [[x inds][y inds][z inds]]
    // boxsize = the size of the box built around coords, dtype = odd integer
    // coords = coordinate of central pixel in 3D, dtype = list/array
    """
    
    # patchsize must be an odd number
    boxend = int(size_patch - np.ceil(size_patch/2))
    # convert coordinates to numpy array for speed
    coords = np.array(coords)
    # obtain x, y, and z coordinates of the patch
    # N.B. these do NOT 'line up' as coordinates - they are merely lists of the x, y, and z coordinates that APPEAR in the box
    inds = np.array([np.arange(coords[i] - boxend, coords[i] + boxend + 1) for i in range(3)])
    
    return inds

def image_of_gaussians(data, size_img, size_patch=5):
    """
    NOTE: NEEDS TO ADJUST CHUNK SIZES IF IMAGE SIZE IS NOT PERFECT CUBE
    Breaks up coordinate data into 3D chunks to decrease runtime,
    Retrieves gaussian contributions to each pixel using coordinate, intensity, and sigma data
    outputs n_chunks arrays each with the data that could contribute to chunk n_chunks. 
    // data is the coordinates of the points that will be made into an image AND their intensity, sigma_xy and sigma_z.
    It is an array with dimensions (6, n_points)
    // size_img is the dimensions of the final image, tuple (size_x, size_y, size_z)
    // n_chunks should be a 3 element vector containing x, y, z chunking values respectively
    //
    """
    
    # this output will contain the final image with illuminated pixels
    img = torch.zeros(tuple(size_img))

    # the size of each chunk
    # // is 'floor division' i.e. divide then round down the result to the nearest int
    size_patch = np.array([size_img[i]//n_chunks[i] for i in range(3)])

    # make an object array to contain all the data inside each chunk
    # (roughly equivalent to matlab cell array, each object/unit/cell can contain anything i.e. an array of any size)
    chunked_data = np.empty([n_chunks[0], n_chunks[1], n_chunks[2]], dtype=object)
    
    # assign an empty array as each object in chunked_data
    for x in range(n_chunks[0]):
        for y in range(n_chunks[1]):
            for z in range(n_chunks[2]):
                    chunked_data[x][y][z] = []
    
    # Now load up those empty arrays with the data (only the data in each chunk)!                
    # This loop goes through all the points in data, and loads up the empty arrays in chunked_data with them as appropriate
    for j in range(len(data[0])):
        # don't need to do this can use the position to do the calculaion I think (see next loops)
        for x in range(n_chunks[0]):
            xstart = (size_img[0]*x)//n_chunks[0]
            for y in range(n_chunks[1]):
                ystart = (size_img[1]*y)//n_chunks[1]
                for z in range(n_chunks[2]):
                    zstart = (size_img[2]*z)//n_chunks[2]
                    
                    # if the point is inside the chunk, append it to that chunk!
                    if ((data[0][j] >= xstart - overlap and data[0][j] < (xstart + size_patch[0] + overlap)) and
                        (data[1][j] >= ystart - overlap and data[1][j] < (ystart + size_patch[1] + overlap)) and
                        (data[2][j] >= zstart - overlap and data[2][j] < (zstart + size_patch[2] + overlap))):
                        # edited to include the sigma & intensity information
                        chunked_data[x][y][z].append([data[0][j], data[1][j], data[2][j], data[3][j], data[4][j], data[5][j]])


    # NOTE this needs to be fixed to accomodate potentially varying sizes of chunk
    # creates a matrix of indices for each dimension (x, y, and z)
    N = torch.tensor(range(size_patch[0]))
    zi = N.expand(size_patch[0], size_patch[1], size_patch[2])
    xi = zi.transpose(0, 2)
    yi = zi.transpose(1, 2)
    
    
    # This loop calculates the contributions from each local gaussian to each chunk
    for x in range(n_chunks[0]):
        xstart = (size_img[0]*x)//n_chunks[0]
        for y in range(n_chunks[1]):
            ystart = (size_img[1]*y)//n_chunks[1]
            for z in range(n_chunks[2]):
                zstart = (size_img[2]*z)//n_chunks[2]

                intensityspot = torch.zeros((size_patch[0], size_patch[1], size_patch[2]))
                
                # NOTE do we want to include a patch calculation here?

                for cx, cy, cz, cintensity, csig_xy, csig_z in data:
                    # define the normalisation constant for the gaussian
                    const_norm = cintensity/((csig_xy**3)*(2*np.pi)**1.5)
                    # add the gaussian contribution to the spot
                    intensityspot += const_norm*torch.exp(-(((xi + xstart - cx)**2)/(2*csig_xy**2)
                                                        +   ((yi + ystart - cy)**2)/(2*csig_xy**2)
                                                        +   ((zi + zstart - cz)**2)/(2*csig_z **2)))
                    
                
                img[xstart:xstart + size_patch[0], ystart:ystart + size_patch[1], zstart:zstart + size_patch[2]] = intensityspot
    
    return np.array(img)

    # note assumes cuboidal data prob not correct
    # get all pixels corresponding to chunk + 4 sigma
    # BEWARE EDGE EFFECTS
    # this assumes all chunk dimenstions same size
    # find which data points are inside that chunk


# Initialize timer
time1 = time()

# file specs:
# how many images do you want?
nimg = 1
# file name root:
filename = "mtubs_sim_"
# bittage of final image - 8, 16, 32, or 64?
img_bit = 32

# image + random walk specs:
# number of steps per walk:
t = 2000
# size of final image in pixels:
size_img = np.array([96, 96, 96])
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 0.5
# how sharply can the path bend each step?
sharpest = (np.pi*max_step)/10

# PSF specs: 
# What is the mean intensity (in AU) and its uncertainty (as a fraction of the mean value)?
intensity_mean = 5
int_unc = 0.2
# What is the mean sigma in xy (in voxels) and the sigma uncertainty (as a fraction of the mean value)?
sigma_xy_mean = 1
sig_unc = 0.2
# What is the mean sigma in z (in voxels)
sigma_z_mean = 3*sigma_xy_mean

# how many chunks are we splitting the data into along each dimension? (found to be 5 for 96x96x96 voxels)
n_chunks = np.array([5, 5, 5])
# how much do the chunks overlap?
overlap = 7*sigma_xy_mean


for i in range(nimg):
    data = random_walk(t, size_img, max_step, sharpest)

    # broadcast intensity & sigma values into distributed arrays
    intensity = np.array([intensity_mean*(1 + r.uniform(-int_unc, int_unc)) for i in range(len(data[0]))])
    sigma_xy  = np.array([sigma_xy_mean *(1 + r.uniform(-sig_unc, sig_unc)) for i in range(len(data[0]))])
    sigma_z   = np.array([sigma_z_mean  *(1 + r.uniform(-sig_unc, sig_unc)) for i in range(len(data[0]))])

    # put all of the coordinates, intensity, sigma_xy, and sigma_z data into a single structure    
    data = np.concatenate(([data[0]], [data[1]], [data[2]], [intensity], [sigma_xy], [sigma_z]), axis=0)
    data = np.array(data)

    # This function breaks the data into "chunks" for efficiency, then uses it to 'fill up' the empty image array:
    mtubs = image_of_gaussians(data, size_img, n_chunks, overlap)

    # normalise all the brightness values, then scale them up so that the brightest value is 255:
    mtubs = (mtubs/np.amax(mtubs))*255
        
    # tiff writing in python gets the axes wrong; rotate the image before writing so it doesn't!
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

print("The image size, in voxels, is " + str(size_img))
print("The overlap, in voxels, is " + str(overlap))
print("The number of total steps is " + str(t))
print("the mean xy-sigma is " + str(sigma_xy_mean) + ", and the mean z-sigma is " + str(sigma_z_mean))
print("Done! To make " + str(nimg) + " " + str(img_bit) + "-bit images with " + str(n_chunks[0]) + " chunks/axis only took " +
      str(time2 - time1) + " seconds - wow!")
    

