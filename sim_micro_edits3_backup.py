import gc
gc.collect()
import random as r
import numpy as np
from time import time
from tifffile import imsave
import numpy

"""
This code generates 3D images of simulated microtubules. Workflow:
1. Uses a 3D random walk with constant step sizes and limited 'turn sharpness' to generate coordinates
2. Creates an empty 3 dimensional array
3. Uses a pooling-style approach to convert the random walk coordinates into array 'signal' values.
   This is done by convolving each coordinate with a gaussian to simulate the PSF. 
   Signal is then pooled from a surrounding patch of pre-determined size into the central voxel
4. Scales up the signal to match the desired image bittage and saves the final array as a tiff

To do:
- Optimise for speed

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

def random_walk(n, t, size, max_step=0.25, sharpest=2*np.pi, reinitialise=True):
    """
    Sets up a random walk in three dimensions:
    // n = number of random walks, dtype = uint
    // t = number of steps taken on each walk, dtype = uint
    // size = dimensions of space in which walk takes place, presented as [xsize, ysize, zsize]. 
    Faster if fed in as a numpy array.
    // max_step = the size of each step in the walk, dtype = uint
    // sharpest = the sharpest turn between each step, dtype = float
    // reinitialise = whether or not the walk is reinitialised at a random location when it leaves the space (saves memory), dtype = bool
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
    
    #I think this is missing a factor out the front no?
    gauss3D = intensity*np.exp(-(((coords[0] - mean[0])**2)/(2*(sigma_xy**2)) + 
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
        
    # make all lists are defined as numpy arrays to make everything faster
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
    # initialise empty array to contain the full image
    mtubs = np.zeros((size[0] + 2*boxend, size[1] + 2*boxend, size[2] + 2*boxend))

    for i in range(boxend, size[0]):
        for j in range(boxend, size[1]):
            for k in range(boxend, size[2]):
                coords = [i, j, k]
                # add all signal contributions from surrounding voxels to voxel [i, j, k]
                #try taking out +
                mtubs[i, j, k] += signal_contribution(coords, data, intensity, sigma_xy, sigma_z, boxsize)
            
    return mtubs

def chunkify(data, size, chunknum, sigmaxy, sigmaz):
    #chunknum should be a 3 element vector containing x, y, z chunking values respectively
    #chunk data into bits
    #output chunknum arrays each with the data that could contribute to chunk chunknum
    #to start with let's assume we want to chunk by same factor in x,y, and z directions - correct later
    #contribution distance cuttoff depends on sigma so these need to be input too
    chunkedxstart = []
    chunkedystart = []
    chunkedzstart = []
    chunkeddata = []
    currentchunk = 0
    #go through all chunks
    for x in range(chunknum):
        for y in range(chunknum):
            for z in range(chunknum):
                currentchunk += 1
                #I never actually use i do I want it?
                currentchunkdata = []
               

                chunksize = size[0]//chunknum
                xstart = (size[0]*x)//chunknum
                ystart = (size[1]*y)//chunknum
                zstart = (size[2]*z)//chunknum
                chunkedxstart.append(xstart)
                chunkedystart.append(ystart)
                chunkedzstart.append(zstart)
                #note this currently doesn't go out beyone the chunk to 4 sigma needs correcting!!!
                for j in range(len(data[0])):
                    if data[0][j] >= xstart and data[0][j] < (xstart + chunksize) and data[1][j] >= ystart and data[1][j] < (ystart + chunksize) and data[2][j] >= zstart and data[2][j] < (zstart + chunksize):
                        currentchunkdata.append([data[0][j],data[1][j],data[2][j]])
                chunkeddata.append(currentchunkdata)
            
            

    #get all pixels corresponding to chunk + 4 sigma
    #BEWARE EDGE EFFECTS
    #this assumes all chunk dimenstions same size
    #find which data points are inside that chunk
    
    return [chunkedxstart, chunkedystart, chunkedzstart, chunksize, chunkeddata]

def chunkandconv(data,size,chunknum,intensity,sigmaxy,sigmaz):
    #chunknum should be a 3 element vector containing x, y, z chunking values respectively
    #chunk data into bits
    #output chunknum arrays each with the data that could contribute to chunk chunknum
    #to start with let's assume we want to chunk by same factor in x,y, and z directions - correct later
    #contribution distance cuttoff depends on sigma so these need to be input too
    convimage = numpy.zeros(size)

    chunksize = size[0]//chunknum

    intensity = 100.0
    sigma_xy = 1.0
    sigma_z = 1.0

    chunkeddata = numpy.empty([chunknum,chunknum,chunknum],dtype=numpy.object)
    
    print("I'm so slow")

    for x in range(chunknum):
        for y in range(chunknum):
            for z in range(chunknum):
                    chunkeddata[x][y][z] = []
                    
    print("I suck")

    #go through all the points
    for j in range(len(data[0])):   
        print("j ",j)
        #go through all chunks
        #don't need to do this can use the position to do the calculaion I think (see next loops)
        for x in range(chunknum):
            xstart = (size[0]*x)//chunknum
            for y in range(chunknum):
                ystart = (size[1]*y)//chunknum
                for z in range(chunknum):
                    zstart = (size[2]*z)//chunknum
                    #note this currently doesn't go out beyone the chunk to 4 sigma needs correcting!!!
                    if data[0][j] >= xstart and data[0][j] < (xstart + chunksize) and data[1][j] >= ystart and data[1][j] < (ystart + chunksize) and data[2][j] >= zstart and data[2][j] < (zstart + chunksize):
                        chunkeddata[x][y][z].append([data[0][j],data[1][j],data[2][j]])


    #for x in range(chunknum):
     #   for y in range(chunknum):
      #      for z in range(chunknum):


    print("so much")
    #note assumes cuboidal data prob not correct
    for xi in range(size[0]):
        print("I mean really suck",xi)
        for yi in range(size[1]):
            for zi in range(size[2]):
                pixelchunkx = xi//chunksize
                pixelchunky = yi//chunksize
                pixelchunkz = zi//chunksize

                #NOTE need to check if this is the same as chunkcounter
                for s in range(len(chunkeddata[pixelchunkx,pixelchunky,pixelchunkz])):
                    #NOTE in this function coords is the centre of the current pixel and mean is the position of the spot
                    #NOTE should we calculate from centre of pixels at 0.5,0.5? probably
                    convimage[xi][yi][zi] += gauss3D([xi,yi,zi], intensity, chunkeddata[pixelchunkx][pixelchunky][pixelchunkz][s], sigma_xy, sigma_z)           
            


    return convimage
    #get all pixels corresponding to chunk + 4 sigma
    #BEWARE EDGE EFFECTS
    #this assumes all chunk dimenstions same size
    #find which data points are inside that chunk
    
    #return [convdata]


# Initialize timer
time1 = time()

# file specs:
# how many images do you want?
nimg = 1
# file name root:
filename = "sim_mtubs"
# bittage of final image - 8, 16, 32, or 64?
img_bit = 32

# PSF specs: 
# What is the mean intensity (AU) and its uncertainty?
intensity_mean = 5
int_unc = 0.2
# What is the mean sigma in xy (voxels?) and the sigma uncertainty?
sigma_xy_mean = 0.5
sig_unc = 0.2
# by what factor is the z axis PSF less precise?
sigfact = 10

# from what surrounding area do we retrieve psf data?
boxsize = 5

# image + random walk specs:
# number of random walks happening:
n = 2
# number of steps per walk:
t = 1000
# size of final image in pixels:
size = np.array([32*3, 32*3, 32*3])
#size = np.array([63, 63, 63])
# step size each iteration (make it <0.5 if you want continuous microtubules):
max_step = 1
# how sharply can the path bend each step?
sharpest = (np.pi*max_step)/(10)
# reinitialise walk in a random location when they leave the AOI?
reinitialise = True


# calculate the highest signal value for the image size (8bit, 16bit etc)
# calculate the mean sigma in z (in voxels)
sigma_z_mean = sigfact*sigma_xy_mean


for i in range(nimg):
    [x, y, z] = random_walk(n, t, size, max_step, sharpest, reinitialise)
        
    time1a = time()
    # columnate all of the datapoint coordinates (currently each walk has its own separate column)
    xf = x.flatten()
    yf = y.flatten()
    zf = z.flatten()
    
    time1b = time()
    # broadcast intensity & sigma values into distributed arrays
    intensity = np.array([intensity_mean*(1 + + r.uniform(-int_unc, int_unc)) for i in range(len(xf))])
    sigma_xy = np.array([sigma_xy_mean*(1 + r.uniform(-sig_unc, sig_unc)) for i in range(len(xf))])
    sigma_z = np.array([sigma_z_mean*(1 + r.uniform(-sig_unc, sig_unc)) for i in range(len(xf))])
    
    time1c = time()
    # put them into a single array
    dataF = np.concatenate(([xf], [yf], [zf]), axis=0)
    dataF = np.array(dataF)
    print("these are the lengths of dataF")
    print(len(dataF))
    print(dataF.shape)
    sizeF = dataF.shape
    print(sizeF[0])
    print(sizeF[1])
    print("these are the maxima and minima ")
    print(min(dataF[0]))
    print(max(dataF[0]))
    print(min(dataF[1]))
    print(max(dataF[1]))
    print(min(dataF[2]))
    print(max(dataF[2]))



    print(size)
    [chunkx,chunky,chunkz,chunksize,stuff2] = chunkify(dataF,size,3,sigma_xy,sigma_z)

    print("this is the stuff from the chunkify function")
    print(chunkx,chunky,chunkz,chunksize,len(stuff2),len(stuff2[0]),len(stuff2[1]))
  
    q = 0
    for p in range(len(stuff2)):
        q = q+len(stuff2[p])
        #print(stuff2[p])
        print("********************************************************")

    time1d = time()

    print("this should be the same as the total number of points ",q)


    #def chunkandconv(data,size,chunknum,intensity,mean,sigmaxy,sigmaz):

    mtubs = chunkandconv(dataF,size,3,5,sigma_xy,sigma_z)

    #mtubs = plot2mat3D(dataF, size, intensity, sigma_xy, sigma_z, boxsize)
    # normalise all the brightness values, then scale them up to (dependent on bittage of final image)

    time1e = time() 
    # write to file
    filename_ind = filename + str(i) + ".tif"
    print("Writing to tiff: " + str(i + 1))
    if img_bit == 8:
        imsave(filename_ind, mtubs.astype(np.uint8))
    elif img_bit == 16:
        imsave(filename_ind, mtubs.astype(np.uint16))
    elif img_bit == 32:
        imsave(filename_ind, mtubs.astype(np.uint32))
    elif img_bit == 64:
        imsave(filename_ind, mtubs.astype(np.uint64))

time2 = time()

print("The image size, in voxels, is " + str(size))
print("The patch size, in voxels, is " + str(boxsize))
print("The number of total steps is " + str(n) + "*" + str(t))
print("Done! To make " + str(nimg) + " " + str(img_bit) + "-bit images with these parameters only took " +
      str(time2 - time1) + "seconds, a mere " + str((time2 - time1)/60) + " minutes - wow!")
print("intermediate times are ", str((time1a - time1)/60),  str((time1b - time1)/60),  str((time1c - time1)/60),  str((time1d - time1)/60),  str((time1e - time1)/60))
