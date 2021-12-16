#%%

# This code 
# 1. generates a 3D cylinder manifold using a scatter plot
# 2. rotates the cylinder a random amount along all 3 DoF
# 3. plots 'PSF' scatter distributions (gaussian distributed) at a random subselection of points from the manifold scatter
# CYLINDERS ONLY
# However, the dimensions of the cylinder, the density of plotted points on its surface, and
#          the parameters for the PSF scatters can be controlled in the INPUT section below


import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d


# CYLINDER PARAMETER INPUT
# -----------------------------------------------------------------------------

# circle radius:
r = 10
# cylinder height:
h = 30
# number of points on the circle perimeter:
Ncirc = 60


# GAUSSIAN PARAMETER INPUT
# -----------------------------------------------------------------------------

# number of fluorophores emitting
Nem = 8
# standard deviation:
sig = 1
# number of points to plot:
N = 100


# MAKING THE CYLINDER
# -----------------------------------------------------------------------------

# number of points on the circle radius:
Nrad = int(Ncirc/(2*np.pi))
# number of points for a filled circle:
Nfcirc = int(Nrad*Ncirc)
# number of points along cylinder side:
Nhcyl = int(Ncirc*(h/r))

# define the points along the circle radius:
r_pts = np.linspace(0, r, Nrad)
# rotation to create circle:
theta = np.linspace(0, 2*np.pi, Ncirc)


# parametric equation of circle:
x = r*np.cos(theta)
y = r*np.sin(theta)
# points along cylinder side
z = np.linspace(-h/2, h/2, Nhcyl)

# parametric equation of cylinder sides:
xs = np.zeros([Nhcyl*Ncirc])
ys = np.zeros([Nhcyl*Ncirc])
zs = np.zeros([Nhcyl*Ncirc])
for i in range(Nhcyl):
    xs[i*Ncirc:(i + 1)*Ncirc] = x
    ys[i*Ncirc:(i + 1)*Ncirc] = y
    zs[i*Ncirc:(i + 1)*Ncirc] = np.ones(Ncirc)*z[i]

# parametric equation of cylinder faces:
xf = np.zeros([Nfcirc])
yf = np.zeros([Nfcirc])
for i in range(Nrad):
    xf[i*Ncirc:(i + 1)*Ncirc] = r_pts[i]*np.cos(theta)
    yf[i*Ncirc:(i + 1)*Ncirc] = r_pts[i]*np.sin(theta)

# the cylinder faces will be plotted at the z-extrema:
xf = np.concatenate((xf, xf), axis=0)
yf = np.concatenate((yf, yf), axis=0)
zf = np.concatenate((-h/2*np.ones([Nfcirc]), h/2*np.ones([Nfcirc])), axis=0)


# concatenate all the data into single structure
x = np.concatenate((xf, xs), axis=0)
y = np.concatenate((yf, ys), axis=0)
z = np.concatenate((zf, zs), axis=0)
cyl = np.array([x, y, z])

# remove duplicates
cyl = np.unique(cyl, axis=1)

# Transpose for clarity
cyl = cyl.T


# define the axes in 3d space
fig = plt.figure()
ax = plt.axes(projection="3d")

# plot the points
ax.scatter(cyl[:, 0], cyl[:, 1], cyl[:, 2], color='r')

# set the axis limits
lims = [1.5*min(z), 1.5*max(z)]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_zlim(lims)

# label the axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# ROTATE THE CYLINDER
# -----------------------------------------------------------------------------

# obtain random rotations (in radians) for each DoF
phiX = np.pi*2*np.random.rand()
phiY = np.pi*2*np.random.rand()
phiZ = np.pi*2*np.random.rand()

# define rotation matrices:
Rx = np.array([[1, 0, 0],
               [0, np.cos(phiX), -np.sin(phiX)],
               [0, np.sin(phiX), np.cos(phiX)]])
Ry = np.array([[np.cos(phiY), 0, np.sin(phiY)],
               [0, 1, 0],
               [-np.sin(phiY), 0, np.cos(phiY)]])
Rz = np.array([[np.cos(phiZ), -np.sin(phiZ), 0],
               [np.sin(phiZ), np.cos(phiZ), 0],
               [0, 0, 1]])

# apply rotation:
cylR = np.zeros(cyl.shape)
for i in range(int(np.size(cyl)/3)):
    cylR[i, :] = Rx.dot(cyl[i, :])
    cylR[i, :] = Ry.dot(cylR[i, :])
    cylR[i, :] = Rz.dot(cylR[i, :])


# define the axes in 3d space:
fig = plt.figure()
ax = plt.axes(projection="3d")

# plot the points:
ax.scatter(cylR[:, 0], cylR[:, 1], cylR[:, 2], color='r')

# set the axis limits:
lims = [1.5*min(z), 1.5*max(z)]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_zlim(lims)

# label the axes:
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# INSERT GAUSSIANS!
# -----------------------------------------------------------------------------

# select random points from cylinder to be the emitters
# random integer matrix to serve as indices:
rows = np.random.randint(np.size(cylR)/3, size=Nem)
# use idx to extract coordinates of those random points:
emitters = cylR[rows, :]

# define the axes in 3d space:
fig = plt.figure()
ax = plt.axes(projection="3d")

# PSF distributions at each emitter:
PSFs = np.zeros([N, 3, Nem])
for i in range(Nem):
    # x, y & z are distributed normally (where sig_z = 2sig_x = 2sig_y):
    PSFs[:, 0, i] = np.random.normal(emitters[i, 0], sig, N)
    PSFs[:, 1, i] = np.random.normal(emitters[i, 1], sig, N)
    PSFs[:, 2, i] = np.random.normal(emitters[i, 2], 2*sig, N)
    # plot the points:
    ax.scatter(PSFs[:, 0, i], PSFs[:, 1, i], PSFs[:, 2, i], color='b')

# set the limits:
lims = [1.5*min(z), 1.5*max(z)]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_zlim(lims)

# label the axes:
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
