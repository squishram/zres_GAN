**__AI For Forging Isotropic Resolution for 3D Micrographs (AFFIRM3D)__**

# User Section

## What is this?
  - AFFIRM3D is a generative adversarial network (GAN) for improving the z-resolution of a 3D microscopy image until it matches its x/y-resolution
  - In theory, AFFIRM3D will be suitable for use on all kinds of anisotropic images, including SMLM and confocal images
  - It will be trained using both simulated and experimental data

## FAQ
### AIs for use in micrograph augmentation are typically biased by the training process to search for particular structures. How does AFFIRM3D overcome this?
   - AFFIRM3D's loss function is heavily weighted in favour of what we call 'The Fourier Loss'
   - The Fourier Loss is calculated as the difference between the (high-pass filtered) fourier-transformed z-projection (the 'z-spectrum') and its xy counterparts
   - When training on experimental data, The Fourier Loss is the *only* loss used to train the generator network
### Surely there needs to be some sensitivity to the actual locations of the structures in the training process?
   - The network undergoes a 'pre-training' process using simulated data, where the actual pixel values of the ground truth are also used to obtain the loss for training the network
   - Importantly, The Fourier Loss is still much more heavily weighted than the signal space loss


# Developer Section

## TODO

### preliminary: seeing the effect of the microtubule density
  - [x] simulate high-density microtubules
  - [x] simulate low-density microtubules
  - [x] test on cubic (non-undersampled) data, see what difference it makes!
    * low-density microtubules didn't really work

### 1: training on data with undersampled z
  - [x] simulate undersampled
  - [x] interpolate z-spectrum so it can be taken away from the xy-spectrum
  - [x] fix <0 interpolated values to 0 to get rid of NaNs
  - [x] offset pixel base values in image by a set value to get rid of this - try an increase of +100 to pixel values
  - [x] try a monotonic cubic interpolator because cubic spline ones tend to overshoot and get negative values
  - [x] monotonic cubic interpolators seem to miss the, ah, finer details of the spectrum - maybe try OVER-sampling it, and then downsampling it back down to the level of the others?
  - [ ] this doesn't seem to make any major difference - perhaps it's because the loss is calculated from $mean(x-spectrum, y-spectrum)$ - maybe the loss should be calculated from them separately?
  - [ ] would it be worth upsampling ALL the spectra? would this give the loss a finer attention to detail?
  - [ ] pass some of my data through it and see what happens

### 2: getting some appropriate data
  - [x] simulate microtubules (undersampled z, with noise)
  - [ ] simulate mitochondria (or any other 2D structure) (undersampled z, with noise)
  - [ ] get my own 3D experimental data
  - [ ] Jonas Ries data
  - [ ] Double-Helix data
  - [ ] Biplane data
  - [ ] OpenCell data (Chan/Zucc database)

### 3: saving and retrieving models
  - [x] figure out how to do this (simple syntax)
  - [ ] figure out a training pattern
    1. train on simulated data, save model
    1. train on experimental data, save model
    1. test on simulated and experimental data
  - [ ] try it!

## Overview
  - training on simulated data seems like the best idea
  - varying the size of the z-psf across simulated data
  - but testing/verifying on real data is essential ofc
