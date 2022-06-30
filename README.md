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
  - [ ] train network on them
    * issue: cannot calculate the fourier loss because the z-spectrum has less dimensions for undersampled data making
     $loss = (y-spectrum + x-spectrum) - (2 * z-spectrum)$
     impossible
       + solution 1: use cubic crops of the image
       + solution 2: bilinear interpolation of z-spectrum
        * progress made: figured out how to do a bilinear interpolation
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
