#!/usr/bin/env python3

"""
This module has the method that performs a valid
convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    perfom a concolution of a grayscale image
    images is a np.ndarray with shape(m, h, w)
    m - no. of images
    h - height in pixels of images
    w - width in pixels of images
    kernel is nmpy.ndarray with shape(kh, kw)
    kh - height of kernel
    kw - width of kernel
    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # # Calculate the output size for "valid" convolution
    # out_h = h - kh + 1
    # out_w = w - kw + 1

    # # Initialize the output array
    # convolved_images = np.zeros((m, out_h, out_w))

    # # Perform "valid" convolution with two for loops
    # for i in range(out_h):
    #     for j in range(out_w):
    #         # Extract a patch from the image
    #         patch = images[:, i:i + kh, j:j + kw]
    #         # Perform element-wise multiplication and sum
    #         convolved_images[:, i, j] = np.sum(patch * kernel, axis=(1, 2))
    #         print(convolved_images)

    # return convolved_images

    ph = int(kh / 2)
    pw = int(kw / 2)
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convoluted = np.zeros((m, h, w))
    for h in range(h):
        for w in range(w):
            square = padded[:, h: h + kh, w: w + kw]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
