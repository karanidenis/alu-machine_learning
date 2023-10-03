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
    output_shape = (m, h - kh + 1, w - kw + 1)
    convolved_images = np.zeros(output_shape)

    for i in range(m):
        for j in range(h - kh + 1):
            for k in range(w - kw + 1):
                patch = images[i, j:j + kh, k:k + kw]
                convolved_images[i, j, k] = np.sum(patch * kernel)
                print(convolved_images)

    return convolved_images
