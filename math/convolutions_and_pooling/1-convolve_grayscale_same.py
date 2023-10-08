#!/usr/bin/env python3

"""
This module has the method that performs a valid
convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    m, h, w = images.shape
    kh, kw = kernel.shape
    # # Calculate the padding required to maintain the same output size
    # padding_h = kh // 2
    # padding_w = kw // 2
    # padded_images = np.pad(
    #     images, ((0, 0), (padding_h, padding_h), (padding_w, padding_w)),
    #     mode='constant')

    # # Perform convolution
    # convolved_images = np.zeros((m, h, w))
    # for i in range(m):
    #     for j in range(h):
    #         for k in range(w):
    #             patch = padded_images[i, j:j + kh, k:k + kw]
    #             convolved_images[i, j, k] = np.sum(patch * kernel)
    #             print(convolved_images)

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
