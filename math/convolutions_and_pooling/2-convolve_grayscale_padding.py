#!/usr/bin/env python3

"""
This module has the method that performs a covolution of
convolution on grayscale images with padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    convolution with padding
    """
    # m, h, w = images.shape
    # kh, kw = kernel.shape
    # ph, pw = padding
    # # Calculate the output size with padding
    # out_h = h + 2 * ph - kh + 1
    # out_w = w + 2 * pw - kw + 1

    # padded_images = np.pad(
    #     images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    # print(padded_images.shape)

    # # initialize convolved image
    # convolved_images = np.zeros((m, out_h, out_w))

    # for i in range(m):
    #     for j in range(out_h):
    #         for k in range(out_w):
    #             # Extract a patch from the image
    #             patch = padded_images[i, j:j + kh, k:k + kw]
    #             convolved_images[i, j, k] = np.sum(patch * kernel)
    #             print(convolved_images)

    # return convolved_images
    kh, kw = kernel.shape
    m, hm, wm = images.shape
    ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    ch = hm + (2 * ph) - kh + 1
    cw = wm + (2 * pw) - kw + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            square = padded[:, h: h + kh, w: w + kw]
            insert = np.sum(square * kernel, axis=1).sum(axis=1)
            convoluted[:, h, w] = insert
    return convoluted
