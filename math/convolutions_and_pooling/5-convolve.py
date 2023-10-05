#!/usr/bin/env python3

"""
This module has the method that performs
convolution on grayscale images
with multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Convolution with multiple kernels
    c - no. of channels
    nc - no. of kernels/ filters
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    # stride
    sh, sw = stride

    if padding == 'same':
        ph = max(0, (h - 1) * stride[0] + kh - h)
        pw = max(0, (h-1) * stride[1] + kw - w)
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple):
        ph, pw = padding

    # Calculate the output dimensions
    oh = int((h - kh + 2 * ph) / sh) + 1
    ow = int((w - kw + 2 * pw) / sw) + 1

    # Pad the input images
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Initialize the output tensor
    convolved_images = np.zeros((m, oh, ow, nc))

    # Perform the convolution
    for i in range(oh):
        for j in range(ow):
            for k in range(nc):
                # Extract a patch from the image
                patch = images_padded[:, i * sh:i *
                                      sh + kh, j * sw:j * sw + kw, :]
                # Convolve using tensordot to handle multiple channels
                convolved_images[:, i, j, k] = np.tensordot(
                    patch, kernels[:, :, :, k], axes=([1, 2, 3], [0, 1, 2]))

    return convolved_images
