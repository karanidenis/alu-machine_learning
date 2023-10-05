#!/usr/bin/env python3

"""
This module has the method that performs
convolution on grayscale images with channels
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    # stride
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    elif isinstance(padding, tuple):
        ph, pw = padding
    else:
        ph, pw = 0, 0

    # output
    out_h = int((h - kh + 2 * ph) / sh) + 1
    out_w = int((w - kw + 2 * pw) / sw) + 1

    # initialise convolved images
    convolved_images = np.zeros((m, out_h, out_w))

    # Perform the convolution
    for i in range(out_h):
        for j in range(out_w):
            for k in range(m):
                for ch in range(c):
                    # Extract a patch from the image
                    patch = images[k, i * sh:i *
                                   sh + kh, j * sw:j * sw + kw, :]
                    convolved_images[k, i,
                                     j] += np.sum(patch * kernel[:, :, ch])

    return convolved_images
